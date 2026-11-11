"""Candidate equivalence verifier. Experimental; never bypasses permission gates."""
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np

from .semantic import normalize, intent_direction
from .calibrated_v5b import probabilities
from .selection_v5b import NeedsSelector, NliReplyPolicy, KIND_HYPOTHESES


def text_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def relation_key(a, b):
    return hashlib.sha256(json.dumps([a,b], ensure_ascii=False).encode()).hexdigest()


def requested_direction(text):
    s = normalize(text)
    if re.search(r'не (?:отключ|выключ|отмен|прекращ)', s):
        return 'enable'
    if re.search(r'не (?:получа|присыла|отправля|показыва|включ)|отключ|выключ|отпис|прекрат|перестан', s):
        return 'disable'
    if re.search(r'(?:хочу|нужн|начн|начни|прошу|позвол).{0,30}(?:получа|присыла|отправля|уведом|письм)|включ|возобнов', s):
        return 'enable'
    return intent_direction(s)


def functionality_state(text):
    s = normalize(text)
    # Negated failure is not evidence of a broken function. Keep separate
    # positive clauses, e.g. "вход не сломан, но экспорт перестал работать".
    failure_text = re.sub(r'\bне\s+слома[а-яё]*\b', ' ', s)
    # A past working function followed by a failure is distinct from a missing feature.
    if re.search(r'раньше.{0,35}работ|перестал|упал.{0,20}ошиб|после обновлен.{0,45}не |слома', failure_text):
        return 'broken'
    if re.search(r'такой (?:функции|возможности|опции|настройки).{0,15}нет|пока (?:нет|отсутств)|котор[а-я]+ сейчас нет|впервые (?:добав|сдел)', s):
        return 'missing'
    return None


def compatible_direction(a, b):
    x, y = requested_direction(a), requested_direction(b)
    if x is not None and y is not None and x != y:
        return False
    x, y = functionality_state(a), functionality_state(b)
    return x is None or y is None or x == y


def pair_inputs(store, a, b):
    values = store.data.get('reranker', {})
    forward, reverse = values.get(relation_key(a,b)), values.get(relation_key(b,a))
    if forward is None or reverse is None:
        return None
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in (forward,reverse)):
        raise ValueError('Invalid verifier evidence')
    f, r = store.relation(a,b), store.relation(b,a)
    cosine = store.cosine('qwen',a,b)
    features = [cosine, min(forward,reverse), max(forward,reverse),
                min(f['entailment'],r['entailment']),
                max(f['contradiction'],r['contradiction'])]
    if not all(math.isfinite(v) for v in features):
        raise ValueError('Nonfinite pair features')
    return features


def kind_inputs(store, text):
    vector = store.vectors['qwen'][text_key(text)]
    scores = store.kind_scores(text)
    if set(scores) != set(KIND_HYPOTHESES):
        raise ValueError('Incomplete kind evidence')
    return np.concatenate([vector, [scores[k] for k in KIND_HYPOTHESES]])


def transform_pair_inputs(values, transform='identity'):
    values = list(values)
    if transform == 'identity':
        return values
    if transform == 'poly2':
        return values+[a*b for i,a in enumerate(values) for b in values[i:]]
    raise ValueError('Unknown pair feature transform')


class CandidateFeatures:
    def __init__(self, store, heads):
        self.store = store
        self.data = store.data
        self.vectors = store.vectors
        self.heads = json.loads(Path(heads).read_text()) if isinstance(heads,(str,Path)) else heads

    def kind_scores(self, text):
        return probabilities(self.heads['kind'], kind_inputs(self.store,text))

    def risk_scores(self, text):
        return self.store.risk_scores(text)

    def cosine(self, backend, a, b):
        return self.store.cosine(backend,a,b)

    def relation(self, a, b):
        return self.store.relation(a,b)

    def pair_score(self, a, b):
        if not compatible_direction(a,b):
            return None
        values = pair_inputs(self.store,a,b)
        if values is None:
            return None
        head = self.heads['pair']
        values = transform_pair_inputs(values, head.get('feature_transform','identity'))
        if len(values) != len(head['coefficients']):
            raise ValueError('Pair head dimensionality mismatch')
        z = sum(v*w for v,w in zip(values,head['coefficients'])) + head['intercept']
        return 1/(1+math.exp(-max(-700,min(700,z))))

    def for_reply(self):
        if 'reply_pair' not in self.heads:
            return self
        return type(self)(self.store,dict(self.heads,pair=self.heads['reply_pair']))


class CandidateSelector(NeedsSelector):
    mode = 'local-qwen-verifier-v5c-experiment'

    def predict(self, rows):
        if self.config.get('cluster_policy') == 'history':
            from .clustering_v5c import predict_history
            return predict_history(self, rows)
        return super().predict(rows)

    def context_allows_kind(self, text, predicted, target):
        if target == 'unknown':
            return False
        if predicted == 'unknown':
            scores = self.features.kind_scores(text)
            return (scores['unknown'] <= self.config.get('context_unknown_max', 0.)
                    and scores.get(target, 0.) >= self.config.get('context_kind_min', .1))
        if predicted == target:
            return True
        slack = self.config.get('context_kind_margin', 0.)
        if slack <= 0:
            return False
        scores = self.features.kind_scores(text)
        return (scores.get(target, 0.) >= self.config.get('context_kind_min', .1)
                and scores[predicted]-scores[target] <= slack)

    def choose_kind(self, text):
        scores = self.features.kind_scores(text)
        ordered = sorted(scores.items(),key=lambda item:item[1],reverse=True)
        if not ordered or set(scores) != set(KIND_HYPOTHESES) or any(
                isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0 <= v <= 1
                for v in scores.values()):
            raise ValueError('Invalid kind classifier output')
        kind, score = ordered[0]
        if score < self.config['kind_min'] or score-ordered[1][1] < self.config['kind_margin']:
            return 'unknown', score
        return kind, score

    def analyze(self, row, catalog):
        if any(k in row for k in ['gold','expected','gold_kind','gold_topic']):
            raise ValueError('Gold cannot enter selection')
        text = row['text']
        kind, confidence = self.choose_kind(text)
        initial_kind = kind
        matches = []
        if kind != 'unknown' or self.config.get('context_unknown_max', 0.) > 0:
            for topic in catalog[:100]:
                if not self.context_allows_kind(text, kind, topic['kind']):
                    continue
                got = self.score_refs(text,(topic.get('examples') or [topic['quote']])[:3])
                if got and kind == 'unknown' and got[0] < self.config.get('unknown_pair_min', 1.):
                    got = None
                if got:
                    matches.append((got[0],got[1],topic))
        matches.sort(key=lambda match:(match[0],match[1]),reverse=True)
        if len(matches) > 1 and matches[0][0]-matches[1][0] < self.config['margin']:
            kind = 'unknown'
        elif matches:
            kind = matches[0][2]['kind']
        source = row.get('source_id') or row['id']
        topic_key = 'v5c-'+hashlib.sha256((row['product']+':'+kind+':'+source).encode()).hexdigest()[:20]
        label = text[:120]
        if kind == 'unknown':
            topic_key, label = 'unknown','Нужен разбор'
        elif matches:
            topic_key, label = matches[0][2]['topic_key'],matches[0][2]['label']
            kind = matches[0][2]['kind']
            confidence = self.features.kind_scores(text)[kind]
        self.trace = dict(kind=kind,initial_kind=initial_kind,kind_score=confidence,matches=[
            dict(match_score=a,support_score=b,topic=t['topic_key']) for a,b,t in matches[:3]])
        return dict(topic_key=topic_key,label=label,kind=kind,quote=text[:300],confidence=confidence)

    def score_refs(self, text, examples):
        examples = list(dict.fromkeys(examples))
        ranked = sorted([(self.features.cosine('qwen',text,x),x) for x in examples],reverse=True)
        if not ranked or ranked[0][0] < self.config['cosine_min']:
            return None
        policy = self.config.get('reference_policy', 'all')
        scores = [self.features.pair_score(text,ref) for _,ref in ranked[:3 if policy == 'best' else 2]]
        if policy == 'best':
            available = [score for score in scores if score is not None]
            if not available:
                return None
            score = max(available)
        elif policy == 'all':
            if any(score is None for score in scores):
                return None
            score = min(scores)
        else:
            raise ValueError('Unknown reference evidence policy')
        if score < self.config['pair_min']:
            return None
        return score, ranked[0][0]


class CandidateReplyPolicy(NliReplyPolicy):
    def __init__(self, features, *, config):
        from .reply_policy import ReplyPolicy
        if hasattr(features,'for_reply'):
            features = features.for_reply()
        self.selector = CandidateSelector(features,backend='qwen',config=config)
        self.features = features
        self.policy = ReplyPolicy(self.same_topic)

    def same_topic(self, payload):
        refs, text = payload['references'],payload['query']['text']
        if any(ref.get('kind') != payload['kind'] for ref in refs):
            return False
        kind,_ = self.selector.choose_kind(text)
        if kind == 'unknown':
            return None
        if (not self.selector.context_allows_kind(text,kind,payload['kind'])
                or any(not compatible_direction(text,ref['text']) for ref in refs)):
            return False
        return self.selector.score_refs(text,[ref['text'] for ref in refs[:3]]) is not None
