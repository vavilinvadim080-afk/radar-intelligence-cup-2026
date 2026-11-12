"""Separate v6 candidate; the published v5c experiment remains unchanged."""
import math
from .selection_v5c import CandidateFeatures, CandidateSelector, CandidateReplyPolicy, pair_inputs, transform_pair_inputs
from .reply_policy import ReplyPolicy, FIELDS, _reject_gold
from .semantics_v6 import compatible_direction, authority_override


class V6Features(CandidateFeatures):
    def __init__(self, store, heads):
        super().__init__(store,heads)
        self._pairs, self._kinds = {}, {}

    def kind_scores(self, text):
        if text not in self._kinds:
            self._kinds[text] = super().kind_scores(text)
        return self._kinds[text]

    def pair_score(self, a, b):
        key = frozenset((a,b))
        if key in self._pairs:
            return self._pairs[key]
        score = None
        if compatible_direction(a,b):
            values = pair_inputs(self.store,a,b)
            if values is not None:
                head = self.heads['pair']
                values = transform_pair_inputs(values,head.get('feature_transform','identity'))
                if len(values) != len(head['coefficients']):
                    raise ValueError('Pair head dimensionality mismatch')
                z = sum(v*w for v,w in zip(values,head['coefficients'])) + head['intercept']
                score = 1/(1+math.exp(-max(-700,min(700,z))))
        self._pairs[key] = score
        return score


class V6Selector(CandidateSelector):
    mode = 'local-qwen-v6-diagnostic-experiment'

    def predict(self, rows):
        if self.config.get('cluster_policy')=='history':
            from .clustering_v6 import predict_history
            return predict_history(self,rows)
        return super().predict(rows)

    def context_allows_kind(self, text, predicted, target):
        if predicted not in {'unknown',target}:
            values = self.features.kind_scores(text)
            ordered = sorted(values.values(),reverse=True)
            if (values[predicted] >= self.config.get('protect_kind_score',.75)
                    and ordered[0]-ordered[1] >= self.config.get('protect_kind_margin',.2)):
                return False
        return super().context_allows_kind(text,predicted,target)


class V6ReplyPolicy(CandidateReplyPolicy):
    def __init__(self, features, *, config):
        if hasattr(features,'for_reply'):
            features = features.for_reply()
        self.features = features
        self.selector = V6Selector(features,backend='qwen',config=config)
        self.policy = ReplyPolicy(self.same_topic)

    def same_topic(self, payload):
        refs,text = payload['references'],payload['query']['text']
        if any(ref.get('kind')!=payload['kind'] for ref in refs):
            return False
        kind,_ = self.selector.choose_kind(text)
        if kind=='unknown':
            return None
        if (not self.selector.context_allows_kind(text,kind,payload['kind'])
                or any(not compatible_direction(text,ref['text']) for ref in refs)):
            return False
        return self.selector.score_refs(text,[ref['text'] for ref in refs[:3]]) is not None

    def decide(self, payload):
        _reject_gold(payload)
        if set(payload)-FIELDS:
            raise ValueError('Unknown policy fields')
        query = payload.get('query',{})
        if query.get('id') not in payload.get('seen_message_ids',[]) and authority_override(query.get('text','')):
            return dict(action='operator',reason='попытка изменить правила допуска',same_topic=None,eligible=False)
        return super().decide(payload)
