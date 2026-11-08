"""Optional local semantic prototype: kind, bounded retrieval, abstention.

The corporate gateway is separate. No topic inventory is trained: each new
meaning starts a catalog entry and later paraphrases can join that entry.
"""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(join_cosine=.55, join_margin=.05, reranker_min=-4., use_reranker=True,
                content_weight=.65)


def normalize(text):
    return ' '.join(text.lower().replace('ё', 'е').split())


def intent_direction(text):
    s = normalize(text)
    if re.search(r'не (?:отключ|выключ|отмен|прекращ)', s):
        return 'enable'
    if re.search(r'не (?:включ|присыла|отправля)|отключ|выключ|отпис|перестан|прекрат|прекращ', s):
        return 'disable'
    if re.search(r'включ|подписк|начните|начина|возобнов', s):
        return 'enable'
    return None


def compatible_intent(a, b):
    x, y = intent_direction(a), intent_direction(b)
    return x is None or y is None or x == y


OBJECT_CUES={
    'numeric-summary':r'итог|суммир|сумм[ауы]|количеств|складыва|общий объем',
    'list-filter':r'фильтр|отбор|условие поиска|выбир.{0,20}(?:выполн|заверш)',
    'upload':r'загруз|прикреп|вложени',
    'download':r'выгруз|выгруск|экспорт|скач|сохран.{0,25}(?:файл|спис|переч|истори)',
    'authentication':r'авторизац|вход|войти|парол|логин',
    'notification':r'уведом|оповещ|подписк|отписк|письма',
}


def semantic_objects(text):
    s=normalize(text)
    return {key for key,pattern in OBJECT_CUES.items() if re.search(pattern,s)}


def compatible_object(a,b):
    x,y=semantic_objects(a),semantic_objects(b)
    return not x or not y or bool(x&y)


def kind_cue(text):
    s = normalize(text)
    if re.search(r'(?:отдельн|нов[а-я]*|самостоятельн)[^.!?]{0,45}(?:сервис|услуг|продукт|площадк)', s):
        return 'service'
    if re.search(r'не (?:ошибк|поломк)[^.!?]{0,35}новый продукт', s):
        return 'service'
    if re.search(r'(?:кнопк|функц|значк|пол[яе]|экспорт|выгрузк|действи)[^.!?]{0,25}исправн|работает (?:верно|правильно)|работают (?:верно|правильно)|(?:ошибки|пропажи|ошибок) нет|не сломан|не исправление', s):
        return 'usability'
    if re.search(r'(?:работает|работают)[^.!?]{0,20}но[^.!?]{0,35}(?:найти|чит|непонят|трудно|знач)', s):
        return 'usability'
    if intent_direction(s) and re.search(r'уведом|оповещ|письм|сообщ|событ',s):return 'wish'
    if re.search(r'раньше|после (?:релиза|обновления)|перестал[аио]?\b|слома[лн]|ошибк[аиу]? (?:500|404)|зацикли|не (?:приход|начисл|отображ)', s):
        return 'bug'
    if re.search(r'(?:добав|ввести|расширить|включить|выключить|подписк|отписк|новый способ|новая команда|новую возможность|такой (?:функции|опции|кнопки).{0,15}нет)', s):
        return 'wish'
    return None


BOILERPLATE=set('я мы нам мне у в во на к по с со из от для и а но или что это этого так как чтобы при то все всё только уже ещё еще пока сейчас раньше теперь после новый новая новое новые нового новую новой новым нужен нужна нужно нужны хочу хотим хочется хотели предлагаю предлагаем пожалуйста прошу просьба можно может добавьте сделать такой такая такую такого такой такой отдельный отдельная отдельную отдельного самостоятельный самостоятельная самостоятельную самостоятельного существующий существующая существующей существующую штатный штатная штатного штатная сервис сервиса услуга услугу услуги продукт продукта функция функции функцию возможность возможности возможность проблема проблемы нет есть работает работало работала работали работал работает работают правильно верно исправно неверно исправны исправна исправный исправление исправлять случай такой это бы быть был была было были более менее не ни о об без перед который которая котором которых хотя просто нужен сделать делать надо фаил файл файлом файла файлa'.split())


def content_text(text):
    words=re.findall(r'[\w<>]+',normalize(text))
    content=' '.join(w for w in words if w not in BOILERPLATE and len(w)>2)
    return content if len(content)>8 else text


def pair_features(a,b,ca,cb):
    return ([abs(x-y) for x,y in zip(a,b)]+[x*y for x,y in zip(a,b)]+
            [abs(x-y) for x,y in zip(ca,cb)]+[x*y for x,y in zip(ca,cb)]+
            [sum(x*y for x,y in zip(a,b)),sum(x*y for x,y in zip(ca,cb))])


class PairHead:
    def __init__(self,path=None):
        self.data=json.loads(Path(path or ROOT/'data/semantic-pair-v5.json').read_text())

    def __call__(self,a,b,ca,cb):
        f=pair_features(a,b,ca,cb)
        if len(f)!=len(self.data['coefficients']):raise ValueError('Pair feature dimensions differ')
        z=self.data['intercept']+sum(x*w for x,w in zip(f,self.data['coefficients']))
        return 1/(1+math.exp(-max(-700,min(700,z))))


class KindHead:
    def __init__(self, path=None):
        self.data = json.loads(Path(path or ROOT/'data/semantic-kind-v5.json').read_text())

    def __call__(self, text, vector):
        d = self.data
        if len(vector) != d['dimension']:
            raise ValueError('Encoder and kind head dimensions differ')
        logits = [b + sum(w*x for w,x in zip(weights, vector))
                  for b,weights in zip(d['intercept'], d['coefficients'])]
        exp = [math.exp(v-max(logits)) for v in logits]
        probabilities = [v/sum(exp) for v in exp]
        best = max(range(len(exp)), key=exp.__getitem__)
        cue = kind_cue(text)
        # Explicit declared functionality state has precedence over statistical guess.
        chosen = cue or d['classes'][best]
        return chosen, probabilities[d['classes'].index(chosen)]


class SemanticAnalyzer:
    mode = 'local-semantic-prototype'
    minimum_confidence = 0
    supports_examples = True

    def __init__(self, models, *, kind_predictor=None, config=None):
        self.models = models
        self.kind_predictor = kind_predictor or KindHead()
        self.config = dict(DEFAULTS, **(config or {}))
        self.pair_head=PairHead() if self.config.get('use_pair_head') else None
        self.last_trace = {}

    @staticmethod
    def result(text, kind, confidence, key=None, label=None):
        return dict(topic_key=key or ('unknown' if kind=='unknown' else
                    'sem-'+hashlib.sha256((kind+':'+normalize(text)).encode()).hexdigest()[:20]),
                    label=label or ('Нужен разбор' if kind=='unknown' else text[:120]),
                    kind=kind, quote=text[:300], confidence=max(0., min(1., confidence)))

    def __call__(self, text, catalog):
        vector = self.models.embed([text])[0]
        kind, confidence = self.kind_predictor(text, vector)
        self.last_trace = dict(kind=kind, kind_score=confidence,
                               score_description='uncalibrated kind score; not probability of correctness')
        if kind == 'unknown':
            return self.result(text, 'unknown', confidence)
        candidates = []
        weight=self.config['content_weight']
        content_vector=self.models.embed([content_text(text)])[0] if weight else vector
        for topic in catalog[:100]:
            if topic['kind'] != kind:
                continue
            examples = list(dict.fromkeys(topic.get('examples') or [topic['label']]))[:3]
            examples = [x for x in examples if compatible_intent(text, x) and compatible_object(text,x)]
            if not examples:
                continue
            embeddings = self.models.embed(examples)
            contents=self.models.embed([content_text(x) for x in examples]) if weight else embeddings
            scores = [self.pair_head(vector,v,content_vector,c) if self.pair_head else
                       (1-weight)*sum(a*b for a,b in zip(vector,v))+weight*sum(a*b for a,b in zip(content_vector,c))
                       for v,c in zip(embeddings,contents)]
            i = max(range(len(scores)), key=scores.__getitem__)
            candidates.append((scores[i], topic, examples[i]))
        candidates.sort(key=lambda x: x[0], reverse=True)
        self.last_trace['nearest'] = [dict(topic_key=t['topic_key'], match_score=s,
                                        score_type='uncalibrated_pair' if self.pair_head else 'mixed_cosine') for s,t,_ in candidates[:3]]
        if not candidates or candidates[0][0] < self.config['join_cosine']:
            return self.result(text, kind, confidence)
        best, topic, example = candidates[0]
        if len(candidates)>1 and best-candidates[1][0] < self.config['join_margin']:
            self.last_trace['abstention'] = 'ambiguous_neighbors'
            return self.result(text, 'unknown', confidence)
        if self.config['use_reranker']:
            relevance = self.models.rerank([(text, example)])[0]
            self.last_trace['reranker_logit'] = relevance
            if relevance < self.config['reranker_min']:
                return self.result(text, kind, confidence)
        return self.result(text, kind, confidence, topic['topic_key'], topic['label'])


class SemanticReplyMatcher:
    def __init__(self,models,*,threshold=.6):
        self.models=models;self.kind=KindHead();self.head=PairHead();self.threshold=threshold
        self.last_trace={}

    def __call__(self,p):
        self.last_trace={}
        refs=p.get('references',[]);query=p.get('query',{})
        if not refs or not query.get('text','').strip():return None
        if query.get('product')!=p.get('product') or any(r.get('product')!=p.get('product') for r in refs):return False
        from .privacy.mask import mask_text
        texts=[mask_text(t,ner='auto',ctx_names=True)[0] for t in [query['text']]+[r['text'] for r in refs[:3]]]
        vectors=self.models.embed(texts)
        kind,confidence=self.kind(texts[0],vectors[0])
        if kind=='unknown':return None
        if kind!=p.get('kind') or any(r.get('kind')!=p.get('kind') for r in refs):return False
        if any(not compatible_intent(texts[0],t) for t in texts[1:]):return False
        if any(not compatible_object(texts[0],t) for t in texts[1:]):return False
        cv=self.models.embed([content_text(t) for t in texts])
        scores=[self.head(vectors[0],v,cv[0],c) for v,c in zip(vectors[1:],cv[1:])]
        score=max(scores)
        self.last_trace=dict(kind=kind,kind_score=confidence,pair_score=score,
                             threshold=self.threshold,score_type='uncalibrated_pair')
        return score>=self.threshold
