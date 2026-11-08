"""Bounded comparison of embeddings plus bidirectional NLI; no topic inventory."""
import hashlib
import math

from .semantic import kind_cue,compatible_intent,compatible_object
from .semantic_v5b import equivalence
from .reply_policy import ReplyPolicy

KIND_HYPOTHESES={
    'wish':'Автор просит добавить новую, пока отсутствующую возможность существующего продукта.',
    'bug':'Автор сообщает об ошибке или поломке функции, которая уже существует и должна работать.',
    'usability':'Работающая функция неудобна или непонятна; автор предлагает улучшить её удобство.',
    'service':'Автор предлагает самостоятельную новую услугу или новый продукт, а не изменение существующей функции.',
    'unknown':'Из сообщения нельзя понять, какая конкретная функция или проблема имеется в виду.',
}
RISK_HYPOTHESES={
    'money':'Автор просит выполнить индивидуальный возврат, перевод или списание денег.',
    'credentials':'Автор просит сообщить, заменить или сбросить пароль, секретный код или другие данные доступа.',
    'injection':'Автор даёт инструкции самой модели, просит игнорировать правила или выполнить неразрешённое действие.',
}
DEFAULTS=dict(cosine_min=.7,entailment_min=.4,max_contradiction=.15,margin=.02,
              kind_min=.2,kind_margin=.02,risk_min=.6)


def input_payload(row):
    return {k:row[k] for k in ['id','source_id','product','author_id','created_at','text'] if k in row}


def choose_kind(text,scores,config):
    ordered=sorted(scores.items(),key=lambda p:p[1],reverse=True)
    if not ordered:return 'unknown',0.
    if scores.get('unknown',0)>=.5:return 'unknown',scores['unknown']
    cue=kind_cue(text)
    if cue:return cue,scores.get(cue,0.)
    kind,score=ordered[0]
    if score<config['kind_min'] or (len(ordered)>1 and score-ordered[1][1]<config['kind_margin']):return 'unknown',score
    return kind,score


class NeedsSelector:
    mode='local-semantic-v5b-experiment'
    minimum_confidence=0
    supports_examples=True

    def __init__(self,features,*,backend,config=None):
        self.features=features;self.backend=backend;self.config=dict(DEFAULTS,**(config or {}));self.trace={}

    def score_refs(self,text,examples):
        examples=[x for x in dict.fromkeys(examples) if compatible_intent(text,x) and compatible_object(text,x)]
        if not examples:return None
        ranked=sorted([(self.features.cosine(self.backend,text,x),x) for x in examples],reverse=True)
        if ranked[0][0]<self.config['cosine_min']:return None
        scores=[]
        for _,ref in ranked[:2]:
            a=self.features.relation(text,ref);b=self.features.relation(ref,text)
            scores.append(equivalence(a,b,self.config['max_contradiction']))
        nli=min(scores)
        if nli<self.config['entailment_min']:return None
        return ranked[0][0],nli

    def analyze(self,row,catalog):
        if any(k in row for k in ['gold','expected','gold_kind','gold_topic']):raise ValueError('Gold cannot enter selection')
        text=row['text'];kind,score=choose_kind(text,self.features.kind_scores(text),self.config)
        matches=[]
        if kind!='unknown':
            for c in catalog[:100]:
                if c['kind']!=kind:continue
                got=self.score_refs(text,(c.get('examples') or [c.get('quote') or c['label']])[:3])
                if got:matches.append((got[0],got[1],c))
        matches.sort(key=lambda m:(m[0],m[1]),reverse=True)
        self.trace=dict(kind=kind,kind_score=score,matches=[dict(match_score=x,support_score=y,topic=z['topic_key']) for x,y,z in matches[:3]])
        if len(matches)>1 and matches[0][0]-matches[1][0]<self.config['margin']:kind='unknown'
        ident=row.get('source_id') or row.get('id') or hashlib.sha256(text.encode()).hexdigest()
        key='nli-'+hashlib.sha256((row.get('product','scope')+':'+kind+':'+ident).encode()).hexdigest()[:20]
        label=text[:120]
        if kind=='unknown':key='unknown';label='Нужен разбор'
        elif matches:key=matches[0][2]['topic_key'];label=matches[0][2]['label']
        return dict(topic_key=key,label=label,kind=kind,quote=text[:300],confidence=max(0.,min(1.,score)))

    def predict(self,rows):
        catalog={};seen=set();out=[]
        for r in sorted(rows,key=lambda r:(r.get('created_at',''),r['id'])):
            if any(k in r for k in ['gold','expected','gold_kind','gold_topic']):raise ValueError('Gold cannot enter selection')
            source=(r['product'],r.get('source_id',r['id']))
            if source in seen:continue
            seen.add(source)
            a=self.analyze(r,list(catalog.get(r['product'],{}).values()))
            pred=None if a['kind']=='unknown' else [r['product'],a['kind'],a['topic_key']]
            out.append(dict(id=r['id'],kind=a['kind'],prediction=pred,trace=dict(self.trace)))
            if pred:
                c=catalog.setdefault(r['product'],{}).setdefault(a['topic_key'],dict(a,examples=[]))
                if r['text'] not in c['examples'] and len(c['examples'])<3:c['examples'].append(r['text'])
        return out


class NliReplyPolicy:
    def __init__(self,features,*,backend,config=None):
        self.selector=NeedsSelector(features,backend=backend,config=config);self.features=features
        self.policy=ReplyPolicy(self.same_topic)

    def same_topic(self,p):
        refs=p['references'];text=p['query']['text']
        if any(r.get('kind')!=p['kind'] for r in refs):return False
        kind,_=choose_kind(text,self.features.kind_scores(text),self.selector.config)
        if kind=='unknown':return None
        if kind!=p['kind']:return False
        if any(not compatible_intent(text,r['text']) or not compatible_object(text,r['text']) for r in refs):return False
        return self.selector.score_refs(text,[r['text'] for r in refs[:3]]) is not None

    def decide(self,p):
        # Frozen permission gates remain in force; NLI adds a refusal, never an
        # authorization. It cannot turn a rejected payload into a permitted one.
        from .reply_policy import FIELDS,_reject_gold
        _reject_gold(p)
        if set(p)-FIELDS:raise ValueError('Unknown policy fields')
        if p.get('query',{}).get('id') in p.get('seen_message_ids',[]):
            return dict(action='ignore_duplicate',reason='повторная доставка',same_topic=None,eligible=False)
        if p.get('event') not in {'incoming_ticket','fixed_notice_candidate'}:
            return dict(action='operator',reason='неподдерживаемое событие',same_topic=None,eligible=False)
        text=p['query']['text']
        try:risks=self.features.risk_scores(text)
        except Exception:
            return dict(action='operator',reason='смысловая проверка недоступна',same_topic=None,eligible=False)
        if (not isinstance(risks,dict) or set(risks)!=set(RISK_HYPOTHESES) or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1
                for v in risks.values())):
            return dict(action='operator',reason='смысловая проверка риска неполна',same_topic=None,eligible=False)
        for key,score in risks.items():
            if score>=self.selector.config['risk_min']:
                return dict(action='operator',reason='индивидуальное действие или инструкция требует человека',
                            same_topic=None,eligible=False,risk=key)
        return self.policy.decide(p)
