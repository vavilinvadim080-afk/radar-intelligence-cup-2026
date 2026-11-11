"""Opt-in streaming adapter; batch-history benchmark is not an online quality claim.

Only eight retrieved references reach the verifier. All three workers run
sequentially; missing models/evidence never become a rules-based success.
"""
import hashlib
import json
from pathlib import Path
import threading
import numpy as np
from .selection_v5c import CandidateFeatures,CandidateSelector,CandidateReplyPolicy,relation_key,text_key
from .selection_v5b import KIND_HYPOTHESES,RISK_HYPOTHESES
from .semantic_v5b import BackendClient
from .reranker_v5c import RerankerClient

ROOT=Path(__file__).resolve().parents[1]


class LiveStore:
    def __init__(self,client_factory,reranker_factory):
        self.client_factory=client_factory;self.reranker_factory=reranker_factory
        self.data={'reranker':{}};self.vectors={'qwen':{}}
        self.relations={};self.kinds={};self.risks={}

    def prepare(self,text,references):
        reference_texts=list(dict.fromkeys(references))
        texts=list(dict.fromkeys([text]+reference_texts))
        if len(texts)>301 or any(not isinstance(t,str) or not t.strip() or len(t)>4000 for t in texts):
            raise ValueError('Experimental catalog exceeds text bounds')
        self.data={'reranker':{}};self.vectors={'qwen':{}};self.relations={};self.kinds={};self.risks={}
        with self.client_factory('qwen') as client:
            for start in range(0,len(texts),8):
                chunk=texts[start:start+8];vectors=client.embed(chunk)
                if len(vectors)!=len(chunk):raise ValueError('Incomplete embedding batch')
                for t,v in zip(chunk,vectors):
                    vector=np.asarray(v,dtype=float)
                    if vector.shape!=(1024,) or not np.isfinite(vector).all():raise ValueError('Invalid live vector')
                    self.vectors['qwen'][text_key(t)]=vector
        refs=sorted(reference_texts,key=lambda ref:(-self.cosine('qwen',text,ref),ref))[:8]
        pairs=list(dict.fromkeys(pair for ref in refs for pair in ((text,ref),(ref,text))))
        requests=list(pairs);dest=[('pair',a,b) for a,b in pairs]
        for kind,hypothesis in KIND_HYPOTHESES.items():
            requests.append((text,hypothesis));dest.append(('kind',kind))
        for risk,hypothesis in RISK_HYPOTHESES.items():
            requests.append((text,hypothesis));dest.append(('risk',risk))
        with self.client_factory('nli') as client:
            for start in range(0,len(requests),16):
                chunk=requests[start:start+16];values=client.nli(chunk)
                if len(values)!=len(chunk):raise ValueError('Incomplete NLI batch')
                for target,value in zip(dest[start:start+16],values):
                    if target[0]=='kind':self.kinds[target[1]]=value['entailment']
                    elif target[0]=='risk':self.risks[target[1]]=value['entailment']
                    else:self.relations[target[1:]]=value
        if pairs:
            with self.reranker_factory() as client:
                for start in range(0,len(pairs),4):
                    chunk=pairs[start:start+4];values=client.rerank(chunk)
                    if len(values)!=len(chunk):raise ValueError('Incomplete verifier batch')
                    self.data['reranker'].update((relation_key(a,b),value) for (a,b),value in zip(chunk,values))

    def cosine(self,backend,a,b):
        if backend!='qwen':raise ValueError('Pinned Qwen encoder required')
        return float(self.vectors[backend][text_key(a)]@self.vectors[backend][text_key(b)])

    def relation(self,a,b):return self.relations[(a,b)]
    def kind_scores(self,text):return self.kinds
    def risk_scores(self,text):return self.risks


class LiveAnalyzer:
    mode='local-verifier-v5c-experiment'
    minimum_confidence=0
    supports_examples=True

    def __init__(self,heads,config,quality,*,client_factory=None,reranker_factory=None):
        self.heads=heads;self.config=config;self.quality=quality;self.lock=threading.RLock()
        self.store=LiveStore(client_factory or (lambda name:BackendClient(name,cache_dir=ROOT/'runtime/v5c-online'/name)),
                             reranker_factory or (lambda:RerankerClient(cache_dir=ROOT/'runtime/v5c-online/reranker')))

    def __call__(self,text,catalog):
        catalog=catalog[:100]
        refs=[ref for topic in catalog for ref in (topic.get('examples') or [topic.get('quote') or topic['label']])[:3]]
        with self.lock:
            self.store.prepare(text,refs)
            features=CandidateFeatures(self.store,self.heads)
            selector=CandidateSelector(features,backend='qwen',config=self.config['needs'])
            ident=hashlib.sha256(text.encode()).hexdigest()
            return selector.analyze(dict(id=ident,source_id=ident,product='scoped-live',text=text),catalog)

    def decide_reply(self,payload,*,config):
        from .reply_policy import _reject_gold
        from .privacy.mask import mask_text
        _reject_gold(payload)
        gate=self.quality['models']['candidate']['gates']
        if not gate['target_80_met'] or not gate['zero_false_positive_replies']:
            return dict(action='operator',eligible=False,same_topic=None,reason='Эксперимент не прошёл допуск автоответа')
        p=dict(payload);p['query']=dict(p['query'],text=mask_text(p['query']['text'],ner='auto',ctx_names=True)[0])
        p['references']=[dict(r,text=mask_text(r['text'],ner='auto',ctx_names=True)[0]) for r in p.get('references',[])]
        with self.lock:
            self.store.prepare(p['query']['text'],[r['text'] for r in p['references'][:3]])
            return CandidateReplyPolicy(CandidateFeatures(self.store,self.heads),config=config).decide(p)


def experimental_analyzer():
    import os
    if os.environ.get('RADAR_ALLOW_EXPERIMENTAL')!='1':raise ValueError('Explicit experimental opt-in required')
    quality=ROOT/'data/quality-v5c.json';freeze=ROOT/'reports/quality-v5c/CONFIG-FREEZE.json'
    if not quality.is_file() or not freeze.is_file():raise ValueError('Independent control is required')
    q=json.loads(quality.read_text());f=json.loads(freeze.read_text())
    if q['provenance']['freeze_sha256']!=hashlib.sha256(freeze.read_bytes()).hexdigest():raise ValueError('Control freeze differs')
    for name in ('data/semantic-heads-v5c.json','data/semantic-config-v5c.json','radar_app/selection_v5c.py',
                 'scripts/reranker_worker_v5c.py','scripts/reranker_worker_mlx_v5c.py'):
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=f['files'][name]:raise ValueError('Frozen runtime source differs')
    return LiveAnalyzer(ROOT/'data/semantic-heads-v5c.json',json.loads((ROOT/'data/semantic-config-v5c.json').read_text()),q)
