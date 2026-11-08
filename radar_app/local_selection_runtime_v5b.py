"""Optional online adapter for the frozen local experiment; no external calls.

One model process at a time. Hash caches are owned by BackendClient. A failed
model is surfaced to the caller; it is never silently replaced with rules.
"""
import json
from pathlib import Path
import threading

import numpy as np

from .calibrated_v5b import CalibratedFeatures,CalibratedSelector,CalibratedReplyPolicy,key
from .selection_v5b import RISK_HYPOTHESES
from .semantic_v5b import BackendClient

ROOT=Path(__file__).resolve().parents[1]


class LiveStore:
    def __init__(self,backend,client_factory):
        if backend not in {'e5','qwen'}:raise ValueError('A pinned semantic encoder is required')
        self.backend=backend;self.client_factory=client_factory;self.data={};self.vectors={backend:{}}
        self.relations={};self.risks={}

    def prepare(self,text,references):
        texts=list(dict.fromkeys([text]+list(references)))
        if len(texts)>301:raise ValueError('Catalog exceeds the bounded experiment')
        self.vectors[self.backend]={};self.relations={};self.risks={}
        with self.client_factory(self.backend) as client:
            for start in range(0,len(texts),8):
                chunk=texts[start:start+8]
                self.vectors[self.backend].update((key(t),np.asarray(v)) for t,v in zip(chunk,client.embed(chunk)))
        # Encoder is closed before NLI starts, including on exceptions.
        pairs=[];dest=[]
        for ref in texts[1:]:
            pairs.extend([(text,ref),(ref,text)]);dest.extend([(text,ref),(ref,text)])
        for risk,hypothesis in RISK_HYPOTHESES.items():
            pairs.append((text,hypothesis));dest.append(risk)
        with self.client_factory('nli') as client:
            for start in range(0,len(pairs),16):
                values=client.nli(pairs[start:start+16])
                for target,value in zip(dest[start:start+16],values):
                    if isinstance(target,str):self.risks[target]=value['entailment']
                    else:self.relations[target]=value

    def cosine(self,backend,a,b):return float(self.vectors[backend][key(a)]@self.vectors[backend][key(b)])
    def relation(self,a,b):
        if a==b:return dict(entailment=1.,neutral=0.,contradiction=0.)
        return self.relations[(a,b)]
    def risk_scores(self,text):return self.risks


class LiveAnalyzer:
    mode='local-calibrated-v5b-experiment'
    minimum_confidence=0
    supports_examples=True

    def __init__(self,*,backend,heads_path,config,client_factory=BackendClient):
        self.backend=backend;self.config=config;self.heads_path=heads_path
        self.store=LiveStore(backend,client_factory);self.lock=threading.RLock()

    def __call__(self,text,catalog):
        with self.lock:
            catalog=catalog[:100]
            refs=[ref for c in catalog for ref in (c.get('examples') or [c.get('quote') or c['label']])[:3]]
            self.store.prepare(text,refs)
            features=CalibratedFeatures(self.store,self.heads_path).for_backend(self.backend)
            analyzer=CalibratedSelector(features,backend=self.backend,config=self.config)
            return analyzer.analyze({'text':text},catalog)

    def decide_reply(self,payload,*,config):
        from .reply_policy import _reject_gold
        from .privacy.mask import mask_text
        _reject_gold(payload)
        p=dict(payload);p['query']=dict(p['query'],text=mask_text(p['query']['text'],ner='auto',ctx_names=True)[0])
        p['references']=[dict(r,text=mask_text(r['text'],ner='auto',ctx_names=True)[0]) for r in p.get('references',[])]
        with self.lock:
            self.store.prepare(p['query']['text'],[r['text'] for r in p['references'][:3]])
            features=CalibratedFeatures(self.store,self.heads_path).for_backend(self.backend)
            return CalibratedReplyPolicy(features,backend=self.backend,config=config).decide(p)


def experimental_analyzer():
    import os
    if os.environ.get('RADAR_ALLOW_EXPERIMENTAL')!='1':raise ValueError('Explicit experimental opt-in is required')
    config=json.loads((ROOT/'data/semantic-config-v5b.json').read_text());backend=config['selected_backend']
    if backend not in {'e5','qwen'}:raise ValueError('The selected experiment has no semantic encoder')
    if not (ROOT/'data/quality-v5b.json').exists():raise ValueError('Independent control has not completed')
    return LiveAnalyzer(backend=backend,heads_path=ROOT/'data/semantic-heads-v5b.json',config=config['backends'][backend]['needs'])
