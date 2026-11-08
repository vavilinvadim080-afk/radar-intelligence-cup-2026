"""Small supervised heads over fixed embeddings; all training is tune-only."""
import json
import hashlib
from pathlib import Path

import numpy as np

from .selection_v5b import NeedsSelector,NliReplyPolicy


def key(text):return hashlib.sha256(text.encode()).hexdigest()


def probabilities(head,vector):
    z=np.asarray(head['coefficients'])@vector+np.asarray(head['intercept'])
    exp=np.exp(z-z.max());p=exp/exp.sum()
    return dict(zip(head['classes'],p.tolist()))


def pair_vector(a,b,forward,reverse):
    return np.concatenate([np.abs(a-b),a*b,[a@b,
        min(forward['entailment'],reverse['entailment']),
        max(forward['contradiction'],reverse['contradiction'])]])


class CalibratedFeatures:
    def __init__(self,store,heads_path):
        self.store=store;self.data=store.data;self.vectors=store.vectors
        self.heads=json.loads(Path(heads_path).read_text());self.backend=None;self._pairs={}

    def for_backend(self,backend):
        self.backend=backend;return self

    def kind_scores(self,text):
        return probabilities(self.heads[self.backend]['kind'],self.vectors[self.backend][key(text)])

    def cosine(self,backend,a,b):return self.store.cosine(backend,a,b)
    def relation(self,a,b):return self.store.relation(a,b)
    def risk_scores(self,text):return self.store.risk_scores(text)

    def pair_score(self,backend,a,b):
        cache=(backend,*sorted([key(a),key(b)]))
        if cache not in self._pairs:
            v=pair_vector(self.vectors[backend][key(a)],self.vectors[backend][key(b)],
                          self.relation(a,b),self.relation(b,a))
            h=self.heads[backend]['pair'];z=float(np.asarray(h['coefficients'])@v+h['intercept'])
            self._pairs[cache]=float(1/(1+np.exp(-np.clip(z,-700,700))))
        return self._pairs[cache]


class CalibratedSelector(NeedsSelector):
    mode='local-calibrated-v5b-experiment'

    def score_refs(self,text,examples):
        from .semantic import compatible_intent,compatible_object
        ranked=sorted([(self.features.cosine(self.backend,text,x),x) for x in dict.fromkeys(examples)
                       if compatible_intent(text,x) and compatible_object(text,x)],reverse=True)
        if not ranked or ranked[0][0]<self.config['cosine_min']:return None
        score=min(self.features.pair_score(self.backend,text,ref) for _,ref in ranked[:2])
        if score<self.config.get('pair_min',.6):return None
        return score,ranked[0][0]


class CalibratedReplyPolicy(NliReplyPolicy):
    def __init__(self,features,*,backend,config=None):
        from .reply_policy import ReplyPolicy
        self.selector=CalibratedSelector(features,backend=backend,config=config)
        self.features=features;self.policy=ReplyPolicy(self.same_topic)


class EmbeddingOnlySelector(CalibratedSelector):
    def score_refs(self,text,examples):
        from .semantic import compatible_intent,compatible_object
        scores=[self.features.cosine(self.backend,text,x) for x in examples
                if compatible_intent(text,x) and compatible_object(text,x)]
        if not scores or max(scores)<self.config['cosine_min']:return None
        return max(scores),1.
