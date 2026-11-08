"""Separate v5b experiment; frozen v5 code and results remain unchanged."""
import math
from pathlib import Path

from .semantic_client import SemanticClient

ROOT=Path(__file__).resolve().parents[1]
BACKENDS={'e5','qwen','nli'}


def check_nli(row):
    if (not isinstance(row,list) or len(row)!=3
            or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or not 0<=x<=1 for x in row)
            or abs(sum(row)-1)>1e-5):
        raise RuntimeError('Invalid NLI class probabilities')
    return dict(zip(['entailment','neutral','contradiction'],row))


def equivalence(forward,reverse,max_contradiction=.15):
    if max(forward['contradiction'],reverse['contradiction'])>max_contradiction:return 0.
    return min(forward['entailment'],reverse['entailment'])


class BackendClient(SemanticClient):
    def __init__(self,backend,**kw):
        if backend not in BACKENDS:raise ValueError('Unknown pinned local backend')
        self.backend=backend
        kw.setdefault('worker_script',ROOT/'scripts/semantic_worker_v5b.py')
        kw.setdefault('cache_dir',ROOT/'runtime'/('semantic-v5b-'+backend))
        super().__init__(**kw)

    def _request(self,op,**args):
        return super()._request(op,backend=self.backend,**args)

    def nli(self,pairs):
        if self.backend!='nli':raise ValueError('NLI backend required')
        if not isinstance(pairs,(list,tuple)) or len(pairs)>32:raise ValueError('At most 32 pairs required')
        for p in pairs:
            if not isinstance(p,(list,tuple)) or len(p)!=2:raise ValueError('Text pair required')
            self._texts(p)
        if not pairs:return []
        with self._lock:
            keys=[self._key('nli',list(p)) for p in pairs]
            values={k:self._read(k) for k in keys}
            missing=list(dict((k,list(p)) for k,p in zip(keys,pairs) if values[k] is None).items())
            if missing:
                answer=self._request('nli',pairs=[p for _,p in missing])
                if not isinstance(answer,list) or len(answer)!=len(missing):raise RuntimeError('Invalid NLI response size')
                for (key,_),v in zip(missing,answer):
                    check_nli(v);values[key]=v;self._write(key,v)
            return [check_nli(values[k]) for k in keys]
