"""Explicit offline semantic adapter for the original streaming detector."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .semantic_v5b import BackendClient

ROOT=Path(__file__).resolve().parents[1]
FOLDERS={'e5':'e5-small-v5b','qwen':'qwen-embedding-v5b'}


def embedding_identity(kind):
    if kind=='tfidf':return dict(kind=kind)
    if kind not in {'local-e5','local-qwen'}:raise ValueError('Explicit offline backend required')
    backend=kind.removeprefix('local-')
    m=json.loads((ROOT/'runtime/models'/FOLDERS[backend]/'MANIFEST.json').read_text())
    return dict(kind=kind,model=m['model'],revision=m['revision'],
                weights_sha256=m['files']['model.safetensors']['sha256'],
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                worker_sha256=hashlib.sha256((ROOT/'scripts/semantic_worker_v5b.py').read_bytes()).hexdigest())


class OfflineSemanticEmbedder:
    def __init__(self,backend,*,client_factory=BackendClient,batch=8):
        if backend not in FOLDERS:raise ValueError('Unknown offline embedding backend')
        self.backend=backend;self.client_factory=client_factory;self.batch=batch;self.metadata=None

    @property
    def name(self):
        d=self.metadata or embedding_identity('local-'+self.backend)
        return 'local-'+self.backend+'@'+d['revision']

    def fit(self,texts):return self

    def transform(self,texts):
        from radar.mask import strip_tokens
        clean=[strip_tokens(t).strip() for t in texts]
        keys=list(dict.fromkeys(t for t in clean if t))
        lookup={}
        # The model process is released before the browser starts showing the
        # replay. Later transforms can reuse its bounded vector cache.
        with self.client_factory(self.backend) as client:
            self.metadata=client.describe();dimension=self.metadata['dimension']
            for start in range(0,len(keys),self.batch):
                chunk=keys[start:start+self.batch];lookup.update(zip(chunk,client.embed(chunk)))
        if not clean:return np.empty((0,dimension),dtype=np.float32)
        return np.asarray([lookup[t] if t else [0.]*dimension for t in clean],dtype=np.float32)
