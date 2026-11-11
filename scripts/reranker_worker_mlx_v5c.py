"""Pinned FP16 MLX verifier with a shared causal prefix and four-row batches.

The original safetensors stay unchanged. Right padding follows every valid
position; causal attention cannot carry it backwards into the selected token.
"""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys

from scripts.reranker_worker_v5c import (
    ROOT, MODEL, REVISION, WEIGHT_SHA, PREFIX, SUFFIX, INSTRUCTION,
    validate_pairs, check_score, format_pair,
)
from radar_app.model_artifacts_v5b import validate_artifacts


def right_pad(rows):
    if (not isinstance(rows,list) or not 1 <= len(rows) <= 4
            or any(not isinstance(row,list) or not 1 <= len(row) <= 512
                   or any(type(token) is not int or token < 0 for token in row) for row in rows)):
        raise ValueError('One to four bounded token sequences required')
    count=len(rows)
    padded_rows=rows+[rows[0]]*(4-count)
    width=((max(map(len,padded_rows))+31)//32)*32
    return ([r+[0]*(width-len(r)) for r in padded_rows],
            [len(r)-1 for r in padded_rows],count)


def strip_prefix(rows,prefix):
    if not prefix or any(row[:len(prefix)] != prefix or len(row) <= len(prefix) for row in rows):
        raise ValueError('Every exact token sequence must start with the shared prefix')
    return [row[len(prefix):] for row in rows]


class Worker:
    def __init__(self):
        self.path=ROOT/'runtime/models/qwen-reranker-v5c'
        self.manifest=json.loads((self.path/'MANIFEST.json').read_text())
        if (self.manifest['model'] != MODEL or self.manifest['revision'] != REVISION
                or self.manifest['files']['model.safetensors']['sha256'] != WEIGHT_SHA):
            raise ValueError('Unexpected pinned verifier')
        validate_artifacts(self.path,self.manifest,verify_weight_sha256=False)
        self.model=None
        self.calls=0

    def describe(self):
        main=ROOT/'scripts/reranker_worker_v5c.py'
        own=Path(__file__)
        return dict(backend='qwen-reranker-v5c',dimension=1,model=MODEL,revision=REVISION,
            weights_sha256=WEIGHT_SHA,artifacts=copy.deepcopy(self.manifest['files']),
            instruction=INSTRUCTION,prefix=PREFIX,suffix=SUFFIX,max_tokens=512,
            device='mlx-metal',engine='mlx',dtype='float16',quantized=False,
            attention='native MLX causal attention; right padding; exact shared prefix',
            cpu_threads=2,batch_size=4,score='sigmoid(logit_yes-logit_no); not a calibrated probability',
            worker_sha256=hashlib.sha256(main.read_bytes()).hexdigest(),
            engine_sha256=hashlib.sha256(own.read_bytes()).hexdigest(),
            dependency_sha256={str(own.relative_to(ROOT)):hashlib.sha256(own.read_bytes()).hexdigest()},
            versions={name:importlib.metadata.version(name) for name in ['mlx','mlx-metal','mlx-lm','tokenizers']})

    def load(self):
        if self.model is not None:
            return
        validate_artifacts(self.path,self.manifest,verify_weight_sha256=True)
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(self.path/'tokenizer.json'))
        encode=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
        yes,no=[tokenizer.token_to_id(t) for t in ['yes','no']]
        if type(yes) is not int or type(no) is not int or yes==no:
            raise ValueError('Invalid yes/no token IDs')
        import mlx.core as mx
        from mlx.utils import tree_map
        from mlx_lm.utils import load_model
        from mlx_lm.models.cache import make_prompt_cache
        if not mx.metal.is_available():
            raise RuntimeError('MLX Metal unavailable; no implicit fallback')
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(3*1024**3)
        mx.set_cache_limit(128*1024**2)
        model,config=load_model(self.path,lazy=True,strict=True,trust_remote_code=False)
        if config.get('model_type') != 'qwen3' or config.get('tie_word_embeddings') is not True:
            raise ValueError('Unexpected model architecture')
        model.update(tree_map(lambda x:x.astype(mx.float16) if mx.issubdtype(x.dtype,mx.floating) else x,model.parameters()))
        model.eval();mx.eval(model.parameters());mx.clear_cache()
        two_rows=model.model.embed_tokens(mx.array([yes,no],dtype=mx.int32))
        mx.eval(two_rows)
        prefix_tokens=encode(PREFIX)
        suffix_tokens=encode(SUFFIX)
        # Remove the last two header tokens to avoid a BPE boundary at the query.
        shared=prefix_tokens+encode(f'<Instruct>: {INSTRUCTION}\n<Query>:')[:-2]
        cache=make_prompt_cache(model)
        hidden=model.model(mx.array([shared],dtype=mx.int32),cache=cache)
        mx.eval(hidden,[c.state for c in cache]);del hidden
        snapshot=copy.deepcopy(cache);mx.eval([c.state for c in snapshot])
        self.mx,self.tokenizer,self.encode=mx,tokenizer,encode
        self.prefix_tokens,self.suffix_tokens=prefix_tokens,suffix_tokens
        self.shared,self.base,self.snapshot,self.two_rows=shared,cache,snapshot,two_rows
        self.model=model  # Commit readiness only after every stage succeeded.

    def check_base(self):
        for a,b in zip(self.base,self.snapshot):
            if (a.offset != len(self.shared) or b.offset != a.offset
                    or not self.mx.array_equal(a.keys,b.keys).item()
                    or not self.mx.array_equal(a.values,b.values).item()):
                raise RuntimeError('Shared prefix cache changed')

    def rerank(self,pairs):
        validate_pairs(pairs)
        self.load()
        rows=[self.prefix_tokens+self.encode(format_pair(a,b))+self.suffix_tokens for a,b in pairs]
        if any(len(row)>512 for row in rows):
            raise ValueError('Token limit exceeded; no silent truncation')
        tails=strip_prefix(rows,self.shared)
        padded,positions,count=right_pad(tails)
        mx=self.mx
        cache=copy.deepcopy(self.base)
        for entry in cache:
            entry.keys=mx.repeat(entry.keys,4,axis=0)
            entry.values=mx.repeat(entry.values,4,axis=0)
        hidden=self.model.model(mx.array(padded,dtype=mx.int32),cache=cache)
        last=hidden[mx.arange(4),mx.array(positions,dtype=mx.int32)]
        logits=(last@self.two_rows.T).astype(mx.float32)
        scores=mx.sigmoid(logits[:,0]-logits[:,1]);mx.eval(scores)
        result=[check_score(s) for s in scores.tolist()[:count]]
        self.calls+=1
        if self.calls==1 or self.calls%100==0:
            self.check_base()
        return result

    def run(self,op,request):
        if op=='describe':
            return self.describe()
        if op=='rerank':
            return self.rerank(request.get('pairs'))
        if op=='stats':
            if self.model is not None:
                self.check_base()
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return dict(loaded=self.model is not None,device='mlx-metal',calls=self.calls,
                        peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024),
                        peak_mlx_bytes=self.mx.get_peak_memory() if self.model is not None else 0,
                        base_cache_unchanged=True if self.model is not None else None)
        raise ValueError('Unsupported verifier operation')
