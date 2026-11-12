"""Offline pinned 4-bit Qwen3 4B experiment. Batch one, exact final-token logits."""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys

from scripts.reranker_worker_v5c import PREFIX, SUFFIX, INSTRUCTION, validate_pairs, check_score

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'vserifsaglam/Qwen3-Reranker-4B-4bit-MLX'
REVISION = '9655b27c01d2ff1c49f7e672a04b70d630161b46'
WEIGHT_SHA = '8003a82d273cfbf540f3dd96775dd9065d438a9d91ca4983e7a9c4d0dd9e1448'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


class Worker:
    def __init__(self, instruction=INSTRUCTION):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('An explicit instruction is required')
        self.path = ROOT/'runtime/models/qwen-reranker-4b-v7'
        self.manifest = json.loads((self.path/'MANIFEST.json').read_text())
        if (self.manifest['model'] != MODEL or self.manifest['revision'] != REVISION
                or self.manifest['files']['model.safetensors']['sha256'] != WEIGHT_SHA):
            raise ValueError('Unexpected pinned quantized model')
        self.instruction = instruction
        self.model = None
        self.calls = 0

    def describe(self):
        return dict(model=MODEL, revision=REVISION, weight_sha256=WEIGHT_SHA,
            artifacts=self.manifest['files'], instruction=self.instruction,
            prefix=PREFIX, suffix=SUFFIX, engine='mlx-metal', quantization_bits=4,
            batch_size=1, max_tokens=512, automatic_release=False,
            projection='official tied embed_tokens.as_linear on the final hidden token',
            score='sigmoid(logit_yes-logit_no); not a calibrated probability',
            worker_sha256=sha(Path(__file__)), versions={name:importlib.metadata.version(name)
                for name in ['mlx', 'mlx-metal', 'mlx-lm', 'tokenizers']})

    def load(self):
        if self.model is not None:
            return
        for name, expected in self.manifest['files'].items():
            path = self.path/name
            if path.stat().st_size != expected['size'] or sha(path) != expected['sha256']:
                raise ValueError('Pinned model artifact differs: '+name)
        from tokenizers import Tokenizer
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from mlx_lm.models.cache import make_prompt_cache
        if not mx.metal.is_available():
            raise RuntimeError('Metal unavailable; no implicit fallback')
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(3*1024**3)
        mx.set_cache_limit(128*1024**2)
        tokenizer = Tokenizer.from_file(str(self.path/'tokenizer.json'))
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
        yes, no = [tokenizer.token_to_id(t) for t in ['yes', 'no']]
        if type(yes) is not int or type(no) is not int or yes == no:
            raise ValueError('Invalid yes/no token IDs')
        model, config = load_model(self.path, lazy=True, strict=True, trust_remote_code=False)
        if (config.get('model_type') != 'qwen3' or config.get('tie_word_embeddings') is not True
                or config.get('quantization', {}).get('bits') != 4):
            raise ValueError('Unexpected quantized architecture')
        model.eval()
        mx.eval(model.parameters())
        if not hasattr(model.model.embed_tokens, 'scales'):
            raise ValueError('Quantized embedding expected; no full weight conversion')
        prefix, suffix = encode(PREFIX), encode(SUFFIX)
        shared = prefix+encode(f'<Instruct>: {self.instruction}\n<Query>:')[:-2]
        cache = make_prompt_cache(model)
        hidden = model.model(mx.array([shared], dtype=mx.int32), cache=cache)
        mx.eval(hidden, [c.state for c in cache])
        self.mx, self.encode = mx, encode
        self.prefix, self.suffix, self.shared = prefix, suffix, shared
        self.base, self.yes, self.no = cache, yes, no
        self.model = model
        mx.clear_cache()

    def rerank(self, pairs):
        validate_pairs(pairs)
        if len(pairs) != 1:
            raise ValueError('4B experiment is bounded to one pair per batch')
        self.load()
        a, b = pairs[0]
        row = self.prefix+self.encode(f'<Instruct>: {self.instruction}\n<Query>: {a}\n<Document>: {b}')+self.suffix
        if len(row) > 512 or row[:len(self.shared)] != self.shared:
            raise ValueError('Token limit or exact shared-prefix boundary differs')
        mx = self.mx
        cache = copy.deepcopy(self.base)
        hidden = self.model.model(mx.array([row[len(self.shared):]], dtype=mx.int32), cache=cache)
        # The official projection is retained; only the last position is needed.
        logits = self.model.model.embed_tokens.as_linear(hidden[:, -1]).astype(mx.float32)
        scores = mx.sigmoid(logits[:, self.yes]-logits[:, self.no])
        mx.eval(scores)
        self.calls += 1
        if any(c.offset != len(self.shared) for c in self.base):
            raise RuntimeError('Shared prefix cache was mutated')
        return [check_score(scores.item())]

    def verify_shared_cache(self, pair):
        """Compare the optimized cached call with a full uncached causal forward."""
        cached = self.rerank([pair])[0]
        a, b = pair
        row = self.prefix+self.encode(f'<Instruct>: {self.instruction}\n<Query>: {a}\n<Document>: {b}')+self.suffix
        mx = self.mx
        hidden = self.model.model(mx.array([row], dtype=mx.int32))
        logits = self.model.model.embed_tokens.as_linear(hidden[:, -1]).astype(mx.float32)
        value = mx.sigmoid(logits[:, self.yes]-logits[:, self.no])
        mx.eval(value)
        uncached = check_score(value.item())
        difference = abs(cached-uncached)
        if difference > .01:
            raise RuntimeError('Cached and uncached quantized forwards differ')
        return dict(cached=cached, uncached=uncached, absolute_difference=difference,
                    score_tolerance=.01, status='PASS')

    def stats(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(calls=self.calls, peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
            peak_mlx_bytes=self.mx.get_peak_memory() if self.model is not None else 0,
            batch_size=1, prefix_cache_unchanged=all(c.offset == len(self.shared) for c in self.base)
                if self.model is not None else None)
