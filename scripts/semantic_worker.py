"""Offline, bounded JSON-lines inference using already cached safetensors.

Run with the existing Torch environment; the application never imports Torch.
Only masked text should be sent here. Scores are similarities, not probabilities.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
ENCODER = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
RERANKER = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'


def snapshot(model):
    root = Path(os.environ.get('HF_HUB_CACHE', Path.home() / '.cache/huggingface/hub'))
    repo = root / ('models--' + model.replace('/', '--'))
    ref = repo / 'refs/main'
    if ref.is_file():
        path = repo / 'snapshots' / ref.read_text().strip()
    else:
        paths = sorted((repo / 'snapshots').glob('*'))
        if len(paths) != 1:
            raise RuntimeError('A unique cached model revision is required')
        path = paths[0]
    if not (path / 'model.safetensors').is_file():
        raise RuntimeError('Cached safetensors are unavailable')
    return path


class Models:
    def __init__(self):
        self.encoder_path, self.reranker_path = snapshot(ENCODER), snapshot(RERANKER)
        self.torch = None
        self.encoder = self.encoder_tokenizer = None
        self.reranker = self.reranker_tokenizer = None

    def describe(self):
        return dict(encoder=ENCODER, encoder_revision=self.encoder_path.name,
                    reranker=RERANKER, reranker_revision=self.reranker_path.name,
                    dimension=384, pooling='attention-mask mean + L2 normalization',
                    encoder_max_tokens=256, reranker_max_tokens=512,
                    score_type='raw relevance logit; not probability', offline=True,
                    attention='eager',
                    reranker_minimum_batch=2,
                    torch=importlib.metadata.version('torch'),
                    transformers=importlib.metadata.version('transformers'))

    def load(self, name):
        if self.torch is None:
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
            self.torch = torch
        if getattr(self, name) is None:
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
            path = getattr(self, name + '_path')
            loader = AutoModel if name == 'encoder' else AutoModelForSequenceClassification
            setattr(self, name + '_tokenizer', AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=False))
            model = loader.from_pretrained(str(path), local_files_only=True,
                                           use_safetensors=True, trust_remote_code=False,
                                           attn_implementation='eager')
            model.eval()
            setattr(self, name, model)

    @staticmethod
    def texts(texts):
        if (not isinstance(texts, list) or not 1 <= len(texts) <= 32
                or any(not isinstance(t, str) or not t.strip() or len(t) > 4000 for t in texts)):
            raise ValueError('Invalid text batch')

    def embed(self, texts):
        self.texts(texts)
        self.load('encoder')
        tokens = self.encoder_tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        if tokens['input_ids'].shape[1] > 256:
            raise ValueError('Encoder token limit; split or review the ticket')
        with self.torch.inference_mode():
            hidden = self.encoder(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return vectors.tolist()

    def rerank(self, pairs):
        if not isinstance(pairs, list) or not 1 <= len(pairs) <= 32:
            raise ValueError('Invalid pair batch')
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('Invalid pair')
            self.texts(pair)
        self.load('reranker')
        size=len(pairs)
        # This installed ARM Torch build exits with SIGBUS for a one-pair
        # forward pass. Batch-two is verified; the duplicate is never returned.
        batch=pairs if size>1 else pairs+pairs
        tokens = self.reranker_tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                         padding=True, truncation=False, return_tensors='pt')
        if tokens['input_ids'].shape[1] > 512:
            raise ValueError('Reranker token limit; split or review the ticket')
        with self.torch.inference_mode():
            logits = self.reranker(**tokens).logits
        if logits.shape[1] != 1:
            raise RuntimeError('Expected a single relevance logit')
        return logits[:size, 0].tolist()


def main():
    models = Models()
    for line in sys.stdin.buffer:
        request = {}
        try:
            if len(line) > 1024**2:
                raise ValueError('Request exceeds limit')
            request = json.loads(line)
            op = request.get('op')
            if op == 'describe':
                result = models.describe()
            elif op == 'embed':
                result = models.embed(request.get('texts'))
            elif op == 'rerank':
                result = models.rerank(request.get('pairs'))
            else:
                raise ValueError('Unknown operation')
            response = dict(id=request.get('id'), ok=True, result=result)
        except Exception as exc:
            # Neither source text nor model tracebacks go to protocol/log files.
            response = dict(id=request.get('id'), ok=False, error=type(exc).__name__)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
