"""One pinned backend per process. No remote inference or executable model code."""
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import copy

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from radar_app.model_artifacts_v5b import pinned_identity,validate_artifacts,verify_weights

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path(__file__).resolve().parents[1]
SPECS={
    'e5':('e5-small-v5b',384,'mean','query: '),
    'qwen':('qwen-embedding-v5b',1024,'last-token',
        'Instruct: Given a customer support request, retrieve requests expressing the same user intent, affected function, and direction of desired change.\nQuery: '),
    'nli':('nli-minilm-v5b',384,'NLI',None),
}


class Worker:
    def __init__(self):
        self.backend=None;self.model=None;self.tokenizer=None;self.torch=None
        self._descriptor=None

    def bind(self,backend):
        if backend not in SPECS or self.backend not in (None,backend):raise ValueError('A process owns one named backend')
        if self.backend is not None:return
        identity=pinned_identity(backend,root=ROOT)
        if backend=='nli' and identity['config']['id2label']!={'0':'entailment','1':'neutral','2':'contradiction'}:
            raise ValueError('NLI class order differs')
        self.backend=backend;folder,self.dimension,self.pooling,self.prefix=SPECS[backend]
        self.path=ROOT/'runtime/models'/folder
        self.identity=identity;self.manifest=identity['manifest'];self.config=identity['config']
        self._worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def describe(self):
        if self.backend is None:raise RuntimeError('Bind a backend before describe')
        if self._descriptor is None:self._descriptor=dict(backend=self.backend,encoder=self.manifest['model'],
             revision=self.manifest['revision'],dimension=self.dimension,pooling=self.pooling,
             prefix=self.prefix,max_tokens=512,offline=True,attention='eager',cpu_threads=2,
             dtype='float16' if self.backend=='qwen' else 'float32',
             labels=['entailment','neutral','contradiction'] if self.backend=='nli' else None,
             torch=importlib.metadata.version('torch'),transformers=importlib.metadata.version('transformers'),
             weights_sha256=self.identity['weights_sha256'],
             artifacts_fingerprint=self.identity['artifacts_fingerprint'],
             artifact_files=copy.deepcopy(self.identity['files']),
             manifest_sha256=self.identity['manifest_sha256'],worker_sha256=self._worker_sha256)
        return copy.deepcopy(self._descriptor)

    def load(self):
        if self.model is not None:return
        validate_artifacts(self.path,self.manifest,verify_weight_sha256=True)
        import torch
        from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
        torch.set_num_threads(2);torch.set_num_interop_threads(1);self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(str(self.path),local_files_only=True,trust_remote_code=False)
        if self.backend=='qwen':self.tokenizer.padding_side='left'
        loader=AutoModelForSequenceClassification if self.backend=='nli' else AutoModel
        dtype=torch.float16 if self.backend=='qwen' else torch.float32
        self.model=loader.from_pretrained(str(self.path),local_files_only=True,use_safetensors=True,
                                         trust_remote_code=False,attn_implementation='eager',dtype=dtype)
        self.model.eval()

    @staticmethod
    def validate(texts):
        if (not isinstance(texts,list) or not 1<=len(texts)<=32
                or any(not isinstance(t,str) or not t.strip() or len(t)>4000 for t in texts)):
            raise ValueError('Invalid bounded text batch')

    def run(self,op,request):
        if op=='describe':return self.describe()
        if op=='stats':
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return dict(loaded=self.model is not None,peak_rss_bytes=int(peak if sys.platform=='darwin' else peak*1024),
                        parameters_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()) if self.model is not None else 0,
                        cpu_threads=2)
        if op=='embed' and self.backend!='nli':
            texts=request.get('texts');self.validate(texts);self.load()
            count=len(texts)
            batch=texts+texts if count==1 else texts
            tokens=self.tokenizer([self.prefix+t for t in batch],padding=True,truncation=False,return_tensors='pt')
            if tokens['input_ids'].shape[1]>512:raise ValueError('Token limit; review or split input')
            with self.torch.inference_mode():
                extra={'use_cache':False} if self.backend=='qwen' else {}
                hidden=self.model(**tokens,**extra).last_hidden_state
                if self.pooling=='last-token':pooled=hidden[:,-1]
                else:
                    mask=tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                    pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                return self.torch.nn.functional.normalize(pooled.float(),p=2,dim=1)[:count].tolist()
        if op=='nli' and self.backend=='nli':
            pairs=request.get('pairs')
            if not isinstance(pairs,list) or not 1<=len(pairs)<=32:raise ValueError('Invalid pair batch')
            for p in pairs:
                if not isinstance(p,list) or len(p)!=2:raise ValueError('Invalid pair')
                self.validate(p)
            self.load();count=len(pairs);batch=pairs+pairs if count==1 else pairs
            tokens=self.tokenizer([p[0] for p in batch],[p[1] for p in batch],padding=True,truncation=False,return_tensors='pt')
            if tokens['input_ids'].shape[1]>512:raise ValueError('Token limit; review or split input')
            with self.torch.inference_mode():
                logits=self.model(**tokens).logits
                return self.torch.softmax(logits,dim=-1)[:count].tolist()
        raise ValueError('Unsupported backend operation')


def main():
    worker=Worker()
    for line in sys.stdin.buffer:
        r={}
        try:
            if len(line)>1024**2:raise ValueError('Request exceeds limit')
            r=json.loads(line);worker.bind(r.get('backend'))
            response=dict(id=r.get('id'),ok=True,result=worker.run(r.get('op'),r))
        except Exception as exc:response=dict(id=r.get('id'),ok=False,error=type(exc).__name__)
        print(json.dumps(response,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
