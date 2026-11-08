"""Prepare fixed-model features without reading any control answers."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'mvp-v13-adapted')]
from radar.embed import TfidfEmbedder
from radar_app.privacy.mask import mask_text
from radar_app.reply_policy import payload_only
from radar_app.selection_v5b import KIND_HYPOTHESES,RISK_HYPOTHESES,input_payload
from radar_app.semantic_v5b import BackendClient

OUT=ROOT/'reports/quality-v5b'


def key(text):return hashlib.sha256(text.encode()).hexdigest()
def pair_key(a,b):return hashlib.sha256(json.dumps([a,b],ensure_ascii=False).encode()).hexdigest()
def read(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def feature_contract():
    from radar_app.model_artifacts_v5b import pinned_identity
    return hashlib.sha256(json.dumps(dict(
        worker_sha256=hashlib.sha256((ROOT/'scripts/semantic_worker_v5b.py').read_bytes()).hexdigest(),
        models={name:pinned_identity(name)['artifacts_fingerprint'] for name in ['e5','qwen','nli']},
        hypotheses=[KIND_HYPOTHESES,RISK_HYPOTHESES],
        masking={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                 ['radar_app/privacy/mask.py','radar_app/privacy/mask_ner.py']}
    ),sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def plain_metadata(metadata):
    return {k:{f:v for f,v in m.items() if f!='resource_measurement'} for k,m in metadata.items()}


def validate_feature_provenance(data,*,sources,needs,reply,contract,metadata):
    if data.get('source_sha256')!=sources:raise RuntimeError('Feature source hashes differ')
    if data.get('needs_inputs')!=needs or data.get('reply_inputs')!=reply:raise RuntimeError('Feature input rows differ')
    if data.get('preparation_contract_sha256')!=contract:raise RuntimeError('Feature contract differs')
    if plain_metadata(data['metadata'])!=plain_metadata(metadata):raise RuntimeError('Feature model metadata differs')
    texts=set([r['text'] for r in needs]+[r['query']['text'] for r in reply]+[x['text'] for r in reply for x in r['references']])
    for backend in ['tfidf','e5','qwen']:
        values=data['vectors'][backend]
        if set(values)!={key(t) for t in texts}:raise RuntimeError('Feature vector inventory differs')
        for vector in values.values():
            if len(vector)!=metadata[backend]['dimension'] or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in vector):
                raise RuntimeError('Invalid feature vector')
            norm=sum(v*v for v in vector)
            if abs(norm-1)>1e-3 and not(backend=='tfidf' and norm==0):raise RuntimeError('Feature vectors must be normalized')
    def scores(values,labels):
        if set(values)!=set(labels) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in values.values()):
            raise RuntimeError('Missing or invalid semantic scores')
    for r in needs+[r['query'] for r in reply]:scores(data['kinds'][key(r['text'])],KIND_HYPOTHESES)
    for r in reply:scores(data['risks'][key(r['query']['text'])],RISK_HYPOTHESES)
    groups={}
    for r in needs:groups.setdefault(r['product'],set()).add(r['text'])
    pairs=set()
    for group in groups.values():
        for a,b in itertools.combinations(sorted(group),2):pairs.update([(a,b),(b,a)])
    for r in reply:
        for ref in r['references']:
            a,b=r['query']['text'],ref['text']
            if a!=b:pairs.update([(a,b),(b,a)])
    if {pair_key(a,b) for a,b in pairs}!=set(data['relations']):raise RuntimeError('Feature relation inventory differs')
    for relation in data['relations'].values():
        scores(relation,['entailment','neutral','contradiction'])
        if abs(sum(relation.values())-1)>1e-5:raise RuntimeError('Invalid relation probabilities')


class FeatureStore:
    def __init__(self,path):
        import numpy as np
        self.data=json.loads(Path(path).read_text());self.vectors={}
        for backend,vs in self.data['vectors'].items():self.vectors[backend]={k:np.asarray(v) for k,v in vs.items()}
        self._cos={}
    def cosine(self,backend,a,b):
        k=(backend,key(a),key(b))
        if k not in self._cos:self._cos[k]=float(self.vectors[backend][k[1]]@self.vectors[backend][k[2]])
        return self._cos[k]
    def kind_scores(self,text):return self.data['kinds'][key(text)]
    def risk_scores(self,text):return self.data['risks'].get(key(text),{})
    def relation(self,a,b):
        if a==b:return dict(entailment=1.,neutral=0.,contradiction=0.)
        return self.data['relations'][pair_key(a,b)]


def masked_inputs(needs,reply):
    clean={}
    def mask(t):
        if t not in clean:clean[t]=mask_text(t,ner='auto',ctx_names=True)[0]
        return clean[t]
    n=[dict(input_payload(r),text=mask(r['text'])) for r in needs]
    replies=[]
    for r in reply:
        p=payload_only(r)
        p['query']=dict(p['query'],text=mask(p['query']['text']))
        p['references']=[dict(x,text=mask(x['text'])) for x in p['references']]
        replies.append(p)
    return n,replies


def main():
    ap=argparse.ArgumentParser();ap.add_argument('split',choices=['tune','control']);ap.add_argument('--stage',choices=['embeddings','nli','all'],default='all');args=ap.parse_args()
    if args.split=='control' and not (OUT/'CONFIG-FREEZE.json').exists():raise RuntimeError('Control features require a frozen configuration')
    paths=([ROOT/'evaluation/v5/tune.jsonl',ROOT/'evaluation/v5/reply-tune.jsonl'] if args.split=='tune' else
           [ROOT/'evaluation/v5b/control-inputs.jsonl',ROOT/'evaluation/v5b/reply-control-inputs.jsonl'])
    needs,reply=masked_inputs(*[read(p) for p in paths])
    queries=list(dict.fromkeys([r['text'] for r in needs]+[r['query']['text'] for r in reply]))
    texts=list(dict.fromkeys(queries+[x['text'] for r in reply for x in r['references']]))
    path=OUT/(args.split+'-features.json');OUT.mkdir(parents=True,exist_ok=True)
    result=json.loads(path.read_text()) if path.exists() else dict(vectors={},kinds={},risks={},relations={},metadata={},timings={})
    contract=feature_contract()
    if result.get('preparation_contract_sha256') not in (None,contract):raise RuntimeError('Feature preparation contract changed')
    for name,meta in result['metadata'].items():
        if name!='tfidf' and meta.get('worker_sha256')!=hashlib.sha256((ROOT/'scripts/semantic_worker_v5b.py').read_bytes()).hexdigest():
            raise RuntimeError('Prepared model features have a stale worker')
    result['preparation_contract_sha256']=contract
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if result.get('source_sha256') not in (None,sources):raise RuntimeError('Feature source bytes changed')
    result.update(needs_inputs=needs,reply_inputs=reply,
                  source_sha256=sources,
                  note='Masked synthetic inputs only. No control answer file opened. Pair precomputation is not online latency.')
    def save():path.write_text(json.dumps(result,ensure_ascii=False))
    if args.stage in {'embeddings','all'}:
        training,_=masked_inputs(read(ROOT/'evaluation/v5/tune.jsonl'),[])
        tf=TfidfEmbedder().fit([r['text'] for r in training]);vectors=tf.transform(texts)
        result['vectors']['tfidf']={key(t):v.tolist() for t,v in zip(texts,vectors)}
        result['metadata']['tfidf']=dict(kind='char ngram3–5 + SVD + L2',fit_source='evaluation/v5/tune.jsonl',dimension=int(vectors.shape[1]),training_only=True)
        save()
        for backend in ['e5','qwen']:
            if backend in result['vectors']:continue
            values={};timings=[]
            with BackendClient(backend) as client:
                for start in range(0,len(texts),8):
                    chunk=texts[start:start+8];t=time.perf_counter();vectors=client.embed(chunk)
                    timings.append(dict(n=len(chunk),seconds=time.perf_counter()-t))
                    values.update((key(text),v) for text,v in zip(chunk,vectors))
                    if start%40==0:print(args.split,backend,start,'/',len(texts),flush=True)
                result['metadata'][backend]=dict(client.describe(),resource_measurement=client._request('stats'))
            result['vectors'][backend]=values;result['timings'][backend]=timings;save()
    if args.stage in {'nli','all'}:
        pairs=[];dest=[]
        for text in queries:
            for kind,hyp in KIND_HYPOTHESES.items():
                if kind not in result['kinds'].get(key(text),{}):pairs.append((text,hyp));dest.append(('kinds',key(text),kind))
        for r in reply:
            text=r['query']['text']
            for risk,hyp in RISK_HYPOTHESES.items():
                if risk not in result['risks'].get(key(text),{}):pairs.append((text,hyp));dest.append(('risks',key(text),risk))
        groups={}
        for r in needs:groups.setdefault(r['product'],set()).add(r['text'])
        wanted=set()
        for group in groups.values():
            for a,b in itertools.combinations(sorted(group),2):wanted.update([(a,b),(b,a)])
        for r in reply:
            a=r['query']['text']
            for ref in r['references']:
                b=ref['text']
                if a!=b:wanted.update([(a,b),(b,a)])
        for a,b in sorted(wanted):
            k=pair_key(a,b)
            if k not in result['relations']:pairs.append((a,b));dest.append(('relations',k,None))
        times=[]
        with BackendClient('nli') as client:
            for start in range(0,len(pairs),16):
                t=time.perf_counter();batch=client.nli(pairs[start:start+16]);times.append(dict(n=len(batch),seconds=time.perf_counter()-t))
                for (scope,k,label),v in zip(dest[start:start+16],batch):
                    if scope=='relations':result[scope][k]=v
                    else:result[scope].setdefault(k,{})[label]=v['entailment']
                if start%160==0:save();print(args.split,'NLI',start,'/',len(pairs),flush=True)
            result['metadata']['nli']=dict(client.describe(),resource_measurement=client._request('stats'))
        result['timings'].setdefault('nli',[]).extend(times);save()
    if args.stage=='all':
        validate_feature_provenance(result,sources=sources,needs=needs,reply=reply,
                                    contract=contract,metadata=result['metadata'])
    print('READY',args.split,args.stage,len(texts),'texts',len(result['relations']),'directed relations',flush=True)


if __name__=='__main__':main()
