"""Develop heads on old tune families, without opening new control inputs/gold."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'mvp-v13-adapted')]
from radar_app.calibrated_v5b import pair_vector
from scripts.prepare_features_v5b import FeatureStore,key,read


def main():
    if (ROOT/'reports/quality-v5b/CONFIG-FREEZE.json').exists():raise RuntimeError('No training after control freeze')
    store=FeatureStore(ROOT/'reports/quality-v5b/tune-features.json')
    raw=read(ROOT/'evaluation/v5/tune.jsonl');gold={r['id']:r['gold'] for r in raw}
    unique={}
    for r in store.data['needs_inputs']:unique.setdefault((r['product'],r['source_id']),r)
    rows=list(unique.values());labels=np.array([gold[r['id']]['kind'] for r in rows])
    families=np.array([gold[r['id']]['topic_id'] or r['id'] for r in rows]);result={}
    for backend in ['tfidf','e5','qwen']:
        x=np.array([store.vectors[backend][key(r['text'])] for r in rows]);trials=[]
        for c in [1.,5.,20.]:
            correct=0
            for family in sorted(set(families)):
                train=families!=family;model=LogisticRegression(C=c,class_weight='balanced',max_iter=800).fit(x[train],labels[train])
                correct+=int(sum(model.predict(x[~train])==labels[~train]))
            trials.append(dict(C=c,accuracy=correct/len(rows),n=len(rows)))
        c=max(trials,key=lambda t:t['accuracy'])['C']
        model=LogisticRegression(C=c,class_weight='balanced',max_iter=800).fit(x,labels)
        kind=dict(classes=model.classes_.tolist(),coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),C=c,trials=trials)
        pairs=[];truth=[];groups=[]
        for i,r in enumerate(rows):
            for j,s in enumerate(rows[:i]):
                a,b=gold[r['id']],gold[s['id']]
                if not a['topic_id'] or not b['topic_id']:continue
                if (r['product'],a['kind'])!=(s['product'],b['kind']):continue
                pairs.append(pair_vector(x[i],x[j],store.relation(r['text'],s['text']),store.relation(s['text'],r['text'])))
                truth.append(int(a['topic_id']==b['topic_id']));groups.append((a['topic_id'],b['topic_id']))
        px=np.asarray(pairs);y=np.asarray(truth);families_pair=sorted({g for p in groups for g in p});trials_pair=[]
        for c in [.01,.1,1.]:
            predicted=[];actual=[]
            for fold in range(4):
                held=set(families_pair[fold::4]);train=np.array([not(a in held or b in held) for a,b in groups]);test=~train
                m=LogisticRegression(C=c,class_weight='balanced',max_iter=800).fit(px[train],y[train])
                predicted.extend(m.predict_proba(px[test])[:,1]);actual.extend(y[test])
            for threshold in [.4,.5,.6,.7,.8,.9]:
                p,r,f,_=precision_recall_fscore_support(actual,np.asarray(predicted)>=threshold,average='binary',zero_division=0)
                trials_pair.append(dict(C=c,threshold=threshold,precision=p,recall=r,f1=f,n=len(actual)))
        good=[t for t in trials_pair if t['precision']>=.75]
        chosen=max(good or trials_pair,key=lambda t:t['f1'])
        model=LogisticRegression(C=chosen['C'],class_weight='balanced',max_iter=800).fit(px,y)
        pair=dict(coefficients=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),
                  selected=chosen,trials=trials_pair,n=len(y),positive=int(sum(y)),
                  features='abs(a-b),a*b,cosine,min_bidirectional_entailment,max_contradiction')
        result[backend]=dict(kind=kind,pair=pair,encoder=store.data['metadata'][backend],
            training_sha256=hashlib.sha256((ROOT/'evaluation/v5/tune.jsonl').read_bytes()).hexdigest(),
            feature_source_sha256=hashlib.sha256((ROOT/'reports/quality-v5b/tune-features.json').read_bytes()).hexdigest(),
            note='Synthetic tune-only heads. Scores are uncalibrated; family-CV diagnostics are not independent acceptance.')
        print(backend,'kind',max(trials,key=lambda t:t['accuracy']),'pair',chosen,flush=True)
    (ROOT/'data/semantic-heads-v5b.json').write_text(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
