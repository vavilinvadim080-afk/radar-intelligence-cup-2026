"""Sealed reply-only experiment, with complete policy gates and persisted predictions."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/quality-v7b-reply'
DATA=ROOT/'evaluation/v7b-transport'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())
def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def metric(rows,predictions,gold):
    by_id={r['id']:r for r in predictions}
    if set(by_id)!={r['id'] for r in rows} or len(by_id)!=len(predictions):raise ValueError('Incomplete predictions')
    result={}
    for event,positive in [('incoming_ticket','queue_standard'),('fixed_notice_candidate','queue_fixed_notice')]:
        subset=[r for r in rows if r['event']==event]
        tp=fp=fn=tn=0
        for r in subset:
            expected=gold[r['id']]['safe_action']==positive;actual=by_id[r['id']]['action']==positive
            tp+=bool(expected and actual);fp+=bool(not expected and actual)
            fn+=bool(expected and not actual);tn+=bool(not expected and not actual)
        result[event]=dict(n=len(subset),tp=tp,fp=fp,fn=fn,tn=tn,
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
    return result


def tune():
    if (OUT/'CONFIG-FREEZE.json').exists():raise ValueError('Candidate is frozen; TUNE cannot be changed')
    from radar_app.reply_match_v7 import ReplyMatcherV7,pair_key,requests_for
    from radar_app.reply_safety_v7 import unsafe_reason
    probe=ROOT/'reports/quality-v7-probe'
    requests=read(probe/'requests-only.json')['requests'];ids={r['case'] for r in requests}
    source=read(ROOT/'reports/quality-v5c/tune-features.json')
    rows=[r for r in source['reply_inputs'] if r['id'] in ids];risks=source['risks']
    raw=jsonl(ROOT/'evaluation/v5c/reply-tune-inputs.jsonl')
    blocked={r['query']['id']:unsafe_reason(r['query']['text']) for r in raw if unsafe_reason(r['query']['text'])}
    scores_by_id={r['id']:r['score'] for r in read(probe/'qwen4b-original-scores.json')['scores']}
    scores={pair_key(r['query'],r['document']):scores_by_id[r['id']] for r in requests}
    if not set(requests_for(rows,risks,blocked))<=set(scores):raise ValueError('Missing TUNE pairs')
    predictions=[dict(id=r['id'],**ReplyMatcherV7(scores,risks=risks,blocked=blocked).decide(r)) for r in rows]
    write(OUT/'tune-predictions-only.json',predictions)
    gold={r['id']:r['gold'] for r in jsonl(ROOT/'evaluation/v5c/reply-tune-gold.jsonl')}
    result=dict(scope='TUNE24full-policy probe; not independent acceptance',metrics=metric(rows,predictions,gold),
        prior_probe_error='eligibility() omitted product gate and learned risk gate; probe labels were same_topic, not permitted replies; not full-service metrics',
        selection='585settings explored; final4Bmax-direction/best-reference threshold0.95; no CV',automatic_release=False)
    write(OUT/'TUNE-RESULTS.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


def freeze():
    if (OUT/'CONFIG-FREEZE.json').exists():raise ValueError('Candidate already frozen')
    metrics=read(OUT/'TUNE-RESULTS.json')['metrics']['incoming_ticket']
    if metrics['fp'] or (metrics['recall'] or 0)<.8:raise ValueError('TUNE candidate does not meet its declared gate')
    paths=list((ROOT/'radar_app').rglob('*.py'))+[Path(__file__),ROOT/'scripts/reranker_worker_v5c.py',
        ROOT/'scripts/reranker_worker_mlx_v5c.py',ROOT/'scripts/reranker_worker_mlx_4b_v7.py',
        ROOT/'scripts/prepare_features_v5b.py',ROOT/'scripts/semantic_worker_v5b.py',ROOT/'data/reply-config-v7.json',ROOT/'tests/test_reply_match_v7.py',ROOT/'tests/test_reply_safety_v7.py']
    for folder in ['qwen-reranker-4b-v7','qwen-reranker-v5c','nli-minilm-v5b']:
        directory=ROOT/'runtime/models'/folder;manifest=read(directory/'MANIFEST.json')
        paths.append(directory/'MANIFEST.json')
        paths.extend(directory/name for name in manifest['files'])
    values={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(OUT/'CONFIG-FREEZE.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Reply-only; both model arms use same full policy and threshold; no needs quality claim',
        files=values,dataset_lock_sha256=sha(DATA/'LOCK.json'),config=read(ROOT/'data/reply-config-v7.json'),
        tune_result_sha256=sha(OUT/'TUNE-RESULTS.json'),automatic_release=False))
    print('Frozen',len(values),'files before opening new control inputs',flush=True)


def verify_freeze(weights=False):
    frozen=read(OUT/'CONFIG-FREEZE.json')
    if frozen['dataset_lock_sha256']!=sha(DATA/'LOCK.json'):raise ValueError('Control lock changed')
    for name,expected in frozen['files'].items():
        if not weights and name.endswith('.safetensors'):continue
        if sha(ROOT/name)!=expected:raise ValueError('Frozen source differs:'+name)
    for name,entry in read(DATA/'LOCK.json')['files'].items():
        expected=entry['sha256'] if isinstance(entry,dict) else entry
        if sha(DATA/name)!=expected:raise ValueError('Locked data differs:'+name)
    return frozen


def prepared():
    verify_freeze()
    if not (OUT/'control-inputs-masked.json').exists():
        from scripts.prepare_features_v5b import masked_inputs
        from radar_app.reply_safety_v7 import unsafe_reason
        raw=jsonl(DATA/'reply-inputs.jsonl')
        # Reject transport incompatibility before any model call or gold access.
        if len({r['id'] for r in raw})!=len(raw):raise ValueError('Reply scenario IDs must be unique')
        for row in raw:
            if row.get('event') not in {'incoming_ticket','fixed_notice_candidate'}:
                raise ValueError('Control event does not match the working policy contract')
            if type(row.get('query',{}).get('consent')) is not bool:
                raise ValueError('Query consent must be explicit')
            if any(type(ref.get('source_verified')) is not bool for ref in row.get('references',[])):
                raise ValueError('Reference source_verified must be explicit')
        blocked={r['query']['id']:unsafe_reason(r['query']['text']) for r in raw if unsafe_reason(r['query']['text'])}
        write(OUT/'control-preflight.json',dict(input_sha256=sha(DATA/'reply-inputs.jsonl'),blocked=blocked))
        _,rows=masked_inputs([],raw)
        write(OUT/'control-inputs-masked.json',rows)
    return read(OUT/'control-inputs-masked.json')


def risk():
    from radar_app.semantic_v5b import BackendClient
    from radar_app.selection_v5b import RISK_HYPOTHESES
    rows=prepared();target=OUT/'control-risks.json'
    if target.exists():raise ValueError('Risk scores already recorded')
    queries={hashlib.sha256(r['query']['text'].encode()).hexdigest():r['query']['text'] for r in rows}
    client=BackendClient('nli',cache_dir=OUT/'nli-cache')
    try:
        metadata=client.describe();result={};start=time.monotonic()
        for key,text in queries.items():
            values=client.nli([[text,hypothesis] for hypothesis in RISK_HYPOTHESES.values()])
            result[key]={name:v['entailment'] for name,v in zip(RISK_HYPOTHESES,values)}
            print('risk',len(result),'/',len(queries),'elapsed',round(time.monotonic()-start,1),flush=True)
        write(target,dict(input_sha256=sha(OUT/'control-inputs-masked.json'),metadata=metadata,risks=result))
    finally:client.close()


def risk_values():
    data=read(OUT/'control-risks.json')
    if data['input_sha256']!=sha(OUT/'control-inputs-masked.json'):raise ValueError('Risk inputs changed')
    return data['risks']


def preflight():
    data=read(OUT/'control-preflight.json')
    if data['input_sha256']!=sha(DATA/'reply-inputs.jsonl'):raise ValueError('Preflight inputs changed')
    return data['blocked']


def infer(model):
    from radar_app.reply_match_v7 import requests_for
    if model=='4b':from scripts.reranker_worker_mlx_4b_v7 import Worker
    else:from scripts.reranker_worker_mlx_v5c import Worker
    rows=prepared();requests=requests_for(rows,risk_values(),preflight());worker=Worker();target=OUT/f'scores-{model}.json'
    data=read(target) if target.exists() else dict(input_sha256=sha(OUT/'control-inputs-masked.json'),model=worker.describe(),scores={})
    if data['input_sha256']!=sha(OUT/'control-inputs-masked.json') or data['model']!=worker.describe():raise ValueError('Score provenance changed')
    if set(data['scores'])-set(requests):raise ValueError('Unknown saved score')
    remaining=[k for k in requests if k not in data['scores']];batch=1 if model=='4b' else 4;start=time.monotonic()
    for offset in range(0,len(remaining),batch):
        keys=remaining[offset:offset+batch];values=worker.rerank([requests[k] for k in keys])
        if len(values)!=len(keys) or any(not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('Invalid scores')
        data['scores'].update(zip(keys,values));write(target,data)
        if len(data['scores'])%4==0 or len(data['scores'])==len(requests):print(model,len(data['scores']),'/',len(requests),'elapsed',round(time.monotonic()-start,1),flush=True)
    write(OUT/f'resources-{model}.json',worker.stats() if model=='4b' else worker.run('stats',{}))


def predict():
    if (OUT/'control-predictions-only.json').exists():raise ValueError('Control predictions already persisted')
    from radar_app.reply_match_v7 import requests_for,ReplyMatcherV7
    rows=prepared();risks=risk_values();blocked=preflight();needed=set(requests_for(rows,risks,blocked));predictions={}
    for model in ['06b','4b']:
        data=read(OUT/f'scores-{model}.json')
        if set(data['scores'])!=needed:raise ValueError('Cannot predict from incomplete inference')
        if data['input_sha256']!=sha(OUT/'control-inputs-masked.json'):raise ValueError('Inference inputs changed')
        policy=ReplyMatcherV7(data['scores'],threshold=read(ROOT/'data/reply-config-v7.json')['threshold'],risks=risks,blocked=blocked)
        predictions[model]=[dict(id=r['id'],**policy.decide(r)) for r in rows]
    target=OUT/'control-predictions-only.json';write(target,dict(models=predictions,input_sha256=sha(DATA/'reply-inputs.jsonl'),
        freeze_sha256=sha(OUT/'CONFIG-FREEZE.json'),risk_sha256=sha(OUT/'control-risks.json'),preflight_sha256=sha(OUT/'control-preflight.json'),scores_sha256={m:sha(OUT/f'scores-{m}.json') for m in predictions}))
    (OUT/'control-predictions-only.sha256').write_text(sha(target)+'\n');print('Complete predictions persisted before gold',flush=True)


def score():
    if (OUT/'CONTROL-RESULTS.json').exists():raise ValueError('Independent result already recorded')
    verify_freeze(weights=True)
    target=OUT/'control-predictions-only.json';pred=read(target)
    if sha(target)!=(OUT/'control-predictions-only.sha256').read_text().strip():raise ValueError('Predictions digest differs')
    if pred['freeze_sha256']!=sha(OUT/'CONFIG-FREEZE.json') or pred['input_sha256']!=sha(DATA/'reply-inputs.jsonl'):raise ValueError('Provenance changed')
    if pred['risk_sha256']!=sha(OUT/'control-risks.json'):raise ValueError('Risk scores changed')
    if pred['preflight_sha256']!=sha(OUT/'control-preflight.json'):raise ValueError('Preflight changed')
    for m,expected in pred['scores_sha256'].items():
        if sha(OUT/f'scores-{m}.json')!=expected:raise ValueError('Scores changed after prediction')
    # Sole opening of control labels, after complete predictions of both arms.
    gold={r['id']:r['gold'] for r in jsonl(DATA/'sealed/reply-gold.jsonl')};rows=jsonl(DATA/'reply-inputs.jsonl')
    result=dict(scope='Independent synthetic reply-only control;24ordinary+4fixed;no needs-clustering measurement',
        models={m:metric(rows,p,gold) for m,p in pred['models'].items()},predictions_sha256=sha(target),automatic_release=False,
        limitations=['Only6ordinary and2fixed positive cases','Synthetic author labels; no corporate data or expert labels',
                     'Both arms use v7policy;06b arm is not the full frozenv5c service'])
    write(OUT/'CONTROL-RESULTS.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['tune','freeze','risk','infer-4b','infer-06b','predict','score']);args=ap.parse_args()
    if args.stage.startswith('infer-'):infer(args.stage[6:])
    else:globals()[args.stage]()
