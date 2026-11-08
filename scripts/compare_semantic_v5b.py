"""Tune three fixed encoders; lock decisions before an unseen control run.

All inference features are prepared without evaluation answers. Control answers
are first opened after a complete predictions-only artifact is on disk.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'mvp-v13-adapted')]
from radar_app.evaluation import pair_metrics
from radar_app.selection_v5b import DEFAULTS,NeedsSelector,NliReplyPolicy,choose_kind
from radar_app.calibrated_v5b import CalibratedFeatures,CalibratedSelector,CalibratedReplyPolicy,EmbeddingOnlySelector
from scripts.prepare_features_v5b import FeatureStore,read,masked_inputs,feature_contract,validate_feature_provenance,plain_metadata
from scripts.evaluate_reply_tune_v5 import binary

OUT=ROOT/'reports/quality-v5b'
CONFIG=ROOT/'data/semantic-config-v5b.json'
BACKENDS=['tfidf','e5','qwen']
CODE=['scripts/compare_semantic_v5b.py','scripts/prepare_features_v5b.py',
      'scripts/semantic_worker_v5b.py','radar_app/selection_v5b.py',
      'radar_app/semantic_v5b.py','radar_app/semantic.py','radar_app/semantic_client.py',
      'radar_app/reply_policy.py','radar_app/reply_safety.py','radar_app/evaluation.py',
      'radar_app/privacy/mask.py','radar_app/privacy/mask_ner.py',
      'mvp-v13-adapted/radar/embed.py','scripts/evaluate_reply_tune_v5.py',
      'data/semantic-config-v5b.json']
CODE += ['runtime/models/'+folder+'/MANIFEST.json' for folder in
         ['e5-small-v5b','qwen-embedding-v5b','nli-minilm-v5b']]
CODE += ['radar_app/calibrated_v5b.py','scripts/train_heads_v5b.py','data/semantic-heads-v5b.json']
CODE += ['radar_app/model_artifacts_v5b.py','radar_app/local_selection_runtime_v5b.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def f1(m):
    p,r=m.get('precision') or 0,m.get('recall') or 0
    return 2*p*r/(p+r) if p+r else 0.


def score_needs(pred,gold):
    lookup={r['id']:r.get('gold',r) for r in gold}
    return dict(n=len(pred),pair_metrics=pair_metrics(
        [lookup[r['id']]['topic_id'] for r in pred],
        [tuple(r['prediction']) if r['prediction'] else None for r in pred]),
        kind_accuracy=sum(r['kind']==lookup[r['id']]['kind'] for r in pred)/len(pred),
        assigned=sum(r['prediction'] is not None for r in pred))


def score_reply(pred,gold):
    lookup={r['id']:r.get('gold',r) for r in gold}
    wanted=[lookup[r['id']]['safe_action'] for r in pred]
    actions=[r['action'] for r in pred]
    m=binary([v=='queue_standard' for v in wanted],[v=='queue_standard' for v in actions])
    return dict(n=len(pred),**m,expected_standard=wanted.count('queue_standard'),
        queued_standard=actions.count('queue_standard'),
        action_accuracy=sum(a==b for a,b in zip(actions,wanted))/len(pred),
        fixed_notice=binary([v=='queue_fixed_notice' for v in wanted],
                            [v=='queue_fixed_notice' for v in actions]),
        unsafe_standard_replies=m['fp'])


def reply_predict(features,backend,config):
    policy=CalibratedReplyPolicy(features,backend=backend,config=config)
    return [dict(id=r['id'],**policy.decide(r)) for r in features.data['reply_inputs']]


def needs_rank(result):
    m=result['metrics']['pair_metrics'];precision=m['precision'] or 0.
    return (precision>=.85,f1(m),precision,m['recall'] or 0.)


def reply_rank(result):
    m=result['metrics']
    return (m['fp']==0,m['recall'] or 0.,m['precision'] or 0.,m['action_accuracy'])


def tune():
    if (OUT/'CONFIG-FREEZE.json').exists():raise RuntimeError('Already frozen; no tuning on this control')
    features=CalibratedFeatures(FeatureStore(OUT/'tune-features.json'),ROOT/'data/semantic-heads-v5b.json')
    needs=read(ROOT/'evaluation/v5/tune.jsonl');reply=read(ROOT/'evaluation/v5/reply-tune.jsonl')
    truth={r['id']:r['gold']['kind'] for r in needs}
    unique={}
    for row in sorted(features.data['needs_inputs'],key=lambda r:(r['created_at'],r['id'])):
        unique.setdefault((row['product'],row['source_id']),row)
    results={};selected={}
    for backend in BACKENDS:
        features.for_backend(backend);kinds=[]
        for minimum,margin in itertools.product([0.,.2,.4],[0.,.02,.1]):
            config=dict(DEFAULTS,kind_min=minimum,kind_margin=margin)
            accuracy=sum(choose_kind(r['text'],features.kind_scores(r['text']),config)[0]==truth[r['id']]
                         for r in unique.values())/len(unique)
            kinds.append(dict(config=config,accuracy=accuracy))
        kind=max(kinds,key=lambda x:(x['accuracy'],x['config']['kind_min'],x['config']['kind_margin']))
        candidates=[]
        for cosine,pair_min,margin in itertools.product(
                [.0,.3,.5,.65,.75,.85,.9],[.4,.5,.6,.7,.8,.9,.95],[0.,.02,.05]):
            config=dict(kind['config'],cosine_min=cosine,pair_min=pair_min,margin=margin)
            prediction=CalibratedSelector(features,backend=backend,config=config).predict(features.data['needs_inputs'])
            candidates.append(dict(config=config,metrics=score_needs(prediction,needs)))
        best=max(candidates,key=needs_rank)
        replies=[]
        for cosine,pair_min in itertools.product([.0,.3,.5,.65,.75,.85,.9],[.4,.5,.6,.7,.8,.9,.95]):
            config=dict(kind['config'],cosine_min=cosine,pair_min=pair_min,margin=0.,risk_min=.8)
            replies.append(dict(config=config,metrics=score_reply(reply_predict(features,backend,config),reply)))
        best_reply=max(replies,key=reply_rank)
        # Same learned kind head and guards; pair head is removed entirely.
        ablations=[]
        for cosine,margin in itertools.product([.2,.3,.4,.5,.6,.7,.8,.9,.94],[0.,.02,.05]):
            config=dict(kind['config'],cosine_min=cosine,margin=margin)
            p=EmbeddingOnlySelector(features,backend=backend,config=config).predict(features.data['needs_inputs'])
            ablations.append(dict(config=config,metrics=score_needs(p,needs)))
        ablation=max(ablations,key=needs_rank)
        results[backend]=dict(kind_selection=kind,needs=best,autoreply=best_reply,embedding_only=ablation,
            tried=dict(needs=len(candidates),autoreply=len(replies),ablation=len(ablations)))
        selected[backend]=dict(needs=best['config'],reply=best_reply['config'],ablation=ablation['config'])
        print(backend,json.dumps(results[backend],ensure_ascii=False),flush=True)
    # Choose before control. Quality dominates; E5 wins an exact tie to reduce RAM.
    preference={'e5':2,'qwen':1,'tfidf':0}
    backend=max(BACKENDS,key=lambda b:(needs_rank(results[b]['needs']),reply_rank(results[b]['autoreply']),preference[b]))
    report=dict(split='tune',results=results,selected_backend=backend,
                rule='Needs pair F1 subject to precision >= .85 where achievable; reply zero false standard replies before recall. Encoder selected before control.')
    write(OUT/'tune-comparison.json',report)
    write(CONFIG,dict(version='5b',selected_backend=backend,backends=selected,
                      status='experiment; control pending',selection=report['rule']))


def freeze():
    path=OUT/'CONFIG-FREEZE.json'
    if path.exists():raise RuntimeError('Control configuration is already frozen')
    config=json.loads(CONFIG.read_text())
    tune=json.loads((OUT/'tune-features.json').read_text())
    paths=[ROOT/'evaluation/v5/tune.jsonl',ROOT/'evaluation/v5/reply-tune.jsonl']
    needs,reply=masked_inputs(*[read(p) for p in paths])
    validate_feature_provenance(tune,sources={str(p.relative_to(ROOT)):sha(p) for p in paths},
        needs=needs,reply=reply,contract=feature_contract(),metadata=tune['metadata'])
    heads=json.loads((ROOT/'data/semantic-heads-v5b.json').read_text())
    for backend,head in heads.items():
        if head['feature_source_sha256']!=sha(OUT/'tune-features.json'):raise RuntimeError('Heads were trained on another feature file')
        if head['training_sha256']!=sha(ROOT/'evaluation/v5/tune.jsonl'):raise RuntimeError('Heads were trained on another source')
        if plain_metadata({backend:head['encoder']})!=plain_metadata({backend:tune['metadata'][backend]}):raise RuntimeError('Head encoder provenance differs')
    write(path,dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        files={n:sha(ROOT/n) for n in CODE},selected_backend=config['selected_backend'],
        dataset_lock_sha256=sha(ROOT/'evaluation/v5b/CONTROL-LOCK.json'),
        tune_features_sha256=sha(OUT/'tune-features.json'),
        models=plain_metadata(tune['metadata']),preparation_contract_sha256=feature_contract(),
        rule='One final control; predictions on disk before gold; no post-control tuning or winner switching.'))
    print('FROZEN',config['selected_backend'],flush=True)


def check_freeze():
    frozen=json.loads((OUT/'CONFIG-FREEZE.json').read_text())
    if any(sha(ROOT/n)!=value for n,value in frozen['files'].items()):raise RuntimeError('Frozen code/configuration changed')
    if sha(ROOT/'evaluation/v5b/CONTROL-LOCK.json')!=frozen['dataset_lock_sha256']:raise RuntimeError('Control lock changed')
    if sha(OUT/'tune-features.json')!=frozen['tune_features_sha256']:raise RuntimeError('Frozen tune features changed')
    if feature_contract()!=frozen['preparation_contract_sha256']:raise RuntimeError('Frozen model artifacts changed')
    lock=json.loads((ROOT/'evaluation/v5b/CONTROL-LOCK.json').read_text())
    for name,meta in lock['files'].items():
        if sha(ROOT/'evaluation/v5b'/name)!=meta['sha256']:raise RuntimeError('Locked dataset file changed: '+name)
    return frozen


def control():
    frozen=check_freeze();target=OUT/'control-predictions-only.json'
    if target.exists():raise RuntimeError('Control predictions already exist; do not overwrite')
    features=CalibratedFeatures(FeatureStore(OUT/'control-features.json'),ROOT/'data/semantic-heads-v5b.json');config=json.loads(CONFIG.read_text())
    paths=[ROOT/'evaluation/v5b/control-inputs.jsonl',ROOT/'evaluation/v5b/reply-control-inputs.jsonl']
    needs,reply=masked_inputs(*[read(p) for p in paths])
    validate_feature_provenance(features.data,sources={str(p.relative_to(ROOT)):sha(p) for p in paths},
        needs=needs,reply=reply,contract=frozen['preparation_contract_sha256'],metadata=frozen['models'])
    predictions=dict(needs={},autoreply={},embedding_only={})
    for backend in BACKENDS:
        features.for_backend(backend)
        c=config['backends'][backend]
        predictions['needs'][backend]=CalibratedSelector(features,backend=backend,config=c['needs']).predict(features.data['needs_inputs'])
        predictions['autoreply'][backend]=reply_predict(features,backend,c['reply'])
        predictions['embedding_only'][backend]=EmbeddingOnlySelector(features,backend=backend,config=c['ablation']).predict(features.data['needs_inputs'])
    seen=set();expected=[]
    for row in sorted(needs,key=lambda r:(r['created_at'],r['id'])):
        identity=(row['product'],row['source_id'])
        if identity not in seen:expected.append(row['id']);seen.add(identity)
    for scope in ['needs','embedding_only']:
        for backend,pred in predictions[scope].items():
            if [r['id'] for r in pred]!=expected:raise RuntimeError('Control needs denominator differs')
    for pred in predictions['autoreply'].values():
        if [r['id'] for r in pred]!=[r['id'] for r in reply]:raise RuntimeError('Control reply denominator differs')
    write(target,predictions)
    # No answers are read until the complete prediction artifact is persisted.
    gold=read(ROOT/'evaluation/v5b/control-gold.jsonl')
    reply_gold=read(ROOT/'evaluation/v5b/reply-control-gold.jsonl')
    result=dict(version='5b',status='measured',selected_before_control=frozen['selected_backend'],
        population='Fresh unseen synthetic semantic families; no GPN tickets or internal model',
        needs={b:score_needs(p,gold) for b,p in predictions['needs'].items()},
        autoreply={b:score_reply(p,reply_gold) for b,p in predictions['autoreply'].items()},
        embedding_only={b:score_needs(p,gold) for b,p in predictions['embedding_only'].items()},
        protocol=dict(predictions_before_gold=True,expert_annotation=False,
                      needs_input_rows=len(features.data['needs_inputs']),reply_input_rows=len(features.data['reply_inputs']),
                      freeze_sha256=sha(OUT/'CONFIG-FREEZE.json'),control_lock_sha256=frozen['dataset_lock_sha256'],
                      control_features_sha256=sha(OUT/'control-features.json')),
        model_metadata=features.data['metadata'],
        limitations=['Pair precision/recall measures grouping, not cards or factual correctness',
                     'Small synthetic positive sample; zero observed errors is not proof of production safety',
                     'Precomputed NLI pair grid is not online throughput',
                     'Local prepared queues only, no external delivery'])
    chosen=frozen['selected_backend'];n=result['needs'][chosen]['pair_metrics'];r=result['autoreply'][chosen]
    result['gate']=dict(needs=(n['precision'] or 0)>=.85 and (n['recall'] or 0)>=.7,
                        autoreply=r['fp']==0 and (r['recall'] or 0)>=.7,
                        auto_enable=False,reason='Real data and human acceptance are unavailable')
    write(ROOT/'data/quality-v5b.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['tune','freeze','control']);args=ap.parse_args()
    {'tune':tune,'freeze':freeze,'control':control}[args.action]()
