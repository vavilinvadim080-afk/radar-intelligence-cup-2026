"""Recount persisted reply results and verify source/data hashes without neural weights."""
from pathlib import Path
import hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/quality-v7b-reply'
DATA=ROOT/'evaluation/v7b-transport'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
frozen=read(OUT/'CONFIG-FREEZE.json');excluded=[]
for name,expected in frozen['files'].items():
 p=ROOT/name
 if name.endswith('.safetensors') and not p.exists():excluded.append(name);continue
 assert p.is_file() and digest(p)==expected,('frozen source differs',name)
assert digest(DATA/'LOCK.json')==frozen['dataset_lock_sha256']
for name,entry in read(DATA/'LOCK.json')['files'].items():
 assert digest(DATA/name)==(entry['sha256'] if isinstance(entry,dict) else entry),name
predpath=OUT/'control-predictions-only.json';pred=read(predpath)
assert digest(predpath)==(OUT/'control-predictions-only.sha256').read_text().strip()
assert pred['freeze_sha256']==digest(OUT/'CONFIG-FREEZE.json')
assert pred['input_sha256']==digest(DATA/'reply-inputs.jsonl')
assert pred['risk_sha256']==digest(OUT/'control-risks.json')
assert pred['preflight_sha256']==digest(OUT/'control-preflight.json')
for m,expected in pred['scores_sha256'].items():
 assert digest(OUT/f'scores-{m}.json')==expected
 values=read(OUT/f'scores-{m}.json')
 assert values['input_sha256']==digest(OUT/'control-inputs-masked.json')
 assert all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in values['scores'].values())
inputs=lines(DATA/'reply-inputs.jsonl');ids={r['id'] for r in inputs}
assert len(ids)==len(inputs)==28
# No labels are parsed before complete persisted predictions and their hashes.
for values in pred['models'].values():assert len(values)==len(ids) and {r['id'] for r in values}==ids
gold={r['id']:r['gold'] for r in lines(DATA/'sealed/reply-gold.jsonl')};assert set(gold)==ids
result={}
for model,values in pred['models'].items():
 byid={r['id']:r['action'] for r in values};result[model]={}
 for event,positive in [('incoming_ticket','queue_standard'),('fixed_notice_candidate','queue_fixed_notice')]:
  subset=[r for r in inputs if r['event']==event];tp=fp=fn=tn=0
  for row in subset:
   actual=byid[row['id']]==positive;expected=gold[row['id']]['safe_action']==positive
   tp+=bool(actual and expected);fp+=bool(actual and not expected);fn+=bool(not actual and expected);tn+=bool(not actual and not expected)
  result[model][event]=dict(n=len(subset),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
report=read(OUT/'CONTROL-RESULTS.json');assert result==report['models'];assert report['automatic_release'] is False
assert digest(predpath)==report['predictions_sha256']
published=read(ROOT/'data/quality-reply-v7b.json');assert published['models']==result and published['automatic_release'] is False
assert published['provenance']['report_sha256']==digest(OUT/'CONTROL-RESULTS.json')
print(json.dumps(dict(status='PASS',models=result,neural_inference=False,missing_weights_excluded=excluded,scope='Reply-only; both model arms use v7 policy, not full frozenv5c service'),ensure_ascii=False,indent=2))
