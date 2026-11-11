"""Independently recount published v5c predictions using only Python's standard library.

Checks public frozen sources and supplied dataset hashes. Model weights and
private dataset-authoring records are explicitly excluded, not called verified.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def confusion(expected,predicted):
    tp=sum(a and b for a,b in zip(expected,predicted));fp=sum(not a and b for a,b in zip(expected,predicted))
    fn=sum(a and not b for a,b in zip(expected,predicted));tn=sum(not a and not b for a,b in zip(expected,predicted))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)


def checked_path(base,name):
    p=(base/name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(base.resolve()):
        raise ValueError('Unsafe artifact path')
    return p


def check(root=ROOT):
    root=Path(root);directory=root/'reports/quality-v5c'
    report=read(directory/'control-metrics.json');freeze=read(directory/'CONFIG-FREEZE.json')
    path=directory/'control-predictions-only.json';pred=read(path)
    assert digest(path)==report['provenance']['predictions_sha256']
    assert digest(directory/'CONFIG-FREEZE.json')==report['provenance']['freeze_sha256']
    assert digest(root/'evaluation/v5c/CONTROL-LOCK.json')==freeze['dataset_lock_sha256']
    checked=[];omitted=[]
    for name,sha in freeze['files'].items():
        if name.startswith('runtime/models/') and Path(name).name!='MANIFEST.json':
            omitted.append(name);continue
        assert digest(checked_path(root,name))==sha,name
        checked.append(name)
    dataset=root/'evaluation/v5c';lock=read(dataset/'CONTROL-LOCK.json')
    for name,entry in lock['files'].items():
        if name in {'sealed/authoring.py','sealed/adjudication.json','sealed/source-lock.json'}:
            omitted.append('evaluation/v5c/'+name);continue
        sha=entry.get('sha256') if isinstance(entry,dict) else entry
        assert digest(checked_path(dataset,name))==sha,name
    needs=jsonl(dataset/'control-inputs.jsonl');replies=jsonl(dataset/'reply-control-inputs.jsonl')
    gold={r['id']:r['gold'] for r in jsonl(dataset/'sealed/control-gold.jsonl')}
    rgold={r['id']:r['gold'] for r in jsonl(dataset/'sealed/reply-control-gold.jsonl')}
    unique={}
    for row in sorted(needs,key=lambda r:(r['created_at'],r['id'])):
        unique.setdefault((row['product'],row.get('source_id',row['id'])),row)
    rows=list(unique.values());assert len(rows)==184 and len(replies)==132
    recounted={}
    for model in ('baseline','candidate'):
        body=pred[model];np={r['id']:r for r in body['needs']};rp={r['id']:r for r in body['reply']}
        assert len(np)==len(body['needs']) and set(np)=={r['id'] for r in rows}
        assert len(rp)==len(body['reply']) and set(rp)=={r['id'] for r in replies}
        truth={r['id']:None if gold[r['id']]['kind']=='unknown' else (r['product'],gold[r['id']]['kind'],gold[r['id']]['topic_id']) for r in rows}
        actual={r['id']:None if np[r['id']]['prediction'] is None else tuple(np[r['id']]['prediction']) for r in rows}
        ids=[r['id'] for r in rows];pairs=list(combinations(ids,2))
        m=confusion([truth[a] is not None and truth[a]==truth[b] for a,b in pairs],
                    [actual[a] is not None and actual[a]==actual[b] for a,b in pairs])
        expected=report['models'][model]
        for k,v in m.items():assert expected['needs']['pairs'][k]==v,(model,k)
        accuracy=sum(np[i]['kind']==gold[i]['kind'] for i in ids)/len(ids)
        assert expected['needs']['kind_accuracy']==accuracy
        result=dict(kind_accuracy=accuracy,pairs=m)
        for event,key,action in [('incoming_ticket','ordinary_reply','queue_standard'),
                                 ('fixed_notice_candidate','fixed_notice','queue_fixed_notice')]:
            subset=[r for r in replies if r['event']==event]
            values=confusion([rgold[r['id']]['safe_action']==action for r in subset],
                             [rp[r['id']]['action']==action for r in subset])
            for k,v in values.items():assert expected[key][k]==v,(model,key,k)
            result[key]=values
        assert expected['gates']['auto_enable'] is False
        recounted[model]=result
    return dict(status='PASS',models=recounted,verified_frozen_files=len(checked),
                excluded_from_hash_verification=omitted,neural_inference_performed=False)


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
