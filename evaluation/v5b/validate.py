"""Structural/integrity audit only. No matcher, embedding or application imports."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import unicodedata

ROOT=Path(__file__).resolve().parent
DEFAULT=dict(needs_rows=120,families=24,family_size=4,unknown=20,duplicate=4,
             reply_rows=80,standard_positive=20,reply_families=6)
NEED_KEYS={'id','product','author_id','source_id','created_at','received_at','channel','text'}
REPLY_KEYS={'id','event','product','kind','references','query','approved','state','fix_confirmed','human_closed','fixed_notice_approved','seen_message_ids'}
NG_KEYS={'product','kind','topic_id','direction','route','is_duplicate_delivery','synthetic','reason'}
RG_KEYS={'same_topic','eligibility','safe_action','reference_topic_id','query_kind','query_direction','reason','synthetic'}
KINDS={'wish','bug','usability','service','unknown'}
PRODUCTS={'portal','webneft'}
FORBIDDEN={'gold','topic_id','same_topic','eligibility','safe_action','reference_topic_id','query_kind','query_direction','reason','stratum','family','expected'}
OLD_NAMES=['control-inputs.jsonl','tune.jsonl','reply-control-inputs.jsonl','reply-tune.jsonl']
FILES=['control-inputs.jsonl','control-gold.jsonl','reply-control-inputs.jsonl','reply-control-gold.jsonl']

def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def digest(path):return sha256(path.read_bytes()).hexdigest()

def leaking(obj):
    if isinstance(obj,dict):return bool(set(obj)&FORBIDDEN) or any(leaking(v) for v in obj.values())
    if isinstance(obj,list):return any(leaking(v) for v in obj)
    return False

def validate_bundle(needs,needs_gold,replies,reply_gold,expected=None):
    spec=expected or DEFAULT;errors=set();seen={};families=Counter();unknown=duplicates=0
    if len(needs)!=len(needs_gold) or len(replies)!=len(reply_gold):return ['gold_alignment']
    for row,envelope in zip(needs,needs_gold):
        if leaking(row):errors.add('input_label_leak')
        if set(row)!=NEED_KEYS or not all(isinstance(row.get(k),str) and row[k] for k in NEED_KEYS):errors.add('needs_schema')
        if set(envelope)!={'id','gold'} or row.get('id')!=envelope.get('id'):errors.add('gold_alignment')
        g=envelope.get('gold',{})
        if set(g)!=NG_KEYS or g.get('synthetic') is not True or g.get('kind') not in KINDS or g.get('product')!=row.get('product') or row.get('product') not in PRODUCTS:errors.add('needs_gold_schema')
        try:
            created=datetime.fromisoformat(row['created_at']);received=datetime.fromisoformat(row['received_at'])
            if not created.tzinfo or not received.tzinfo or received<created:errors.add('needs_timestamp')
        except (KeyError,ValueError,TypeError):errors.add('needs_timestamp')
        identity={k:v for k,v in row.items() if k!='received_at'}
        if row.get('id') in seen:
            duplicates+=1
            if identity!=seen[row['id']][0]:errors.add('duplicate_content_drift')
            if g.get('is_duplicate_delivery') is not True:errors.add('duplicate_gold_flag')
            if row.get('received_at','')<=seen[row['id']][1]:errors.add('duplicate_delivery_order')
        else:
            seen[row.get('id')]=(identity,row.get('received_at',''))
            if g.get('is_duplicate_delivery') is not False:errors.add('duplicate_gold_flag')
            if g.get('kind')=='unknown':
                unknown+=1
                if g.get('topic_id') is not None or g.get('route')!='review' or g.get('direction')!='undetermined':errors.add('unknown_contract')
            else:
                families[g.get('topic_id')]+=1
                if not isinstance(g.get('topic_id'),str) or not g['topic_id'].startswith(f"{row.get('product')}:{g.get('kind')}:") or g.get('route')!='candidate':errors.add('needs_topic_contract')
    if len(needs)!=spec['needs_rows'] or len(families)!=spec['families'] or set(families.values())!={spec['family_size']} or unknown!=spec['unknown'] or duplicates!=spec['duplicate']:errors.add('needs_counts')
    positive=0;reference_families=set();reply_ids=set()
    for row,envelope in zip(replies,reply_gold):
        if leaking(row):errors.add('input_label_leak')
        if set(row)!=REPLY_KEYS or not isinstance(row.get('query'),dict) or not isinstance(row.get('references'),list):errors.add('reply_schema');continue
        if set(envelope)!={'id','gold'} or row.get('id')!=envelope.get('id'):errors.add('gold_alignment')
        if row.get('id') in reply_ids:errors.add('reply_duplicate_case_id')
        reply_ids.add(row.get('id'));g=envelope.get('gold',{});q=row['query'];refs=row['references']
        if set(g)!=RG_KEYS or type(g.get('same_topic')) is not bool or type(g.get('eligibility')) is not bool or g.get('synthetic') is not True or g.get('query_kind') not in KINDS or not isinstance(g.get('reason'),list) or not g.get('reason'):errors.add('reply_gold_schema')
        if any(type(row.get(k)) is not bool for k in ['approved','fix_confirmed','human_closed','fixed_notice_approved']) or type(q.get('consent')) is not bool:errors.add('reply_boolean_schema')
        reference_families.add(g.get('reference_topic_id'))
        if g.get('eligibility')!=(g.get('safe_action')=='queue_standard'):errors.add('standard_eligibility_contract')
        if g.get('eligibility'):
            positive+=1
            valid_refs=len(refs)>=2 and len({r.get('author_id') for r in refs})>=2 and all(r.get('source_verified') is True and r.get('product')==row.get('product') and r.get('kind')=='bug' for r in refs)
            if not (g.get('same_topic') and row.get('event')=='incoming_ticket' and row.get('state')=='open' and row.get('approved') is True and q.get('consent') is True and q.get('product')==row.get('product') and not row.get('fix_confirmed') and q.get('id') not in row.get('seen_message_ids',[]) and g.get('query_kind')=='bug' and g.get('query_direction')=='restore' and valid_refs):errors.add('unsafe_standard_positive')
        if g.get('safe_action')=='queue_fixed_notice' and not (row.get('event')=='fixed_notice_candidate' and row.get('state') in {'closed','fix_verified'} and row.get('fix_confirmed') is True and row.get('human_closed') is True and row.get('fixed_notice_approved') is True):errors.add('unsafe_fixed_notice')
        if g.get('safe_action')=='ignore_duplicate' and q.get('id') not in row.get('seen_message_ids',[]):errors.add('duplicate_reply_contract')
    if len(replies)!=spec['reply_rows'] or positive!=spec['standard_positive'] or len(reference_families)!=spec['reply_families']:errors.add('reply_counts')
    return sorted(errors)

def normalized(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())

def source_audit(needs,replies):
    old=ROOT.parent/'v5';texts=set();old_topics=set();sources={}
    for name in OLD_NAMES:
        path=old/name;rows=read_rows(path);sources[str(path.relative_to(ROOT.parents[1]))]=dict(sha256=digest(path),rows=len(rows))
        for x in rows:
            if 'text' in x:texts.add(normalized(x['text']))
            if 'query' in x:texts.add(normalized(x['query']['text']))
            for ref in x.get('references',[]):texts.add(normalized(ref['text']))
            # Only the explicitly permitted tune files contain annotations here.
            if name.endswith('tune.jsonl'):
                t=x.get('gold',{}).get('topic_id') or x.get('gold',{}).get('reference_topic_id')
                if t:old_topics.add(t)
    newtexts=[x['text'] for x in needs]+[x['query']['text'] for x in replies]+[ref['text'] for x in replies for ref in x['references']]
    overlap=sum(normalized(x) in texts for x in newtexts)
    return sources,old_topics,overlap

def verify_lock():
    lock=json.loads((ROOT/'CONTROL-LOCK.json').read_text());bad=[]
    for name,entry in lock['files'].items():
        if not (ROOT/name).exists() or digest(ROOT/name)!=entry['sha256']:bad.append(name)
    for name,entry in lock['old_read_allowlist'].items():
        if digest(ROOT.parents[1]/name)!=entry['sha256']:bad.append('old_source_changed')
    return bad

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lock',action='store_true');parser.add_argument('--verify-lock',action='store_true');args=parser.parse_args()
    if args.verify_lock:
        bad=verify_lock();print(json.dumps(dict(lock_valid=not bad,changed_files=bad),ensure_ascii=False));return bool(bad)
    needs,ng,replies,rg=[read_rows(ROOT/name) for name in FILES]
    errors=validate_bundle(needs,ng,replies,rg)
    sources,old_topics,overlap=source_audit(needs,replies)
    if overlap:errors.append('old_text_overlap')
    new_topics={x['gold']['topic_id'] for x in ng if x['gold']['topic_id']}
    if new_topics&old_topics:errors.append('old_topic_id_overlap')
    authors={t:set() for t in new_topics}
    for row,g in zip(needs,ng):
        if g['gold']['topic_id']:authors[g['gold']['topic_id']].add(row['author_id'])
    if set(map(len,authors.values()))!={3}:errors.append('author_repeat_distribution')
    stats=dict(needs_rows=len(needs),needs_unique_ids=len({x['id'] for x in needs}),needs_families=len(new_topics),unknown=sum(x['gold']['kind']=='unknown' for x in ng),duplicate_deliveries=sum(x['gold']['is_duplicate_delivery'] for x in ng),reply_rows=len(replies),standard_positive=sum(x['gold']['eligibility'] for x in rg),standard_negative=sum(not x['gold']['eligibility'] for x in rg),reply_families=len({x['gold']['reference_topic_id'] for x in rg}),fixed_notice_positive=sum(x['gold']['safe_action']=='queue_fixed_notice' for x in rg),old_exact_text_overlap=overlap)
    report=dict(passed=not errors,errors=sorted(set(errors)),counts=stats,old_read_allowlist=sources,no_model_imports=True,labels_not_in_inputs=True,semantic_nonoverlap='Manually reviewed against allowed inputs/tune; automatic audit checks exact text/topic ids only')
    (ROOT/'VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=not errors,errors=report['errors'],counts=stats),ensure_ascii=False))
    if errors:return 1
    if args.lock:
        names=FILES+['PROTOCOL.md','SCHEMA.json','validate.py','test_validate.py','VALIDATION.json','README.md','private/authoring.py','private/adjudication.jsonl','private/semantic-review.md']
        lock=dict(version='v5b',created_utc=datetime.now(timezone.utc).isoformat(),counts=stats,files={n:dict(sha256=digest(ROOT/n),bytes=(ROOT/n).stat().st_size) for n in names},old_read_allowlist=sources,freeze_contract='Control locked before backend test; parent may read gold only after configuration freeze; no matcher/model read or executed by author',gold_format='id plus gold envelope; gold fields copied from the permitted v5 tune schema')
        if (ROOT/'CONTROL-LOCK.json').exists():
            if verify_lock():raise SystemExit('Existing lock differs; do not overwrite control')
        else:(ROOT/'CONTROL-LOCK.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
