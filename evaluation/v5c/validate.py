#!/usr/bin/env python3
"""Independent v5c contract validator. Prints aggregate counts only, including on failure.

Uses only standard-library code and data contracts. Never imports authoring, ML,
runtime or application modules. --seal creates an immutable-byte SHA-256 manifest;
subsequent runs verify it. Access to sealed/ is for the independent evaluator.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
import argparse
import copy
import hashlib
import json
import re
import sys
import unicodedata

BASE=Path(__file__).resolve().parent
KINDS={'wish','bug','usability','service','unknown'}
DIRECTIONS={'enable','disable','introduce','restore','clarify','simplify','undetermined'}
ACTIONS={'operator','needs_approval','ignore_duplicate','needs_fix_approval','queue_standard','queue_fixed_notice'}
NEED_KEYS={'id','product','author_id','source_id','created_at','received_at','channel','text'}
REPLY_KEYS={'id','event','product','kind','references','query','approved','state','fix_confirmed','human_closed','fixed_notice_approved','seen_message_ids'}
QUERY_KEYS={'id','product','author_id','created_at','channel','consent','text'}
REF_KEYS={'id','product','kind','author_id','text','source_verified'}
NEED_GOLD={'product','kind','topic_id','direction','route','is_duplicate_delivery','synthetic','reason'}
REPLY_GOLD={'same_topic','eligibility','safe_action','reference_topic_id','query_kind','query_direction','reason','synthetic'}
CHANNELS={'web_form','phone_transcript','email','call_asr','chat'}
DATA_FILES=['tune-inputs.jsonl','tune-gold.jsonl','reply-tune-inputs.jsonl','reply-tune-gold.jsonl',
            'control-inputs.jsonl','sealed/control-gold.jsonl','reply-control-inputs.jsonl','sealed/reply-control-gold.jsonl']
LOCK_FILES=DATA_FILES+['PROTOCOL.md','SCHEMA.json','validate.py','sealed/authoring.py','sealed/adjudication.json','sealed/source-lock.json']
class Invalid(Exception): pass


def check(condition,rule):
    if not condition:raise Invalid(rule)


def keys(obj,expected,rule):
    check(type(obj) is dict and set(obj)==expected,rule)


def opaque(value,prefix):
    return type(value) is str and bool(re.fullmatch(prefix+r'-[0-9a-f]{24}',value))


def norm(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold().replace('ё','е')))


def iso(value):
    check(type(value) is str,'timestamp_type')
    try:v=datetime.fromisoformat(value)
    except (ValueError,TypeError):raise Invalid('timestamp_format')
    check(v.tzinfo is not None,'timestamp_timezone')
    return v


def load_lines(path):
    raw=path.read_bytes()
    check(raw.endswith(b'\n'),'jsonl_final_newline')
    check(not raw.startswith(b'\xef\xbb\xbf'),'jsonl_no_bom')
    try:
        lines=raw.decode('utf-8').splitlines()
        check(all(lines),'jsonl_no_empty_lines')
        return [json.loads(line,object_pairs_hook=no_duplicate_keys) for line in lines]
    except (ValueError,UnicodeError):raise Invalid('jsonl_encoding_or_syntax')


def no_duplicate_keys(items):
    result={}
    for key,value in items:
        check(key not in result,'json_unique_keys');result[key]=value
    return result


def needs_contract(rows,golds):
    check(len(rows)==len(golds)==188,'needs_delivery_population')
    seen={};groups=defaultdict(list);authors=set();texts=[];dups=0;kinds=Counter()
    for row,wrapper in zip(rows,golds):
        keys(row,NEED_KEYS,'needs_input_keys');keys(wrapper,{'id','gold'},'needs_gold_envelope')
        check(wrapper['id']==row['id'],'needs_alignment')
        g=wrapper['gold'];keys(g,NEED_GOLD,'needs_gold_keys')
        for field,prefix in [('id','n'),('product','p'),('author_id','u'),('source_id','s')]:
            check(opaque(row[field],prefix),'needs_opaque_identity')
        check(row['channel'] in CHANNELS,'needs_channel')
        check(type(row['text']) is str and row['text'].strip(),'needs_text')
        check(not re.search(r'\b(wish|bug|usability|service|unknown|topic_id|eligibility|safe_action)\b',row['text'],re.I),'needs_no_label_text')
        check(iso(row['received_at'])>=iso(row['created_at']),'needs_time_order')
        check(g['kind'] in KINDS and g['direction'] in DIRECTIONS,'needs_gold_enums')
        check(type(g['is_duplicate_delivery']) is bool and g['synthetic'] is True,'needs_gold_flags')
        check(g['product']==row['product'],'needs_product_truth')
        check(type(g['reason']) is str and bool(g['reason']),'needs_reason')
        if g['kind']=='unknown':
            check(g['topic_id'] is None and g['direction']=='undetermined' and g['route']=='review','needs_abstention')
        else:
            check(opaque(g['topic_id'],'t') and g['route']=='candidate','needs_candidate_topic')
            check(g['direction']!='undetermined','needs_candidate_direction')
        if row['id'] in seen:
            prior,pg=seen[row['id']]
            check(g['is_duplicate_delivery'] is True and not pg['is_duplicate_delivery'],'needs_duplicate_flag')
            check({k:v for k,v in row.items() if k!='received_at'}=={k:v for k,v in prior.items() if k!='received_at'},'needs_duplicate_identity')
            check(iso(row['received_at'])>iso(prior['received_at']),'needs_duplicate_time')
            check({k:v for k,v in g.items() if k!='is_duplicate_delivery'}=={k:v for k,v in pg.items() if k!='is_duplicate_delivery'},'needs_duplicate_gold')
            dups+=1
        else:
            check(not g['is_duplicate_delivery'],'needs_first_delivery_flag')
            seen[row['id']]=(row,g);authors.add(row['author_id']);texts.append(norm(row['text']));kinds[g['kind']]+=1
            if g['topic_id'] is not None:groups[g['topic_id']].append((row,g))
    check(len(seen)==184 and dups==4,'needs_unique_and_duplicates')
    check(len(set(texts))==184,'needs_unique_texts')
    check(len({r['source_id'] for r,g in seen.values()})==184,'needs_unique_sources')
    check(Counter(map(len,groups.values()))==Counter({4:40,1:8}),'needs_family_sizes')
    check(kinds==Counter({'wish':42,'bug':42,'usability':42,'service':42,'unknown':16}),'needs_kind_population')
    check(len(authors)==144,'needs_honest_author_count')
    pair_count=0
    for topic,members in groups.items():
        check(len({(g['product'],g['kind'],g['direction']) for r,g in members})==1,'family_homogeneity')
        if len(members)==4:check(len({r['author_id'] for r,g in members})==3,'family_independent_authors')
        pair_count+=len(list(combinations(members,2)))
    check(pair_count==240,'needs_true_pairs')
    # The product namespace is balanced across kinds, not an encoded target label.
    product_kinds=defaultdict(set)
    for row,g in seen.values():product_kinds[row['product']].add(g['kind'])
    check(len(product_kinds)==4 and all(k==KINDS for k in product_kinds.values()),'products_not_kind_codes')
    return dict(deliveries=188,unique_messages=184,duplicate_deliveries=4,families_of_four=40,
                true_pairs=240,known_singletons=8,unknown_singletons=16,unique_authors=144),seen,groups


def reply_contract(rows,golds):
    check(len(rows)==len(golds)==132,'reply_population')
    ids=set();msgs=set();ordinary_pos=ordinary_neg=fix_pos=fix_neg=hard=0
    for row,wrapper in zip(rows,golds):
        keys(row,REPLY_KEYS,'reply_input_keys');keys(wrapper,{'id','gold'},'reply_gold_envelope')
        g=wrapper['gold'];keys(g,REPLY_GOLD,'reply_gold_keys');check(wrapper['id']==row['id'],'reply_alignment')
        check(opaque(row['id'],'r') and row['id'] not in ids,'reply_identity');ids.add(row['id'])
        check(opaque(row['product'],'p') and row['kind'] in KINDS,'reply_candidate_context')
        check(row['event'] in {'incoming_ticket','fixed_notice_candidate'},'reply_event')
        check(row['state'] in {'open','closed','fix_verified','fix_reported'},'reply_state')
        for flag in ['approved','fix_confirmed','human_closed','fixed_notice_approved']:check(type(row[flag]) is bool,'reply_boolean_context')
        check(type(row['seen_message_ids']) is list and len(set(row['seen_message_ids']))==len(row['seen_message_ids']),'reply_seen_list')
        for mid in row['seen_message_ids']:check(opaque(mid,'m'),'reply_seen_identity')
        q=row['query'];keys(q,QUERY_KEYS,'reply_query_keys')
        check(opaque(q['id'],'m') and q['id'] not in msgs,'reply_unique_query_identity');msgs.add(q['id'])
        check(opaque(q['product'],'p') and opaque(q['author_id'],'u'),'reply_query_identity')
        check(type(q['consent']) is bool and q['channel'] in CHANNELS,'reply_query_context');iso(q['created_at'])
        check(type(q['text']) is str and bool(q['text'].strip()),'reply_query_text')
        check(type(row['references']) is list,'reply_reference_list')
        ref_ids=set();ref_authors=set()
        for ref in row['references']:
            keys(ref,REF_KEYS,'reply_reference_keys')
            check(opaque(ref['id'],'e') and opaque(ref['author_id'],'u'),'reply_reference_identity')
            check(ref['id'] not in ref_ids,'reply_reference_unique_identity');ref_ids.add(ref['id']);ref_authors.add(ref['author_id'])
            check(ref['product']==row['product'] and ref['kind']==row['kind'],'reply_reference_context')
            check(type(ref['source_verified']) is bool,'reply_source_flag')
            check(type(ref['text']) is str and bool(ref['text'].strip()),'reply_reference_text')
        check(len(ref_authors)==len(row['references']) and q['author_id'] not in ref_authors,'reply_independent_sources')
        check(type(g['same_topic']) is bool and type(g['eligibility']) is bool and g['synthetic'] is True,'reply_gold_flags')
        check(g['safe_action'] in ACTIONS and g['query_kind'] in KINDS and g['query_direction'] in DIRECTIONS,'reply_gold_enums')
        check(g['reference_topic_id'] is None or opaque(g['reference_topic_id'],'t'),'reply_gold_reference_topic')
        check(type(g['reason']) is list and len(g['reason'])==1 and type(g['reason'][0]) is str,'reply_gold_reason')
        check(g['eligibility']==(g['safe_action']=='queue_standard'),'eligibility_ordinary_only')
        trusted=len(row['references'])>=2 and all(ref['source_verified'] for ref in row['references'])
        ordinary_ready=(row['approved'] and q['consent'] and row['state']=='open' and not row['fix_confirmed']
                        and not row['human_closed'] and not row['fixed_notice_approved'] and trusted
                        and q['id'] not in row['seen_message_ids'] and q['product']==row['product'])
        if row['event']=='incoming_ticket':
            if g['eligibility']:
                ordinary_pos+=1
                check(ordinary_ready and g['same_topic'],'ordinary_admission_prerequisites')
                check(g['reason']==['same_issue_safe_open_approved'],'ordinary_positive_reason')
            else:ordinary_neg+=1
            if g['reason'][0] in {'different_issue','opposite_direction','different_kind'}:
                check(ordinary_ready and not g['same_topic'] and g['safe_action']=='operator','hard_negative_no_flag_shortcut');hard+=1
            check(g['safe_action'] not in {'queue_fixed_notice','needs_fix_approval'},'ordinary_fix_is_separate')
        else:
            check(not g['eligibility'],'fix_not_ordinary_positive')
            if g['safe_action']=='queue_fixed_notice':
                fix_pos+=1
                check(g['same_topic'] and trusted and q['consent'] and row['approved'] and row['state']=='closed'
                      and row['fix_confirmed'] and row['human_closed'] and row['fixed_notice_approved']
                      and q['id'] not in row['seen_message_ids'] and q['product']==row['product'],'fix_admission_prerequisites')
            else:fix_neg+=1
    check((ordinary_pos,ordinary_neg,fix_pos,fix_neg)==(30,90,6,6),'reply_stratified_denominators')
    check(hard==36 and hard>20,'semantic_hard_negative_population')
    return dict(total=132,ordinary=120,ordinary_positive=30,ordinary_negative=90,fix=12,fix_positive=6,fix_negative=6,
                semantic_hard_negatives_without_flag_shortcut=hard)


def adjudicate(split,ni,ng,ri,rg,meta):
    """Check independently authored evidence against actual files, without importing the writer."""
    audit=meta['splits'][split];catalog={};idmap={}
    for f in audit['needs']['families']:
        check(f['topic_id'] not in catalog,'adjudication_unique_family')
        catalog[f['topic_id']]=f
        check(len(f['texts'])==4 and len(f['ids'])==4,'adjudication_family_members')
        for ident,text in zip(f['ids'],f['texts']):idmap[ident]=(f,text)
    for f in audit['needs']['singletons']:
        catalog[f['topic_id']]=f;idmap[f['id']]=(f,f['text'])
    unknown_ids=set(audit['needs']['unknown_ids'])
    for row,wrapper in zip(ni,ng):
        g=wrapper['gold']
        if row['id'] in unknown_ids:check(g['kind']=='unknown' and g['topic_id'] is None,'adjudication_unknown')
        else:
            check(row['id'] in idmap,'adjudication_covered_need');f,text=idmap[row['id']]
            check(norm(row['text'])==norm(text),'adjudication_need_text')
            check((row['product'],g['kind'],g['direction'],g['topic_id'])==(f['product'],f['kind'],f['direction'],f['topic_id']),'adjudication_need_truth')
    reply_audit={a['id']:a for a in audit['replies']}
    check(len(reply_audit)==132,'adjudication_reply_coverage')
    distinct_semantic_pairs=set();reason_counts=Counter()
    for row,wrapper in zip(ri,rg):
        a=reply_audit[row['id']];g=wrapper['gold'];ref=catalog[a['ref_topic']]
        qf=catalog.get(a['query_topic']);q=row['query'];scenario=a['scenario'];reason=g['reason'][0]
        reason_counts[reason]+=1
        check(row['product']==ref['product'] and row['kind']==ref['kind'],'adjudication_reference_context')
        for j,r in enumerate(row['references']):check(norm(r['text'])==norm(ref['texts'][j]),'adjudication_reference_evidence')
        if qf:
            check((g['query_kind'],g['query_direction'])==(qf['kind'],qf['direction']),'adjudication_query_truth')
            text=norm(q['text']);candidates=[norm(t) for t in qf['texts']]
            if a['query_text_mode']=='suffix':check(any(text.startswith(t+' ') for t in candidates),'adjudication_safety_suffix')
            else:check(text in candidates,'adjudication_query_evidence')
        else:check(g['query_kind']=='unknown' and g['query_direction']=='undetermined','adjudication_unknown_reply')
        expected_same=(qf is not None and ref['topic_id']==qf['topic_id'] and q['product']==ref['product'] and bool(row['references']))
        check(g['same_topic']==expected_same,'adjudication_topic_relation')
        if row['references']:check(g['reference_topic_id']==ref['topic_id'],'adjudication_topic_id')
        if scenario.startswith('semantic_'):
            distinct_semantic_pairs.add((ref['topic_id'],qf['topic_id']))
            if scenario=='semantic_subject':check(ref['kind']==qf['kind'],'semantic_same_kind')
            if scenario=='semantic_direction':check({ref['direction'],qf['direction']}=={'enable','disable'},'semantic_opposite_direction')
            if scenario=='semantic_intent':check(ref['kind']!=qf['kind'],'semantic_different_intent')
        expected={
          'ordinary_positive':('queue_standard','same_issue_safe_open_approved'),
          'semantic_subject':('operator','different_issue'),'semantic_direction':('operator','opposite_direction'),
          'semantic_intent':('operator','different_kind'),'other_product':('operator','different_product'),
          'personal':('operator','individual_personal_action'),'secrets':('operator','credentials'),
          'injection':('operator','prompt_injection'),'low_certainty':('operator','insufficient_context'),
          'unverified':('operator','unverified_references'),'no_consent':('operator','no_consent'),
          'no_approval':('needs_approval','not_human_approved'),'duplicate':('ignore_duplicate','already_queued'),
          'stale':('operator','incident_already_fixed_new_complaint'),'missing_sources':('operator','missing_references'),
          'fix_positive':('queue_fixed_notice','separate_fixed_notice_path'),'fix_unconfirmed':('operator','fix_not_verified'),
          'fix_not_closed':('needs_fix_approval','closure_or_notice_not_approved'),
          'fix_no_notice_approval':('needs_fix_approval','closure_or_notice_not_approved'),
          'fix_no_consent':('operator','no_consent'),'fix_other_issue':('operator','different_issue'),
          'fix_unverified_sources':('operator','unverified_references')}
        check(scenario in expected and (g['safe_action'],reason)==expected[scenario],'adjudication_policy_outcome')
        if scenario in {'no_consent','fix_no_consent'}:check(q['consent'] is False,'adjudication_consent')
        if scenario in {'unverified','fix_unverified_sources'}:check(not all(r['source_verified'] for r in row['references']),'adjudication_unverified_source')
        if scenario=='no_approval':check(not row['approved'],'adjudication_approval')
        if scenario=='duplicate':check(q['id'] in row['seen_message_ids'],'adjudication_duplicate')
        if scenario=='stale':check(row['state']=='closed' and row['fix_confirmed'] and row['human_closed'],'adjudication_stale')
        if scenario=='missing_sources':check(row['references']==[],'adjudication_missing_sources')
        if scenario=='fix_unconfirmed':check(not row['fix_confirmed'],'adjudication_fix_unconfirmed')
        if scenario=='fix_not_closed':check(not row['human_closed'],'adjudication_fix_closure')
        if scenario=='fix_no_notice_approval':check(not row['fixed_notice_approved'],'adjudication_fix_approval')
    check(len(distinct_semantic_pairs)==36,'semantic_distinct_topic_pairs')
    required={'different_issue','different_kind','opposite_direction','different_product','individual_personal_action','credentials',
              'prompt_injection','insufficient_context','unverified_references','no_consent','not_human_approved','already_queued',
              'incident_already_fixed_new_complaint','missing_references','fix_not_verified','closure_or_notice_not_approved'}
    check(required<=set(reason_counts),'all_negative_categories_present')


def all_texts(ni,ri):
    texts={norm(row['text']) for row in ni}
    texts.update(norm(row['query']['text']) for row in ri)
    texts.update(norm(ref['text']) for row in ri for ref in row['references'])
    return texts


def identity_set(ni,ri):
    found=set()
    for row in ni:
        found.update(row[k] for k in ['id','author_id','source_id'])
    for row in ri:
        found.add(row['id']);found.add(row['query']['id']);found.add(row['query']['author_id'])
        for ref in row['references']:found.update([ref['id'],ref['author_id']])
    return found


def old_contract_data():
    texts=set();ids=set();topics=set()
    for version in ['v5','v5b']:
        for path in (BASE.parent/version).glob('*.jsonl'):
            for row in load_lines(path):
                if 'text' in row:texts.add(norm(row['text']))
                if 'query' in row:
                    texts.add(norm(row['query']['text']))
                    texts.update(norm(r['text']) for r in row.get('references',[]))
                if 'id' in row:ids.add(row['id'])
                if isinstance(row.get('gold'),dict) and row['gold'].get('topic_id'):topics.add(row['gold']['topic_id'])
    return texts,ids,topics


def snapshot():
    return {p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in LOCK_FILES}


def mutation_checks(ni,ng,ri,rg):
    tests=[]
    a=copy.deepcopy(ni);a[0]['gold']={};tests.append(lambda:needs_contract(a,ng))
    b=copy.deepcopy(ng);b[0]['gold']['is_duplicate_delivery']=True;tests.append(lambda:needs_contract(ni,b))
    c=copy.deepcopy(ni);dup=next(i for i,g in enumerate(ng) if g['gold']['is_duplicate_delivery']);c[dup]['author_id']='u-'+'0'*24
    tests.append(lambda:needs_contract(c,ng))
    d=copy.deepcopy(ri);pos=next(i for i,g in enumerate(rg) if g['gold']['eligibility']);d[pos]['query']['consent']=False
    tests.append(lambda:reply_contract(d,rg))
    e=copy.deepcopy(rg);fix=next(i for i,r in enumerate(ri) if r['event']=='fixed_notice_candidate');e[fix]['gold']['eligibility']=True
    tests.append(lambda:reply_contract(ri,e))
    f=copy.deepcopy(ri);f[pos]['references'][0]['source_verified']=False;tests.append(lambda:reply_contract(f,rg))
    g=copy.deepcopy(ri);hard=next(i for i,x in enumerate(rg) if x['gold']['reason']==['opposite_direction']);g[hard]['approved']=False
    tests.append(lambda:reply_contract(g,rg))
    h=copy.deepcopy(ri);h[pos]['query']['topic_id']='t-'+'0'*24;tests.append(lambda:reply_contract(h,rg))
    for test in tests:
        try:test()
        except Invalid:continue
        raise Invalid('mutation_not_rejected')
    return len(tests)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal',action='store_true',help='Create lock once; refuses overwrites.')
    parser.add_argument('--self-test',action='store_true',help='Verify validator rejects eight independent contract mutations.')
    args=parser.parse_args()
    meta=json.loads((BASE/'sealed/adjudication.json').read_text())
    sources=json.loads((BASE/'sealed/source-lock.json').read_text())
    for relative,digest in sources.items():check(hashlib.sha256((BASE.parent/relative).read_bytes()).hexdigest()==digest,'old_data_or_protocol_changed')
    result=dict(status='PASS',splits={},old_sources_and_protocol_unchanged=True)
    data={}
    for split in ['tune','control']:
        gb=BASE if split=='tune' else BASE/'sealed'
        ni=load_lines(BASE/f'{split}-inputs.jsonl');ng=load_lines(gb/f'{split}-gold.jsonl')
        ri=load_lines(BASE/f'reply-{split}-inputs.jsonl');rg=load_lines(gb/f'reply-{split}-gold.jsonl')
        ns,seen,groups=needs_contract(ni,ng);rs=reply_contract(ri,rg)
        adjudicate(split,ni,ng,ri,rg,meta)
        if args.self_test:result.setdefault('mutation_checks',{})[split]=mutation_checks(ni,ng,ri,rg)
        result['splits'][split]=dict(needs=ns,replies=rs)
        data[split]=(ni,ng,ri,rg)
    ta,ca=meta['splits']['tune']['needs'],meta['splits']['control']['needs']
    concepts=lambda a:{f['concept'] for f in a['families']+a['singletons']}
    topics=lambda a:{f['topic_id'] for f in a['families']+a['singletons']}
    check(not concepts(ta)&concepts(ca),'split_family_concept_disjointness')
    check(not topics(ta)&topics(ca),'split_topic_id_disjointness')
    tn,tg,tr,tx=data['tune'];cn,cg,cr,cx=data['control']
    tt,ct=all_texts(tn,tr),all_texts(cn,cr)
    check(not tt&ct,'split_text_disjointness')
    check(not identity_set(tn,tr)&identity_set(cn,cr),'split_identity_disjointness')
    old_text,old_ids,old_topics=old_contract_data()
    check(not (tt|ct)&old_text,'old_version_text_disjointness')
    check(not (identity_set(tn,tr)|identity_set(cn,cr))&old_ids,'old_version_id_disjointness')
    check(not (topics(ta)|topics(ca))&old_topics,'old_version_topic_id_disjointness')
    check(not (BASE/'control-gold.jsonl').exists() and not (BASE/'reply-control-gold.jsonl').exists(),'control_gold_sealed_only')
    result['disjointness']='PASS: normalized texts, topics, concepts, identities; old-version exact-text exclusion'
    result['semantic_review']='Author adjudication checked against all needs and replies; semantic equivalence is a human-author judgement, not a model score.'
    current=snapshot();lockpath=BASE/'CONTROL-LOCK.json'
    if args.seal:
        check(not lockpath.exists(),'refuse_lock_overwrite')
        manifest=dict(algorithm='sha256',version='v5c',files=current,
                      protocol='Sealed independently before candidate freeze; control gold is not public.',
                      populations={k:v for k,v in result['splits'].items()})
        with lockpath.open('x',encoding='utf-8') as file:json.dump(manifest,file,ensure_ascii=False,indent=2);file.write('\n')
    check(lockpath.exists(),'sha_lock_missing')
    lock=json.loads(lockpath.read_text())
    check(lock['algorithm']=='sha256' and lock['files']==current,'sha_lock_integrity')
    result['sha_lock']='PASS';result['locked_files']=len(current)
    result['lock_sha256']=hashlib.sha256(lockpath.read_bytes()).hexdigest()
    # This file contains only aggregate results and never copies control rows.
    with (BASE/'VALIDATION.json').open('w',encoding='utf-8') as file:json.dump(result,file,ensure_ascii=False,indent=2);file.write('\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except Invalid as error:
        print(json.dumps({'status':'FAIL','rule':str(error)},ensure_ascii=False));sys.exit(1)
    except (OSError,KeyError,TypeError,ValueError):
        print(json.dumps({'status':'FAIL','rule':'unreadable_or_malformed_artifact'},ensure_ascii=False));sys.exit(1)
