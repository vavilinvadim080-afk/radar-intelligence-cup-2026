"""Schema/lock checks only. No matcher, analyzer, encoder or model imports."""
from pathlib import Path
import hashlib,json,collections

ROOT=Path(__file__).resolve().parent
FILES=['tune.jsonl','control-inputs.jsonl','control-gold.jsonl','reply-tune.jsonl','reply-control-inputs.jsonl','reply-control-gold.jsonl']
FORBIDDEN={'gold','expected','same_topic','eligibility','safe_action','topic_id','query_kind','query_direction','duplicate_of','is_duplicate_delivery','reason'}
KINDS={'wish','bug','service','usability','unknown'}

def read(name):return [json.loads(s) for s in (ROOT/name).read_text().splitlines() if s.strip()]
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def no_gold(obj):
    if isinstance(obj,dict):
        require(not (set(obj)&FORBIDDEN),'Gold field leaked into control input')
        for v in obj.values():no_gold(v)
    elif isinstance(obj,list):
        for v in obj:no_gold(v)
def joined(inputs,labels):
    im={r['id']:r for r in inputs};gm={r['id']:r['gold'] for r in labels}
    require(len(im)==len(inputs) and len(gm)==len(labels),'Duplicate evaluation ID')
    require(im.keys()==gm.keys(),'Input/gold ID mismatch')
    return [dict(v,gold=gm[k]) for k,v in im.items()]
def grouping(rows):
    require(len(rows)==120,'Wrong needs count')
    require(len({r['id'] for r in rows})==120,'Duplicate needs ID')
    families=set();groups=collections.defaultdict(list);duplicates=0
    byid={r['id']:r for r in rows}
    for r in rows:
        g=r['gold'];require(g['kind'] in KINDS,'Invalid kind');require(g['product']==r['product'],'Product label mismatch')
        if g['kind']=='unknown':require(g['topic_id'] is None and g['route']=='review','Ambiguity forced into one topic')
        else:
            prefix=r['product']+':'+g['kind']+':'
            require(g['topic_id'].startswith(prefix),'Topic ID lacks product/kind')
            families.add(g['topic_id'].split(':',2)[2]);groups[g['topic_id']].append(r)
        if g['is_duplicate_delivery']:
            duplicates+=1;original=byid[g['duplicate_of']]
            require(all(r[k]==original[k] for k in ['product','author_id','source_id','text']),'Duplicate delivery changed content')
    require(len(families)==24 and len(groups)==24,'Wrong semantic family count')
    require(duplicates==4,'Missing duplicate delivery tests')
    require(sum(r['gold']['kind']=='unknown' for r in rows)==20,'Wrong ambiguity count')
    for group in groups.values():
        authors=[r['author_id'] for r in group if not r['gold']['is_duplicate_delivery']]
        require(len(authors)==4 and len(set(authors))==3,'Author repetition missing')
    return families
def reply(rows):
    require(len(rows)==80 and len({r['id'] for r in rows})==80,'Wrong reply count/IDs')
    for r in rows:
        g=r['gold'];same=g['same_topic'];eligible=g['eligibility'];action=g['safe_action']
        require(same in [True,False,None] and isinstance(eligible,bool),'Invalid independent reply labels')
        if action=='queue_standard':
            require(same is True and eligible and r['approved'] and r['state']=='open' and r['query']['consent'],'Unsafe standard reply gold')
            require(r['references'] and all(x['source_verified'] for x in r['references']),'Standard reply lacks verified references')
            require(r['query']['product']==r['product'],'Standard reply wrong product')
        if action=='queue_fixed_notice':require(r['fix_confirmed'] and r['human_closed'] and r['fixed_notice_approved'] and not eligible and r['event']=='fixed_notice_candidate','Fixed notice lacks separate approvals')
        if action=='ignore_duplicate':require(r['query']['id'] in r['seen_message_ids'],'Duplicate action without previous enqueue')
        if same is not True:require(action=='operator','Nonmatch was auto-queued')
        if r['state']!='open':require(action!='queue_standard','Fix got ordinary template')
        if not r['query']['consent']:require(action=='operator' and not eligible,'No-consent query was eligible')
    require(any(r['gold']['same_topic'] is True and not r['gold']['eligibility'] for r in rows),'Eligibility collapsed into topic match')
    require(any(r['gold']['same_topic'] is False and r['gold']['eligibility'] for r in rows),'Semantic mismatch collapsed into eligibility')

def main():
    nt=read('tune.jsonl');nc_in=read('control-inputs.jsonl');nc_g=read('control-gold.jsonl')
    rt=read('reply-tune.jsonl');rc_in=read('reply-control-inputs.jsonl');rc_g=read('reply-control-gold.jsonl')
    for x in nc_in+rc_in:no_gold(x)
    nc=joined(nc_in,nc_g);rc=joined(rc_in,rc_g)
    tf,cf=grouping(nt),grouping(nc);require(not tf&cf,'Tune/control semantic family overlap')
    reply(rt);reply(rc)
    require(not {r['text'].strip().casefold() for r in nt}&{r['text'].strip().casefold() for r in nc},'Needs text leak between splits')
    require(not {r['query']['text'].strip().casefold() for r in rt}&{r['query']['text'].strip().casefold() for r in rc},'Reply query leak between splits')
    counts=dict(needs_tune=len(nt),needs_control=len(nc),reply_tune=len(rt),reply_control=len(rc),needs_families_tune=len(tf),needs_families_control=len(cf))
    manifest=ROOT/'CONTROL-LOCK.json'
    if manifest.exists():
        lock=json.loads(manifest.read_text())
        for n in FILES:require(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==lock['files'][n]['sha256'],'Locked file hash changed: '+n)
    print(json.dumps(dict(status='pass',counts=counts,control_payload_gold_free=True,semantic_families_disjoint=True),ensure_ascii=False))

if __name__=='__main__':main()
