"""Mechanical transport checks; aggregate-only output, no inference or gold oracle."""
import sys
sys.dont_write_bytecode=True
import json,hashlib,uuid,copy,collections,types,importlib.util
from pathlib import Path
B=Path(__file__).resolve().parent
S=B.parent/'v7-holdout'
CONTRACT=B.parent.parent/'radar_app/reply_policy.py'
FILES={'needs-inputs.jsonl','reply-inputs.jsonl','sealed/needs-gold.jsonl','sealed/reply-gold.jsonl'}
KINDS={'bug','wish','usability','service','unknown'}
ACTIONS={'queue_standard','queue_fixed_notice','operator','ignore_duplicate','needs_approval','needs_fix_approval'}
def check(x):
 if not x:raise ValueError('check_failed')
def desc(p):
 data=p.read_bytes();return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
def load(p):return json.loads(p.read_text())
def rows(p):
 data=p.read_text(encoding='utf-8');check(data.endswith('\n'))
 return [json.loads(x) for x in data.splitlines()]
def ident(v):
 u=uuid.UUID(v);check(u.version==4 and str(u)==v)
def noanswers(x):
 if isinstance(x,dict):
  check(not set(x)&{'gold','expected','same_topic','eligibility','safe_action','topic_id','synthetic','reason'})
  for v in x.values():noanswers(v)
 elif isinstance(x,list):
  for v in x:noanswers(v)
def kinds(x):
 if isinstance(x,dict):
  if 'kind' in x:check(x['kind'] in KINDS)
  for v in x.values():kinds(v)
 elif isinstance(x,list):
  for v in x:kinds(v)
def normalize_input(x):
 x=copy.deepcopy(x);x['event']={'incoming':'incoming_ticket','fix_confirmed':'fixed_notice_candidate'}.get(x['event'],x['event'])
 for ref in x['references']:
  if 'verified' in ref:
   check('source_verified' not in ref);ref['source_verified']=ref.pop('verified')
 return x
def normalize_gold(x):
 x=copy.deepcopy(x)
 if x['gold']['safe_action']=='abstain':x['gold']['safe_action']='operator'
 check(x['gold']['safe_action'] in ACTIONS)
 return x
def run():
 audit=load(B/'sealed/TRANSPORT-AUDIT.json');lock=load(B/'LOCK.json')
 check(set(audit['source_files'])==FILES and set(lock['files'])==FILES|{'PROTOCOL.md'} and lock['config_freeze'] is False)
 for f in FILES:check(desc(S/f)==audit['source_files'][f])
 check(desc(CONTRACT)==audit['contract'])
 for f,d in lock['files'].items():check(desc(B/f)==d)
 for f in ['needs-inputs.jsonl','sealed/needs-gold.jsonl']:check((B/f).read_bytes()==(S/f).read_bytes())
 n=rows(B/'needs-inputs.jsonl');ng=rows(B/'sealed/needs-gold.jsonl');r=rows(B/'reply-inputs.jsonl');rg=rows(B/'sealed/reply-gold.jsonl')
 oldr=rows(S/'reply-inputs.jsonl');oldg=rows(S/'sealed/reply-gold.jsonl')
 check(len(n)==len(ng)==42 and len(r)==len(rg)==len(oldr)==len(oldg)==28)
 check([normalize_input(x) for x in oldr]==r and [normalize_gold(x) for x in oldg]==rg)
 if oldg==rg:check((B/'sealed/reply-gold.jsonl').read_bytes()==(S/'sealed/reply-gold.jsonl').read_bytes())
 check(all(x['gold']['same_topic']==y['gold']['same_topic'] and x['gold']['eligibility']==y['gold']['eligibility'] for x,y in zip(oldg,rg)))
 byid=collections.defaultdict(list);target={};families=collections.defaultdict(list);products=set();authors=set();sources=set();unk=0
 for x,g in zip(n,ng):
  noanswers(x);kinds(g);check(set(x)=={'id','product','author_id','source_id','created_at','received_at','channel','text'})
  check(set(g)=={'id','gold'} and g['id']==x['id'] and set(g['gold'])=={'kind','topic_id','synthetic'} and g['gold']['synthetic'] is True)
  for k in ['id','product','author_id','source_id']:ident(x[k])
  if byid[x['id']]:
   check({k:v for k,v in byid[x['id']][0].items() if k!='received_at'}=={k:v for k,v in x.items() if k!='received_at'})
   check(byid[x['id']][0]['received_at']!=x['received_at'] and target[x['id']]==g)
  else:
   target[x['id']]=g;products.add(x['product']);authors.add(x['author_id']);sources.add(x['source_id'])
   if g['gold']['kind']=='unknown':check(g['gold']['topic_id'] is None);unk+=1
   else:ident(g['gold']['topic_id']);families[g['gold']['topic_id']].append(x)
  byid[x['id']].append(x)
 check(len(byid)==40 and sorted(map(len,byid.values()))==[1]*38+[2]*2 and len(products)==4 and unk==8 and len(families)==8 and all(len(v)==4 for v in families.values()))
 check(len(authors)==39 and len(sources)==40)
 familykind=collections.Counter(target[v[0]['id']]['gold']['kind'] for v in families.values());check(set(familykind.values())=={2} and len(familykind)==4)
 check(any(len({x['author_id'] for x in v})==3 and len({x['source_id'] for x in v})==4 for v in families.values()))
 package_name='_sealed_transport_contract'
 pkg=types.ModuleType(package_name);pkg.__path__=[str(CONTRACT.parent)];sys.modules[package_name]=pkg
 spec=importlib.util.spec_from_file_location(package_name+'.reply_policy',CONTRACT)
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 calls=0
 def matcher(_):
  nonlocal calls
  calls+=1;return True
 policy=mod.ReplyPolicy(matcher)
 actions=collections.Counter();events=collections.Counter();processed=0;call_rows=0;pre_nn=0;replyids=set()
 for x,g in zip(r,rg):
  check(set(x)==mod.FIELDS and mod.payload_only(x)==x);noanswers(x);kinds(x)
  check(set(g)=={'id','gold'} and g['id']==x['id'] and set(g['gold'])=={'same_topic','eligibility','safe_action','synthetic','reason'})
  check(type(g['gold']['same_topic']) is bool and type(g['gold']['eligibility']) is bool and g['gold']['safe_action'] in ACTIONS and g['gold']['synthetic'] is True)
  ident(x['id']);check(x['id'] not in replyids);replyids.add(x['id']);ident(x['product']);check(x['product'] in products)
  check(x['event'] in {'incoming_ticket','fixed_notice_candidate'});events[x['event']]+=1
  q=x['query'];check(set(q)=={'id','product','source_id','author_id','text','consent','created_at'})
  for k in ['id','product','source_id','author_id']:ident(q[k])
  for ref in x['references']:
   check(set(ref)=={'id','source_id','author_id','product','text','kind','source_verified','consent','created_at'})
   check(type(ref['source_verified']) is bool and 'verified' not in ref)
   for k in ['id','source_id','author_id','product']:ident(ref[k])
   check(ref['id'] in byid)
   check(all(ref[k]==byid[ref['id']][0][k] for k in ['id','source_id','author_id','product','text','created_at']))
  if x['event']=='incoming_ticket' and policy.eligibility(mod.payload_only(x))[0]:pre_nn+=1
  before=calls;decision=policy.decide(mod.payload_only(x));call_rows+=calls>before;processed+=1
  check(decision['action'] in ACTIONS);actions[decision['action']]+=1
 check(events=={'incoming_ticket':24,'fixed_notice_candidate':4} and processed==28 and call_rows>0 and calls>=call_rows)
 # No comparison of smoke decisions with gold: true matcher is not a semantic oracle.
 for f in FILES:check(desc(S/f)==audit['source_files'][f])
 return {'status':'PASS','counts':{'needs_rows':42,'needs_unique':40,'needs_retransmissions':2,'reply_rows':28,'ordinary_rows':24,'fixed_rows':4,'products':4,'families':8,'unknown_singletons':8,'needs_authors_unique':39,'decisions_processed':processed,'matcher_call_rows':call_rows,'matcher_calls':calls,'ordinary_pre_nn_eligibility':pre_nn},'policy_action_counts':dict(sorted(actions.items())),'checks':{'source_integrity':'PASS','contract_integrity':'PASS','lock':'PASS','schemas':'PASS','uuid_and_enums':'PASS','alignment_and_duplicates':'PASS','whitelisted_diff_only':'PASS','same_topic_eligibility_unchanged':'PASS','gold_separation':'PASS','actual_policy_smoke':'PASS'}}
if __name__=='__main__':
 try:result=run()
 except Exception:result={'status':'FAIL','error':'validation_failed'}
 print(json.dumps(result,sort_keys=True))
 sys.exit(0 if result['status']=='PASS' else 1)
