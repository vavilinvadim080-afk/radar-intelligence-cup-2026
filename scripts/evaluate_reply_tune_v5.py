"""Choose standard-reply matching threshold on tuning data only."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from radar_app.semantic import SemanticReplyMatcher
from radar_app.semantic_client import SemanticClient
from radar_app.reply_policy import ReplyPolicy,payload_only


def binary(gold,pred):
    tp=sum(g and p for g,p in zip(gold,pred));fp=sum(not g and p for g,p in zip(gold,pred));fn=sum(g and not p for g,p in zip(gold,pred))
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)


def evaluate(rows,matcher):
    policy=ReplyPolicy(matcher);details=[];truth=[];pred=[]
    for r in rows:
        p=payload_only(r);out=policy.decide(p)
        truth.append(r['gold']['safe_action']=='queue_standard');pred.append(out['action']=='queue_standard')
        details.append(dict(id=r['id'],expected_action=r['gold']['safe_action'],**out))
    metrics=binary(truth,pred)
    return dict(n=len(rows),**metrics,
                action_accuracy=sum(x['expected_action']==x['action'] for x in details)/len(details),
                unsafe_standard_replies=sum(x['action']=='queue_standard' and x['expected_action']!='queue_standard' for x in details),
                details=details)


def main():
    rows=[json.loads(x) for x in (ROOT/'evaluation/v5/reply-tune.jsonl').read_text().splitlines() if x]
    results={}
    with SemanticClient() as c:
        for threshold in [.4,.5,.6,.7,.8]:
            r=evaluate(rows,SemanticReplyMatcher(c,threshold=threshold));results[str(threshold)]=r
            print(threshold,{k:v for k,v in r.items() if k!='details'},flush=True)
    (ROOT/'reports/quality-v5/reply-tune-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
