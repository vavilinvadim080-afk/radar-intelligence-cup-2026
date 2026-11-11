"""Bounded regrouping of already observed history; no unobserved future or labels."""
import hashlib
import itertools
from .selection_v5c import compatible_direction


def predict_history(selector, deliveries):
    rows, seen = [], {}
    for row in sorted(deliveries, key=lambda r:(r.get('created_at',''),r['id'])):
        if any(k in row for k in ('gold','expected','gold_kind','gold_topic')):
            raise ValueError('Gold cannot enter history clustering')
        source = (row['product'],row.get('source_id',row['id']))
        if source in seen:
            if seen[source]['text'] != row['text']:
                raise ValueError('Conflicting repeated history source')
            continue
        seen[source] = row
        rows.append(row)
    if len(rows) > 2000:
        raise ValueError('Experimental history clustering is bounded to 2000 observations')
    original = [selector.choose_kind(row['text'])[0] for row in rows]
    scores = [selector.features.kind_scores(row['text']) for row in rows]
    owner = list(range(len(rows)))
    groups = {i:[i] for i in owner}
    edges = []
    pair_scores = {}
    for i,j in itertools.combinations(range(len(rows)),2):
        if rows[i]['product'] != rows[j]['product']:
            continue
        a,b=rows[i]['text'],rows[j]['text']
        score = selector.features.pair_score(a,b) if compatible_direction(a,b) else None
        pair_scores[i,j] = score
        if score is not None and score >= selector.config['pair_min']:
            edges.append((score,i,j))
    for strength,i,j in sorted(edges, key=lambda edge:(-edge[0],edge[1],edge[2])):
        left,right = owner[i],owner[j]
        if left == right:
            continue
        members = groups[left]+groups[right]
        if all(original[k]=='unknown' for k in members):
            continue
        if any(original[k]=='unknown' for k in members) and strength < selector.config.get('unknown_pair_min',1.):
            continue
        target = max((k for k in scores[i] if k!='unknown'),
                     key=lambda kind:sum(scores[k][kind] for k in members))
        if not all(selector.context_allows_kind(rows[k]['text'],original[k],target) for k in members):
            continue
        veto = False
        for a,b in itertools.product(groups[left],groups[right]):
            if not compatible_direction(rows[a]['text'],rows[b]['text']):
                veto = True
                break
            value = pair_scores.get(tuple(sorted((a,b))))
            if value is not None and value < selector.config.get('pair_veto',selector.config['pair_min']):
                veto = True
                break
        if veto:
            continue
        keep,remove=min(left,right),max(left,right)
        groups[keep] = members
        del groups[remove]
        for k in members:
            owner[k] = keep
    out = []
    for i,row in enumerate(rows):
        members = groups[owner[i]]
        kind = (max((k for k in scores[i] if k!='unknown'),
                    key=lambda label:sum(scores[k][label] for k in members))
                if len(members)>1 else original[i])
        first = rows[min(members)]
        ident = first['product']+':'+first.get('source_id',first['id'])
        topic = 'history-'+hashlib.sha256(ident.encode()).hexdigest()[:20]
        prediction = None if kind=='unknown' else [row['product'],kind,topic]
        out.append(dict(id=row['id'],kind=kind,prediction=prediction,
                        trace=dict(initial_kind=original[i], history_members=len(members),
                                   scope='all observations supplied to this call; no gold')))
    return out
