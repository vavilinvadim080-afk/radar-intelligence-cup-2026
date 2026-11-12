"""Evidence-complete agglomeration with a recorded reason for each refusal."""
from collections import Counter
import hashlib
import itertools
from .semantics_v6 import compatible_direction


def predict_history(selector, deliveries):
    rows, sources = [], {}
    for row in sorted(deliveries, key=lambda r:(r.get('created_at',''),r['id'])):
        if any(k in row for k in ('gold','expected','gold_kind','gold_topic')):
            raise ValueError('Gold cannot enter history clustering')
        key = row['product'], row.get('source_id',row['id'])
        if key in sources:
            if sources[key]['text'] != row['text']:
                raise ValueError('Conflicting repeated history source')
            continue
        sources[key] = row
        rows.append(row)
    if len(rows) > 2000:
        raise ValueError('Experimental history clustering is bounded to 2000 observations')
    initial = [selector.choose_kind(row['text'])[0] for row in rows]
    scores = [selector.features.kind_scores(row['text']) for row in rows]
    groups = {i:[i] for i in range(len(rows))}
    evidence, refusals = {}, Counter()
    pair_min = selector.config['pair_min']
    aggregation = selector.config.get('group_aggregation','average')
    if aggregation not in {'average','complete'}:
        raise ValueError('Unknown group aggregation')
    for i,j in itertools.combinations(range(len(rows)),2):
        if rows[i]['product'] == rows[j]['product']:
            evidence[i,j] = (selector.features.pair_score(rows[i]['text'],rows[j]['text'])
                             if compatible_direction(rows[i]['text'],rows[j]['text']) else None)

    def proposed(left, right):
        members = groups[left] + groups[right]
        if rows[members[0]]['product'] != rows[members[-1]]['product']:
            return None, 'different_product'
        if all(initial[i] == 'unknown' for i in members):
            return None, 'all_unknown'
        target = max((k for k in scores[members[0]] if k != 'unknown'),
                     key=lambda kind:sum(scores[i][kind] for i in members))
        if not all(selector.context_allows_kind(rows[i]['text'],initial[i],target) for i in members):
            return None, 'kind_conflict'
        values = []
        for i,j in itertools.product(groups[left],groups[right]):
            if not compatible_direction(rows[i]['text'],rows[j]['text']):
                return None, 'opposite_direction_or_state'
            value = evidence.get(tuple(sorted((i,j))))
            if value is None:
                return None, 'missing_pair_evidence'
            values.append(value)
        if any(initial[i] == 'unknown' for i in members) and max(values) < selector.config.get('unknown_pair_min',1.):
            return None, 'unknown_without_strong_evidence'
        if min(values) < selector.config.get('pair_veto',pair_min):
            return None, 'weak_cross_pair'
        value = min(values) if aggregation == 'complete' else sum(values)/len(values)
        return ((value,target),None) if value >= pair_min else (None,'group_score_below_threshold')

    while True:
        candidates = []
        for left,right in itertools.combinations(sorted(groups),2):
            result, reason = proposed(left,right)
            if result is not None:
                candidates.append((result[0],left,right,result[1]))
            else:
                refusals[reason] += 1
        if not candidates:
            break
        value,left,right,target = min(candidates,key=lambda x:(-x[0],x[1],x[2]))
        groups[left].extend(groups.pop(right))
    membership = {i:members for members in groups.values() for i in members}
    out = []
    for i,row in enumerate(rows):
        members = membership[i]
        kind = (max((k for k in scores[i] if k!='unknown'),key=lambda k:sum(scores[j][k] for j in members))
                if len(members)>1 else initial[i])
        first = rows[min(members)]
        identity = first['product']+':'+first.get('source_id',first['id'])
        topic = 'history-v6-'+hashlib.sha256(identity.encode()).hexdigest()[:20]
        out.append(dict(id=row['id'],kind=kind,prediction=None if kind=='unknown' else [row['product'],kind,topic],
                        trace=dict(initial_kind=initial[i],history_members=len(members),context_kind_changed=kind!=initial[i],
                                   aggregation=aggregation,scope='observations supplied to this call; no gold')))
    selector.history_trace = dict(refusals=dict(refusals),groups=len(groups),messages=len(rows),
                                  note='Refusals count proposed group merges, not unique message pairs')
    return out
