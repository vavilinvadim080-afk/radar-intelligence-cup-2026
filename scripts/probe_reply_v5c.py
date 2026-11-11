"""Calibrate a separate reply equivalence head using tune-only hard negatives.

Validation holds out whole reference families from both classifiers and pair fitting.
Permission/lifecycle/consent labels never train the semantic head.
"""
import itertools
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from scripts import train_service_v5c as t
from radar_app.selection_v5c import transform_pair_inputs


def reply_examples(store, rows, gold):
    examples = {}
    for row in rows:
        a = row['query']['text']
        if row['query']['product'] != row['product']:
            continue
        label = gold[row['id']]['same_topic']
        if not isinstance(label,bool):
            continue
        for ref in row['references'][:3]:
            if ref['product'] != row['product']:
                continue
            b = ref['text']
            values = t.pair_inputs(store,a,b)
            if values is None:
                continue
            key = tuple(sorted((a,b)))
            if key in examples and examples[key]['label'] != int(label):
                raise ValueError('Contradictory semantic pair labels in tune')
            examples[key] = dict(values=values,label=int(label))
    return list(examples.values())


def fit_reply_head(needs_pairs, reply_pairs, transform, c, reply_weight):
    rows = list(needs_pairs)+list(reply_pairs)
    x=np.asarray([transform_pair_inputs(p['values'],transform) for p in rows])
    y=[p['label'] for p in rows]
    weights=[1.]*len(needs_pairs)+[reply_weight]*len(reply_pairs)
    model=LogisticRegression(C=c,class_weight='balanced',max_iter=800,random_state=0).fit(x,y,sample_weight=weights)
    return dict(classes=[0,1],coefficients=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),
                feature_transform=transform,C=c,reply_weight=reply_weight,n=len(rows),
                needs_pairs=len(needs_pairs),reply_pairs=len(reply_pairs))


def prepare_folds(store, needs, needs_gold, replies):
    by_text={r['text']:t.family_key(r,needs_gold) for r in needs}
    parents={family:family for family in by_text.values()}
    def root(family):
        while parents[family]!=family:
            family=parents[family]
        return family
    def union(a,b):
        a,b=root(a),root(b)
        parents[max(a,b)]=min(a,b)
    contexts=[];queries={}
    for row in replies:
        families={by_text[ref['text']] for ref in row['references']}
        if not families:
            family='empty-reference:'+row['id']
            parents[family]=family
            families={family}
        first=min(families)
        for family in families:
            union(first,family)
        query=row['query']['text']
        if query in queries:
            union(first,queries[query])
        queries[query]=first
        contexts.append(first)
    groups=[root(family) for family in contexts]
    if len(set(groups))<5:
        raise ValueError('At least five disjoint reply context components are required')
    folds=[]
    all_pairs=t.training_pairs(store,needs,needs_gold)
    for fold,(train_indices,validation) in enumerate(GroupKFold(5).split(replies,groups=groups)):
        held_out={groups[i] for i in validation}
        forbidden={family for family in parents if root(family) in held_out}
        # Also exclude any validation query that is an existing needs family.
        forbidden.update(by_text[replies[i]['query']['text']] for i in validation
                         if replies[i]['query']['text'] in by_text)
        need_indices=[i for i,r in enumerate(needs) if t.family_key(r,needs_gold) not in forbidden]
        need_set=set(need_indices)
        pair_indices=[i for i,p in enumerate(all_pairs) if {p['left'],p['right']} <= need_set]
        training=[replies[i] for i in train_indices if not any(
            by_text.get(text) in forbidden for text in [replies[i]['query']['text']]+
            [r['text'] for r in replies[i]['references']])]
        folds.append(dict(fold=fold,train_replies=training,validation=[replies[i] for i in validation],
                          train_needs=need_indices,train_pairs=[all_pairs[i] for i in pair_indices],
                          pair_indices=pair_indices,all_pairs=all_pairs,held_out_families=sorted(forbidden)))
    return folds


def main():
    t.assert_unfrozen();t.verify_baseline()
    sources,code=t.training_sources(),t.semantic_code_hashes()
    store=t.load_features('tune');gold,reply_gold=t.load_tune_gold(store)
    needs=t.unique_needs(store.data['needs_inputs']);replies=store.data['reply_inputs']
    matching=t.read_json(t.OUT/'tune-matching-ablation.json')['selected']
    folds=prepare_folds(store,needs,gold,replies)
    trials=[]
    for transform,c,weight in itertools.product(('identity','poly2'),(.1,1.,10.),(1.,5.)):
        prepared=[]
        for fold in folds:
            heads=t.fit_heads(store,needs,gold,fold['all_pairs'],fold['train_needs'],fold['pair_indices'],
                              matching['kind_C'],matching['pair_C'])
            heads['reply_pair']=fit_reply_head(fold['train_pairs'],reply_examples(store,fold['train_replies'],reply_gold),
                                               transform,c,weight)
            prepared.append((t.CachedCandidateFeatures(store,heads),fold['validation']))
        for threshold,risk,reference_policy in itertools.product((.3,.5,.7,.85,.95),(.6,.8),('all','best')):
            config=dict(matching['config'],cluster_policy='online',pair_min=threshold,risk_min=risk,
                        reference_policy=reference_policy)
            predicted=[]
            for features,validation in prepared:
                predicted.extend(t.predict_replies(t.CandidateReplyPolicy(features,config=config),validation))
            metrics=t.score_replies(predicted,replies,reply_gold)
            trials.append(dict(transform=transform,C=c,reply_weight=weight,config=config,metrics=metrics))
        print('READY reply calibration',transform,c,weight,flush=True)
    best=max(trials,key=lambda trial:t.reply_rank(trial['metrics']))
    result=dict(status='tune_only',selected=best,trials=trials,
                training_sources_sha256=sources,semantic_code_sha256=code,
                folds=[dict(fold=f['fold'],held_out_families=f['held_out_families'],
                            validation_ids=[r['id'] for r in f['validation']],
                            training_ids=[r['id'] for r in f['train_replies']]) for f in folds],
                caveat='Whole reference families held out in both kind and semantic pair heads. '
                       'Reply tune same_topic supervises semantic verification; permission labels never train it. '
                       'Model/threshold selection uses this CV, so independent control is still required.')
    if sources!=t.training_sources() or code!=t.semantic_code_hashes():
        raise RuntimeError('Tune sources or semantic code changed during reply ablation')
    t.save_checkpoint(t.OUT/'tune-reply-ablation.json',result)
    print(t.json.dumps(best,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    main()
