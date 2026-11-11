"""Bounded matching ablation on tune families only; never opens control data."""
import itertools
from scripts import train_service_v5c as train


def main():
    train.assert_unfrozen()
    train.verify_baseline()
    sources,code=train.training_sources(),train.semantic_code_hashes()
    store = train.load_features('tune')
    gold, _ = train.load_tune_gold(store)
    rows = train.unique_needs(store.data['needs_inputs'])
    pairs = train.training_pairs(store, rows, gold)
    folds = train.grouped_folds(rows, gold)
    trials = []
    for pair_c in train.C_VALUES:
        prepared = []
        for fold in folds:
            indices, _, _ = train.partition_pairs(pairs, fold['train'], fold['validation'])
            heads = train.fit_heads(store, rows, gold, pairs, fold['train'], indices, 10., pair_c)
            prepared.append((fold['fold'], train.CachedCandidateFeatures(store, heads),
                             [rows[i] for i in fold['validation']]))
        for reference_policy, slack, threshold, kind_floor in itertools.product(
                ('all', 'best'), (0., .15, .35, 1.), (.2, .4, .6, .85), (.03, .1)):
            config = dict(train.DEFAULTS, kind_min=0., kind_margin=0., cosine_min=0.,
                          margin=0., reference_policy=reference_policy,
                          context_kind_margin=slack, pair_min=threshold, context_kind_min=kind_floor)
            predicted = []
            for fold_id, features, validation in prepared:
                out = train.CandidateSelector(features, backend='qwen', config=config).predict(validation)
                predicted.extend(train.prefix_predictions(out, fold_id))
            trials.append(dict(kind_C=10., pair_C=pair_c, config=config,
                               metrics=train.score_needs(predicted, gold)))
        if pair_c == 1.:
            for unknown_max, unknown_pair_min in itertools.product((.5, .8), (.75, .85, .95)):
                config = dict(train.DEFAULTS, kind_min=0., kind_margin=0., cosine_min=0.,
                              margin=.02, reference_policy='best', context_kind_margin=1.,
                              pair_min=.6, context_kind_min=.03, context_unknown_max=unknown_max,
                              unknown_pair_min=unknown_pair_min)
                predicted = []
                for fold_id, features, validation in prepared:
                    out = train.CandidateSelector(features, backend='qwen', config=config).predict(validation)
                    predicted.extend(train.prefix_predictions(out, fold_id))
                trials.append(dict(kind_C=10., pair_C=pair_c, config=config,
                                   metrics=train.score_needs(predicted, gold)))
            for unknown_max, threshold, veto in itertools.product((0., .5, .8), (.4, .6, .8), (.2, .4)):
                config = dict(train.DEFAULTS, kind_min=0., kind_margin=0., cosine_min=0.,
                              margin=0., cluster_policy='history', context_kind_margin=1.,
                              pair_min=threshold, context_kind_min=.03, context_unknown_max=unknown_max,
                              unknown_pair_min=.85, pair_veto=veto)
                predicted = []
                for fold_id, features, validation in prepared:
                    out = train.CandidateSelector(features, backend='qwen', config=config).predict(validation)
                    predicted.extend(train.prefix_predictions(out, fold_id))
                trials.append(dict(kind_C=10., pair_C=pair_c, config=config,
                                   metrics=train.score_needs(predicted, gold)))
        print('READY matching ablation pair C', pair_c, flush=True)
    result = dict(status='tune_only', trials=trials,
                  selected=max(trials, key=train.needs_rank),
                  training_sources_sha256=sources, semantic_code_sha256=code,
                  caveat='Nested selection is not performed. These tune-CV trials select a candidate only. '
                         'Control remains sealed; no acceptance metrics are reported.')
    if sources!=train.training_sources() or code!=train.semantic_code_hashes():
        raise RuntimeError('Tune sources or semantic code changed during ablation')
    train.save_checkpoint(train.OUT / 'tune-matching-ablation.json', result)
    print(train.json.dumps(result['selected'], ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
