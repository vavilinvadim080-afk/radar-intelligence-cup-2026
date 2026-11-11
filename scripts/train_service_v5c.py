"""Train and tune small v5c heads over saved features; freeze before control prediction.

No neural model, worker, network client or control-gold parser is started here.
Commands: train, tune, freeze, control. The independent scorer owns control gold.
"""
import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'mvp-v13-adapted')]

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from radar_app.calibrated_v5b import CalibratedFeatures, CalibratedReplyPolicy, CalibratedSelector
from radar_app.evaluation import pair_metrics
from radar_app.selection_v5b import DEFAULTS, KIND_HYPOTHESES
from radar_app.selection_v5c import (
    CandidateFeatures, CandidateReplyPolicy, CandidateSelector, kind_inputs, pair_inputs,
)
from scripts.prepare_features_v5c import (
    CODE as PREPARATION_CODE, FeatureStore, candidate_pairs, contract_details,
    feature_contract, fingerprint, key, masked_inputs, pair_key, plain_metadata,
    read, reject_gold, save_checkpoint, sha256_file, validate_feature_provenance,
)

OUT = ROOT / 'reports/quality-v5c'
HEADS = ROOT / 'data/semantic-heads-v5c.json'
CONFIG = ROOT / 'data/semantic-config-v5c.json'
C_VALUES = (.1, 1., 10.)
KIND_MIN = (0., .2, .4)
KIND_MARGIN = (0., .02)
PAIR_MIN = (.4, .6, .75, .85, .95)
COSINE_MIN = (0., .3, .5)
ACTIONS = {'operator', 'needs_approval', 'ignore_duplicate', 'needs_fix_approval',
           'queue_standard', 'queue_fixed_notice'}
MODEL_DIRS = dict(qwen='qwen-embedding-v5b', nli='nli-minilm-v5b', reranker='qwen-reranker-v5c')
RERANKER_WORKER = 'scripts/reranker_worker_v5c.py'
MLX_WORKER = 'scripts/reranker_worker_mlx_v5c.py'
CODE = tuple(sorted(set(PREPARATION_CODE) | {
    'scripts/train_service_v5c.py', 'scripts/score_service_v5c.py',
    'scripts/probe_matching_v5c.py', 'scripts/probe_reply_v5c.py',
    'scripts/finalize_candidate_v5c.py',
    'radar_app/selection_v5c.py', 'radar_app/clustering_v5c.py', 'radar_app/calibrated_v5b.py',
    'radar_app/semantic.py', 'radar_app/reply_safety.py', 'radar_app/evaluation.py',
    'mvp-v13-adapted/radar/embed.py',
}))


class FoldError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def root_path(name):
    relative = Path(name)
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Path must stay within the project root')
    if 'sealed' in relative.parts or 'gold' in relative.name.lower():
        raise RuntimeError('Prediction/freeze checks cannot open gold or sealed files')
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_exclusive(path, value):
    """Commit a complete JSON atomically, refusing replacement even under a race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError('Artifact exists; refusing to overwrite: ' + path.name) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def assert_unfrozen():
    if (OUT / 'CONFIG-FREEZE.json').exists() or (OUT / 'control-predictions-only.json').exists():
        raise RuntimeError('Already frozen; no training or tuning after control commitment')


def verify_baseline():
    manifest = read_json(OUT / 'baseline-v5b-preserved.json')
    if manifest.get('changed') != [] or manifest.get('selected_backend') != 'qwen':
        raise RuntimeError('Frozen baseline inventory is invalid')
    files = dict(manifest['files'])
    files['data/quality-v5b.json'] = manifest['quality_sha256']
    if len(files) != 24:
        raise RuntimeError('Baseline preservation requires all 24 original hashes')
    for name, expected in files.items():
        if sha256_file(root_path(name)) != expected:
            raise RuntimeError('Preserved v5b file changed: ' + name)
    return files


def unique_needs(rows):
    unique, ids = {}, {}
    for row in sorted(rows, key=lambda r: (r.get('created_at', ''), r['id'])):
        source = (row['product'], row.get('source_id', row['id']))
        if row['id'] in ids and ids[row['id']] != source:
            raise ValueError('Needs ID maps to different source identities')
        ids[row['id']] = source
        if source in unique:
            old = unique[source]
            if any(old.get(k) != row.get(k) for k in ('id', 'author_id', 'created_at', 'text')):
                raise ValueError('Conflicting repeated needs delivery')
        else:
            unique[source] = row
    return list(unique.values())


def load_features(split, frozen=None):
    if split == 'control' and frozen is None:
        raise RuntimeError('Control features require verified freeze')
    directory = ROOT / 'evaluation/v5c'
    paths = [directory / f'{split}-inputs.jsonl', directory / f'reply-{split}-inputs.jsonl']
    sources = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    if frozen is not None:
        locked = read_json(directory / 'CONTROL-LOCK.json')['files']
        for path in paths:
            entry = locked.get(path.name)
            expected = entry.get('sha256') if isinstance(entry, dict) else entry
            if sources[str(path.relative_to(ROOT))] != expected:
                raise RuntimeError('Locked control input changed')
    raw = [read(path) for path in paths]
    reject_gold(raw)
    needs, replies = masked_inputs(*raw)
    store = FeatureStore(OUT / f'{split}-features.json')
    validate_feature_provenance(store.data, sources=sources, needs=needs, reply=replies,
                                contract=feature_contract(), metadata=store.data['metadata'])
    if frozen is not None and plain_metadata(store.data['metadata']) != frozen['models']:
        raise RuntimeError('Control model descriptors differ from frozen tune descriptors')
    return store


def training_sources():
    directory = ROOT / 'evaluation/v5c'
    names = ('tune-inputs.jsonl', 'reply-tune-inputs.jsonl', 'tune-gold.jsonl', 'reply-tune-gold.jsonl')
    return {str((directory / n).relative_to(ROOT)): sha256_file(directory / n) for n in names}


def semantic_code_hashes():
    return {name: sha256_file(root_path(name)) for name in CODE}


def load_tune_gold(store):
    result = []
    for name, rows, duplicate_ok in [
        ('tune-gold.jsonl', store.data['needs_inputs'], True),
        ('reply-tune-gold.jsonl', store.data['reply_inputs'], False),
    ]:
        envelopes = read(ROOT / 'evaluation/v5c' / name)
        if len(envelopes) != len(rows):
            raise ValueError('Tune gold/input populations differ')
        lookup = {}
        for envelope, row in zip(envelopes, rows):
            if set(envelope) != {'id', 'gold'} or envelope['id'] != row['id']:
                raise ValueError('Tune gold must align with input delivery IDs')
            ident, target = envelope['id'], envelope['gold']
            if ident in lookup:
                if not duplicate_ok or any(lookup[ident].get(k) != target.get(k)
                                           for k in ('kind', 'topic_id', 'product')):
                    raise ValueError('Conflicting or duplicate tune gold IDs')
            else:
                lookup[ident] = target
        result.append(lookup)
    return tuple(result)


def family_key(row, gold):
    target = gold[row['id']]
    if target['kind'] == 'unknown':
        return 'unknown:' + row['id']
    topic = target.get('topic_id')
    if not isinstance(topic, str) or not topic:
        raise ValueError('Known needs kind requires a gold topic ID')
    return 'topic:' + topic


def grouped_folds(rows, gold):
    groups = [family_key(row, gold) for row in rows]
    if len(set(groups)) < 5:
        raise FoldError('At least five independent families are required', dict(status='failed', folds=[]))
    return [dict(fold=i, train=train.tolist(), validation=validation.tolist())
            for i, (train, validation) in enumerate(GroupKFold(n_splits=5).split(rows, groups=groups))]


def training_pairs(store, rows, gold):
    graph = {tuple(pair) for pair in candidate_pairs(rows, [], store.vectors['qwen'])}
    needs_texts = {row['text'] for row in rows}
    reply_graph = set()
    for reply in store.data.get('reply_inputs', []):
        a = reply['query']['text']
        if a not in needs_texts:
            continue
        for reference in reply['references']:
            b = reference['text']
            if b in needs_texts:
                reply_graph.add(tuple(sorted((a, b))))
    extra_reply_graph = reply_graph - graph
    pairs = []
    for left, right in itertools.combinations(range(len(rows)), 2):
        a, b = rows[left], rows[right]
        edge = tuple(sorted((a['text'], b['text'])))
        if a['product'] != b['product'] or (edge not in graph and edge not in extra_reply_graph):
            continue
        values = pair_inputs(store, a['text'], b['text'])
        if values is None:
            continue
        ga, gb = gold[a['id']], gold[b['id']]
        same = (ga['kind'] != 'unknown' and gb['kind'] != 'unknown'
                and ga.get('topic_id') is not None and ga['topic_id'] == gb.get('topic_id'))
        pairs.append(dict(left=left, right=right, values=list(values), label=int(same),
                          extra_reply_edge=edge in extra_reply_graph))
    return pairs


def extra_reply_counts(pairs, indices=None):
    selected = pairs if indices is None else (pairs[i] for i in indices)
    labels = [pair['label'] for pair in selected if pair['extra_reply_edge']]
    return dict(n=len(labels), positive=sum(labels), negative=len(labels) - sum(labels))


def partition_pairs(pairs, train_indices, validation_indices):
    train_rows, val_rows = set(train_indices), set(validation_indices)
    train, validation, mixed = [], [], 0
    for index, pair in enumerate(pairs):
        endpoints = {pair['left'], pair['right']}
        if endpoints <= train_rows:
            train.append(index)
        elif endpoints <= val_rows:
            validation.append(index)
        else:
            mixed += 1
    return train, validation, mixed


def logistic_head(x, y, c, *, binary):
    if len(set(y)) < 2:
        raise ValueError('Cannot fit a single-class logistic head')
    model = LogisticRegression(C=c, class_weight='balanced', max_iter=800, random_state=0).fit(x, y)
    common = dict(C=c, class_weight='balanced', max_iter=800, n=len(y), classes=model.classes_.tolist())
    if binary:
        if list(model.classes_) != [0, 1]:
            raise ValueError('Pair labels must be binary zero/one')
        return dict(common, coefficients=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))
    coefficients, intercept = model.coef_.tolist(), model.intercept_.tolist()
    if len(model.classes_) == 2:
        coefficients = [[0.] * len(coefficients[0]), coefficients[0]]
        intercept = [0., intercept[0]]
    return dict(common, coefficients=coefficients, intercept=intercept)


def fit_heads(store, rows, gold, pairs, train_indices, pair_indices, kind_c, pair_c):
    x = np.asarray([kind_inputs(store, rows[i]['text']) for i in train_indices], dtype=float)
    y = [gold[rows[i]['id']]['kind'] for i in train_indices]
    if x.ndim != 2 or x.shape[1] != 1029 or set(y) != set(KIND_HYPOTHESES):
        raise ValueError('Kind training requires 1024 Qwen + 5 NLI features and all five classes')
    px = np.asarray([pairs[i]['values'] for i in pair_indices], dtype=float)
    py = [pairs[i]['label'] for i in pair_indices]
    if px.ndim != 2 or px.shape[1] != 5:
        raise ValueError('Pair training requires exactly five features')
    return dict(kind=logistic_head(x, y, kind_c, binary=False),
                pair=logistic_head(px, py, pair_c, binary=True))


class CachedCandidateFeatures(CandidateFeatures):
    """Reuse deterministic head scores across threshold trials within one fold."""
    def __init__(self, store, heads):
        super().__init__(store, heads)
        self._kind_cache, self._pair_cache = {}, {}

    def kind_scores(self, text):
        if text not in self._kind_cache:
            self._kind_cache[text] = super().kind_scores(text)
        return self._kind_cache[text]

    def pair_score(self, a, b):
        if (a, b) not in self._pair_cache:
            self._pair_cache[a, b] = super().pair_score(a, b)
        return self._pair_cache[a, b]


def prefix_predictions(predictions, fold):
    prefixed = copy.deepcopy(predictions)
    for row in prefixed:
        if row['prediction'] is not None:
            product, kind, group = row['prediction']
            row['prediction'] = [product, kind, f'fold-{fold}:{group}']
    return prefixed


def score_needs(predictions, gold):
    if not predictions or len({p['id'] for p in predictions}) != len(predictions):
        raise ValueError('Needs predictions must have nonempty unique ID coverage')
    truth = [None if gold[p['id']]['kind'] == 'unknown' else gold[p['id']]['topic_id'] for p in predictions]
    metrics = pair_metrics(truth, [tuple(p['prediction']) if p['prediction'] is not None else None for p in predictions])
    return dict(n=len(predictions), kind_accuracy=sum(p['kind'] == gold[p['id']]['kind']
                                                     for p in predictions) / len(predictions),
                pair_metrics=metrics)


def needs_rank(trial):
    metrics = trial['metrics']
    pair = metrics['pair_metrics']
    precision, recall = pair['precision'] or 0., pair['recall'] or 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision >= .8, f1, metrics['kind_accuracy'], recall, precision


def grouped_search(store, rows, gold):
    folds = grouped_folds(rows, gold)
    pairs = training_pairs(store, rows, gold)
    report = dict(status='ok', n_splits=5, n=len(rows), pairs=len(pairs), folds=[], trials=[],
                  extra_reply_edges=extra_reply_counts(pairs),
                  method='Whole gold-topic families held out; unknown singletons use their own IDs. '
                         'Pair train endpoints both in train; validation endpoints both in validation. '
                         'Extra reply edges require two existing needs endpoints and use only needs gold labels. '
                         'Input-only retrieval graph is shared; held-out labels never enter either head. '
                         'Thresholds and C selected on grouped CV; not an independent acceptance test.')
    partitions = []
    for fold in folds:
        train, validation, crossed = partition_pairs(pairs, fold['train'], fold['validation'])
        summary = dict(fold=fold['fold'], train_ids=[rows[i]['id'] for i in fold['train']],
                       validation_ids=[rows[i]['id'] for i in fold['validation']],
                       train_families=sorted({family_key(rows[i], gold) for i in fold['train']}),
                       validation_families=sorted({family_key(rows[i], gold) for i in fold['validation']}),
                       train_pairs=len(train), validation_pairs=len(validation), excluded_cross_pairs=crossed,
                       train_pair_classes=sorted({pairs[i]['label'] for i in train}),
                       validation_pair_classes=sorted({pairs[i]['label'] for i in validation}))
        summary['train_extra_reply_edges'] = extra_reply_counts(pairs, train)
        summary['validation_extra_reply_edges'] = extra_reply_counts(pairs, validation)
        summary['excluded_cross_extra_reply_edges'] = {
            name: total - summary['train_extra_reply_edges'][name] - summary['validation_extra_reply_edges'][name]
            for name, total in report['extra_reply_edges'].items()}
        report['folds'].append(summary)
        if summary['train_pair_classes'] != [0, 1] or summary['validation_pair_classes'] != [0, 1]:
            report.update(status='failed', reason='single-class pair train/validation fold')
            raise FoldError('single-class pair train/validation fold', report)
        partitions.append((train, validation))
    for kind_c, pair_c in itertools.product(C_VALUES, repeat=2):
        prepared = []
        for fold, (train_pairs, _) in zip(folds, partitions):
            try:
                heads = fit_heads(store, rows, gold, pairs, fold['train'], train_pairs, kind_c, pair_c)
            except ValueError as error:
                report.update(status='failed', reason=str(error), failed_fold=fold['fold'])
                raise FoldError(str(error), report) from error
            features = CachedCandidateFeatures(store, heads)
            prepared.append((fold['fold'], features, [rows[i] for i in fold['validation']]))
        for kind_min, kind_margin, pair_min, cosine_min in itertools.product(
                KIND_MIN, KIND_MARGIN, PAIR_MIN, COSINE_MIN):
            config = dict(DEFAULTS, kind_min=kind_min, kind_margin=kind_margin,
                          pair_min=pair_min, cosine_min=cosine_min, margin=0.)
            predictions = []
            for fold_id, features, val_rows in prepared:
                predicted = CandidateSelector(features, backend='qwen', config=config).predict(val_rows)
                if {p['id'] for p in predicted} != {r['id'] for r in val_rows}:
                    raise RuntimeError('Fold selector omitted validation rows')
                predictions.extend(prefix_predictions(predicted, fold_id))
            report['trials'].append(dict(kind_C=kind_c, pair_C=pair_c, config=config,
                                         metrics=score_needs(predictions, gold)))
        print('CV', 'kind_C', kind_c, 'pair_C', pair_c, 'trials', len(report['trials']), flush=True)
    selected = max(report['trials'], key=needs_rank)
    report['selected'] = selected
    report['selection_rule'] = 'Pair precision >= .80 first, then pair F1, actual selector kind accuracy, recall, precision.'
    refitted = fit_heads(store, rows, gold, pairs, list(range(len(rows))), list(range(len(pairs))),
                         selected['kind_C'], selected['pair_C'])
    refitted['pair']['extra_reply_edges'] = report['extra_reply_edges']
    refitted['features'] = dict(kind='Qwen1024 + five NLI kind hypotheses',
                               pair=['cosine', 'min_reranker', 'max_reranker', 'min_entailment', 'max_contradiction'])
    return report, refitted


def binary_metrics(expected, predicted):
    tp = sum(a and b for a, b in zip(expected, predicted))
    fp = sum(not a and b for a, b in zip(expected, predicted))
    fn = sum(a and not b for a, b in zip(expected, predicted))
    tn = sum(not a and not b for a, b in zip(expected, predicted))
    return dict(n=len(expected), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None)


def score_replies(predictions, rows, gold):
    indexed = {p['id']: p for p in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['id'] for r in rows}:
        raise ValueError('Reply predictions must cover every input ID exactly once')
    result = {}
    for event, scope, action in [('incoming_ticket', 'ordinary', 'queue_standard'),
                                  ('fixed_notice_candidate', 'fixed_notice', 'queue_fixed_notice')]:
        group = [r for r in rows if r['event'] == event]
        result[scope] = binary_metrics([gold[r['id']]['safe_action'] == action for r in group],
                                        [indexed[r['id']]['action'] == action for r in group])
    result['action_accuracy'] = (sum(indexed[r['id']]['action'] == gold[r['id']]['safe_action'] for r in rows)
                                 / len(rows) if rows else None)
    return result


def reply_rank(metrics):
    ordinary = metrics['ordinary']
    return (ordinary['fp'] == 0, ordinary['recall'] or 0.,
            ordinary['precision'] or 0., metrics['action_accuracy'] or 0.)


def predict_replies(policy, rows):
    return [dict(id=row['id'], **policy.decide(row)) for row in rows]


def baseline_predictions(store):
    config = read_json(ROOT / 'data/semantic-config-v5b.json')['backends']['qwen']
    features = CalibratedFeatures(store, ROOT / 'data/semantic-heads-v5b.json').for_backend('qwen')
    return dict(needs=CalibratedSelector(features, backend='qwen', config=config['needs']).predict(store.data['needs_inputs']),
                reply=predict_replies(CalibratedReplyPolicy(features, backend='qwen', config=config['reply']),
                                      store.data['reply_inputs']))


def predict_models(store):
    config = read_json(CONFIG)
    heads = read_json(HEADS)
    features = CachedCandidateFeatures(store, heads)
    candidate = dict(needs=CandidateSelector(features, backend='qwen', config=config['needs']).predict(store.data['needs_inputs']),
                     reply=predict_replies(CandidateReplyPolicy(features, config=config['reply']), store.data['reply_inputs']))
    return dict(baseline=baseline_predictions(store), candidate=candidate)


def validate_predictions(predictions, store):
    if set(predictions) != {'baseline', 'candidate'}:
        raise RuntimeError('Both baseline and candidate predictions are required')
    inputs = {task: {r['id']: r for r in store.data[field]}
              for task, field in [('needs', 'needs_inputs'), ('reply', 'reply_inputs')]}
    for name, body in predictions.items():
        if set(body) != {'needs', 'reply'}:
            raise RuntimeError('Both prediction tasks are required: ' + name)
        for task, rows in body.items():
            ids = [row['id'] for row in rows]
            if len(set(ids)) != len(ids) or set(ids) != set(inputs[task]):
                raise RuntimeError('Incomplete or duplicate prediction IDs: ' + name + '/' + task)
            for row in rows:
                if task == 'needs':
                    group = row.get('prediction')
                    if (row.get('kind') not in KIND_HYPOTHESES or 'prediction' not in row
                            or (group is not None and (not isinstance(group, list) or len(group) != 3))):
                        raise RuntimeError('Invalid needs prediction payload')
                    if group is not None and (row['kind'] == 'unknown' or not all(isinstance(v, str) and v for v in group)
                                              or group[1] != row['kind']
                                              or group[0] != inputs[task][row['id']]['product']):
                        raise RuntimeError('Invalid needs group scope/kind')
                elif row.get('action') not in ACTIONS:
                    raise RuntimeError('Invalid reply prediction payload')


def check_training_provenance(heads, config, store):
    provenance = heads['provenance']
    if (provenance['feature_source_sha256'] != sha256_file(OUT / 'tune-features.json')
            or provenance['training_sources_sha256'] != training_sources()
            or provenance['preparation_contract_sha256'] != feature_contract()
            or config['training']['feature_source_sha256'] != provenance['feature_source_sha256']
            or config['training']['heads_sha256'] != sha256_file(HEADS)
            or provenance['semantic_code_sha256'] != semantic_code_hashes()
            or plain_metadata(store.data['metadata']) != provenance['models']):
        raise RuntimeError('Training provenance changed; refit before freeze')


def train():
    assert_unfrozen()
    verify_baseline()
    sources = training_sources()
    code = semantic_code_hashes()
    store = load_features('tune')
    gold, _ = load_tune_gold(store)
    rows = unique_needs(store.data['needs_inputs'])
    try:
        cv, heads = grouped_search(store, rows, gold)
    except FoldError as error:
        save_checkpoint(OUT / 'tune-grouped-results.json', dict(status='failed', **{
            key: value for key, value in error.report.items() if key != 'status'}))
        raise
    if training_sources() != sources or semantic_code_hashes() != code:
        raise RuntimeError('Tune sources/code changed during grouped training')
    provenance = dict(training_sources_sha256=sources, feature_source_sha256=sha256_file(OUT / 'tune-features.json'),
                      semantic_code_sha256=code,
                      preparation_contract_sha256=feature_contract(), models=plain_metadata(store.data['metadata']),
                      training_ids=[r['id'] for r in rows], training_families=sorted({family_key(r, gold) for r in rows}))
    heads['provenance'] = provenance
    save_checkpoint(HEADS, heads)
    config = dict(version='5c', selected_backend='qwen', status='heads_trained_reply_pending',
                  needs=cv['selected']['config'],
                  training=dict(feature_source_sha256=provenance['feature_source_sha256'], heads_sha256=sha256_file(HEADS)),
                  selection=cv['selection_rule'])
    save_checkpoint(CONFIG, config)
    save_checkpoint(OUT / 'tune-grouped-results.json', dict(status='heads_trained_reply_pending', needs_cv=cv,
                    training=provenance, caveat='Grouped CV drives model/threshold selection, not independent acceptance.'))
    return config


def tune():
    assert_unfrozen()
    verify_baseline()
    store = load_features('tune')
    gold, reply_gold = load_tune_gold(store)
    heads, config = read_json(HEADS), read_json(CONFIG)
    check_training_provenance(heads, config, store)
    report = read_json(OUT / 'tune-grouped-results.json')
    if report.get('status') not in {'heads_trained_reply_pending', 'tuned'}:
        raise RuntimeError('Successful grouped training is required before reply tuning')
    features = CachedCandidateFeatures(store, heads)
    trials = []
    for cosine, pair_min, risk_min in itertools.product(COSINE_MIN, PAIR_MIN, (.6, .8)):
        candidate = dict(config['needs'], cosine_min=cosine, pair_min=pair_min, risk_min=risk_min, margin=0.)
        predictions = predict_replies(CandidateReplyPolicy(features, config=candidate), store.data['reply_inputs'])
        trials.append(dict(config=candidate, metrics=score_replies(predictions, store.data['reply_inputs'], reply_gold)))
    best = max(trials, key=lambda trial: reply_rank(trial['metrics']))
    config.update(reply=best['config'], status='tuned_control_pending',
                  reply_selection='Tune resubstitution: zero ordinary false positives first, then ordinary recall.')
    predictions = CandidateSelector(features, backend='qwen', config=config['needs']).predict(store.data['needs_inputs'])
    baseline = baseline_predictions(store)
    report.update(status='tuned', needs_resubstitution=score_needs(predictions, gold),
                  reply_resubstitution=dict(**best, evaluation='resubstitution on reply tune; heads refitted on all needs tune families; '
                                             'no claim of family-held-out reply generalization'),
                  reply_trials=trials,
                  baseline_tune=dict(needs=score_needs(baseline['needs'], gold),
                                     reply=score_replies(baseline['reply'], store.data['reply_inputs'], reply_gold)),
                  auto_enable=False)
    check_training_provenance(heads, config, store)
    save_checkpoint(OUT / 'tune-grouped-results.json', report)
    save_checkpoint(CONFIG, config)
    return report


def engine_dependencies(metadata):
    """Local helper sources declared by the actual worker, as root-relative SHA pins."""
    descriptor = metadata.get('reranker', {})
    dependencies = descriptor.get('dependency_sha256', {})
    if not isinstance(dependencies, dict):
        raise RuntimeError('Reranker dependency SHA inventory must be a mapping')
    for name, digest in dependencies.items():
        if (not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise RuntimeError('Invalid reranker dependency path/SHA')
        root_path(name)
    return dict(dependencies)


def freeze_inventory(details, preserved, *, metadata=None):
    dependencies = engine_dependencies(metadata or {})
    names = set(CODE) | set(preserved) | {
        str(HEADS.relative_to(ROOT)), str(CONFIG.relative_to(ROOT)),
        'reports/quality-v5c/baseline-v5b-preserved.json',
        'reports/quality-v5c/tune-features.json', 'reports/quality-v5c/tune-grouped-results.json',
        'evaluation/v5c/tune-inputs.jsonl', 'evaluation/v5c/reply-tune-inputs.jsonl',
        'evaluation/v5c/PROTOCOL.md', 'evaluation/v5c/SCHEMA.json', 'evaluation/v5c/validate.py',
    }
    if root_path(MLX_WORKER).exists():
        names.add(MLX_WORKER)
    for artifact in ('tune-matching-ablation.json','tune-reply-ablation.json'):
        relative='reports/quality-v5c/'+artifact
        if root_path(relative).is_file():
            names.add(relative)
    names.update(dependencies)
    for backend, directory in MODEL_DIRS.items():
        base = 'runtime/models/' + directory + '/'
        names.add(base + 'MANIFEST.json')
        for name in details['pins'][backend]['files']:
            names.add(base + name)
    files = {name: sha256_file(root_path(name)) for name in sorted(names)}
    for name, expected in dependencies.items():
        if files[name] != expected:
            raise RuntimeError('Reranker dependency source differs from actual descriptor: ' + name)
    for backend, directory in MODEL_DIRS.items():
        for name, pin in details['pins'][backend]['files'].items():
            if files['runtime/models/' + directory + '/' + name] != pin['sha256']:
                raise RuntimeError('Pinned model artifact changed: ' + backend + '/' + name)
    return files


def frozen_model_pins(details, metadata, files):
    """Bind the measured engine to its source without importing either runtime."""
    pins = copy.deepcopy(details['pins'])
    descriptor = metadata.get('reranker')
    if descriptor is None:
        return pins
    engine = descriptor.get('engine', 'torch')
    if engine not in {'torch', 'mlx'}:
        raise RuntimeError('Unsupported reranker engine descriptor')
    if descriptor.get('device') == 'mlx-metal' and engine != 'mlx':
        raise RuntimeError('Reranker engine descriptor contradicts actual MLX device')
    if descriptor.get('worker_sha256') != files.get(RERANKER_WORKER):
        raise RuntimeError('Reranker dispatcher SHA differs from actual descriptor')
    path = MLX_WORKER if engine == 'mlx' else RERANKER_WORKER
    expected = files.get(path)
    claimed = descriptor.get('engine_sha256')
    if engine == 'torch' and claimed is None:
        claimed = descriptor['worker_sha256']
    if expected is None or claimed != expected:
        raise RuntimeError('Reranker engine source SHA is missing or differs from actual descriptor')
    dependencies = engine_dependencies(metadata)
    if any(files.get(name) != digest for name, digest in dependencies.items()):
        raise RuntimeError('Reranker engine dependency SHA is missing or differs from files inventory')
    pins['reranker'].update(engine=engine, engine_path=path, engine_sha256=expected,
                           dependency_sha256=dependencies,
                           descriptor_sha256=fingerprint(plain_metadata(metadata)['reranker']))
    return pins


def verify_tune_lock(sources):
    directory = ROOT / 'evaluation/v5c'
    locked = read_json(directory / 'CONTROL-LOCK.json')['files']
    for relative, digest in sources.items():
        name = str((ROOT / relative).relative_to(directory))
        entry = locked.get(name)
        expected = entry.get('sha256') if isinstance(entry, dict) else entry
        if digest != expected:
            raise RuntimeError('Tune source differs from independently locked dataset: ' + name)


def freeze():
    assert_unfrozen()
    preserved = verify_baseline()
    store = load_features('tune')
    heads, config = read_json(HEADS), read_json(CONFIG)
    check_training_provenance(heads, config, store)
    report = read_json(OUT / 'tune-grouped-results.json')
    if config.get('status') != 'tuned_control_pending' or report.get('status') != 'tuned':
        raise RuntimeError('Complete grouped training and reply tuning before freeze')
    if (report['needs_cv']['selected']['config'] != config['needs']
            or report['reply_resubstitution']['config'] != config['reply']):
        raise RuntimeError('Selected configuration differs from tune report')
    verify_tune_lock(heads['provenance']['training_sources_sha256'])
    details = contract_details()
    files = freeze_inventory(details, preserved, metadata=store.data['metadata'])
    frozen = dict(version='v5c', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  files=files, dataset_lock_path='evaluation/v5c/CONTROL-LOCK.json',
                  dataset_lock_sha256=sha256_file(ROOT / 'evaluation/v5c/CONTROL-LOCK.json'),
                  preparation_contract_sha256=feature_contract(),
                  tune_features_sha256=sha256_file(OUT / 'tune-features.json'),
                  models=plain_metadata(store.data['metadata']),
                  pins=frozen_model_pins(details, store.data['metadata'], files),
                  training_sources_sha256=heads['provenance']['training_sources_sha256'],
                  preserved_v5b_files=preserved, selected_backend='qwen',
                  rule='One control run. Persist complete baseline/candidate predictions before any control gold. '
                       'No post-control threshold, model, family or denominator changes. auto_enable remains false.')
    write_exclusive(OUT / 'CONFIG-FREEZE.json', frozen)
    return frozen


def verify_freeze():
    path = OUT / 'CONFIG-FREEZE.json'
    if not path.is_file():
        raise RuntimeError('Control prediction requires a verified freeze')
    frozen = read_json(path)
    if not isinstance(frozen.get('files'), dict) or not frozen['files']:
        raise RuntimeError('Freeze needs the complete files inventory')
    for name, expected in frozen['files'].items():
        if sha256_file(root_path(name)) != expected:
            raise RuntimeError('Frozen semantic/config/artifact file changed: ' + name)
    if 'pins' in frozen:
        expected = frozen_model_pins({'pins': frozen['pins']}, frozen.get('models', {}), frozen['files'])
        if expected != frozen['pins']:
            raise RuntimeError('Frozen reranker engine pins differ from descriptor/files inventory')
    verify_baseline()
    lock = root_path(frozen.get('dataset_lock_path', 'evaluation/v5c/CONTROL-LOCK.json'))
    if sha256_file(lock) != frozen.get('dataset_lock_sha256'):
        raise RuntimeError('Frozen dataset lock changed')
    if feature_contract() != frozen.get('preparation_contract_sha256'):
        raise RuntimeError('Frozen preparation contract changed')
    return frozen


def control():
    target = OUT / 'control-predictions-only.json'
    if target.exists() or target.with_suffix('.sha256').exists():
        raise RuntimeError('Control predictions exist; refusing to overwrite')
    frozen = verify_freeze()
    feature_path = OUT / 'control-features.json'
    feature_sha = sha256_file(feature_path)
    store = load_features('control', frozen=frozen)
    predictions = predict_models(store)
    validate_predictions(predictions, store)
    if verify_freeze() != frozen:
        raise RuntimeError('Freeze changed during control prediction')
    if feature_sha != sha256_file(feature_path):
        raise RuntimeError('Control features changed during prediction')
    for name, expected in store.data.get('source_sha256', {}).items():
        if sha256_file(root_path(name)) != expected:
            raise RuntimeError('Control inputs changed during prediction')
    write_exclusive(target, predictions)
    with target.with_suffix('.sha256').open('x', encoding='ascii') as stream:
        stream.write(sha256_file(target) + '  ' + target.name + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'tune', 'freeze', 'control'])
    args = parser.parse_args()
    {'train': train, 'tune': tune, 'freeze': freeze, 'control': control}[args.action]()
    print('READY', args.action, flush=True)


if __name__ == '__main__':
    main()
