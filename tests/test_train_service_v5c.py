"""Grouped training and durable prediction guards; all data and heads are local fixtures."""
import copy
import importlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def service():
    return importlib.import_module('scripts.train_service_v5c')


@pytest.fixture
def dataset(service):
    rows, gold = [], {}
    kinds = ['wish', 'bug', 'usability', 'service']
    for family in range(10):
        for member, prefix in enumerate(('a', 'z')):
            ident = f'n-{family}-{member}'
            rows.append(dict(id=ident, source_id=ident, product='p', text=f'{prefix}-{family:02}',
                             created_at=f'2026-10-{family + 1:02}T00:00:0{member}+00:00'))
            gold[ident] = dict(kind=kinds[family % 4], topic_id=f'topic-{family}')
    for i in range(5):
        ident = f'unknown-{i}'
        rows.append(dict(id=ident, source_id=ident, product='p', text=f'u-{i}', created_at=''))
        gold[ident] = dict(kind='unknown', topic_id=None)
    vectors = {service.key(row['text']): np.r_[1., np.zeros(1023)] for row in rows}
    by_text = {row['text']: gold[row['id']] for row in rows}

    class Store:
        data = dict(needs_inputs=rows, reply_inputs=[], metadata={'qwen': {'fixture': True}}, reranker={})

        def __init__(self):
            self.vectors = {'qwen': vectors}
            for a, b in itertools.permutations(by_text, 2):
                same = by_text[a]['topic_id'] is not None and by_text[a]['topic_id'] == by_text[b]['topic_id']
                self.data['reranker'][service.pair_key(a, b)] = .95 if same else .1

        def cosine(self, backend, a, b):
            assert backend == 'qwen'
            return 1.

        def kind_scores(self, text):
            return {label: .9 if label == by_text[text]['kind'] else .025 for label in service.KIND_HYPOTHESES}

        def relation(self, a, b):
            same = a == b or (by_text[a]['topic_id'] is not None and by_text[a]['topic_id'] == by_text[b]['topic_id'])
            return dict(entailment=.9 if same else .1, neutral=.05 if same else .8, contradiction=.05 if same else .1)

        def risk_scores(self, text):
            return dict(money=0., credentials=0., injection=0.)

    return rows, gold, Store()


@pytest.fixture
def reply_dataset(dataset):
    rows, gold, store = dataset
    store.data['reply_inputs'] = [
        dict(query={'text': 'a-09'}, references=[{'text': 'z-09'}], approved=False),
        dict(query={'text': 'z-09'}, references=[{'text': 'z-08'}], approved=True),
        dict(query={'text': 'z-08'}, references=[{'text': 'z-09'}], approved=False),
        dict(query={'text': 'a-00'}, references=[{'text': 'z-00'}]),
        dict(query={'text': 'not a need'}, references=[{'text': 'z-09'}]),
        dict(query={'text': 'z-09'}, references=[{'text': 'not a need'}]),
    ]
    return rows, gold, store


@pytest.fixture
def workspace(service, tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ROOT', tmp_path)
    monkeypatch.setattr(service, 'OUT', tmp_path / 'reports/quality-v5c')
    monkeypatch.setattr(service, 'HEADS', tmp_path / 'data/semantic-heads-v5c.json')
    monkeypatch.setattr(service, 'CONFIG', tmp_path / 'data/semantic-config-v5c.json')
    monkeypatch.setattr(service, 'CODE', ('code.py',))
    monkeypatch.setattr(service, 'feature_contract', lambda: 'contract')
    monkeypatch.setattr(service, 'verify_baseline', lambda: {})
    (tmp_path / 'code.py').write_text('fixed semantic code')
    service.OUT.mkdir(parents=True)
    (service.OUT / 'tune-features.json').write_text('{}')
    return tmp_path


def test_five_grouped_folds_hold_out_whole_families_once(service, dataset):
    rows, gold, _ = dataset
    folds = service.grouped_folds(rows, gold)
    assert len(folds) == 5
    visits = []
    for fold in folds:
        train = {service.family_key(rows[i], gold) for i in fold['train']}
        validation = {service.family_key(rows[i], gold) for i in fold['validation']}
        assert not train & validation
        assert train | validation == {service.family_key(row, gold) for row in rows}
        visits.extend(fold['validation'])
    assert sorted(visits) == list(range(len(rows)))
    assert len({service.family_key(row, gold) for row in rows if gold[row['id']]['kind'] == 'unknown'}) == 5


def test_pair_partitions_exclude_both_held_out_ends_and_mixed_boundaries(service, dataset):
    rows, gold, store = dataset
    pairs = service.training_pairs(store, rows, gold)
    assert {pair['label'] for pair in pairs} == {0, 1}
    assert any(gold[rows[p['left']]['id']]['kind'] != gold[rows[p['right']]['id']]['kind'] for p in pairs)
    for fold in service.grouped_folds(rows, gold):
        train, validation, crossed = service.partition_pairs(pairs, fold['train'], fold['validation'])
        train_rows, val_rows = set(fold['train']), set(fold['validation'])
        assert all({pairs[i]['left'], pairs[i]['right']} <= train_rows for i in train)
        assert all({pairs[i]['left'], pairs[i]['right']} <= val_rows for i in validation)
        assert len(train) + len(validation) + crossed == len(pairs)
        assert crossed > 0


def test_training_pairs_require_top_eight_graph_and_both_reranker_directions(service, dataset):
    rows, gold, store = dataset
    pairs = service.training_pairs(store, rows, gold)
    # Last two a-* messages and z-* messages cannot retrieve one another under the cosine tie rule.
    assert not any(rows[p['left']]['text'] == 'z-08' and rows[p['right']]['text'] == 'z-09' for p in pairs)
    first = pairs[0]
    a, b = rows[first['left']]['text'], rows[first['right']]['text']
    store.data['reranker'].pop(service.pair_key(b, a))
    after = service.training_pairs(store, rows, gold)
    assert len(after) == len(pairs) - 1


def test_fold_heads_are_fitted_only_to_training_rows_and_pair_endpoints(service, reply_dataset, monkeypatch):
    rows, gold, store = reply_dataset
    records = []
    real = service.fit_heads

    def tracked(store, rows, gold, pairs, train_indices, pair_indices, kind_c, pair_c):
        records.append((set(train_indices), {endpoint for i in pair_indices
                                             for endpoint in (pairs[i]['left'], pairs[i]['right'])}))
        return real(store, rows, gold, pairs, train_indices, pair_indices, kind_c, pair_c)

    monkeypatch.setattr(service, 'fit_heads', tracked)
    monkeypatch.setattr(service, 'C_VALUES', (.1,))
    monkeypatch.setattr(service, 'KIND_MIN', (0.,))
    monkeypatch.setattr(service, 'KIND_MARGIN', (0.,))
    monkeypatch.setattr(service, 'PAIR_MIN', (.4,))
    monkeypatch.setattr(service, 'COSINE_MIN', (0.,))
    report, heads = service.grouped_search(store, rows, gold)
    assert report['n_splits'] == 5 and report['status'] == 'ok'
    assert report['extra_reply_edges'] == dict(n=2, positive=1, negative=1)
    assert len(records) == 6  # five independent fits and one final refit
    pairs = service.training_pairs(store, rows, gold)
    for record, fold, summary in zip(records, service.grouped_folds(rows, gold), report['folds']):
        assert record[0] == set(fold['train']) and record[1] <= record[0]
        assert not record[1] & set(fold['validation'])
        for scope in ('train', 'validation'):
            indices = set(fold[scope])
            labels = [p['label'] for p in pairs if p['extra_reply_edge']
                      and p['left'] in indices and p['right'] in indices]
            assert summary[scope + '_extra_reply_edges'] == dict(
                n=len(labels), positive=labels.count(1), negative=labels.count(0))
        for name, total in report['extra_reply_edges'].items():
            assert sum(summary[scope + '_extra_reply_edges'][name]
                       for scope in ('train', 'validation', 'excluded_cross')) == total
    assert set(heads) >= {'kind', 'pair'}
    assert len(heads['kind']['coefficients'][0]) == 1029
    assert len(heads['pair']['coefficients']) == 5
    assert heads['pair']['extra_reply_edges'] == report['extra_reply_edges']
    assert heads['kind']['class_weight'] == 'balanced' and heads['kind']['max_iter'] == 800
    assert report['selected']['metrics']['n'] == len(rows)
    assert report['selected']['metrics']['kind_accuracy'] >= 0


def test_reply_edges_add_only_indexed_needs_pairs_and_use_needs_gold(service, reply_dataset):
    rows, gold, store = reply_dataset
    replies = copy.deepcopy(store.data['reply_inputs'])
    store.data['reply_inputs'] = []
    original = service.training_pairs(store, rows, gold)
    store.data['reply_inputs'] = replies
    pairs = service.training_pairs(store, rows, gold)
    added = [pair for pair in pairs if pair['extra_reply_edge']]
    assert len(pairs) == len(original) + 2
    assert {frozenset((rows[p['left']]['text'], rows[p['right']]['text'])): p['label'] for p in added} == {
        frozenset(('a-09', 'z-09')): 1,
        frozenset(('z-08', 'z-09')): 0,
    }
    # Reply approvals and contextual kind cannot change labels derived from needs.
    for reply in replies:
        reply['approved'] = not reply.get('approved', False)
        reply['kind'] = 'service'
        for ref in reply['references']:
            ref['kind'] = 'unknown'
    assert service.training_pairs(store, rows, gold) == pairs
    store.data['reranker'].pop(service.pair_key('z-09', 'z-08'))
    remaining = service.training_pairs(store, rows, gold)
    assert len(remaining) == len(pairs) - 1
    assert [pair['label'] for pair in remaining if pair['extra_reply_edge']] == [1]


def test_reply_extra_edges_respect_both_held_out_endpoints(service, reply_dataset):
    rows, gold, store = reply_dataset
    pairs = service.training_pairs(store, rows, gold)
    extras = {i for i, pair in enumerate(pairs) if pair['extra_reply_edge']}
    assert len(extras) == 2
    for held in [{'topic-9'}, {'topic-8'}, {'topic-8', 'topic-9'}, {'topic-0'}]:
        validation = {i for i, row in enumerate(rows) if gold[row['id']]['topic_id'] in held}
        train = set(range(len(rows))) - validation
        train_pairs, val_pairs, _ = service.partition_pairs(pairs, train, validation)
        for index in extras:
            endpoints = {pairs[index]['left'], pairs[index]['right']}
            assert (index in train_pairs) == (endpoints <= train)
            assert (index in val_pairs) == (endpoints <= validation)
            if endpoints & train and endpoints & validation:
                assert index not in train_pairs and index not in val_pairs


def test_fold_group_keys_are_prefixed_before_aggregation(service):
    value = [dict(id='x', kind='bug', prediction=['p', 'bug', 'same-key'])]
    a, b = service.prefix_predictions(value, 0), service.prefix_predictions(value, 1)
    assert a[0]['prediction'] != b[0]['prediction']
    assert value[0]['prediction'] == ['p', 'bug', 'same-key']


def test_single_class_validation_pairs_produce_explicit_failure_report(service, dataset, workspace, monkeypatch):
    rows, gold, store = dataset
    monkeypatch.setattr(service, 'load_features', lambda split, frozen=None: store)
    monkeypatch.setattr(service, 'load_tune_gold', lambda store: (gold, {}))
    monkeypatch.setattr(service, 'training_sources', lambda: {})
    original = service.training_pairs

    def positives_only(*args):
        return [pair for pair in original(*args) if pair['label'] == 1]

    monkeypatch.setattr(service, 'training_pairs', positives_only)
    with pytest.raises(service.FoldError, match='single.class'):
        service.train()
    report = json.loads((service.OUT / 'tune-grouped-results.json').read_text())
    assert report['status'] == 'failed' and report['folds']
    assert not service.HEADS.exists()


def test_precision_priority_uses_real_cv_accuracy_as_tiebreaker(service):
    def trial(p, recall, accuracy):
        return dict(metrics=dict(kind_accuracy=accuracy, pair_metrics=dict(precision=p, recall=recall)))
    assert service.needs_rank(trial(.8, .5, .7)) > service.needs_rank(trial(.79, 1., 1.))
    assert service.needs_rank(trial(.9, .7, .9)) > service.needs_rank(trial(.9, .7, .6))


def test_ordinary_reply_zero_false_positive_precedes_recall_and_fix_is_separate(service):
    rows = [dict(id='ordinary', event='incoming_ticket'), dict(id='fixed', event='fixed_notice_candidate')]
    gold = {'ordinary': {'safe_action': 'operator'}, 'fixed': {'safe_action': 'queue_fixed_notice'}}
    result = service.score_replies([dict(id='ordinary', action='operator'),
                                    dict(id='fixed', action='queue_fixed_notice')], rows, gold)
    assert result['ordinary']['fp'] == 0 and result['fixed_notice']['tp'] == 1
    a = dict(ordinary=dict(fp=0, precision=None, recall=0.), action_accuracy=.5)
    b = dict(ordinary=dict(fp=1, precision=.9, recall=1.), action_accuracy=1.)
    assert service.reply_rank(a) > service.reply_rank(b)


def test_control_requires_freeze_before_inputs_or_features(service, workspace, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('input/features were opened before freeze verification')
    monkeypatch.setattr(service, 'load_features', forbidden)
    with pytest.raises(RuntimeError, match='[Ff]reez|frozen'):
        service.control()


def test_existing_predictions_are_never_overwritten(service, workspace, monkeypatch):
    path = service.OUT / 'control-predictions-only.json'
    path.write_bytes(b'existing immutable predictions')
    monkeypatch.setattr(service, 'verify_freeze', lambda: pytest.fail('Already committed run should stop immediately'))
    with pytest.raises(RuntimeError, match='exist|overwrite'):
        service.control()
    assert path.read_bytes() == b'existing immutable predictions'


def test_control_persists_both_models_without_opening_gold(service, workspace, monkeypatch):
    events = []
    frozen = {'models': {}, 'files': {}}
    monkeypatch.setattr(service, 'verify_freeze', lambda: events.append('freeze') or frozen)
    store = SimpleNamespace(data=dict(needs_inputs=[{'id': 'n', 'product': 'p'}], reply_inputs=[{'id': 'r'}]))
    monkeypatch.setattr(service, 'load_features', lambda split, frozen=None: events.append('features') or store)
    monkeypatch.setattr(service, 'load_tune_gold', lambda *args: pytest.fail('Gold must never enter control prediction'))
    predictions = {name: dict(needs=[dict(id='n', kind='bug', prediction=['p', 'bug', 'g'])],
                              reply=[dict(id='r', action='operator')]) for name in ('baseline', 'candidate')}
    monkeypatch.setattr(service, 'predict_models', lambda store: copy.deepcopy(predictions))
    (service.OUT / 'control-features.json').write_text('fixture features')
    data = service.control()
    assert events[0:2] == ['freeze', 'features']
    path = service.OUT / 'control-predictions-only.json'
    assert data == predictions == json.loads(path.read_text())
    assert path.with_suffix('.sha256').read_text().split()[0] == service.sha256_file(path)


def test_exclusive_json_commit_is_atomic_and_does_not_replace_existing_file(service, tmp_path):
    path = tmp_path / 'predictions.json'
    service.write_exclusive(path, {'first': True})
    with pytest.raises(RuntimeError, match='exist|overwrite'):
        service.write_exclusive(path, {'second': True})
    assert json.loads(path.read_text()) == {'first': True}
    assert list(tmp_path.iterdir()) == [path]


def test_freeze_verification_checks_code_lock_and_contract_without_gold(service, workspace, monkeypatch):
    lock = workspace / 'evaluation/v5c/CONTROL-LOCK.json'
    lock.parent.mkdir(parents=True)
    lock.write_text('{"files":{"sealed/control-gold.jsonl":"not opened"}}')
    frozen = dict(files={'code.py': service.sha256_file(workspace / 'code.py')},
                  dataset_lock_sha256=service.sha256_file(lock), preparation_contract_sha256='contract')
    (service.OUT / 'CONFIG-FREEZE.json').write_text(json.dumps(frozen))
    assert service.verify_freeze() == frozen
    for path in [workspace / 'code.py', lock]:
        original = path.read_bytes()
        path.write_bytes(b'changed')
        with pytest.raises(RuntimeError):
            service.verify_freeze()
        path.write_bytes(original)
    monkeypatch.setattr(service, 'feature_contract', lambda: 'changed')
    with pytest.raises(RuntimeError):
        service.verify_freeze()


def test_baseline_preservation_checks_all_twenty_four_original_hashes(service, workspace, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(service, 'ROOT', workspace)
    monkeypatch.setattr(service, 'OUT', workspace / 'reports/quality-v5c')
    files = {}
    for i in range(23):
        path = workspace / f'old-{i}.py'
        path.write_text(f'original {i}')
        files[path.name] = service.sha256_file(path)
    quality = workspace / 'data/quality-v5b.json'
    quality.parent.mkdir(exist_ok=True)
    quality.write_text('historical quality')
    manifest = dict(files=files, changed=[], selected_backend='qwen', quality_sha256=service.sha256_file(quality))
    (service.OUT / 'baseline-v5b-preserved.json').write_text(json.dumps(manifest))
    assert len(service.verify_baseline()) == 24
    quality.write_text('changed')
    with pytest.raises(RuntimeError):
        service.verify_baseline()


def test_train_tune_freeze_roundtrip_uses_real_cv_heads_and_labels_reply_resubstitution(
        service, dataset, workspace, monkeypatch):
    rows, gold, store = dataset
    monkeypatch.setattr(service, 'load_features', lambda split, frozen=None: store)
    monkeypatch.setattr(service, 'load_tune_gold', lambda store: (gold, {}))
    monkeypatch.setattr(service, 'training_sources', lambda: {})
    monkeypatch.setattr(service, 'C_VALUES', (.1,))
    monkeypatch.setattr(service, 'KIND_MIN', (0.,))
    monkeypatch.setattr(service, 'KIND_MARGIN', (0.,))
    monkeypatch.setattr(service, 'PAIR_MIN', (.4,))
    monkeypatch.setattr(service, 'COSINE_MIN', (0.,))
    baseline = dict(needs=[dict(id=r['id'], kind='unknown', prediction=None) for r in rows], reply=[])
    monkeypatch.setattr(service, 'baseline_predictions', lambda store: baseline)
    config = service.train()
    assert config['status'] == 'heads_trained_reply_pending'
    assert service.HEADS.exists()
    report = service.tune()
    assert report['status'] == 'tuned' and not report['auto_enable']
    assert 'resubstitution' in report['reply_resubstitution']['evaluation']
    assert report['needs_cv']['selected']['metrics']['n'] == len(rows)
    assert report['baseline_tune']['needs']['n'] == len(rows)
    monkeypatch.setattr(service, 'MODEL_DIRS', {})
    monkeypatch.setattr(service, 'contract_details', lambda: {'pins': {}})
    for name in ['PROTOCOL.md', 'SCHEMA.json', 'validate.py', 'tune-inputs.jsonl',
                 'reply-tune-inputs.jsonl', 'CONTROL-LOCK.json']:
        path = workspace / 'evaluation/v5c' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"files": {}}' if name == 'CONTROL-LOCK.json' else '{}')
    (service.OUT / 'baseline-v5b-preserved.json').write_text('{}')
    frozen = service.freeze()
    assert service.verify_freeze() == frozen
    assert 'code.py' in frozen['files']
    assert 'data/semantic-heads-v5c.json' in frozen['files']
    assert all('gold' not in path and 'sealed' not in path for path in frozen['files'])
    with pytest.raises(RuntimeError, match='frozen'):
        service.train()
    with pytest.raises(RuntimeError, match='frozen'):
        service.tune()
    with pytest.raises(RuntimeError, match='frozen'):
        service.freeze()


def test_incomplete_predictions_never_get_committed(service, workspace, monkeypatch):
    frozen = dict(files={})
    monkeypatch.setattr(service, 'verify_freeze', lambda: frozen)
    store = SimpleNamespace(data=dict(needs_inputs=[{'id': 'required'}], reply_inputs=[]))
    monkeypatch.setattr(service, 'load_features', lambda split, frozen=None: store)
    monkeypatch.setattr(service, 'predict_models', lambda store: {
        name: dict(needs=[], reply=[]) for name in ('baseline', 'candidate')})
    (service.OUT / 'control-features.json').write_text('fixture')
    with pytest.raises(RuntimeError, match='prediction IDs'):
        service.control()
    assert not (service.OUT / 'control-predictions-only.json').exists()


def test_freeze_changed_during_prediction_cannot_commit(service, workspace, monkeypatch):
    versions = iter([dict(files={}, serial=1), dict(files={}, serial=2)])
    monkeypatch.setattr(service, 'verify_freeze', lambda: next(versions))
    store = SimpleNamespace(data=dict(needs_inputs=[], reply_inputs=[]))
    monkeypatch.setattr(service, 'load_features', lambda split, frozen=None: store)
    monkeypatch.setattr(service, 'predict_models', lambda store: {
        name: dict(needs=[], reply=[]) for name in ('baseline', 'candidate')})
    (service.OUT / 'control-features.json').write_text('fixture')
    with pytest.raises(RuntimeError, match='Freeze changed'):
        service.control()
    assert not (service.OUT / 'control-predictions-only.json').exists()


def test_freeze_inventory_hashes_every_pinned_artifact_and_rejects_changed_weights(service, workspace, monkeypatch):
    heads, config = service.HEADS, service.CONFIG
    heads.parent.mkdir(parents=True, exist_ok=True)
    heads.write_text('{}')
    config.write_text('{}')
    for relative in ['reports/quality-v5c/baseline-v5b-preserved.json',
                     'reports/quality-v5c/tune-grouped-results.json',
                     'evaluation/v5c/tune-inputs.jsonl', 'evaluation/v5c/reply-tune-inputs.jsonl',
                     'evaluation/v5c/PROTOCOL.md', 'evaluation/v5c/SCHEMA.json', 'evaluation/v5c/validate.py']:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    pins = {}
    for backend, directory in service.MODEL_DIRS.items():
        base = workspace / 'runtime/models' / directory
        base.mkdir(parents=True)
        (base / 'MANIFEST.json').write_text('fixture manifest')
        (base / 'model.safetensors').write_bytes(b'tiny fake weights')
        pins[backend] = dict(files={'model.safetensors': {'sha256': service.sha256_file(base / 'model.safetensors')}})
    files = service.freeze_inventory(dict(pins=pins), {})
    for backend, directory in service.MODEL_DIRS.items():
        assert files[f'runtime/models/{directory}/model.safetensors'] == pins[backend]['files']['model.safetensors']['sha256']
    directory = next(iter(service.MODEL_DIRS.values()))
    (workspace / 'runtime/models' / directory / 'model.safetensors').write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='Pinned model artifact'):
        service.freeze_inventory(dict(pins=pins), {})


@pytest.mark.parametrize('change', ['cross-product', 'invalid-action', 'duplicate-id'])
def test_prediction_payloads_are_validated_before_commit(service, change):
    store = SimpleNamespace(data=dict(needs_inputs=[dict(id='n', product='p')], reply_inputs=[dict(id='r')]))
    body = dict(needs=[dict(id='n', kind='bug', prediction=['p', 'bug', 'group'])],
                reply=[dict(id='r', action='operator')])
    predictions = {name: copy.deepcopy(body) for name in ('baseline', 'candidate')}
    candidate = predictions['candidate']
    if change == 'cross-product':
        candidate['needs'][0]['prediction'][0] = 'another product'
    elif change == 'invalid-action':
        candidate['reply'][0]['action'] = 'send_email'
    else:
        candidate['needs'].append(copy.deepcopy(candidate['needs'][0]))
    with pytest.raises(RuntimeError):
        service.validate_predictions(predictions, store)


def test_freeze_checks_tune_sources_against_independent_lock_without_control_access(service, workspace):
    directory = workspace / 'evaluation/v5c'
    directory.mkdir(parents=True)
    source = directory / 'tune-gold.jsonl'
    source.write_text('synthetic tune labels')
    digest = service.sha256_file(source)
    lock = dict(files={'tune-gold.jsonl': digest,
                       'sealed/control-gold.jsonl': 'must not be opened',
                       'control-inputs.jsonl': 'must not be opened'})
    (directory / 'CONTROL-LOCK.json').write_text(json.dumps(lock))
    service.verify_tune_lock({'evaluation/v5c/tune-gold.jsonl': digest})
    with pytest.raises(RuntimeError, match='independently locked dataset'):
        service.verify_tune_lock({'evaluation/v5c/tune-gold.jsonl': 'changed'})


@pytest.mark.parametrize('mlx_present', [False, True])
def test_freeze_inventory_includes_optional_mlx_worker_without_importing_it(
        service, workspace, monkeypatch, mlx_present):
    monkeypatch.setattr(service, 'MODEL_DIRS', {})
    for relative in ['data/semantic-heads-v5c.json', 'data/semantic-config-v5c.json',
                     'reports/quality-v5c/baseline-v5b-preserved.json',
                     'reports/quality-v5c/tune-grouped-results.json',
                     'evaluation/v5c/tune-inputs.jsonl', 'evaluation/v5c/reply-tune-inputs.jsonl',
                     'evaluation/v5c/PROTOCOL.md', 'evaluation/v5c/SCHEMA.json', 'evaluation/v5c/validate.py']:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    name = 'scripts/reranker_worker_mlx_v5c.py'
    if mlx_present:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('MLX worker must never be imported by freeze')")
    files = service.freeze_inventory({'pins': {}}, {})
    assert (name in files) == mlx_present
    if mlx_present:
        assert files[name] == service.sha256_file(workspace / name)


@pytest.mark.parametrize('engine', ['torch', 'mlx'])
def test_freeze_pins_actual_reranker_engine_source_hash_independent_of_environment(
        service, workspace, monkeypatch, engine):
    dispatcher = 'scripts/reranker_worker_v5c.py'
    implementation = 'scripts/reranker_worker_mlx_v5c.py' if engine == 'mlx' else dispatcher
    files = {dispatcher: 'a' * 64, implementation: 'b' * 64}
    descriptor = dict(engine=engine, device='mlx-metal' if engine == 'mlx' else 'mps',
                      worker_sha256=files[dispatcher], engine_sha256=files[implementation])
    details = dict(pins={'reranker': {'model': 'same pinned weights', 'weights_sha256': 'c' * 64}})
    before = copy.deepcopy(details)
    # Freeze pins the measured descriptor, not whichever engine the caller requests now.
    monkeypatch.setenv('RADAR_RERANK_ENGINE', 'torch' if engine == 'mlx' else 'mlx')
    frozen = service.frozen_model_pins(details, {'reranker': descriptor}, files)
    assert frozen['reranker']['engine'] == engine
    assert frozen['reranker']['engine_path'] == implementation
    assert frozen['reranker']['engine_sha256'] == files[implementation]
    assert details == before


def test_legacy_torch_descriptor_keeps_existing_constructor_contract(service):
    path = 'scripts/reranker_worker_v5c.py'
    pins = service.frozen_model_pins(
        {'pins': {'reranker': {}}}, {'reranker': {'worker_sha256': 'd' * 64, 'device': 'mps'}},
        {path: 'd' * 64})
    assert pins['reranker']['engine'] == 'torch'
    assert pins['reranker']['engine_sha256'] == 'd' * 64


@pytest.mark.parametrize('problem', ['missing-worker', 'missing-engine-sha', 'wrong-engine-sha', 'wrong-dispatcher-sha'])
def test_mlx_freeze_rejects_missing_or_stale_engine_pins(service, problem):
    dispatcher, worker = 'scripts/reranker_worker_v5c.py', 'scripts/reranker_worker_mlx_v5c.py'
    files = {dispatcher: 'a' * 64, worker: 'b' * 64}
    descriptor = dict(engine='mlx', device='mlx-metal', worker_sha256='a' * 64, engine_sha256='b' * 64)
    if problem == 'missing-worker':
        files.pop(worker)
    elif problem == 'missing-engine-sha':
        descriptor.pop('engine_sha256')
    elif problem == 'wrong-engine-sha':
        descriptor['engine_sha256'] = 'c' * 64
    else:
        descriptor['worker_sha256'] = 'c' * 64
    with pytest.raises(RuntimeError, match='[Ee]ngine|dispatcher'):
        service.frozen_model_pins({'pins': {'reranker': {}}}, {'reranker': descriptor}, files)


def test_verify_freeze_checks_mlx_engine_hash_and_pins_before_control_inputs(service, workspace):
    directory = workspace / 'scripts'
    directory.mkdir()
    dispatcher, worker = 'scripts/reranker_worker_v5c.py', 'scripts/reranker_worker_mlx_v5c.py'
    (workspace / dispatcher).write_text('dispatcher')
    (workspace / worker).write_text('MLX engine')
    files = {name: service.sha256_file(workspace / name) for name in (dispatcher, worker)}
    descriptor = dict(engine='mlx', device='mlx-metal', worker_sha256=files[dispatcher], engine_sha256=files[worker])
    lock = workspace / 'evaluation/v5c/CONTROL-LOCK.json'
    lock.parent.mkdir(parents=True)
    lock.write_text('{"files": {"sealed/control-gold.jsonl": "never opened"}}')
    frozen = dict(files=files, dataset_lock_sha256=service.sha256_file(lock),
                  preparation_contract_sha256='contract', models={'reranker': descriptor},
                  pins=service.frozen_model_pins({'pins': {'reranker': {}}}, {'reranker': descriptor}, files))
    path = service.OUT / 'CONFIG-FREEZE.json'
    path.write_text(json.dumps(frozen))
    assert service.verify_freeze() == frozen
    (workspace / worker).write_text('modified engine')
    with pytest.raises(RuntimeError, match='Frozen'):
        service.verify_freeze()
    (workspace / worker).write_text('MLX engine')
    frozen['pins']['reranker']['engine_sha256'] = 'f' * 64
    path.write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError, match='[Ee]ngine'):
        service.verify_freeze()


@pytest.fixture
def mlx_identity(service, workspace, monkeypatch):
    monkeypatch.setattr(service, 'MODEL_DIRS', {})
    monkeypatch.setattr(service, 'CODE', ('code.py', service.RERANKER_WORKER))
    for relative in ['data/semantic-heads-v5c.json', 'data/semantic-config-v5c.json',
                     'reports/quality-v5c/baseline-v5b-preserved.json',
                     'reports/quality-v5c/tune-grouped-results.json',
                     'evaluation/v5c/tune-inputs.jsonl', 'evaluation/v5c/reply-tune-inputs.jsonl',
                     'evaluation/v5c/PROTOCOL.md', 'evaluation/v5c/SCHEMA.json', 'evaluation/v5c/validate.py',
                     service.RERANKER_WORKER, service.MLX_WORKER, 'scripts/mlx_shared.py']:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('identity checks must never import engines/helpers')")
    descriptor = dict(engine='mlx', device='mlx-metal',
                      worker_sha256=service.sha256_file(workspace / service.RERANKER_WORKER),
                      engine_sha256=service.sha256_file(workspace / service.MLX_WORKER),
                      dependency_sha256={'scripts/mlx_shared.py': service.sha256_file(workspace / 'scripts/mlx_shared.py')},
                      versions={'mlx': 'fixture-mlx', 'mlx-lm': 'fixture-lm', 'tokenizers': 'fixture-tokenizers'},
                      dtype='float16', quantization=None, padding='right-causal', prefix_cache=True,
                      resource_measurement={'peak_bytes': 123})
    return dict(pins={'reranker': {}}), {'reranker': descriptor}


@pytest.mark.parametrize('engine', [None, 'torch'])
def test_mlx_device_cannot_fall_back_to_torch_engine_pin(service, engine):
    descriptor = dict(device='mlx-metal', worker_sha256='a' * 64)
    if engine is not None:
        descriptor['engine'] = engine
    with pytest.raises(RuntimeError, match='[Ee]ngine'):
        service.frozen_model_pins({'pins': {'reranker': {}}}, {'reranker': descriptor},
                                 {service.RERANKER_WORKER: 'a' * 64})


def test_freeze_hashes_declared_engine_dependencies_without_importing_them(service, workspace, mlx_identity):
    details, metadata = mlx_identity
    files = service.freeze_inventory(details, {}, metadata=metadata)
    assert files['scripts/mlx_shared.py'] == metadata['reranker']['dependency_sha256']['scripts/mlx_shared.py']
    pins = service.frozen_model_pins(details, metadata, files)['reranker']
    assert pins['dependency_sha256'] == metadata['reranker']['dependency_sha256']
    (workspace / 'scripts/mlx_shared.py').write_text('changed helper, unchanged engine')
    with pytest.raises(RuntimeError, match='[Dd]ependency'):
        service.freeze_inventory(details, {}, metadata=metadata)


@pytest.mark.parametrize('problem', ['missing', 'changed', 'escape', 'invalid-map'])
def test_freeze_rejects_unbound_engine_dependency_identity(service, mlx_identity, problem):
    details, metadata = mlx_identity
    descriptor = metadata['reranker']
    files = {service.RERANKER_WORKER: descriptor['worker_sha256'], service.MLX_WORKER: descriptor['engine_sha256'],
             **descriptor['dependency_sha256']}
    if problem == 'missing':
        files.pop('scripts/mlx_shared.py')
    elif problem == 'changed':
        files['scripts/mlx_shared.py'] = 'f' * 64
    elif problem == 'escape':
        descriptor['dependency_sha256'] = {'../external.py': 'a' * 64}
    else:
        descriptor['dependency_sha256'] = ['scripts/mlx_shared.py']
    with pytest.raises(RuntimeError):
        service.frozen_model_pins(details, metadata, files)


def test_freeze_binds_actual_runtime_hints_and_ignores_resource_measurements(service, mlx_identity):
    details, metadata = mlx_identity
    files = service.freeze_inventory(details, {}, metadata=metadata)
    original = service.frozen_model_pins(details, metadata, files)['reranker']['descriptor_sha256']
    for field, value in [('versions', {'mlx': 'another-version'}), ('padding', 'left'),
                         ('quantization', {'bits': 8}), ('prefix_cache', False)]:
        changed = copy.deepcopy(metadata)
        changed['reranker'][field] = value
        assert service.frozen_model_pins(details, changed, files)['reranker']['descriptor_sha256'] != original
    changed = copy.deepcopy(metadata)
    changed['reranker']['resource_measurement']['peak_bytes'] += 100
    assert service.frozen_model_pins(details, changed, files)['reranker']['descriptor_sha256'] == original


def test_version_hint_tampering_fails_freeze_before_control_inputs(service, workspace, mlx_identity):
    details, metadata = mlx_identity
    files = service.freeze_inventory(details, {}, metadata=metadata)
    lock = workspace / 'evaluation/v5c/CONTROL-LOCK.json'
    lock.write_text('{"files": {"sealed/control-gold.jsonl": "not opened"}}')
    models = service.plain_metadata(metadata)
    frozen = dict(files=files, models=models, pins=service.frozen_model_pins(details, metadata, files),
                  dataset_lock_sha256=service.sha256_file(lock), preparation_contract_sha256='contract')
    path = service.OUT / 'CONFIG-FREEZE.json'
    path.write_text(json.dumps(frozen))
    assert service.verify_freeze()['models'] == models
    frozen['models']['reranker']['versions']['mlx'] = 'tampered-version'
    path.write_text(json.dumps(frozen))
    with pytest.raises(RuntimeError, match='[Ee]ngine'):
        service.verify_freeze()
