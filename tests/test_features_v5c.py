"""Preparation contract tests: synthetic inputs, no model processes or downloads."""
import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def prep():
    return importlib.import_module('scripts.prepare_features_v5c')


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def unit_vector(index=0):
    return [float(i == index) for i in range(1024)]


@pytest.fixture
def sandbox(prep, tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'ROOT', tmp_path)
    monkeypatch.setattr(prep, 'OUT', tmp_path / 'reports/quality-v5c')
    needs = [dict(id=f'n{i}', source_id=f's{i}', product='p', text=f'text {i}')
             for i in range(11)]
    needs.append(dict(id='other', product='other', text='other product'))
    replies = [dict(id='r', kind='bug', query={'text': 'query'},
                    references=[{'text': 'ref', 'kind': 'bug'}, {'text': 'query', 'kind': 'bug'}])]
    data_dir = tmp_path / 'evaluation/v5c'
    data_dir.mkdir(parents=True)
    for name, rows in [('tune-inputs.jsonl', needs), ('reply-tune-inputs.jsonl', replies)]:
        (data_dir / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    # Opening a gold source would fail JSON parsing, even if the filename were overlooked.
    (data_dir / 'tune-gold.jsonl').write_text('DO NOT READ GOLD')
    (data_dir / 'reply-tune-gold.jsonl').write_text('DO NOT READ GOLD')
    pins = {name: dict(model=name, revision='fixed', weights_sha256='a' * 64,
                       worker_sha256='b' * 64, files={'config.json': {'sha256': 'c' * 64}})
            for name in ['qwen', 'nli', 'reranker']}
    monkeypatch.setattr(prep, 'contract_details', lambda: dict(pins=pins, schema=1))
    monkeypatch.setattr(prep, 'masked_inputs', lambda n, r: (n, r))
    calls, active = [], set()
    failures = {}

    class Client:
        def __init__(self, backend, **kwargs):
            self.backend = backend
            assert Path(kwargs['cache_dir']).is_relative_to(prep.OUT)

        def __enter__(self):
            assert not active, 'Two models would be resident at once'
            active.add(self.backend)
            calls.append(('open', self.backend))
            return self

        def __exit__(self, *exc):
            active.remove(self.backend)
            calls.append(('close', self.backend))

        def describe(self):
            p = pins[self.backend]
            return dict(backend=self.backend, model=p['model'], revision=p['revision'],
                        weights_sha256=p['weights_sha256'], worker_sha256=p['worker_sha256'],
                        artifact_files=p['files'], dimension=1024 if self.backend == 'qwen' else 1,
                        cpu_threads=2, device='cpu')

        def _request(self, op):
            assert op == 'stats'
            return dict(loaded=True, cpu_threads=2)

        def invoke(self, op, rows, output):
            calls.append((op, self.backend, len(rows)))
            if failures.get(op) == sum(c[0] == op for c in calls):
                raise RuntimeError('interrupted batch')
            return output

        def embed(self, texts):
            return self.invoke('embed', texts, [unit_vector() for _ in texts])

        def nli(self, pairs):
            return self.invoke('nli', pairs, [dict(entailment=.7, neutral=.2, contradiction=.1)
                                               for _ in pairs])

        def rerank(self, pairs):
            return self.invoke('rerank', pairs, [.73 for _ in pairs])

    monkeypatch.setattr(prep, 'BackendClient', Client)
    monkeypatch.setattr(prep, 'RerankerClient', lambda **kw: Client('reranker', **kw))
    return dict(needs=needs, reply=replies, calls=calls, failures=failures, active=active,
                data_dir=data_dir, path=prep.OUT / 'tune-features.json')


def test_new_module_is_available(prep):
    assert callable(prep.prepare)


def test_all_stages_keep_legacy_store_contract_and_directional_reranker(prep, sandbox):
    data = prep.prepare('tune', 'all')
    assert set(data['vectors']) == {'qwen'}
    assert all(len(v) == 1024 for v in data['vectors']['qwen'].values())
    assert set(data['source_sha256']) == {
        'evaluation/v5c/tune-inputs.jsonl', 'evaluation/v5c/reply-tune-inputs.jsonl'}
    assert data['pins']
    assert set(data['relations']) == set(data['inventory']['relations'])
    assert set(data['reranker']) == set(data['inventory']['reranker'])
    assert data['retrieval']['missing_pair'] == 'forbidden'
    assert data['reranker'][prep.pair_key('query', 'ref')] == .73
    assert data['reranker'][prep.pair_key('ref', 'query')] == .73
    assert prep.pair_key('query', 'query') in data['reranker']
    assert prep.pair_key('text 0', 'other product') not in data['relations']
    for name, limit in [('embed', 8), ('nli', 16), ('rerank', 4)]:
        batches = [c[2] for c in sandbox['calls'] if c[0] == name]
        assert batches and max(batches) <= limit
    assert [c for c in sandbox['calls'] if c[0] in {'open', 'close'}] == [
        ('open', 'qwen'), ('close', 'qwen'), ('open', 'nli'), ('close', 'nli'),
        ('open', 'reranker'), ('close', 'reranker')]
    store = prep.FeatureStore(sandbox['path'])
    assert store.cosine('qwen', 'query', 'ref') == pytest.approx(1.)
    assert store.kind_scores('query') == dict.fromkeys(prep.KIND_HYPOTHESES, .7)
    assert store.risk_scores('query') == dict.fromkeys(prep.RISK_HYPOTHESES, .7)
    assert store.relation('query', 'ref')['entailment'] == .7
    with pytest.raises(KeyError):
        store.reranker_score('query', 'text 0')
    assert not sandbox['active']


def test_retrieval_is_top_eight_union_with_deterministic_ties_and_reply_pairs(prep, sandbox):
    data = prep.prepare('tune', 'embeddings')
    candidates = {tuple(p) for p in data['candidate_pairs']}
    texts = sorted(r['text'] for r in sandbox['needs'] if r['product'] == 'p')
    expected = {tuple(sorted((a, b))) for a in texts for b in [t for t in texts if t != a][:8]}
    expected |= {('query', 'ref'), ('query', 'query')}
    assert candidates == expected
    assert len(data['relations']) == 0
    assert len(data['inventory']['relations']) == 11 * 10 + 2
    assert ('text 8', 'text 9') not in candidates
    assert prep.pair_key('text 8', 'text 9') in data['inventory']['relations']


def test_embedding_checkpoints_resume_only_missing_items(prep, sandbox):
    sandbox['failures']['embed'] = 2
    with pytest.raises(RuntimeError, match='interrupted'):
        prep.prepare('tune', 'embeddings')
    partial = json.loads(sandbox['path'].read_text())
    assert len(partial['vectors']['qwen']) == 8
    assert partial['timings']['qwen'][0]['n'] == 8
    assert not sandbox['active']
    sandbox['failures'].clear()
    start = len(sandbox['calls'])
    completed = prep.prepare('tune', 'embeddings')
    assert sum(c[2] for c in sandbox['calls'][start:] if c[0] == 'embed') == 6
    assert len(completed['vectors']['qwen']) == 14
    start = len(sandbox['calls'])
    prep.prepare('tune', 'embeddings')
    assert not sandbox['calls'][start:]


@pytest.mark.parametrize('stage,operation', [('nli', 'nli'), ('reranker', 'rerank')])
def test_pair_stage_checkpoints_resume_without_repeating_completed_batches(prep, sandbox, stage, operation):
    prep.prepare('tune', 'embeddings')
    sandbox['failures'][operation] = 2
    with pytest.raises(RuntimeError, match='interrupted'):
        prep.prepare('tune', stage)
    partial = json.loads(sandbox['path'].read_text())
    if stage == 'nli':
        assert sum(map(len, partial['kinds'].values())) == 16
    else:
        assert len(partial['reranker']) == 4
    assert not sandbox['active']
    sandbox['failures'].clear()
    data = prep.prepare('tune', stage)
    assert stage in data['completed_stages']
    assert sum(t['n'] for t in data['timings'][stage]) == (
        len(data['inventory']['kinds']) * len(prep.KIND_HYPOTHESES)
        + len(data['inventory']['risks']) * len(prep.RISK_HYPOTHESES)
        + len(data['inventory']['relations']) if stage == 'nli'
        else len(data['inventory']['reranker']))


@pytest.mark.parametrize('change', ['source', 'contract', 'vector', 'extra-vector', 'metadata', 'inventory', 'complete'])
def test_invalid_partial_cache_is_rejected_before_any_worker(prep, sandbox, monkeypatch, change):
    sandbox['failures']['embed'] = 2
    with pytest.raises(RuntimeError):
        prep.prepare('tune', 'embeddings')
    data = json.loads(sandbox['path'].read_text())
    if change == 'source':
        path = sandbox['data_dir'] / 'tune-inputs.jsonl'
        path.write_text(path.read_text() + '\n')
    elif change == 'contract':
        data['preparation_contract_sha256'] = 'different'
    elif change == 'vector':
        data['vectors']['qwen'][next(iter(data['vectors']['qwen']))] = [1.]
    elif change == 'extra-vector':
        data['vectors']['qwen']['not-an-input'] = unit_vector()
    elif change == 'metadata':
        data['metadata']['qwen']['revision'] = 'different'
    elif change == 'inventory':
        data['inventory']['relations'].clear()
    else:
        data['completed_stages'] = ['embeddings']
    write_json(sandbox['path'], data)
    before = len(sandbox['calls'])
    with pytest.raises(RuntimeError):
        prep.prepare('tune', 'embeddings')
    assert sandbox['calls'][before:] == []


@pytest.mark.parametrize('location', ['needs', 'query', 'ref'])
def test_gold_fields_are_rejected_before_masking_or_models(prep, sandbox, location):
    name = 'tune-inputs.jsonl' if location == 'needs' else 'reply-tune-inputs.jsonl'
    path = sandbox['data_dir'] / name
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = rows[0] if location == 'needs' else rows[0]['query'] if location == 'query' else rows[0]['references'][0]
    target['gold'] = {'kind': 'bug'}
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='[Gg]old|answers'):
        prep.prepare('tune', 'all')
    assert not sandbox['calls']


@pytest.mark.parametrize('stage', ['nli', 'reranker'])
def test_pair_stages_require_complete_embeddings(prep, sandbox, stage):
    with pytest.raises(RuntimeError, match='[Ee]mbeddings'):
        prep.prepare('tune', stage)
    assert not sandbox['calls']


def test_control_without_freeze_cannot_read_sources_or_start_models(prep, sandbox, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Control inputs were accessed before freeze verification')
    monkeypatch.setattr(prep, 'read', forbidden)
    with pytest.raises(RuntimeError, match='[Ff]reez|frozen'):
        prep.prepare('control', 'all')
    assert not sandbox['calls']


def test_freeze_checks_every_file_and_dataset_lock_before_control_inputs(prep, sandbox, monkeypatch):
    code = prep.ROOT / 'code.py'
    code.write_text('fixed code')
    lock = sandbox['data_dir'] / 'CONTROL-LOCK.json'
    write_json(lock, {'files': {}})
    frozen = dict(files={'code.py': prep.sha256_file(code)}, dataset_lock_sha256=prep.sha256_file(lock))
    write_json(prep.OUT / 'CONFIG-FREEZE.json', frozen)
    assert prep.verify_freeze()['files'] == frozen['files']
    for path in [code, lock]:
        before = path.read_bytes()
        path.write_text('changed')
        with pytest.raises(RuntimeError):
            prep.verify_freeze()
        path.write_bytes(before)
    frozen['files'] = {'../outside': '0' * 64}
    write_json(prep.OUT / 'CONFIG-FREEZE.json', frozen)
    with pytest.raises(RuntimeError):
        prep.verify_freeze()


def test_complete_inventory_validation_rejects_missing_or_nonfinite_scores(prep, sandbox):
    data = prep.prepare('tune', 'all')
    for scope, mutation in [('reranker', 'missing'), ('relations', 'extra'), ('risks', 'nan')]:
        broken = copy.deepcopy(data)
        if mutation == 'missing':
            broken[scope].pop(next(iter(broken[scope])))
        elif mutation == 'extra':
            broken[scope]['unknown-pair'] = dict(entailment=.7, neutral=.2, contradiction=.1)
        else:
            scores = next(iter(broken[scope].values()))
            scores[next(iter(scores))] = float('nan')
        with pytest.raises(RuntimeError):
            prep.validate_feature_provenance(
                broken, sources=data['source_sha256'], needs=data['needs_inputs'], reply=data['reply_inputs'],
                contract=data['preparation_contract_sha256'], metadata=data['metadata'])


def test_frozen_tune_inputs_have_full_delivery_inventory_without_gold(prep):
    """Read only public input contracts; do not open either tune answer file."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'evaluation/v5c/tune-inputs.jsonl',
             root / 'evaluation/v5c/reply-tune-inputs.jsonl']
    needs, replies = [prep.read(path) for path in paths]
    prep.reject_gold([needs, replies])
    assert len(needs) == 188
    assert len({row['id'] for row in needs}) == 184
    assert len({row['source_id'] for row in needs}) == 184
    assert len(replies) == 132
    inv = prep.inventory(needs, replies)
    assert inv['needs_rows'] == 188 and inv['reply_rows'] == 132
    assert len(inv['texts']) == len({r['text'] for r in needs}
                                    | {r['query']['text'] for r in replies}
                                    | {ref['text'] for r in replies for ref in r['references']})
    assert all('kind' not in row for row in needs)
    assert all('kind' not in row['query'] for row in replies)


def test_valid_control_uses_only_locked_inputs_after_freeze(prep, sandbox, monkeypatch):
    directory = sandbox['data_dir']
    for name in ('tune-inputs.jsonl', 'reply-tune-inputs.jsonl'):
        (directory / name.replace('tune', 'control')).write_bytes((directory / name).read_bytes())
    lock = directory / 'CONTROL-LOCK.json'
    locked = {name: prep.sha256_file(directory / name)
              for name in ('control-inputs.jsonl', 'reply-control-inputs.jsonl')}
    locked['sealed/control-gold.jsonl'] = 'never open this file'
    write_json(lock, dict(files=locked))
    code = prep.ROOT / 'frozen-code.py'
    code.write_text('synthetic frozen code')
    write_json(prep.OUT / 'CONFIG-FREEZE.json', dict(
        files={'frozen-code.py': {'sha256': prep.sha256_file(code)}},
        dataset_lock_sha256=prep.sha256_file(lock),
        preparation_contract_sha256=prep.feature_contract()))
    opened, original = [], prep.read

    def tracked_read(path):
        opened.append(path.name)
        return original(path)

    monkeypatch.setattr(prep, 'read', tracked_read)
    data = prep.prepare('control', 'all')
    assert data['split'] == 'control'
    assert opened == ['control-inputs.jsonl', 'reply-control-inputs.jsonl']
    assert (prep.OUT / 'control-features.json').is_file()
    # Tampering with an input is refused before parsing or opening any worker.
    opened.clear()
    (directory / 'control-inputs.jsonl').write_text('tampered input')
    before = len(sandbox['calls'])
    with pytest.raises(RuntimeError, match='Locked control input'):
        prep.prepare('control', 'all')
    assert not opened and sandbox['calls'][before:] == []


@pytest.mark.parametrize('field', ['preparation_contract_sha256', 'tune_features_sha256'])
def test_freeze_rejects_contract_and_tune_feature_changes(prep, sandbox, field):
    code = prep.ROOT / 'fixed-code.py'
    code.write_text('fixed')
    if field == 'tune_features_sha256':
        write_json(sandbox['path'], {'synthetic': 'features'})
    frozen = dict(files={'fixed-code.py': prep.sha256_file(code)})
    frozen[field] = 'not-the-current-sha'
    write_json(prep.OUT / 'CONFIG-FREEZE.json', frozen)
    with pytest.raises(RuntimeError):
        prep.verify_freeze()


@pytest.mark.parametrize('score', [True, float('nan'), -0.1, 1.1])
def test_partial_reranker_cache_does_not_accept_invalid_scores(prep, sandbox, score):
    prep.prepare('tune', 'embeddings')
    sandbox['failures']['rerank'] = 2
    with pytest.raises(RuntimeError):
        prep.prepare('tune', 'reranker')
    data = json.loads(sandbox['path'].read_text())
    data['reranker'][next(iter(data['reranker']))] = score
    write_json(sandbox['path'], data)
    before = len(sandbox['calls'])
    with pytest.raises(RuntimeError, match='reranker score'):
        prep.prepare('tune', 'reranker')
    assert sandbox['calls'][before:] == []


def test_changed_model_pins_invalidate_partial_cache(prep, sandbox, monkeypatch):
    sandbox['failures']['embed'] = 2
    with pytest.raises(RuntimeError):
        prep.prepare('tune', 'embeddings')
    details = copy.deepcopy(prep.contract_details())
    details['pins']['qwen']['revision'] = 'updated model'
    monkeypatch.setattr(prep, 'contract_details', lambda: details)
    before = len(sandbox['calls'])
    with pytest.raises(RuntimeError, match='contract or pins'):
        prep.prepare('tune', 'embeddings')
    assert sandbox['calls'][before:] == []


def test_future_reranker_changes_preserve_completed_embeddings_and_nli(prep, sandbox, monkeypatch):
    prep.prepare('tune', 'embeddings')
    old = prep.prepare('tune', 'nli')
    details = copy.deepcopy(prep.contract_details())
    details['pins']['reranker']['worker_sha256'] = 'd' * 64
    details['code_sha256'] = {'scripts/reranker_worker_v5c.py': 'new reranker code'}
    monkeypatch.setattr(prep, 'contract_details', lambda: details)
    before = len(sandbox['calls'])
    data = prep.prepare('tune', 'embeddings')
    data = prep.prepare('tune', 'nli')
    assert sandbox['calls'][before:] == []
    assert data['vectors'] == old['vectors'] and data['relations'] == old['relations']
    assert data['metadata']['qwen'] == old['metadata']['qwen']
    assert data['metadata']['nli'] == old['metadata']['nli']
    assert data['pins']['reranker']['worker_sha256'] == 'd' * 64


def test_used_reranker_contract_cannot_change_silently(prep, sandbox, monkeypatch):
    prep.prepare('tune', 'all')
    details = copy.deepcopy(prep.contract_details())
    details['pins']['reranker']['worker_sha256'] = 'd' * 64
    monkeypatch.setattr(prep, 'contract_details', lambda: details)
    with pytest.raises(RuntimeError, match='contract or pins'):
        prep.prepare('tune', 'all')
