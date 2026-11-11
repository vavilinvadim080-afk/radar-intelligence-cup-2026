"""Isolated, resumable v5c features. Gold is never an inference source.

Run embeddings first, then NLI and reranker in separate, closed worker processes.
All generated files, including worker caches, stay under reports/quality-v5c.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'mvp-v13-adapted')]

from scripts.prepare_features_v5b import (
    BackendClient, FeatureStore as BaseFeatureStore, KIND_HYPOTHESES,
    RISK_HYPOTHESES, key, masked_inputs, pair_key, read,
)
from radar_app.model_artifacts_v5b import pinned_identity, validate_artifacts
from radar_app.reranker_v5c import RerankerClient

OUT = ROOT / 'reports/quality-v5c'
BATCH_SIZE = {'qwen': 8, 'nli': 16, 'reranker': 4}
STAGES = ('embeddings', 'nli', 'reranker')
LABELS = ('entailment', 'neutral', 'contradiction')
RETRIEVAL = dict(backend='qwen', metric='cosine', top_k=8,
                 needs_scope='within_product', union='unordered_canonical_pairs',
                 reply_scope='all_query_reference_pairs', missing_pair='forbidden')
CODE = (
    'scripts/prepare_features_v5c.py', 'scripts/prepare_features_v5b.py',
    'scripts/semantic_worker_v5b.py', 'scripts/reranker_worker_v5c.py',
    'radar_app/semantic_client.py', 'radar_app/semantic_v5b.py',
    'radar_app/model_artifacts_v5b.py', 'radar_app/reranker_v5c.py',
    'radar_app/selection_v5b.py', 'radar_app/reply_policy.py',
    'radar_app/privacy/mask.py', 'radar_app/privacy/mask_ner.py',
    'radar_app/privacy/config.py',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def contract_details():
    """Inspect pins and small artifacts only; no inference imports or weight loading."""
    pins = {}
    for name in ('qwen', 'nli'):
        identity = pinned_identity(name, root=ROOT)
        pins[name] = {field: identity[field] for field in (
            'model', 'revision', 'weights_sha256', 'manifest_sha256',
            'artifacts_fingerprint', 'files')}
        pins[name]['worker_sha256'] = sha256_file(ROOT / 'scripts/semantic_worker_v5b.py')
    directory = ROOT / 'runtime/models/qwen-reranker-v5c'
    manifest = json.loads((directory / 'MANIFEST.json').read_text(encoding='utf-8'))
    validate_artifacts(directory, manifest)
    pins['reranker'] = dict(model=manifest['model'], revision=manifest['revision'],
                            weights_sha256=manifest['files']['model.safetensors']['sha256'],
                            files=manifest['files'], manifest_sha256=fingerprint(manifest),
                            worker_sha256=sha256_file(ROOT / 'scripts/reranker_worker_v5c.py'))
    return dict(schema='features-v5c-1', pins=pins,
                code_sha256={name: sha256_file(ROOT / name) for name in CODE},
                hypotheses=dict(kinds=KIND_HYPOTHESES, risks=RISK_HYPOTHESES),
                retrieval=RETRIEVAL, batches=BATCH_SIZE, qwen_dimension=1024,
                devices=dict(qwen='cpu', nli='cpu',
                             reranker=os.environ.get('RADAR_RERANK_DEVICE', 'mps')),
                cpu_threads=2)


def feature_contract():
    return fingerprint(contract_details())


def stage_contracts(details):
    """Future verifier edits do not invalidate either existing semantic worker."""
    code = details.get('code_sha256', {})
    verifier_files = {'scripts/reranker_worker_v5c.py', 'radar_app/reranker_v5c.py'}
    semantic_code = {name: digest for name, digest in code.items() if name not in verifier_files}
    result = {}
    for name in BATCH_SIZE:
        payload = dict(schema=details['schema'], pin=details['pins'][name],
                       code=code if name == 'reranker' else semantic_code,
                       batch=details.get('batches', BATCH_SIZE)[name],
                       device=details.get('devices', {}).get(name),
                       cpu_threads=details.get('cpu_threads', 2))
        if name == 'nli':
            payload['hypotheses'] = details.get('hypotheses')
        if name != 'nli':
            payload['retrieval'] = details.get('retrieval', RETRIEVAL)
        result[name] = fingerprint(payload)
    return result


def reconcile_contract(data, details):
    previous = data.get('contract_details')
    if (not isinstance(previous, dict)
            or fingerprint(previous) != data.get('preparation_contract_sha256')
            or stage_contracts(previous) != data.get('stage_contract_sha256')
            or previous.get('pins') != data.get('pins')):
        raise RuntimeError('Feature preparation contract or pins differ')
    current = stage_contracts(details)
    used = {'qwen'} | set(data['metadata'])
    if any(data['stage_contract_sha256'][name] != current[name] for name in used):
        raise RuntimeError('Feature preparation contract or pins differ for a started stage')
    data.update(contract_details=details, stage_contract_sha256=current,
                preparation_contract_sha256=fingerprint(details), pins=details['pins'])


def plain_metadata(metadata):
    return {name: {field: value for field, value in descriptor.items()
                   if field != 'resource_measurement'} for name, descriptor in metadata.items()}


def reject_gold(value):
    if isinstance(value, dict):
        forbidden = {'gold', 'expected', 'topic_id', 'topic_key', 'same_topic',
                     'eligibility', 'safe_action', 'query_kind', 'query_direction'}
        if any(name in forbidden or name.startswith(('gold_', 'expected_')) for name in value):
            raise ValueError('Gold answers cannot enter feature inputs')
        for child in value.values():
            reject_gold(child)
    elif isinstance(value, list):
        for child in value:
            reject_gold(child)


def root_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Frozen path escapes project root')
    relative = path.relative_to(ROOT.resolve())
    if 'sealed' in relative.parts or 'gold' in path.name.lower():
        raise RuntimeError('Gold/sealed files cannot be feature sources or freeze checks')
    return path


def verify_freeze(path=None):
    """Check every frozen code/config file before touching any control inputs."""
    path = Path(path) if path is not None else OUT / 'CONFIG-FREEZE.json'
    if not path.is_file():
        raise RuntimeError('Control features require a verified freeze')
    frozen = json.loads(path.read_text(encoding='utf-8'))
    files = frozen.get('files')
    if not isinstance(files, dict) or not files:
        raise RuntimeError('Freeze must contain a nonempty files inventory')
    for name, expected in files.items():
        digest = expected.get('sha256') if isinstance(expected, dict) else expected
        if sha256_file(root_path(name)) != digest:
            raise RuntimeError('Frozen file changed: ' + name)
    if 'dataset_lock_sha256' in frozen:
        lock = root_path(frozen.get('dataset_lock_path', 'evaluation/v5c/CONTROL-LOCK.json'))
        if sha256_file(lock) != frozen['dataset_lock_sha256']:
            raise RuntimeError('Frozen dataset lock changed')
    if 'tune_features_sha256' in frozen:
        if sha256_file(OUT / 'tune-features.json') != frozen['tune_features_sha256']:
            raise RuntimeError('Frozen tune features changed')
    if ('preparation_contract_sha256' in frozen
            and feature_contract() != frozen['preparation_contract_sha256']):
        raise RuntimeError('Frozen preparation contract changed')
    return frozen


def source_paths(split):
    directory = ROOT / 'evaluation/v5c'
    return [directory / f'{split}-inputs.jsonl', directory / f'reply-{split}-inputs.jsonl']


def source_hashes(paths):
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def check_locked_sources(paths, frozen):
    if 'dataset_lock_sha256' not in frozen:
        return
    path = root_path(frozen.get('dataset_lock_path', 'evaluation/v5c/CONTROL-LOCK.json'))
    locked = json.loads(path.read_text(encoding='utf-8'))['files']
    for source in paths:
        name = str(source.relative_to(path.parent))
        entry = locked.get(name, locked.get(str(source.relative_to(ROOT))))
        expected = entry.get('sha256') if isinstance(entry, dict) else entry
        if not expected or sha256_file(source) != expected:
            raise RuntimeError('Locked control input changed: ' + name)


def relation_pairs(needs, reply):
    groups = {}
    for row in needs:
        groups.setdefault(row['product'], set()).add(row['text'])
    pairs = set()
    for texts in groups.values():
        for a, b in itertools.combinations(sorted(texts), 2):
            pairs.update(((a, b), (b, a)))
    for row in reply:
        a = row['query']['text']
        for reference in row['references']:
            b = reference['text']
            if a != b:
                pairs.update(((a, b), (b, a)))
    return sorted(pairs)


def candidate_pairs(needs, reply, vectors):
    """Union of each need's top eight neighbors; replies are never pruned."""
    groups, candidates = {}, set()
    for row in needs:
        groups.setdefault(row['product'], set()).add(row['text'])
    for texts in groups.values():
        neighbors = {text: [] for text in texts}
        for a, b in itertools.combinations(sorted(texts), 2):
            cosine = sum(x * y for x, y in zip(vectors[key(a)], vectors[key(b)]))
            neighbors[a].append((cosine, b))
            neighbors[b].append((cosine, a))
        for a, ranked in neighbors.items():
            for _, b in sorted(ranked, key=lambda item: (-item[0], item[1]))[:8]:
                candidates.add(tuple(sorted((a, b))))
    for row in reply:
        for reference in row['references']:
            candidates.add(tuple(sorted((row['query']['text'], reference['text']))))
    return [list(pair) for pair in sorted(candidates)]


def inventory(needs, reply, candidates=None):
    queries = {r['text'] for r in needs} | {r['query']['text'] for r in reply}
    texts = queries | {ref['text'] for r in reply for ref in r['references']}
    reranker = None
    if candidates is not None:
        directed = {(a, b) for a, b in candidates} | {(b, a) for a, b in candidates}
        reranker = {pair_key(a, b): [key(a), key(b)] for a, b in sorted(directed)}
    return dict(texts={key(t): t for t in sorted(texts)},
                kinds=sorted(key(t) for t in queries),
                risks=sorted({key(r['query']['text']) for r in reply}),
                relations={pair_key(a, b): [key(a), key(b)] for a, b in relation_pairs(needs, reply)},
                reranker=reranker, needs_rows=len(needs), reply_rows=len(reply))


def probability(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1)


def check_vector(vector):
    if (not isinstance(vector, list) or len(vector) != 1024
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in vector)
            or abs(sum(v * v for v in vector) - 1) > 1e-3):
        raise RuntimeError('Invalid normalized 1024-dimensional Qwen vector')


def check_relation(scores):
    if (not isinstance(scores, dict) or set(scores) != set(LABELS)
            or not all(probability(v) for v in scores.values())
            or abs(sum(scores.values()) - 1) > 1e-5):
        raise RuntimeError('Invalid NLI relation probabilities')


def check_descriptor(name, descriptor, pins):
    pin = pins[name]
    for field in ('revision', 'weights_sha256', 'worker_sha256'):
        if descriptor.get(field) != pin[field]:
            raise RuntimeError('Feature model metadata differs: ' + name + '/' + field)
    if descriptor.get('encoder', descriptor.get('model')) != pin['model']:
        raise RuntimeError('Feature model identity differs: ' + name)
    if descriptor.get('artifact_files', descriptor.get('artifacts')) != pin['files']:
        raise RuntimeError('Feature model artifact inventory differs: ' + name)
    for field in ('artifacts_fingerprint', 'manifest_sha256'):
        if field in descriptor and descriptor[field] != pin.get(field):
            raise RuntimeError('Feature model fingerprint differs: ' + name)
    if name == 'qwen' and descriptor.get('dimension') != 1024:
        raise RuntimeError('Qwen dimension must be 1024')


def check_cache(data, *, sources, needs, reply, contract, pins):
    """Partial inventories may be subsets; every saved value must already be valid."""
    if data.get('source_sha256') != sources:
        raise RuntimeError('Feature source hashes differ')
    if data.get('needs_inputs') != needs or data.get('reply_inputs') != reply:
        raise RuntimeError('Feature input rows differ')
    if data.get('preparation_contract_sha256') != contract or data.get('pins') != pins:
        raise RuntimeError('Feature preparation contract or pins differ')
    if data.get('retrieval') != RETRIEVAL:
        raise RuntimeError('Feature retrieval contract differs')
    vectors = data.get('vectors', {})
    if set(vectors) != {'qwen'}:
        raise RuntimeError('Only Qwen vectors are allowed')
    inv = inventory(needs, reply)
    if not set(vectors['qwen']) <= set(inv['texts']):
        raise RuntimeError('Feature vector inventory differs')
    for vector in vectors['qwen'].values():
        check_vector(vector)
    embeddings_done = set(vectors['qwen']) == set(inv['texts'])
    candidates = candidate_pairs(needs, reply, vectors['qwen']) if embeddings_done else None
    if data.get('candidate_pairs') != candidates:
        raise RuntimeError('Candidate pair inventory differs')
    inv = inventory(needs, reply, candidates)
    if data.get('inventory') != inv:
        raise RuntimeError('Feature inventory differs')
    finished = data.get('completed_stages')
    if (not isinstance(finished, list) or len(set(finished)) != len(finished)
            or not set(finished) <= set(STAGES)):
        raise RuntimeError('Invalid completed stages')
    complete = dict(embeddings=embeddings_done)
    for scope, labels in [('kinds', KIND_HYPOTHESES), ('risks', RISK_HYPOTHESES)]:
        values = data[scope]
        if not set(values) <= set(inv[scope]):
            raise RuntimeError('Unexpected ' + scope + ' inputs')
        for scores in values.values():
            if (not isinstance(scores, dict) or not set(scores) <= set(labels)
                    or not all(probability(v) for v in scores.values())):
                raise RuntimeError('Invalid ' + scope + ' scores')
    for scope in ('relations', 'reranker'):
        expected = inv[scope] or {}
        if not set(data[scope]) <= set(expected):
            raise RuntimeError('Unexpected ' + scope + ' pairs')
        for scores in data[scope].values():
            if scope == 'relations':
                check_relation(scores)
            elif not probability(scores):
                raise RuntimeError('Invalid reranker score')
    complete['nli'] = (all(set(data[s]) == set(inv[s]) and all(set(v) == set(labels)
                             for v in data[s].values()) for s, labels in
                           [('kinds', KIND_HYPOTHESES), ('risks', RISK_HYPOTHESES)])
                       and set(data['relations']) == set(inv['relations']))
    complete['reranker'] = candidates is not None and set(data['reranker']) == set(inv['reranker'])
    if any(not complete[stage] for stage in finished):
        raise RuntimeError('Completed stage has a partial feature inventory')
    used = dict(qwen=bool(vectors['qwen']), nli=any(data[s] for s in ('kinds', 'risks', 'relations')),
                reranker=bool(data['reranker']))
    if not set(data['metadata']) <= set(BATCH_SIZE):
        raise RuntimeError('Unexpected model metadata')
    for name, descriptor in data['metadata'].items():
        check_descriptor(name, descriptor, pins)
    if any(present and name not in data['metadata'] for name, present in used.items()):
        raise RuntimeError('Feature values have no model descriptor')
    for name, timings in data['timings'].items():
        if name not in BATCH_SIZE or not isinstance(timings, list):
            raise RuntimeError('Unexpected timing inventory')
        for batch in timings:
            n, seconds = batch.get('n'), batch.get('seconds')
            if (isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= BATCH_SIZE[name]
                    or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise RuntimeError('Invalid batch timing')
    return complete


def validate_feature_provenance(data, *, sources, needs, reply, contract, metadata):
    """Strict final validation; compatible keyword contract with v5b's validator."""
    complete = check_cache(data, sources=sources, needs=needs, reply=reply,
                           contract=contract, pins=contract_details()['pins'])
    if not all(complete.values()) or set(data['completed_stages']) != set(STAGES):
        raise RuntimeError('Incomplete feature inventory')
    if plain_metadata(data['metadata']) != plain_metadata(metadata):
        raise RuntimeError('Feature model metadata differs')


class FeatureStore(BaseFeatureStore):
    """Legacy accessors plus explicit sparse verifier lookup; absent never means zero."""
    def reranker_score(self, a, b):
        return self.data['reranker'][pair_key(a, b)]

    def is_candidate_pair(self, a, b):
        return pair_key(a, b) in (self.data['inventory']['reranker'] or {})


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=path.stem + '-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def nli_jobs(data):
    inv, texts = data['inventory'], data['inventory']['texts']
    jobs = []
    for scope, hypotheses in [('kinds', KIND_HYPOTHESES), ('risks', RISK_HYPOTHESES)]:
        for text_key in inv[scope]:
            for label, hypothesis in hypotheses.items():
                if label not in data[scope].get(text_key, {}):
                    jobs.append(((texts[text_key], hypothesis), (scope, text_key, label)))
    for ident, (a, b) in inv['relations'].items():
        if ident not in data['relations']:
            jobs.append(((texts[a], texts[b]), ('relations', ident, None)))
    return jobs


def run_stage(data, stage, path):
    name = 'qwen' if stage == 'embeddings' else stage
    inv, texts = data['inventory'], data['inventory']['texts']
    if stage == 'embeddings':
        jobs = [(text, ident) for ident, text in texts.items() if ident not in data['vectors']['qwen']]
    elif stage == 'nli':
        jobs = nli_jobs(data)
    else:
        jobs = [((texts[a], texts[b]), ident) for ident, (a, b) in inv['reranker'].items()
                if ident not in data['reranker']]
    cache_dir = OUT / 'worker-cache' / name
    client = (RerankerClient(cache_dir=cache_dir) if name == 'reranker'
              else BackendClient(name, cache_dir=cache_dir))
    with client:
        descriptor = client.describe()
        check_descriptor(name, descriptor, data['pins'])
        if name in data['metadata'] and plain_metadata({name: data['metadata'][name]}) != {name: descriptor}:
            raise RuntimeError('Resumed model descriptor changed: ' + name)
        data['metadata'][name] = descriptor
        data['timings'].setdefault(name, [])
        save_checkpoint(path, data)
        for start in range(0, len(jobs), BATCH_SIZE[name]):
            chunk = jobs[start:start + BATCH_SIZE[name]]
            started = time.perf_counter()
            inputs = [job[0] for job in chunk]
            values = (client.embed(inputs) if name == 'qwen' else
                      client.nli(inputs) if name == 'nli' else client.rerank(inputs))
            elapsed = time.perf_counter() - started
            if not isinstance(values, list) or len(values) != len(chunk):
                raise RuntimeError('Invalid feature batch response size')
            for (_, target), value in zip(chunk, values):
                if name == 'qwen':
                    check_vector(value)
                    data['vectors']['qwen'][target] = value
                elif name == 'nli':
                    check_relation(value)
                    scope, ident, label = target
                    if label is None:
                        data[scope][ident] = value
                    else:
                        data[scope].setdefault(ident, {})[label] = value['entailment']
                else:
                    if not probability(value):
                        raise RuntimeError('Invalid reranker score')
                    data['reranker'][target] = value
            data['timings'][name].append(dict(n=len(chunk), seconds=elapsed))
            if name == 'qwen' and len(data['vectors']['qwen']) == len(texts):
                data['candidate_pairs'] = candidate_pairs(data['needs_inputs'], data['reply_inputs'],
                                                          data['vectors']['qwen'])
                data['inventory'] = inventory(data['needs_inputs'], data['reply_inputs'], data['candidate_pairs'])
            save_checkpoint(path, data)
            print(data['split'], name, start + len(chunk), '/', len(jobs),
                  'checkpoint', f'{elapsed:.3f}s', flush=True)
        data['metadata'][name]['resource_measurement'] = client._request('stats')
    # Closing the worker precedes marking the stage complete or opening another one.
    data['completed_stages'].append(stage)
    save_checkpoint(path, data)


def prepare(split, stage='all', *, freeze_path=None):
    if split not in {'tune', 'control'} or stage not in {*STAGES, 'all'}:
        raise ValueError('Unknown split or stage')
    frozen = verify_freeze(freeze_path) if split == 'control' else None
    details = contract_details()
    contract = fingerprint(details)
    paths = source_paths(split)
    if frozen is not None:
        check_locked_sources(paths, frozen)
    sources = source_hashes(paths)
    raw_needs, raw_reply = [read(p) for p in paths]
    reject_gold([raw_needs, raw_reply])
    if any('kind' in row for row in raw_needs) or any('kind' in row['query'] for row in raw_reply):
        raise ValueError('Gold kind cannot enter needs/query inputs')
    needs, reply = masked_inputs(raw_needs, raw_reply)
    if sources != source_hashes(paths):
        raise RuntimeError('Feature source bytes changed while reading')
    path = OUT / f'{split}-features.json'
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
    else:
        data = dict(schema='features-v5c-1', split=split,
                    source_sha256=sources, preparation_contract_sha256=contract,
                    contract_details=details, stage_contract_sha256=stage_contracts(details),
                    pins=details['pins'], needs_inputs=needs, reply_inputs=reply,
                    vectors={'qwen': {}}, kinds={}, risks={}, relations={}, reranker={},
                    metadata={}, timings={}, completed_stages=[], retrieval=dict(RETRIEVAL),
                    candidate_pairs=None, inventory=inventory(needs, reply),
                    note='Masked input-only features. NLI covers the exact baseline; reranker is a sparse candidate graph. '
                         'Missing reranker pairs are forbidden, never score zero. Batch timings are preparation costs, not online latency.')
    if data.get('split') != split or data.get('schema') != 'features-v5c-1':
        raise RuntimeError('Feature cache split/schema differs')
    reconcile_contract(data, details)
    check_cache(data, sources=sources, needs=needs, reply=reply, contract=contract, pins=details['pins'])
    for selected in STAGES if stage == 'all' else [stage]:
        if selected in data['completed_stages']:
            continue
        if selected != 'embeddings' and 'embeddings' not in data['completed_stages']:
            raise RuntimeError('Complete embeddings before NLI or reranker')
        run_stage(data, selected, path)
        check_cache(data, sources=sources, needs=needs, reply=reply, contract=contract, pins=details['pins'])
    if sources != source_hashes(paths):
        raise RuntimeError('Feature source bytes changed during preparation')
    save_checkpoint(path, data)
    if stage == 'all':
        validate_feature_provenance(data, sources=sources, needs=needs, reply=reply,
                                    contract=contract, metadata=data['metadata'])
    print('READY', split, stage, len(data['inventory']['texts']), 'texts',
          len(data['relations']), 'directed NLI relations', len(data['reranker']),
          'directed reranker pairs', flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['tune', 'control'])
    parser.add_argument('--stage', choices=[*STAGES, 'all'], default='all')
    parser.add_argument('--freeze', type=Path, help='Explicit freeze manifest; defaults to CONFIG-FREEZE.json')
    args = parser.parse_args()
    prepare(args.split, args.stage, freeze_path=args.freeze)


if __name__ == '__main__':
    main()
