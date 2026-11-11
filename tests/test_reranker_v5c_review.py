"""Independent v5c contract review: stdlib only, synthetic data, no inference.

Run with python3 -B -m unittest discover -s tests -p test_reranker_v5c_review.py -v.
All regressions run unconditionally, including with RADAR_V5C_REVIEW_STRICT=1.
The fake subprocess and all model/cache fixtures live in TemporaryDirectory.
"""
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from radar_app import model_artifacts_v5b as artifacts
from radar_app.reranker_v5c import RerankerClient


ROOT = Path(__file__).resolve().parents[1]


class ReviewCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='radar-v5c-review-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Worker import changes four offline variables; restore the caller's env.
        self.enterContext(patch.dict(os.environ))
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            top = name.split('.')[0]
            if top in {'torch', 'transformers', 'sentence_transformers'}:
                if not getattr(sys.modules.get(top), '_review_double', False):
                    raise AssertionError('Real inference import forbidden: ' + name)
            return original_import(name, *args, **kwargs)

        self.enterContext(patch.object(builtins, '__import__', guarded_import))
        self.enterContext(patch.object(
            socket.socket, 'connect', side_effect=AssertionError('Network forbidden')))
        spec = importlib.util.spec_from_file_location(
            'v5c_review_worker', ROOT / 'scripts/reranker_worker_v5c.py')
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def artifact_fixture(self):
        directory = self.root / 'runtime/models/qwen-reranker-v5c'
        directory.mkdir(parents=True)
        header = json.dumps({
            'x': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}
        }).encode()
        blobs = {
            'model.safetensors': struct.pack('<Q', len(header)) + header + b'1234',
            'config.json': b'{"model_type":"test_fixture"}',
            'tokenizer.json': b'{"vocab":{"fixture":1}}',
            'tokenizer_config.json': b'{"padding_side":"left"}',
        }
        manifest = {'model': self.mod.MODEL, 'revision': self.mod.REVISION, 'files': {}}
        for name, blob in blobs.items():
            (directory / name).write_bytes(blob)
            manifest['files'][name] = {
                'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
        self.mod.ROOT = self.root
        # Only this private imported module accepts the tiny fixture's weight pin.
        self.mod.WEIGHT_SHA = manifest['files']['model.safetensors']['sha256']
        (directory / 'MANIFEST.json').write_text(json.dumps(manifest))
        return directory, manifest

    def loader_doubles(self, *, token_ids=(11, 12), mps_available=True):
        calls = []
        tokenizer = SimpleNamespace(
            encode=Mock(return_value=[1]),
            convert_tokens_to_ids=Mock(side_effect=lambda token: dict(zip(['yes', 'no'], token_ids))[token]))
        model = SimpleNamespace()
        model.to = Mock(return_value=model)
        model.eval = Mock(return_value=model)
        torch = ModuleType('torch')
        torch._review_double = True
        torch.float16 = 'fixture-float16'
        torch.set_num_threads = lambda n: calls.append(('threads', n))
        torch.set_num_interop_threads = lambda n: calls.append(('interop', n))
        torch.backends = SimpleNamespace(mps=SimpleNamespace(
            is_available=lambda: mps_available))
        torch.mps = SimpleNamespace(
            set_per_process_memory_fraction=lambda n: calls.append(('memory_fraction', n)),
            empty_cache=Mock())
        transformers = ModuleType('transformers')
        transformers._review_double = True
        transformers.AutoTokenizer = SimpleNamespace(
            from_pretrained=Mock(return_value=tokenizer))
        transformers.AutoModelForCausalLM = SimpleNamespace(
            from_pretrained=Mock(return_value=model))
        self.enterContext(patch.dict(sys.modules, torch=torch, transformers=transformers))
        return transformers, tokenizer, model, calls

    def protocol_fixture(self, **configuration):
        config = self.root / 'protocol-config.json'
        requests = self.root / 'requests.jsonl'
        script = self.root / 'protocol-worker.py'
        config.write_text(json.dumps(configuration))
        script.write_text(
            'CONFIG = ' + repr(str(config)) + '\n'
            'REQUESTS = ' + repr(str(requests)) + '\n' + r'''
import json, os, sys, time
from pathlib import Path
cfg = json.loads(Path(CONFIG).read_text())
for line in sys.stdin:
    request = json.loads(line)
    with open(REQUESTS, 'a') as log:
        log.write(json.dumps(request, ensure_ascii=False) + '\n')
    op = request['op']
    if op == 'describe':
        result = dict(dimension=1, model='fixture/verifier',
                      revision=cfg.get('revision', 'A'), max_tokens=512,
                      artifacts={'tokenizer.json': {'sha256': cfg.get('tokenizer', 'a')}})
    elif op == 'stats':
        result = {name: os.environ.get(name) for name in [
            'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY',
            'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}
    elif op == 'rerank':
        mode = cfg.get('mode')
        if mode == 'timeout':
            time.sleep(10)
        if mode == 'exit':
            os._exit(7)
        if mode == 'bad-json':
            print('{not-json}', flush=True)
            continue
        if mode == 'nonobject':
            print(json.dumps(cfg.get('envelope', [])), flush=True)
            continue
        if mode == 'oversized':
            sys.stdout.write('x' * (1024**2 + 1) + '\n')
            sys.stdout.flush()
            continue
        result = cfg.get('answer', [cfg.get('score', .25)] * len(request['pairs']))
    else:
        raise ValueError('unexpected operation')
    response = dict(id=request['id'], ok=True, result=result)
    if op == 'rerank':
        mode = cfg.get('mode')
        if mode == 'wrong-id':
            response['id'] += 1
        if mode == 'refusal':
            response = dict(id=request['id'], ok=False, error='ValueError')
        if mode == 'missing-result':
            del response['result']
        if mode == 'truthy-ok':
            response['ok'] = 1
    print(json.dumps(response), flush=True)
''')
        return script, config, requests

    def client(self, script, **kwargs):
        client = RerankerClient(
            python=sys.executable, worker_script=script,
            cache_dir=self.root / 'cache', **kwargs)
        self.addCleanup(client.close)
        return client

    @staticmethod
    def requests(path, op=None):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return [row for row in rows if op is None or row['op'] == op]


class WorkerContractTests(ReviewCase):
    def test_import_and_describe_do_not_import_inference(self):
        self.artifact_fixture()
        with patch.object(self.mod.importlib.metadata, 'version', return_value='fixture'):
            worker = self.mod.Worker()
            description = worker.describe()
        self.assertIsNone(worker.model)
        self.assertEqual(description['revision'], self.mod.REVISION)
        self.assertEqual(description['max_tokens'], 512)
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'], '1')
        self.assertIn('not a calibrated probability', description['score'])
        description['artifacts']['config.json']['sha256'] = 'caller-edit'
        self.assertNotEqual(worker.manifest['files']['config.json']['sha256'], 'caller-edit')

    def test_all_three_declared_pins_are_enforced(self):
        directory, manifest = self.artifact_fixture()
        for field in ['model', 'revision', 'weight']:
            changed = copy.deepcopy(manifest)
            if field == 'weight':
                changed['files']['model.safetensors']['sha256'] = '0' * 64
            else:
                changed[field] = 'unexpected'
            (directory / 'MANIFEST.json').write_text(json.dumps(changed))
            with self.subTest(pin=field), self.assertRaisesRegex(ValueError, 'pinned'):
                self.mod.Worker()

    def test_same_size_small_artifact_changes_are_rejected(self):
        directory, manifest = self.artifact_fixture()
        for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
            path = directory / name
            before = path.read_bytes()
            path.write_bytes(b'!' + before[1:])
            with self.subTest(artifact=name), self.assertRaisesRegex(RuntimeError, 'checksum'):
                self.mod.Worker()
            path.write_bytes(before)
        artifacts.validate_artifacts(directory, manifest, verify_weight_sha256=True)

    def test_weight_mutations_fail_before_any_inference_import(self):
        directory, _ = self.artifact_fixture()
        worker = self.mod.Worker()
        weight = directory / 'model.safetensors'
        before = weight.read_bytes()
        for blob in [before[:-1], before[:-1] + b'!']:
            weight.write_bytes(blob)
            with self.subTest(size=len(blob)), self.assertRaises(RuntimeError):
                worker.load()
            self.assertIsNone(worker.model)
        weight.write_bytes(before)

    def test_small_artifact_changed_after_initial_binding_is_rejected_at_load(self):
        directory, _ = self.artifact_fixture()
        worker = self.mod.Worker()
        path = directory / 'tokenizer.json'
        before = path.read_bytes()
        path.write_bytes(b'!' + before[1:])
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            worker.load()
        self.assertIsNone(worker.model)

    def test_optional_unpinned_tokenizer_and_escaping_symlink_are_rejected(self):
        directory, manifest = self.artifact_fixture()
        optional = directory / 'added_tokens.json'
        optional.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'Unpinned tokenizer'):
            artifacts.validate_artifacts(directory, manifest)
        optional.unlink()
        outside = self.root / 'outside.json'
        outside.write_bytes((directory / 'config.json').read_bytes())
        (directory / 'config.json').unlink()
        (directory / 'config.json').symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, 'escapes'):
            artifacts.validate_artifacts(directory, manifest)

    def test_oversized_artifact_and_manifest_path_escape_are_rejected(self):
        directory, manifest = self.artifact_fixture()
        changed = copy.deepcopy(manifest)
        changed['files']['tokenizer.json']['bytes'] = artifacts.MAX_SMALL_FILE_BYTES + 1
        with self.assertRaisesRegex(RuntimeError, 'limit'):
            artifacts.validate_artifacts(directory, changed)
        changed = copy.deepcopy(manifest)
        changed['files']['../outside.json'] = {'bytes': 0, 'sha256': '0' * 64}
        with self.assertRaisesRegex(RuntimeError, 'path'):
            artifacts.validate_artifacts(directory, changed)

    def test_loader_uses_local_safetensors_and_disables_remote_code(self):
        directory, _ = self.artifact_fixture()
        os.environ['RADAR_RERANK_DEVICE'] = 'cpu'
        transformers, tokenizer, model, calls = self.loader_doubles()
        worker = self.mod.Worker()
        worker.load()
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            str(directory), local_files_only=True, trust_remote_code=False, padding_side='left')
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            str(directory), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype='fixture-float16', attn_implementation='sdpa')
        model.to.assert_called_once_with('cpu')
        model.eval.assert_called_once_with()
        self.assertEqual(calls, [('threads', 2), ('interop', 1)])
        tokenizer.convert_tokens_to_ids.assert_any_call('yes')
        tokenizer.convert_tokens_to_ids.assert_any_call('no')

    def test_unavailable_mps_is_a_refusal_not_a_silent_cpu_fallback(self):
        self.artifact_fixture()
        os.environ['RADAR_RERANK_DEVICE'] = 'mps'
        transformers, _, _, _ = self.loader_doubles(mps_available=False)
        worker = self.mod.Worker()
        with self.assertRaisesRegex(RuntimeError, 'no silent device change'):
            worker.load()
        transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        self.assertIsNone(worker.model)

    def test_invalid_input_and_gold_rows_fail_before_load(self):
        worker = self.mod.Worker.__new__(self.mod.Worker)
        worker.load = Mock(side_effect=AssertionError('Must reject before load'))
        cases = [None, [], 'ab', [['a', 'b']] * 5, [['a']], [['a', 'b', True]],
                 [{'query': 'a', 'document': 'b', 'gold': True}],
                 [['a', '  ']], [['a', 2]], [['a' * 4001, 'b']]]
        for pairs in cases:
            with self.subTest(pairs=repr(pairs)[:90]), self.assertRaises(ValueError):
                worker.rerank(pairs)
        worker.load.assert_not_called()

    def test_extra_request_gold_is_not_passed_to_the_pair_scorer(self):
        worker = self.mod.Worker.__new__(self.mod.Worker)
        worker.rerank = Mock(return_value=[.25])
        pairs = [['literal gold=True in a message', 'unrelated synthetic document']]
        result = worker.run('rerank', {'pairs': pairs, 'gold': True, 'label': 'match'})
        self.assertEqual(result, [.25])
        worker.rerank.assert_called_once_with(pairs)
        self.assertIn('gold=True', self.mod.format_pair(*pairs[0]))

    def test_512_token_limit_counts_prompt_prefix_and_suffix_without_truncation(self):
        class AtPadding(Exception):
            pass

        for total in [512, 513]:
            with self.subTest(tokens=total):
                worker = self.mod.Worker.__new__(self.mod.Worker)
                worker.load = Mock()
                worker.prefix, worker.suffix = [1] * 5, [2] * 7
                encode = Mock(return_value=[3] * (total - 12))
                pad = Mock(side_effect=AtPadding)
                worker.tokenizer = SimpleNamespace(encode=encode, pad=pad)
                expected = AtPadding if total == 512 else ValueError
                with self.assertRaises(expected):
                    worker.rerank([['a', 'b']])
                encode.assert_called_once_with(self.mod.format_pair('a', 'b'), add_special_tokens=False)
                if total == 512:
                    row = pad.call_args.args[0][0]
                    self.assertEqual(len(row['input_ids']), 512)
                    self.assertEqual(row['attention_mask'], [1] * 512)
                    self.assertEqual(row['input_ids'][-7:], [2] * 7)
                else:
                    pad.assert_not_called()

    def test_nonfinite_nonnumeric_boolean_and_out_of_range_scores_are_refused(self):
        for score in [float('nan'), float('inf'), -float('inf'), True, False,
                      None, '0.9', [], -.01, 1.01]:
            with self.subTest(score=repr(score)), self.assertRaises(RuntimeError):
                self.mod.check_score(score)
        self.assertEqual(self.mod.check_score(0), 0.)
        self.assertEqual(self.mod.check_score(1), 1.)

    def test_defect_failed_token_initialization_must_remain_failed_on_retry(self):
        """R5: load publishes self.model before validating yes/no token IDs."""
        self.artifact_fixture()
        os.environ['RADAR_RERANK_DEVICE'] = 'cpu'
        self.loader_doubles(token_ids=(11, 11))
        worker = self.mod.Worker()
        with self.assertRaisesRegex(RuntimeError, 'token IDs'):
            worker.load()
        with self.assertRaisesRegex(RuntimeError, 'token IDs'):
            worker.load()
        self.assertIsNone(worker.model)
        self.assertFalse(hasattr(worker, 'tokenizer'))
        self.assertFalse(hasattr(worker, 'torch'))
        # A failed transfer must not publish state or leave MPS cache cleanup out.
        os.environ['RADAR_RERANK_DEVICE'] = 'mps'
        transformers, _, model, _ = self.loader_doubles()
        model.to.side_effect = RuntimeError('fixture device allocation failure')
        worker = self.mod.Worker()
        with self.assertRaisesRegex(RuntimeError, 'fixture device allocation failure'):
            worker.load()
        self.assertIsNone(worker.model)
        self.assertFalse(hasattr(worker, 'tokenizer'))
        self.assertFalse(hasattr(worker, 'torch'))
        self.assertFalse(worker.run('stats', {})['loaded'])
        sys.modules['torch'].mps.empty_cache.assert_called_once_with()
        model.to.side_effect = None
        worker.load()
        self.assertIs(worker.model, model)
        self.assertIs(worker.tokenizer, transformers.AutoTokenizer.from_pretrained.return_value)

    def test_defect_nonobject_request_must_not_kill_worker_protocol(self):
        """R4: the exception handler calls .get again on a non-object request."""
        worker = SimpleNamespace(run=Mock(return_value={'loaded': False}))
        for envelope in [[], None, True, 7, 'text']:
            with self.subTest(envelope=envelope):
                stdin = SimpleNamespace(buffer=io.BytesIO(
                    json.dumps(envelope).encode() + b'\n{"id":2,"op":"stats"}\n'))
                stdout = io.StringIO()
                with patch.object(self.mod, 'Worker', return_value=worker), \
                        patch.object(sys, 'stdin', stdin), contextlib.redirect_stdout(stdout), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.mod.main()
                rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
                self.assertFalse(rows[0]['ok'])
                self.assertEqual(rows[1], {'id': 2, 'ok': True, 'result': {'loaded': False}})


class ClientProtocolTests(ReviewCase):
    def test_process_is_offline_and_threads_are_bounded(self):
        script, _, _ = self.protocol_fixture()
        client = self.client(script)
        self.assertEqual(client._request('stats'), {
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})

    def test_order_deduplication_cache_hits_and_no_text_in_cache(self):
        script, _, requests = self.protocol_fixture(answer=[.2, .8])
        client = self.client(script)
        a = ['Не хранить исходный текст', 'Синтетическая пара A']
        b = ['Синтетическая пара B', 'Другая формулировка']
        self.assertEqual(client.rerank([a, b, a]), [.2, .8, .2])
        self.assertEqual(client.rerank([b, a]), [.8, .2])
        self.assertEqual(len(self.requests(requests, 'rerank')), 1)
        for path in (self.root / 'cache').glob('*.json'):
            self.assertNotIn('исходный', path.read_text())

    def test_changed_revision_and_tokenizer_pins_use_distinct_cache_entries(self):
        script, config, requests = self.protocol_fixture(score=.2)
        self.assertEqual(self.client(script).rerank([['a', 'b']]), [.2])
        config.write_text(json.dumps({'revision': 'B', 'score': .8}))
        self.assertEqual(self.client(script).rerank([['a', 'b']]), [.8])
        config.write_text(json.dumps({'revision': 'B', 'tokenizer': 'b', 'score': .6}))
        self.assertEqual(self.client(script).rerank([['a', 'b']]), [.6])
        self.assertEqual(len(self.requests(requests, 'rerank')), 3)

    def test_cache_checksum_corruption_is_recomputed(self):
        script, _, requests = self.protocol_fixture()
        client = self.client(script)
        client.rerank([['a', 'b']])
        path = next((self.root / 'cache').glob('*.json'))
        cached = json.loads(path.read_text())
        cached['result'] = .99
        path.write_text(json.dumps(cached))
        self.assertEqual(client.rerank([['a', 'b']]), [.25])
        self.assertEqual(len(self.requests(requests, 'rerank')), 2)

    def test_invalid_cache_values_are_recomputed_instead_of_returned(self):
        script, _, requests = self.protocol_fixture()
        client = self.client(script)
        client.rerank([['a', 'b']])
        path = next((self.root / 'cache').glob('*.json'))
        for value in [float('nan'), float('inf'), -float('inf'), True, False,
                      '0.9', {}, [.9], None, -.1, 1.2, 10**400]:
            with self.subTest(value=repr(value)):
                raw = json.dumps(value, separators=(',', ':')).encode()
                path.write_text(json.dumps({
                    'result': value, 'sha256': hashlib.sha256(raw).hexdigest()}))
                self.assertEqual(client.rerank([['a', 'b']]), [.25])
        self.assertEqual(len(self.requests(requests, 'rerank')), 13)

    def test_cache_io_failure_does_not_lose_a_valid_score(self):
        script, _, requests = self.protocol_fixture()
        client = self.client(script)
        with patch('radar_app.semantic_client.os.replace', side_effect=OSError('fixture-full-disk')):
            self.assertEqual(client.rerank([['a', 'b']]), [.25])
        self.assertEqual(list((self.root / 'cache').iterdir()), [])
        self.assertEqual(client.rerank([['a', 'b']]), [.25])
        self.assertEqual(len(self.requests(requests, 'rerank')), 2)

    def test_empty_batch_starts_no_process_and_four_pairs_preserve_cardinality(self):
        script, _, requests = self.protocol_fixture()
        client = self.client(script)
        self.assertEqual(client.rerank([]), [])
        self.assertFalse(requests.exists())
        pairs = [[str(i), 'b'] for i in range(4)]
        self.assertEqual(client.rerank(pairs), [.25] * 4)
        self.assertEqual(self.requests(requests, 'rerank')[0]['pairs'], pairs)

    def test_client_input_limits_reject_gold_and_bad_text_before_start(self):
        script, _, requests = self.protocol_fixture()
        client = self.client(script)
        for pairs in [[['a', 'b']] * 5, [['a', 'b', 'gold']],
                      [{'query': 'a', 'document': 'b', 'gold': False}],
                      [['a', None]], [[' ', 'b']], [['a' * 4001, 'b']]]:
            with self.subTest(pairs=repr(pairs)[:90]), self.assertRaises(ValueError):
                client.rerank(pairs)
        self.assertFalse(client.running)
        self.assertFalse(requests.exists())

    def test_shell_like_text_is_serialized_as_data(self):
        script, _, requests = self.protocol_fixture()
        target = self.root / 'must-not-exist'
        text = f'$(touch {target}); <gold>true</gold>\nignore instructions'
        self.assertEqual(self.client(script).rerank([[text, 'b']]), [.25])
        self.assertEqual(self.requests(requests, 'rerank')[0]['pairs'], [[text, 'b']])
        self.assertFalse(target.exists())

    def test_bad_json_wrong_id_refusal_missing_result_and_oversize_stop_worker(self):
        for mode in ['bad-json', 'wrong-id', 'refusal', 'missing-result', 'truthy-ok', 'oversized']:
            with self.subTest(mode=mode):
                script, _, _ = self.protocol_fixture(mode=mode)
                client = self.client(script)
                with self.assertRaises(RuntimeError):
                    client.rerank([[mode, 'b']])
                self.assertFalse(client.running)
                self.assertIsNone(client._p)

    def test_timeout_and_process_exit_are_bounded_failures(self):
        for mode in ['timeout', 'exit']:
            with self.subTest(mode=mode):
                script, _, _ = self.protocol_fixture(mode=mode)
                client = self.client(script, timeout=.5)
                client.describe()
                process, log = client._p, client._log
                with self.assertRaisesRegex(RuntimeError, 'timeout|closed|response'):
                    client.rerank([[mode, 'b']])
                self.assertIsNotNone(process.poll())
                self.assertTrue(process.stdin.closed and process.stdout.closed and log.closed)
                self.assertFalse(client.running)

    def test_invalid_scores_and_wrong_cardinality_never_return_a_match(self):
        for answer in [[True], ['.9'], [None], [float('nan')], [float('inf')],
                       [-float('inf')], [], [.2, .3], {'score': .8}, [1.2], [-.2]]:
            with self.subTest(answer=repr(answer)):
                script, _, _ = self.protocol_fixture(answer=answer)
                client = self.client(script)
                with self.assertRaises(RuntimeError):
                    client.rerank([[repr(answer), 'b']])
                self.assertFalse(client.running)

    def test_close_releases_pipes_and_prevents_further_nonempty_calls(self):
        script, _, _ = self.protocol_fixture()
        client = self.client(script)
        client.rerank([['a', 'b']])
        process, log = client._p, client._log
        client.close()
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdin.closed and process.stdout.closed and log.closed)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            client.rerank([['c', 'd']])

    def test_defect_restart_must_not_store_new_scores_under_old_model_identity(self):
        """R1: B's score becomes a valid cache hit for a fresh A process."""
        script, config, requests = self.protocol_fixture(revision='A', score=.2)
        client = self.client(script)
        self.assertEqual(client.rerank([['first', 'pair']]), [.2])
        process = client._p
        process.terminate()
        process.wait(timeout=2)
        config.write_text(json.dumps({'revision': 'B', 'score': .8}))
        # v5c deliberately refuses a second generation; even cache hits and
        # describe must fail until a new client performs a new attestation.
        for operation in [lambda: client.rerank([['new', 'pair']]),
                          lambda: client.rerank([['first', 'pair']]), client.describe]:
            with self.assertRaisesRegex(RuntimeError, 'new client'):
                operation()
        self.assertEqual(len(self.requests(requests, 'rerank')), 1)
        self.assertFalse(client.running)
        client.close()
        replacement = self.client(script)
        self.assertEqual(replacement.rerank([['new', 'pair']]), [.8])
        replacement.close()
        config.write_text(json.dumps({'revision': 'A', 'score': .2}))
        fresh = self.client(script)
        actual = fresh.rerank([['new', 'pair']])
        self.assertEqual(len(self.requests(requests, 'describe')), 3)
        self.assertEqual(actual, [.2], 'Revision A received a cached score computed by B')

    def test_defect_out_of_range_response_must_not_poison_persistent_cache(self):
        """R2: SemanticClient writes before RerankerClient checks [0, 1]."""
        script, config, requests = self.protocol_fixture(answer=[1.2])
        client = self.client(script)
        key = client._key('rerank', ['a', 'b'])
        with self.assertRaises(RuntimeError):
            client.rerank([['a', 'b']])
        self.assertEqual(list((self.root / 'cache').glob('*.json')), [])
        client.close()
        # An entry created by the old client must also be treated as a miss.
        raw = json.dumps(1.2, separators=(',', ':')).encode()
        (self.root / 'cache' / (key + '.json')).write_text(json.dumps({
            'result': 1.2, 'sha256': hashlib.sha256(raw).hexdigest()}))
        config.write_text(json.dumps({'answer': [.4]}))
        fresh = self.client(script)
        try:
            actual = fresh.rerank([['a', 'b']])
        except RuntimeError as exc:
            self.fail('Rejected out-of-range result persists across clients: ' + str(exc))
        self.assertEqual(actual, [.4])
        self.assertEqual(len(self.requests(requests, 'rerank')), 2)

    def test_defect_nonobject_response_must_stop_owned_process_and_raise_runtime_error(self):
        """R3: .get on [] bypasses _request's exception/cleanup boundary."""
        for envelope in [[], None, True, 7, 'text']:
            with self.subTest(envelope=envelope):
                script, _, _ = self.protocol_fixture(mode='nonobject', envelope=envelope)
                client = self.client(script)
                with self.assertRaises(RuntimeError):
                    client.rerank([[repr(envelope), 'b']])
                self.assertFalse(client.running, 'Malformed-response worker remains alive')

    def test_defect_nested_description_must_not_be_mutable_by_caller(self):
        """R6: dict(self._desc) shares artifacts with the cache-key identity."""
        script, _, _ = self.protocol_fixture()
        client = self.client(script)
        description = client.describe()
        before = client._key('rerank', ['a', 'b'])
        description['artifacts']['tokenizer.json']['sha256'] = 'caller-edited'
        self.assertEqual(client._key('rerank', ['a', 'b']), before)


class PublishedV9MetricsTests(unittest.TestCase):
    """Only published aggregate results and v9 source/PDF JSON, never gold rows."""

    def test_aggregate_precision_recall_and_denominators_recompute(self):
        data = json.loads((ROOT / 'data/quality-v5b.json').read_text())
        self.assertEqual(data['selected_before_control'], 'qwen')
        self.assertIs(data['gate']['needs'], False)
        self.assertIs(data['gate']['autoreply'], False)
        self.assertIs(data['gate']['auto_enable'], False)
        for backend in ['tfidf', 'e5', 'qwen']:
            needs = data['needs'][backend]
            self.assertEqual(needs['n'], 116)
            reply = data['autoreply'][backend]
            self.assertEqual(reply['n'], 80)
            self.assertEqual(reply['expected_standard'], reply['tp'] + reply['fn'])
            for scores in [needs['pair_metrics'], reply, reply['fixed_notice']]:
                with self.subTest(backend=backend, scores=scores):
                    self.assertAlmostEqual(scores['precision'], scores['tp'] / (scores['tp'] + scores['fp']))
                    self.assertAlmostEqual(scores['recall'], scores['tp'] / (scores['tp'] + scores['fn']))

    def test_v9_source_and_both_rendered_pdf_json_editions_keep_failed_gate_and_scope(self):
        if (ROOT / 'presentation/radar.html').is_file():
            published = json.loads((ROOT / 'data/quality-v5c.json').read_text())
            self.assertIs(published['models']['candidate']['gates']['auto_enable'], False)
            for name in ['radar.html', 'radar-notes.html']:
                html = (ROOT / 'presentation' / name).read_text()
                marker = 'DECK.DATA.v11={quality:'
                start = html.index(marker) + len(marker)
                embedded, _ = json.JSONDecoder().raw_decode(html[start:])
                self.assertEqual(embedded, published)
                self.assertIn('Accuracy (верные типы)', html)
                self.assertIn('Precision (точность пар)', html)
                self.assertIn('Recall (полнота пар)', html)
                self.assertIn('Порог 80% по всем показателям потребностей пока не достигнут.', html)
            return
        source = (ROOT / 'data/quality-v5b.json').read_bytes()
        published = json.loads(source)
        for suffix in ['', '-для-жюри']:
            html = (ROOT / f'presentation/Радар-v13-плюс-кейс-v9{suffix}.html').read_text()
            start = html.index('DECK.DATA.v5=') + len('DECK.DATA.v5=')
            embedded, _ = json.JSONDecoder().raw_decode(html[start:])
            self.assertEqual(embedded['quality'], published)
            self.assertIn(hashlib.sha256(source).hexdigest(), html)
            self.assertIn('DECK.DATA.v6={quality:DECK.DATA.v5.quality', html)
            self.assertIn('Автовыпуск отключён; проверяет человек; внешней доставки нет.', html)
        qa = ROOT / 'presentation/v13-plus-v9/qa'
        for name in ['export-v9.json', 'pdf-final-inspection-v9.json']:
            data = json.loads((qa / name).read_text())
            documents = data['documents'] if isinstance(data, dict) else data
            self.assertEqual(len(documents), 2)
            for edition, document in enumerate(documents):
                with self.subTest(json=name, edition=edition):
                    self.assertEqual(len(document['pages']), 13)
                    quality = document['pages'][11]['text']
                    for needs in published['needs'].values():
                        scores = needs['pair_metrics']
                        self.assertIn(f"TP {scores['tp']} / FP {scores['fp']} / FN {scores['fn']}", quality)
                        for value in [needs['kind_accuracy'], scores['precision'], scores['recall']]:
                            percent = f'{value * 100:.1f}'.removesuffix('.0').replace('.', ',') + '%'
                            self.assertIn(percent, quality)
                    self.assertEqual(quality.count('n=116'), 3)
                    self.assertIn('Критерии поиска потребностей пока не пройдены.', quality)
                    self.assertIn('не точность карточек', quality)
                    self.assertIn('не корпоративная Qwen 3.8', quality)
                    self.assertNotIn('Reranker', quality)
                    self.assertIn('Оценка при допущениях', document['pages'][9]['text'])
                    self.assertIn('РАСЧЁТ ПРИ ДОПУЩЕНИЯХ', document['pages'][12]['text'])


if __name__ == '__main__':
    unittest.main()
