"""Isolated full-population fixtures; never opens the real v5c control."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'score_service_v5c.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


class ModuleAvailability(unittest.TestCase):
    def test_scorer_module_exists(self):
        self.assertTrue(MODULE.is_file(), 'Independent scorer implementation is missing')


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('independent_score_v5c', MODULE)
        cls.scorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scorer)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='score-v5c-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'evaluation/v5c'
        self.out = self.root / 'reports/quality-v5c'
        self.predpath = self.out / 'control-predictions-only.json'
        self.freezepath = self.out / 'CONFIG-FREEZE.json'
        self.needs = []
        self.ng = []
        kinds = ['wish', 'bug', 'usability', 'service']
        for family in range(40):
            for member in range(4):
                ident = f'n-{family:02}-{member}'
                product, kind = f'p-{family % 2}', kinds[family % 4]
                self.needs.append(dict(id=ident, product=product, source_id='s-' + ident,
                                       author_id='u-' + ident, created_at=f'2026-10-{member+1:02}T09:00:00+04:00',
                                       received_at=f'2026-10-{member+1:02}T09:00:09+04:00',
                                       channel='chat', text='Fictitious fixture request.'))
                self.ng.append(dict(id=ident, gold=dict(product=product, kind=kind, topic_id=f't-{family}',
                                                       direction='introduce', is_duplicate_delivery=False)))
        for i in range(24):
            ident = f'n-single-{i}'
            self.needs.append(dict(id=ident, product='p-0', source_id='s-' + ident,
                                   author_id='u-' + ident, created_at='2026-10-05T09:00:00+04:00',
                                   received_at='2026-10-05T09:00:09+04:00', channel='chat', text='Other fictitious fixture.'))
            self.ng.append(dict(id=ident, gold=dict(product='p-0', kind='wish' if i < 8 else 'unknown',
                                                   topic_id=f't-single-{i}' if i < 8 else None,
                                                   direction='introduce' if i < 8 else 'undetermined',
                                                   is_duplicate_delivery=False)))
        self.unique_needs = copy.deepcopy(self.needs)
        self.unique_ng = copy.deepcopy(self.ng)
        for i in [0, 4, 8, 12]:
            row = copy.deepcopy(self.needs[i]);row['received_at'] = '2026-10-20T09:00:09+04:00'
            gold = copy.deepcopy(self.ng[i]);gold['gold']['is_duplicate_delivery'] = True
            self.needs.append(row);self.ng.append(gold)
        self.reply = []
        self.rg = []
        for i in range(132):
            fixed = i >= 120
            action = 'queue_standard' if i < 30 else ('queue_fixed_notice' if 120 <= i < 126 else 'operator')
            self.reply.append(dict(id=f'r-{i}', event='fixed_notice_candidate' if fixed else 'incoming_ticket',
                                   product='p-0', kind='bug', approved=True, state='closed' if fixed else 'open',
                                   fix_confirmed=fixed, human_closed=fixed, fixed_notice_approved=fixed,
                                   seen_message_ids=[], references=[dict(id='e-1', product='p-0', kind='bug',
                                   author_id='u-e1', source_verified=True, text='Fixture evidence one'),
                                   dict(id='e-2', product='p-0', kind='bug', author_id='u-e2',
                                   source_verified=True, text='Fixture evidence two')],
                                   query=dict(id=f'm-{i}', product='p-0', consent=True, text='Fixture query')))
            self.rg.append(dict(id=f'r-{i}', gold=dict(same_topic=i < 30 or 120 <= i < 126,
                                eligibility=action == 'queue_standard', safe_action=action,
                                query_kind='bug', query_direction='restore', reference_topic_id='t-ref',
                                reason=['fixture'], synthetic=True)))
        one = dict(needs=[dict(id=r['id'], kind=g['gold']['kind'], prediction=g['gold']['topic_id'])
                          for r, g in zip(self.unique_needs, self.unique_ng)],
                   reply=[dict(id=r['id'], action=g['gold']['safe_action']) for r, g in zip(self.reply, self.rg)])
        self.predictions = dict(baseline=copy.deepcopy(one), candidate=copy.deepcopy(one))
        self.write_dataset()
        self.refresh_lock()
        self.persist_predictions()

    def write_dataset(self, flat=False):
        write_jsonl(self.dataset / 'control-inputs.jsonl', self.needs)
        write_jsonl(self.dataset / 'reply-control-inputs.jsonl', self.reply)
        ng, rg = self.ng, self.rg
        if flat:
            ng = [dict(id=r['id'], **r['gold']) for r in ng]
            rg = [dict(id=r['id'], **r['gold']) for r in rg]
        write_jsonl(self.dataset / 'sealed/control-gold.jsonl', ng)
        write_jsonl(self.dataset / 'sealed/reply-control-gold.jsonl', rg)

    def refresh_lock(self):
        files = ['control-inputs.jsonl', 'reply-control-inputs.jsonl',
                 'sealed/control-gold.jsonl', 'sealed/reply-control-gold.jsonl']
        write_json(self.dataset / 'CONTROL-LOCK.json', dict(algorithm='sha256', files={p: sha(self.dataset / p) for p in files}))
        config = self.root / 'frozen/config.json'
        if not config.exists():write_json(config, dict(candidate='fixture-only'))
        write_json(self.freezepath, dict(files={'frozen/config.json': sha(config)},
                                        dataset_lock_sha256=sha(self.dataset / 'CONTROL-LOCK.json')))

    def persist_predictions(self):
        write_json(self.predpath, self.predictions)

    def evaluate(self):
        return self.scorer.evaluate(self.predpath, self.freezepath, root=self.root, output_dir=self.out)

    def assert_no_reports(self):
        self.assertFalse((self.out / 'control-metrics.json').exists())
        self.assertFalse((self.out / 'errors.json').exists())

    def test_exact_population_and_separate_fix_denominator(self):
        report = self.evaluate()
        for model in ['baseline', 'candidate']:
            metrics = report['models'][model]
            self.assertEqual(metrics['needs']['deliveries'], 188)
            self.assertEqual(metrics['needs']['unique_messages'], 184)
            self.assertEqual(metrics['needs']['kind_accuracy'], 1.0)
            self.assertEqual(metrics['needs']['pairs']['tp'], 240)
            self.assertEqual(metrics['needs']['pairs']['fp'], 0)
            self.assertEqual(metrics['needs']['pairs']['fn'], 0)
            self.assertEqual(metrics['ordinary_reply']['population'], 120)
            self.assertEqual(metrics['ordinary_reply']['positives'], 30)
            self.assertEqual(metrics['ordinary_reply']['tp'], 30)
            self.assertEqual(metrics['fixed_notice']['population'], 12)
            self.assertEqual(metrics['fixed_notice']['tp'], 6)
            self.assertEqual(metrics['action_accuracy_all']['population'], 132)
            self.assertEqual(metrics['action_accuracy_all']['accuracy'], 1.0)
            self.assertTrue(metrics['gates']['target_80_met'])
            self.assertTrue(metrics['gates']['zero_false_positive_replies'])
            self.assertFalse(metrics['gates']['auto_enable'])
        self.assertEqual(json.loads((self.out / 'errors.json').read_text())['models']['candidate']['group_merges'], [])

    def test_dedup_chooses_created_then_id_order(self):
        rows = [dict(id='z', product='p', source_id='s', created_at='2026-01-02T00:00:00+00:00'),
                dict(id='b', product='p', source_id='s', created_at='2026-01-01T00:00:00+00:00'),
                dict(id='a', product='p', source_id='s', created_at='2026-01-01T00:00:00+00:00'),
                dict(id='other', product='q', source_id='s', created_at='2026-01-01T00:00:00+00:00')]
        self.assertEqual([r['id'] for r in self.scorer.deduplicate_needs(rows)], ['a', 'other'])

    def test_missing_needs_id_rejected_before_gold(self):
        self.predictions['candidate']['needs'].pop();self.persist_predictions()
        with patch.object(self.scorer, 'read_gold', side_effect=AssertionError('gold opened early')):
            with self.assertRaisesRegex(self.scorer.ScoreError, 'missing.*needs'):
                self.evaluate()
        self.assert_no_reports()

    def test_missing_reply_id_rejected_before_gold(self):
        self.predictions['baseline']['reply'].pop();self.persist_predictions()
        with patch.object(self.scorer, 'read_gold', side_effect=AssertionError('gold opened early')):
            with self.assertRaisesRegex(self.scorer.ScoreError, 'missing.*reply'):
                self.evaluate()
        self.assert_no_reports()

    def test_duplicate_and_extra_prediction_ids_rejected(self):
        for extra in [self.predictions['candidate']['needs'][1], dict(id='not-in-control', kind='wish', prediction=None)]:
            with self.subTest(extra=extra['id']):
                self.predictions['candidate']['needs'].append(extra);self.persist_predictions()
                with self.assertRaisesRegex(self.scorer.ScoreError, 'extra.*needs' if extra['id'] == 'not-in-control' else 'duplicate.*needs'):
                    self.evaluate()
                self.predictions['candidate']['needs'].pop()
                self.predpath.with_suffix('.sha256').unlink(missing_ok=True)
        self.assert_no_reports()

    def test_predictions_and_sha_exist_before_any_gold_bytes(self):
        original = Path.read_bytes
        golds = {(self.dataset / name).resolve() for name in ['sealed/control-gold.jsonl', 'sealed/reply-control-gold.jsonl']}
        reads = []
        def observed(path):
            if path.resolve() in golds:
                sidecar = self.predpath.with_suffix('.sha256')
                self.assertTrue(self.predpath.is_file())
                self.assertTrue(sidecar.is_file(), 'prediction digest must be durable before gold bytes')
                expected = hashlib.sha256(original(self.predpath)).hexdigest()
                self.assertEqual(sidecar.read_text().split()[0], expected)
                reads.append(path)
            return original(path)
        with patch.object(Path, 'read_bytes', observed):self.evaluate()
        self.assertGreaterEqual(len(reads), 2)

    def test_gold_hash_tamper_rejected_without_reports(self):
        p = self.dataset / 'sealed/control-gold.jsonl';p.write_bytes(p.read_bytes() + b'\n')
        with patch.object(self.scorer, 'read_gold', side_effect=AssertionError('gold parsed before hashes')):
            with self.assertRaisesRegex(self.scorer.ScoreError, 'hash'):
                self.evaluate()
        self.assert_no_reports()

    def test_freeze_required_and_frozen_files_verified(self):
        self.freezepath.unlink()
        with self.assertRaisesRegex(self.scorer.ScoreError, 'freeze'):self.evaluate()
        self.refresh_lock()
        (self.root / 'frozen/config.json').write_text('{"candidate":"tampered"}')
        with patch.object(self.scorer, 'read_gold', side_effect=AssertionError('gold parsed before freeze')):
            with self.assertRaisesRegex(self.scorer.ScoreError, 'hash'):self.evaluate()
        self.assert_no_reports()

    def test_dataset_lock_digest_required(self):
        lock = json.loads(self.freezepath.read_text());lock['dataset_lock_sha256'] = '0' * 64
        write_json(self.freezepath, lock)
        with self.assertRaisesRegex(self.scorer.ScoreError, 'dataset.*hash'):self.evaluate()
        self.assert_no_reports()

    def test_flat_gold_supported(self):
        self.write_dataset(flat=True);self.refresh_lock()
        self.assertEqual(self.evaluate()['models']['candidate']['needs']['kind_accuracy'], 1.0)

    def test_undefined_metrics_are_json_null_never_nan(self):
        for model in self.predictions.values():
            for row in model['needs']:row['prediction'] = None
            for row in model['reply']:row['action'] = 'operator'
        self.persist_predictions();report = self.evaluate()
        c = report['models']['candidate']
        self.assertIsNone(c['needs']['pairs']['precision'])
        self.assertIsNone(c['ordinary_reply']['precision'])
        self.assertIsNone(c['fixed_notice']['precision'])
        self.assertFalse(c['gates']['target_80_met'])
        text = (self.out / 'control-metrics.json').read_text()
        def reject_nonfinite(value):
            self.fail('Non-finite numeric literal in report: ' + value)
        json.loads(text, parse_constant=reject_nonfinite)

    def test_nan_prediction_input_rejected_before_gold(self):
        self.predictions['candidate']['needs'][0]['confidence'] = float('nan');self.persist_predictions()
        with patch.object(self.scorer, 'read_gold', side_effect=AssertionError('gold opened early')):
            with self.assertRaisesRegex(self.scorer.ScoreError, 'finite|NaN|numeric'):self.evaluate()
        self.assert_no_reports()

    def test_all_false_merge_pairs_are_reported_with_context(self):
        for row in self.predictions['candidate']['needs']:
            if row['id'].startswith('n-04-'):row['prediction'] = 't-0'
        self.persist_predictions();report = self.evaluate()
        pairs = report['models']['candidate']['needs']['pairs']
        self.assertEqual((pairs['tp'], pairs['fp'], pairs['fn']), (240, 16, 0))
        errors = json.loads((self.out / 'errors.json').read_text())['models']['candidate']['group_merges']
        self.assertEqual(len(errors), 16)
        self.assertEqual(len(errors[0]['ids']), 2)
        self.assertIn('expected', errors[0]);self.assertIn('predicted', errors[0]);self.assertIn('context', errors[0])
        self.assertNotIn('Fictitious fixture request', json.dumps(errors))

    def test_kind_error_and_group_split_use_full_denominator(self):
        row = self.predictions['candidate']['needs'][0];row['kind'] = 'bug'
        self.persist_predictions();report = self.evaluate()
        self.assertAlmostEqual(report['models']['candidate']['needs']['kind_accuracy'], 183 / 184)
        self.assertEqual(report['models']['candidate']['needs']['pairs']['fn'], 3)
        errors = json.loads((self.out / 'errors.json').read_text())['models']['candidate']
        self.assertEqual(len(errors['needs_kind']), 1);self.assertEqual(len(errors['group_splits']), 3)
        self.assertAlmostEqual(report['candidate_minus_baseline']['needs_kind_accuracy'], -1 / 184)

    def test_unknown_null_predictions_never_form_group(self):
        for row in self.predictions['candidate']['needs']:
            if row['kind'] == 'unknown':row['prediction'] = 'shared-key-that-must-not-group'
        self.persist_predictions()
        self.assertEqual(self.evaluate()['models']['candidate']['needs']['pairs']['fp'], 0)

    def test_high_scores_do_not_override_one_unsafe_ordinary_reply(self):
        self.predictions['candidate']['reply'][30]['action'] = 'queue_standard';self.persist_predictions()
        metrics = self.evaluate()['models']['candidate']
        self.assertTrue(metrics['gates']['target_80_met'])
        self.assertEqual(metrics['ordinary_reply']['fp'], 1)
        self.assertFalse(metrics['gates']['zero_false_positive_replies'])
        self.assertFalse(metrics['gates']['auto_enable'])

    def test_wrong_standard_action_on_fix_is_not_added_to_ordinary_denominator(self):
        self.predictions['candidate']['reply'][120]['action'] = 'queue_standard';self.persist_predictions()
        metrics = self.evaluate()['models']['candidate']
        self.assertEqual(metrics['ordinary_reply']['population'], 120)
        self.assertEqual(metrics['ordinary_reply']['fp'], 0)
        self.assertEqual(metrics['fixed_notice']['fn'], 1)
        self.assertEqual(metrics['cross_path_auto_actions'], 1)
        self.assertEqual(metrics['action_accuracy_all']['correct'], 131)
        self.assertFalse(metrics['gates']['zero_false_positive_replies'])

    def test_unsafe_fix_notice_is_separate_from_ordinary_false_positive(self):
        self.predictions['candidate']['reply'][126]['action'] = 'queue_fixed_notice';self.persist_predictions()
        metrics = self.evaluate()['models']['candidate']
        self.assertEqual(metrics['ordinary_reply']['fp'], 0)
        self.assertEqual(metrics['fixed_notice']['fp'], 1)
        self.assertFalse(metrics['gates']['zero_false_positive_replies'])

    def test_prediction_digest_mismatch_does_not_reseal(self):
        self.predpath.with_suffix('.sha256').write_text('0' * 64 + '  control-predictions-only.json\n')
        with self.assertRaisesRegex(self.scorer.ScoreError, 'prediction.*hash'):self.evaluate()
        self.assert_no_reports()

    def test_prediction_change_during_gold_read_does_not_publish(self):
        original = self.scorer.read_gold
        changed = False
        def read_then_change(path):
            nonlocal changed
            value = original(path)
            if not changed:
                self.predpath.write_bytes(self.predpath.read_bytes() + b' ');changed = True
            return value
        with patch.object(self.scorer, 'read_gold', read_then_change):
            with self.assertRaisesRegex(self.scorer.ScoreError, 'prediction.*changed'):self.evaluate()
        self.assert_no_reports()


    def test_identical_predictions_for_retransmitted_ids_are_deduplicated(self):
        for model in self.predictions.values():
            for i in [0, 4, 8, 12]:model['needs'].append(copy.deepcopy(model['needs'][i]))
        self.persist_predictions()
        result = self.evaluate()['models']['candidate']
        self.assertEqual(result['needs']['unique_messages'], 184)
        self.assertEqual(result['needs']['pairs']['tp'], 240)

    def test_conflicting_prediction_for_retransmission_is_rejected(self):
        extra = copy.deepcopy(self.predictions['candidate']['needs'][0]);extra['prediction'] = 'conflicting-key'
        self.predictions['candidate']['needs'].append(extra);self.persist_predictions()
        with self.assertRaisesRegex(self.scorer.ScoreError, 'conflicting.*prediction'):self.evaluate()
        self.assert_no_reports()

    def test_cli_relative_paths_are_based_on_explicit_root(self):
        completed = subprocess.run([sys.executable, '-B', str(MODULE), '--root', str(self.root),
                                    '--freeze', 'reports/quality-v5c/CONFIG-FREEZE.json'],
                                   cwd=self.root.parent, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)['status'], 'scored')
        self.assertTrue((self.out / 'control-metrics.json').is_file())

    def test_structured_prediction_tuples_preserve_product_and_kind(self):
        inputs = {row['id']: row for row in self.unique_needs}
        for row in self.predictions['baseline']['needs']:
            row['prediction'] = dict(product=inputs[row['id']]['product'], kind=row['kind'], key=row['prediction'])
        for row in self.predictions['candidate']['needs']:
            row['prediction'] = [inputs[row['id']]['product'], row['kind'], row['prediction']]
        self.predictions['candidate']['needs'][0]['prediction'][0] = 'different-product'
        self.persist_predictions();result = self.evaluate()['models']
        self.assertEqual(result['baseline']['needs']['pairs']['fn'], 0)
        self.assertEqual(result['candidate']['needs']['pairs']['fn'], 3)
        self.assertEqual(result['candidate']['needs']['kind_accuracy'], 1.0)

    def test_paths_can_supply_freeze_when_flag_is_omitted(self):
        self.predictions['paths'] = dict(freeze='reports/quality-v5c/CONFIG-FREEZE.json')
        self.persist_predictions()
        result = self.scorer.evaluate(self.predpath, root=self.root, output_dir=self.out)
        self.assertEqual(result['models']['candidate']['needs']['kind_correct'], 184)

    def test_missing_persisted_predictions_does_not_open_gold(self):
        self.predpath.unlink()
        with patch.object(self.scorer, 'read_gold', side_effect=AssertionError('gold opened early')):
            with self.assertRaisesRegex(self.scorer.ScoreError, 'prediction.*exist'):self.evaluate()
        self.assert_no_reports()


if __name__ == '__main__':
    unittest.main()
