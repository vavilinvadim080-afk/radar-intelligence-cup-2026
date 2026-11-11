#!/usr/bin/env python3
"""Independent, offline v5c baseline/candidate scorer; no model or service imports.

Run only after explicit authorization to open control gold:
  python3 -B scripts/score_service_v5c.py --freeze <CONFIG-FREEZE.json>

The predictions-only JSON must already exist. Its required top-level models are
baseline/candidate, each containing needs/reply lists. Needs rows require id,
kind, prediction; reply rows require id, action. A prediction is null, a cluster
key, {key, optional product/kind}, or [product, kind, key]. Unknown kinds and null
keys never form groups. Optional top-level paths supplies project-relative
needs_inputs, needs_gold, reply_inputs, reply_gold, dataset_lock, freeze paths.
Identical need predictions for repeated delivery IDs are deduplicated; conflicting
outputs are rejected. No paths are read until the predictions artifact exists.

Freeze JSON requires files={project-relative-path: sha256} and
 dataset_lock_sha256. The dataset lock's files are relative to its directory.
All hashes, full ID coverage and the predictions SHA sidecar precede gold
parsing. This script prepares local reports; it never enables or sends replies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import tempfile

MODEL_NAMES = ('baseline', 'candidate')
ACTIONS = {'queue_standard', 'queue_fixed_notice', 'operator', 'ignore_duplicate',
           'needs_approval', 'needs_fix_approval'}
DEFAULT_PATHS = {
    'needs_inputs': 'evaluation/v5c/control-inputs.jsonl',
    'needs_gold': 'evaluation/v5c/sealed/control-gold.jsonl',
    'reply_inputs': 'evaluation/v5c/reply-control-inputs.jsonl',
    'reply_gold': 'evaluation/v5c/sealed/reply-control-gold.jsonl',
    'dataset_lock': 'evaluation/v5c/CONTROL-LOCK.json',
}


class ScoreError(ValueError):
    """Invalid or unsealed evaluation artifact; messages never contain gold text."""


def require(condition, message):
    if not condition:
        raise ScoreError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bytes(path, role):
    try:
        return path.read_bytes()
    except OSError as error:
        raise ScoreError(f'{role} file must exist and be readable') from error


def unique_keys(items):
    obj = {}
    for key, value in items:
        require(key not in obj, 'duplicate JSON object key')
        obj[key] = value
    return obj


def parse_json(raw, role):
    def invalid_number(_value):
        raise ScoreError('non-finite numeric JSON value (NaN or Infinity)')
    try:
        value = json.loads(raw, object_pairs_hook=unique_keys, parse_constant=invalid_number)
    except (ValueError, UnicodeError) as error:
        if isinstance(error, ScoreError):
            raise
        raise ScoreError(f'invalid {role} JSON') from error
    # JSON exponent overflow (1e999) is not handled by parse_constant.
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'non-finite numeric JSON value')
        elif isinstance(item, dict):
            for sub in item.values():
                finite(sub)
        elif isinstance(item, list):
            for sub in item:
                finite(sub)
    finite(value)
    return value


def read_jsonl(path, role):
    raw = read_bytes(path, role)
    try:
        lines = raw.decode('utf-8').splitlines()
    except UnicodeError as error:
        raise ScoreError(f'invalid {role} UTF-8') from error
    require(lines and all(line.strip() for line in lines), f'invalid {role} JSONL rows')
    rows = [parse_json(line, role) for line in lines]
    require(all(isinstance(row, dict) for row in rows), f'invalid {role} rows')
    return rows


def read_gold(path):
    """The sole gold parser; called only after persisted predictions and hash checks."""
    return read_jsonl(path, 'gold')


def relative_path(base, value, role):
    require(isinstance(value, str) and value, f'invalid {role} path')
    relative = Path(value)
    require(not relative.is_absolute() and '..' not in relative.parts, f'{role} path must be relative')
    resolved = (base / relative).resolve()
    require(resolved.is_relative_to(base.resolve()), f'{role} path escapes its root')
    return resolved


def valid_sha(value):
    return isinstance(value, str) and bool(re.fullmatch('[0-9a-f]{64}', value))


def verify_files(base, files, role):
    require(isinstance(files, dict) and files, f'{role} files hash map is required')
    verified = {}
    for relative, expected in files.items():
        require(valid_sha(expected), f'invalid {role} file hash')
        path = relative_path(base, relative, role)
        actual = digest(read_bytes(path, role))
        require(actual == expected, f'{role} file hash mismatch')
        verified[path] = expected
    return verified


def atomic_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_predictions(path, expected):
    """Never overwrites an existing prediction commitment with a different digest."""
    sidecar = path.with_suffix('.sha256')
    try:
        with sidecar.open('x', encoding='ascii') as file:
            file.write(f'{expected}  {path.name}\n')
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        pass
    try:
        fields = sidecar.read_text(encoding='ascii').split()
    except (OSError, UnicodeError) as error:
        raise ScoreError('prediction hash sidecar is unreadable') from error
    require(fields and fields[0] == expected, 'prediction hash sidecar mismatch; refusing to reseal')
    require(digest(read_bytes(path, 'prediction')) == expected, 'prediction file changed before gold')
    return sidecar


def created_key(row):
    try:
        value = datetime.fromisoformat(row['created_at'])
    except (KeyError, TypeError, ValueError) as error:
        raise ScoreError('needs created_at must be an ISO timestamp') from error
    require(value.tzinfo is not None, 'needs created_at needs a timezone')
    return value, row['id']


def deduplicate_needs(rows):
    """First in created_at/id order per product/source_id, never per author or text."""
    seen = set()
    unique = []
    for row in sorted(rows, key=created_key):
        require(all(isinstance(row.get(k), str) and row[k] for k in ['id', 'product', 'source_id']),
                'needs identity fields are required')
        identity = row['product'], row['source_id']
        if identity not in seen:
            seen.add(identity)
            unique.append(row)
    return unique


def index_predictions(rows, ids, model, task, delivery_counts=None):
    require(isinstance(rows, list), f'{model} {task} predictions must be a list')
    indexed = {}
    occurrences = Counter()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('id'), str), f'invalid {model} {task} prediction row')
        occurrences[row['id']] += 1
        if row['id'] in indexed:
            require(task == 'needs' and delivery_counts is not None
                    and occurrences[row['id']] <= delivery_counts.get(row['id'], 1),
                    f'duplicate {model} {task} prediction ID')
            require(all(row.get(k) == indexed[row['id']].get(k) for k in ['kind', 'prediction']),
                    f'conflicting {model} needs prediction for retransmission')
        else:
            indexed[row['id']] = row
    require(not ids - indexed.keys(), f'missing {model} {task} prediction IDs')
    require(not indexed.keys() - ids, f'extra {model} {task} prediction IDs')
    for row in rows:
        if task == 'needs':
            require(isinstance(row.get('kind'), str) and row['kind'] and 'prediction' in row,
                    f'invalid {model} needs prediction fields')
        else:
            require(isinstance(row.get('action'), str) and row['action'], f'invalid {model} reply prediction action')
    return indexed


def prediction_group(row, context):
    kind = row['kind']
    value = row['prediction']
    if kind == 'unknown' or value is None:
        return None
    product = context['product']
    if isinstance(value, dict):
        require('key' in value, 'needs prediction object requires key')
        product, kind, key = value.get('product', product), value.get('kind', kind), value['key']
    elif isinstance(value, list):
        require(len(value) == 3, 'needs prediction tuple must contain product/kind/key')
        product, kind, key = value
    else:
        key = value
    if kind == 'unknown' or key is None or key == '':
        return None
    require(isinstance(product, str) and product and isinstance(kind, str) and kind,
            'invalid needs prediction group context')
    require(type(key) in {str, int}, 'needs prediction key must be a string, integer or null')
    # Preserve integer/string distinction without changing a model's actual cluster relation.
    return product, kind, (type(key).__name__, key)


def unwrap_gold(row):
    require(isinstance(row, dict) and isinstance(row.get('id'), str), 'gold row requires ID')
    payload = row.get('gold', row)
    require(isinstance(payload, dict), 'gold payload must be an object')
    return row['id'], payload


def index_gold(rows, ids, task):
    indexed = {}
    for row in rows:
        ident, payload = unwrap_gold(row)
        if ident in indexed:
            require(task == 'needs', 'duplicate reply gold ID')
            fields = ('product', 'kind', 'topic_id', 'direction')
            require(all(indexed[ident].get(k) == payload.get(k) for k in fields), 'conflicting duplicate needs gold')
        else:
            indexed[ident] = payload
    require(set(indexed) == ids, f'{task} gold ID coverage mismatch')
    return indexed


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def binary_metrics(tp, fp, fn, tn):
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None else ratio(2 * tp, 2 * tp + fp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, positives=tp + fn, negatives=fp + tn,
                predicted_positives=tp + fp, precision=precision, recall=recall, f1=f1)


def needs_context(row):
    return {k: row[k] for k in ['product', 'source_id', 'author_id', 'created_at'] if k in row}


def expected_group(row, gold):
    require(isinstance(gold.get('kind'), str) and 'topic_id' in gold, 'needs gold kind/topic_id required')
    if gold['kind'] == 'unknown' or gold['topic_id'] is None:
        return None
    require(isinstance(gold['topic_id'], str) and gold['topic_id'], 'invalid gold topic ID')
    require(gold.get('product', row['product']) == row['product'], 'needs gold product mismatch')
    return row['product'], gold['kind'], ('str', gold['topic_id'])


def score_needs(rows, gold, predicted):
    truth, groups = {}, {}
    kind_errors, merges, splits = [], [], []
    confusion = defaultdict(Counter)
    correct = 0
    for row in rows:
        ident = row['id']
        g, p = gold[ident], predicted[ident]
        truth[ident], groups[ident] = expected_group(row, g), prediction_group(p, row)
        confusion[g['kind']][p['kind']] += 1
        if p['kind'] == g['kind']:
            correct += 1
        else:
            kind_errors.append(dict(id=ident, expected=dict(kind=g['kind'], topic_id=g['topic_id']),
                                    predicted=dict(kind=p['kind'], group=groups[ident]), context=needs_context(row)))
    tp = fp = fn = tn = 0
    for left, right in combinations(rows, 2):
        a, b = left['id'], right['id']
        expected_same = truth[a] is not None and truth[a] == truth[b]
        predicted_same = groups[a] is not None and groups[a] == groups[b]
        if expected_same and predicted_same:
            tp += 1
        elif predicted_same:
            fp += 1
        elif expected_same:
            fn += 1
        else:
            tn += 1
        if expected_same != predicted_same:
            entry = dict(ids=[a, b], expected=dict(same_topic=expected_same, groups=[truth[a], truth[b]]),
                         predicted=dict(same_group=predicted_same, groups=[groups[a], groups[b]]),
                         context=[needs_context(left), needs_context(right)])
            (merges if predicted_same else splits).append(entry)
    require(tp + fn == 240, 'gold true-pair population must be 240')
    metrics = dict(deliveries=188, unique_messages=len(rows), duplicate_deliveries=188 - len(rows),
                   kind_correct=correct, kind_accuracy=ratio(correct, len(rows)),
                   kind_confusion={k: dict(v) for k, v in confusion.items()}, pairs=binary_metrics(tp, fp, fn, tn))
    return metrics, dict(needs_kind=kind_errors, group_merges=merges, group_splits=splits)


def reply_context(row, gold):
    context = {k: row.get(k) for k in ['event', 'product', 'kind', 'approved', 'state', 'fix_confirmed',
                                     'human_closed', 'fixed_notice_approved']}
    context['query'] = {k: row.get('query', {}).get(k) for k in ['id', 'product', 'consent']}
    context['references'] = [{k: r.get(k) for k in ['id', 'product', 'kind', 'source_verified']}
                             for r in row.get('references', [])]
    context['expected_relation'] = {k: gold.get(k) for k in ['same_topic', 'reference_topic_id', 'query_kind', 'query_direction']}
    context['already_seen'] = row.get('query', {}).get('id') in row.get('seen_message_ids', [])
    return context


def score_replies(rows, gold, predicted):
    counts = {'ordinary_reply': Counter(), 'fixed_notice': Counter()}
    populations = Counter()
    errors = []
    confusion = defaultdict(Counter)
    correct = cross_path = 0
    for row in rows:
        ident = row['id']
        g, p = gold[ident], predicted[ident]
        expected, action = g.get('safe_action'), p['action']
        require(expected in ACTIONS, 'invalid reply gold safe_action')
        event = row.get('event')
        require(event in {'incoming_ticket', 'fixed_notice_candidate'}, 'invalid reply event')
        fixed = event == 'fixed_notice_candidate'
        require(not fixed or expected != 'queue_standard', 'fixed notice cannot be an ordinary gold positive')
        require(fixed or expected != 'queue_fixed_notice', 'ordinary event cannot be a fixed-notice gold positive')
        if 'eligibility' in g:
            require(type(g['eligibility']) is bool and g['eligibility'] == (expected == 'queue_standard'),
                    'gold eligibility must refer only to ordinary replies')
        task, positive_action = ('fixed_notice', 'queue_fixed_notice') if fixed else ('ordinary_reply', 'queue_standard')
        populations[task] += 1
        positive, guessed = expected == positive_action, action == positive_action
        counts[task]['tp' if positive and guessed else 'fp' if guessed else 'fn' if positive else 'tn'] += 1
        if (fixed and action == 'queue_standard') or (not fixed and action == 'queue_fixed_notice'):
            cross_path += 1
        confusion[expected][action] += 1
        if expected == action:
            correct += 1
        else:
            errors.append(dict(id=ident, expected=dict(action=expected, eligibility=expected == 'queue_standard', reason=g.get('reason')),
                               predicted=dict(action=action), context=reply_context(row, g)))
    require(populations == Counter(ordinary_reply=120, fixed_notice=12), 'reply populations must be ordinary 120 and fix 12')
    metrics = {}
    for task in counts:
        c = counts[task]
        metrics[task] = dict(population=populations[task], **binary_metrics(c['tp'], c['fp'], c['fn'], c['tn']))
    require(metrics['ordinary_reply']['positives'] == 30 and metrics['fixed_notice']['positives'] == 6,
            'reply gold positives must be ordinary 30 and fix 6')
    metrics['action_accuracy_all'] = dict(population=len(rows), correct=correct, accuracy=ratio(correct, len(rows)),
                                         confusion={k: dict(v) for k, v in confusion.items()})
    metrics['cross_path_auto_actions'] = cross_path
    return metrics, errors


def at_least_80(value):
    return value is not None and value >= 0.80


def model_metrics(needs, replies, ngold, rgold, predictions):
    nm, errors = score_needs(needs, ngold, predictions['needs'])
    rm, errors['reply_actions'] = score_replies(replies, rgold, predictions['reply'])
    target_checks = dict(needs_kind_accuracy=at_least_80(nm['kind_accuracy']),
                         pair_precision=at_least_80(nm['pairs']['precision']), pair_recall=at_least_80(nm['pairs']['recall']),
                         ordinary_precision=at_least_80(rm['ordinary_reply']['precision']),
                         ordinary_recall=at_least_80(rm['ordinary_reply']['recall']))
    zero_ordinary = rm['ordinary_reply']['fp'] == 0
    zero_fix = rm['fixed_notice']['fp'] == 0
    no_cross_path = rm['cross_path_auto_actions'] == 0
    gates = dict(target_80_met=all(target_checks.values()), target_checks=target_checks,
                 zero_ordinary_false_positives=zero_ordinary, zero_fixed_notice_false_positives=zero_fix,
                 no_cross_path_auto_actions=no_cross_path,
                 zero_false_positive_replies=zero_ordinary and zero_fix and no_cross_path, auto_enable=False)
    return dict(needs=nm, **rm, gates=gates), errors


def evaluate(predictions_path, freeze_path=None, *, root=None, output_dir=None):
    """Score persisted predictions; callers explicitly authorize gold access by invoking."""
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    predictions_path = Path(predictions_path)
    predictions_path = (predictions_path if predictions_path.is_absolute() else root / predictions_path).resolve()
    # Fail before consulting input/gold paths if a predictions-only artifact is absent.
    raw = read_bytes(predictions_path, 'prediction')
    predictions = parse_json(raw, 'prediction')
    require(isinstance(predictions, dict) and set(MODEL_NAMES) <= predictions.keys(), 'baseline and candidate predictions required')
    require(not predictions.keys() - set(MODEL_NAMES) - {'paths'}, 'unexpected predictions top-level fields')
    paths = predictions.get('paths', {})
    require(isinstance(paths, dict) and not paths.keys() - set(DEFAULT_PATHS) - {'freeze'}, 'invalid prediction paths')
    resolved = {k: relative_path(root, paths.get(k, default), k) for k, default in DEFAULT_PATHS.items()}
    if freeze_path is None:
        require('freeze' in paths, 'freeze JSON path is required')
        freeze_path = relative_path(root, paths['freeze'], 'freeze')
    freeze_path = Path(freeze_path)
    freeze_path = (freeze_path if freeze_path.is_absolute() else root / freeze_path).resolve()
    freeze_raw = read_bytes(freeze_path, 'freeze')
    freeze = parse_json(freeze_raw, 'freeze')
    require(isinstance(freeze, dict) and valid_sha(freeze.get('dataset_lock_sha256')), 'freeze dataset_lock_sha256 is required')
    require(isinstance(freeze.get('files'), dict) and freeze['files'], 'freeze files hash map is required')
    lock_raw = read_bytes(resolved['dataset_lock'], 'dataset lock')
    require(digest(lock_raw) == freeze['dataset_lock_sha256'], 'dataset lock hash mismatch')
    lock = parse_json(lock_raw, 'dataset lock')
    require(isinstance(lock, dict) and lock.get('algorithm', 'sha256') == 'sha256', 'invalid dataset lock algorithm')
    require(isinstance(lock.get('files'), dict) and lock['files'], 'dataset lock files hash map required')

    # No gold bytes (including hash verification) are opened before this durable commitment.
    prediction_sha = digest(raw)
    sidecar = seal_predictions(predictions_path, prediction_sha)
    nrows = read_jsonl(resolved['needs_inputs'], 'needs input')
    rrows = read_jsonl(resolved['reply_inputs'], 'reply input')
    require(len(nrows) == 188 and len(rrows) == 132, 'control populations must be 188 needs deliveries and 132 replies')
    unique_needs = deduplicate_needs(nrows)
    require(len(unique_needs) == 184, 'deduplicated needs population must be 184')
    nids, rids = {row['id'] for row in nrows}, {row.get('id') for row in rrows}
    require(all(isinstance(ident, str) and ident for ident in rids) and len(rids) == 132, 'reply IDs must be unique')
    indexed = {}
    for model in MODEL_NAMES:
        body = predictions[model]
        require(isinstance(body, dict) and set(body) == {'needs', 'reply'}, f'invalid {model} prediction lists')
        indexed[model] = {task: index_predictions(body[task], ids, model, task, Counter(r['id'] for r in nrows) if task == 'needs' else None)
                          for task, ids in [('needs', nids), ('reply', rids)]}
        # Validate group payloads now, before gold access, including suppressed deliveries.
        for row in nrows:
            prediction_group(indexed[model]['needs'][row['id']], row)
    frozen_files = verify_files(root, freeze['files'], 'freeze')
    dataset_files = verify_files(resolved['dataset_lock'].parent, lock['files'], 'dataset')
    for role in ['needs_inputs', 'needs_gold', 'reply_inputs', 'reply_gold']:
        require(resolved[role] in dataset_files, f'{role} must be covered by dataset lock')
    require(digest(read_bytes(predictions_path, 'prediction')) == prediction_sha, 'prediction file changed before gold')

    ngrows = read_gold(resolved['needs_gold'])
    rgrows = read_gold(resolved['reply_gold'])
    require(len(ngrows) == 188 and len(rgrows) == 132, 'gold delivery populations must match full inputs')
    ngold, rgold = index_gold(ngrows, nids, 'needs'), index_gold(rgrows, rids, 'reply')
    results, errors = {}, {}
    for model in MODEL_NAMES:
        results[model], errors[model] = model_metrics(unique_needs, rrows, ngold, rgold, indexed[model])
    extract = lambda m: dict(needs_kind_accuracy=m['needs']['kind_accuracy'], pair_precision=m['needs']['pairs']['precision'],
                             pair_recall=m['needs']['pairs']['recall'], ordinary_precision=m['ordinary_reply']['precision'],
                             ordinary_recall=m['ordinary_reply']['recall'], fixed_precision=m['fixed_notice']['precision'],
                             fixed_recall=m['fixed_notice']['recall'], action_accuracy_all=m['action_accuracy_all']['accuracy'])
    base, candidate = extract(results['baseline']), extract(results['candidate'])
    delta = {k: candidate[k] - base[k] if candidate[k] is not None and base[k] is not None else None for k in base}
    provenance = dict(predictions_path=str(predictions_path), predictions_sha256=prediction_sha,
                      predictions_sha_path=str(sidecar), freeze_path=str(freeze_path), freeze_sha256=digest(freeze_raw),
                      dataset_lock_path=str(resolved['dataset_lock']), dataset_lock_sha256=digest(lock_raw),
                      frozen_files_verified=len(frozen_files), dataset_files_verified=len(dataset_files))
    report = dict(version='v5c', provenance=provenance, models=results, candidate_minus_baseline=delta,
                  metric_policy=dict(undefined='null; never NaN or an invented perfect score',
                                     deduplication='product/source_id, sorted by created_at then id',
                                     ordinary='120 incoming_ticket rows; positive action queue_standard',
                                     fixed='12 fixed_notice_candidate rows; positive action queue_fixed_notice',
                                     target='0.80 for kind accuracy and grouping/ordinary precision and recall; separate from zero-FP gate',
                                     release='Local evaluation only; auto_enable is always false'))
    # Detect mutation during gold parsing/scoring before publishing gold-derived artifacts.
    require(digest(read_bytes(predictions_path, 'prediction')) == prediction_sha, 'prediction file changed during scoring')
    require(read_bytes(freeze_path, 'freeze') == freeze_raw, 'freeze changed during scoring')
    require(read_bytes(resolved['dataset_lock'], 'dataset lock') == lock_raw, 'dataset lock changed during scoring')
    for files, role in [(frozen_files, 'freeze'), (dataset_files, 'dataset')]:
        for path, expected in files.items():
            require(digest(read_bytes(path, role)) == expected, f'{role} file hash changed during scoring')
    output_dir = Path(output_dir or root / 'reports/quality-v5c')
    output_dir = (output_dir if output_dir.is_absolute() else root / output_dir).resolve()
    atomic_json(output_dir / 'errors.json', dict(version='v5c', provenance=provenance, models=errors))
    atomic_json(output_dir / 'control-metrics.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=Path('reports/quality-v5c/control-predictions-only.json'))
    parser.add_argument('--freeze', type=Path, help='Mandatory freeze JSON, or paths.freeze in predictions JSON')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args.predictions, args.freeze, root=args.root, output_dir=args.output_dir)
    except (ScoreError, OSError) as error:
        parser.exit(2, f'Evaluation refused: {error}\n')
    print(json.dumps(dict(status='scored', predictions_sha256=report['provenance']['predictions_sha256'],
                          output_dir=str(args.output_dir or args.root / 'reports/quality-v5c')), ensure_ascii=False))


if __name__ == '__main__':
    main()
