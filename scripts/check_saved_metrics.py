"""Verify published source hashes and recount saved predictions; no model weights needed."""
from collections import Counter
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check_snapshot():
    frozen=json.loads((ROOT/'reports/quality-v5b/CONFIG-FREEZE.json').read_text())
    for name,expected in frozen['files'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Source snapshot changed: '+name)
    lock_path=ROOT/'evaluation/v5b/CONTROL-LOCK.json'
    if sha(lock_path)!=frozen['dataset_lock_sha256']:raise ValueError('Dataset lock changed')
    lock=json.loads(lock_path.read_text())
    for name in ['control-inputs.jsonl','control-gold.jsonl','reply-control-inputs.jsonl','reply-control-gold.jsonl']:
        entry=lock['files'][name];expected=entry['sha256'] if isinstance(entry,dict) else entry
        if sha(lock_path.parent/name)!=expected:raise ValueError('Published control changed: '+name)
    return frozen
def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]

def choose2(n):
    return n * (n - 1) // 2

def main():
    frozen = check_snapshot()
    out = ROOT / 'reports/quality-v5b'
    predictions = json.loads((out / 'control-predictions-only.json').read_text())
    result = json.loads((ROOT / 'data/quality-v5b.json').read_text())
    gold = {r['id']: r['gold'] for r in read(ROOT / 'evaluation/v5b/control-gold.jsonl')}
    reply = {r['id']: r['gold'] for r in read(ROOT / 'evaluation/v5b/reply-control-gold.jsonl')}
    checks = {}
    for scope in ['needs', 'embedding_only']:
        for backend, rows in predictions[scope].items():
            actual = Counter()
            got = Counter()
            joint = Counter()
            for row in rows:
                g = gold[row['id']]['topic_id']
                p = tuple(row['prediction']) if row['prediction'] else None
                if g is not None:
                    actual[g] += 1
                if p is not None:
                    got[p] += 1
                if g is not None and p is not None:
                    joint[g, p] += 1
            tp = sum(map(choose2, joint.values()))
            fp = sum(map(choose2, got.values())) - tp
            fn = sum(map(choose2, actual.values())) - tp
            m = result[scope][backend]['pair_metrics']
            assert (tp, fp, fn) == (m['tp'], m['fp'], m['fn'])
            assert m['precision'] == (tp / (tp + fp) if tp + fp else None)
            assert m['recall'] == (tp / (tp + fn) if tp + fn else None)
            assert len(rows) == 116 and tp + fn == 144
            assert result[scope][backend]['kind_accuracy'] == sum((r['kind'] == gold[r['id']]['kind'] for r in rows)) / 116
            checks[scope + ':' + backend] = dict(tp=tp, fp=fp, fn=fn, n=116)
    for backend, rows in predictions['autoreply'].items():
        true = {r['id'] for r in rows if reply[r['id']]['safe_action'] == 'queue_standard'}
        queued = {r['id'] for r in rows if r['action'] == 'queue_standard'}
        tp = len(true & queued)
        fp = len(queued - true)
        fn = len(true - queued)
        m = result['autoreply'][backend]
        assert len(rows) == 80 and len(true) == 20
        assert (tp, fp, fn) == (m['tp'], m['fp'], m['fn'])
        assert m['precision'] == (tp / (tp + fp) if tp + fp else None) and m['recall'] == tp / 20
        checks['autoreply:' + backend] = dict(tp=tp, fp=fp, fn=fn, n=80, positives=20)
    assert result['selected_before_control'] == frozen['selected_backend'] == 'qwen'
    assert result['gate']['auto_enable'] is False
    report = dict(status='pass', method='Contingency counters and set intersections, no model inference', checks=checks, result_sha256=sha(ROOT / 'data/quality-v5b.json'), predictions_sha256=sha(out / 'control-predictions-only.json'), freeze_sha256=sha(out / 'CONFIG-FREEZE.json'))
    (out / 'SAVED-METRICS-CHECK.json').write_text(json.dumps(report, indent=2))
    print('PASS', len(checks), 'independent recounts; source snapshot intact; model weights not loaded')

if __name__=='__main__':main()
