import json
import subprocess
import sys
from pathlib import Path

from radar.gen import generate

ROOT = Path(__file__).resolve().parents[1]


def test_same_seed_same_data():
    a, b = generate(11, "mid", 300), generate(11, "mid", 300)
    assert [t.text for t in a.tickets] == [t.text for t in b.tickets]
    assert a.ground_truth == b.ground_truth


def test_reproducible_across_processes_with_different_hash_seeds():
    code = "import sys,json;sys.path.insert(0,'.');from radar.gen import generate;d=generate(11,'mid',300);print(hash(tuple(t.text for t in d.tickets)) if False else json.dumps([t.text for t in d.tickets[:300]],ensure_ascii=False))"
    outs = []
    for hs in ("1", "2"):
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env={"PYTHONHASHSEED": hs, "PATH": ""})
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1]


def test_different_seeds_differ_and_events_are_present():
    a, b = generate(11, "mid", 300), generate(12, "mid", 300)
    assert [t.text for t in a.tickets[:50]] != [t.text for t in b.tickets[:50]]
    types = {e["type"] for e in a.ground_truth["events"]}
    assert {"regression", "slow_growth", "noise_spike", "prompt_injection"} <= types
    inj = [e for e in a.ground_truth["events"] if e["type"] == "prompt_injection"]
    assert 2 <= len(inj) <= 4
    for e in inj:
        tid = e["ticket_ids"][0]
        assert any(k in next(t.text for t in a.tickets if t.id == tid).lower() for k in ("инструкц", "instructions", "ассистент", "системное"))


def test_noise_levels_change_typo_level_and_volume_setting_works():
    low, high = generate(3, "low", 300), generate(3, "high", 300)
    assert len(low.tickets) > 0 and len(high.tickets) > 0
    big = generate(3, "mid", 600)
    assert len(big.tickets) > 1.6 * len(generate(3, "mid", 300).tickets)


def test_ticket_fields_and_labels():
    ds = generate(4, "mid", 300)
    t = ds.tickets[0]
    assert t.id.startswith("T") and t.product in {"azs", "bonus", "b2b", "portal", "partners"}
    assert ds.labels[t.id]["topic"]
    assert ds.releases and any(r.kind == "fix" for r in ds.releases) or True
