"""v12 (2026-10-29): панель «Качество на слепых seeds» берёт числа из shared/metrics-v3.json и shared/metrics-agent-v2-schema.json,
подписи «тревога z=...» на графике не налезают друг на друга.
Тесты читают исходник dashboard.html и файлы shared/; раскладку подписей гоняют в node (если он есть), глазами проверено скриншотом
(presentation/screens/v12/dashboard-*.png). Новых зависимостей нет."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

os.environ["RADAR_MODE"] = "template"

import pytest
from fastapi.testclient import TestClient

from app import server

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "app" / "static" / "dashboard.html").read_text(encoding="utf-8")
V3 = json.loads((ROOT / "shared" / "metrics-v3.json").read_text(encoding="utf-8"))
AG2 = json.loads((ROOT / "shared" / "metrics-agent-v2-schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


def test_api_metrics_summary_has_v3_detector_from_json(client):
    s = client.get("/api/metrics").json()["summary"]["detector_v3"]
    t = V3["final"][0]["table"]["none"]
    assert s["strict_precision"] == t["strict"]["precision"] and s["strict_recall"] == t["strict"]["recall"]
    assert s["work_recall"] == t["work"]["recall"] and s["on_target_share"] == t["target"]["page_share"]
    assert s["false_pages_per_week"] == t["false_pages_per_week"] and s["ttc_median_min"] == t["strict"]["ttc_median_min"]
    assert s["seeds"] == V3["meta"]["final_seeds"]
    # главное значение: 60 / 68 / 80 / 73 / 0,21 / 99 (то, что в REPORT-v3.md, строка «без проверки»)
    assert (round(s["strict_precision"] * 100), round(s["strict_recall"] * 100), round(s["work_recall"] * 100), round(s["on_target_share"] * 100), round(s["false_pages_per_week"], 2), round(s["ttc_median_min"])) == (60, 68, 80, 73, 0.21, 99)


def test_api_metrics_summary_has_agent_v2_schema_from_json(client):
    a = client.get("/api/metrics").json()["summary"]["agent_v2"]
    for k in ("n_cards", "first_try_valid_share", "claims_valid_share", "template_fallback_share", "steps_mean", "latency_s_median", "n_injections", "injection_resisted_share"):
        assert a[k] == AG2[k], k
    assert (a["first_try_valid_share"], a["template_fallback_share"], a["steps_mean"], a["n_injections"], a["injection_resisted_share"]) == (1.0, 0.0, 2, 24, 1.0)


def _fn(name):
    m = re.search(r"/\*LAYOUT-BEGIN\*/(.*?)/\*LAYOUT-END\*/", HTML, re.S)
    assert m, "в dashboard.html нет блока /*LAYOUT-BEGIN*/ ... /*LAYOUT-END*/ с раскладкой подписей"
    return m.group(1)


def test_metrics_block_reads_v3_and_agent_v2_not_old_detector_numbers():
    js = HTML[HTML.index("async function loadMetrics"):HTML.index("/* ---------- обновление")]
    assert "detector_v3" in js and "agent_v2" in js
    assert "s.detector.precision" not in js and "s.detector.recall" not in js and "s.detector.ttc_median_min" not in js, "старые числа детектора на панели"
    for phrase in ("строгая", "в рабочих часах", "по делу", "ложных подъёмов", "seeds 401"):
        assert phrase in HTML, phrase
    assert "metrics-v3.json" in HTML and "metrics-agent-v2" in HTML


@pytest.mark.skipif(shutil.which("node") is None, reason="нужен node")
def test_alert_labels_do_not_overlap_and_stay_inside():
    code = _fn("layoutAlertLabels") + """
const cases = [
  [[{x: 450, w: 98}, {x: 468, w: 84}], 40, 660],            // два соседних срабатывания (z=5.1 и z=6)
  [[{x: 300, w: 98}, {x: 305, w: 98}, {x: 311, w: 98}], 40, 660],
  [[{x: 60, w: 98}, {x: 640, w: 98}], 40, 660],             // у краёв
  [[{x: 52, w: 98}], 40, 660]];
const out = cases.map(([items, L, Rr]) => layoutAlertLabels(items, L, Rr, 20));
console.log(JSON.stringify({ cases: cases.map(c => c[0]), out }));
"""
    r = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    for items, res in zip(d["cases"], d["out"]):
        assert len(res) == len(items)
        boxes = []
        for it, p in zip(items, res):
            left = p["x"] - it["w"] if p["anchor"] == "end" else p["x"]
            assert left >= 40 - 0.5 and left + it["w"] <= 660 + 0.5, (it, p)
            boxes.append((left, left + it["w"], p["row"]))
        if len(items) == 2 and items[0]["x"] == 450:       # две чёрты рядом: подпись второй не должна пересекать чёрту первой
            xs = [it["x"] for it in items]
            for k, (lf, rt, _) in enumerate(boxes):
                assert not any(lf < v < rt for m, v in enumerate(xs) if m != k), ("подпись пересекает чужую чёрту", boxes, xs)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                horiz = min(a[1], b[1]) - max(a[0], b[0]) > 0
                assert not (horiz and a[2] == b[2]), ("подписи в одной строке пересекаются", boxes)


def test_old_help_text_names_new_sources():
    assert "shared/metrics-v3.json" in HTML
