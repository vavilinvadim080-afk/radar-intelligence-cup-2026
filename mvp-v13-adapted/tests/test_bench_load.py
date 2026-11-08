"""Замер нагрузки: на крошечном потоке скрипт должен отработать и отдать все поля, на которые ссылается REPORT-load.md."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bench_child_outputs_required_fields_on_tiny_stream():
    p = subprocess.run([sys.executable, str(ROOT / "eval" / "bench_load.py"), "--child", "150", "--child-days", "5"],
                       capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr[-800:]
    r = json.loads(p.stdout.strip().splitlines()[-1])
    for k in ("tickets", "prep", "detector_warmup", "detector_active", "memory_mb", "volume_per_day", "days"):
        assert k in r
    assert r["tickets"] > 300 and r["prep"]["tickets_per_s"] > 0 and r["detector_active"]["tickets"] > 0
    assert r["memory_mb"]["peak_rss"] >= r["memory_mb"]["rss_after_generate"] > 0


def test_machine_block_has_date_cores_python_and_load():
    sys.path.insert(0, str(ROOT / "eval"))
    import bench_load
    m = bench_load.machine()
    for k in ("platform", "cpu", "cores_logical", "python", "loadavg_at_start"):
        assert m[k] not in (None, "")
