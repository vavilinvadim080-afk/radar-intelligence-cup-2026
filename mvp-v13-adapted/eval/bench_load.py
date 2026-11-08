"""Воспроизводимый замер нагрузки: одно ядро, синтетика из radar/gen, потоки 3 тыс., 30 тыс., 100 тыс. обращений в день.

Запуск из mvp/:  PYTHONPATH=. .venv/bin/python eval/bench_load.py            (все три потока, десятки минут)
                 PYTHONPATH=. .venv/bin/python eval/bench_load.py --volumes 3000 --days 3000:14 --out /tmp/x.json
Пишет ../shared/metrics-load.json. Каждый поток считается в отдельном процессе, чтобы пик памяти не копился между потоками.

Что меряется (для каждого потока):
  prep   подготовка: маскирование всех текстов + обучение эмбеддера на прогреве + эмбеддинг всех текстов (radar.pipeline._prepare)
  push   детектор: RadarStream.push по всем обращениям + finish. Отдельно прогрев (детектор не считает z, только копит) и
         рабочая часть (после warmup_days): рабочая часть и есть нагрузка детектора, прогрев её занижает
  mem    пик RSS процесса (ru_maxrss), RSS после генерации данных, прирост на обращение
Что НЕ меряется: задержка одного обращения от прихода до карточки, живой поток с параллельными клиентами, качество детекции.
Одно ядро: BLAS/OpenMP ограничены одним потоком (переменные окружения до импорта numpy), процесс один.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "shared" / "metrics-load.json"
DEFAULT_DAYS = {3000: 28, 30000: 7, 100000: 5}


def rss_now_mb() -> float:
    """Текущий RSS процесса (ps), МБ. ru_maxrss даёт только пик, а нужен ещё и уровень после каждой фазы."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024.0


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024     # macOS отдаёт байты, Linux килобайты


def child(volume: int, days: int, seed: int) -> dict:
    sys.path.insert(0, str(ROOT))
    from radar.config import Config
    from radar.gen import generate
    from radar.pipeline import RadarStream, _prepare

    t = time.perf_counter()
    ds = generate(seed, "mid", volume, days)
    gen_s = time.perf_counter() - t
    n = len(ds.tickets)
    rss_gen = rss_now_mb()

    cfg = Config()
    t = time.perf_counter()
    tickets, texts, reverse, emb, X = _prepare(ds.tickets, cfg)
    prep_s = time.perf_counter() - t
    rss_prep = rss_now_mb()

    start = tickets[0].ts
    t0 = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    stream = RadarStream(ds.releases, cfg, X.shape[1], t0, start)
    warm_end = start + timedelta(days=cfg.warmup_days)
    warm_s = work_s = 0.0
    n_warm = n_work = 0
    tc = time.perf_counter
    for tk, tx, v in zip(tickets, texts, X):
        a = tc()
        stream.push(tk, tx, v)
        d = tc() - a
        if tk.ts < warm_end:
            warm_s += d
            n_warm += 1
        else:
            work_s += d
            n_work += 1
    a = tc()
    stream.finish(tickets[-1].ts)
    work_s += tc() - a
    rss_end = rss_now_mb()
    return {
        "volume_per_day": volume, "days": days, "seed": seed, "tickets": n, "generate_s": round(gen_s, 1),
        "prep": {"seconds": round(prep_s, 2), "tickets_per_s": round(n / prep_s, 1), "ms_per_ticket": round(1000 * prep_s / n, 3)},
        "detector_all": {"seconds": round(warm_s + work_s, 2), "tickets_per_s": round(n / (warm_s + work_s), 1)},
        "detector_warmup": {"tickets": n_warm, "seconds": round(warm_s, 2), "tickets_per_s": round(n_warm / warm_s, 1) if warm_s else None},
        "detector_active": {"tickets": n_work, "seconds": round(work_s, 2), "tickets_per_s": round(n_work / work_s, 1) if work_s else None,
                            "ms_per_ticket": round(1000 * work_s / n_work, 3) if n_work else None},
        "memory_mb": {"rss_after_generate": round(rss_gen), "rss_after_prepare": round(rss_prep), "rss_after_detector": round(rss_end),
                      "peak_rss": round(peak_rss_mb()),
                      "kb_per_ticket_growth_prepare_to_end": round(1024 * (rss_end - rss_prep) / n, 2),
                      "kb_per_ticket_peak_over_generated": round(1024 * (peak_rss_mb() - rss_gen) / n, 2)},
        "bursts": len(stream.bursts), "bursts_alert": sum(b.alert for b in stream.bursts),
    }


def memprobe(volume: int = 3000, days: int = 14, seed: int = 1) -> dict:
    """Что именно держит память в RadarStream: tracemalloc на небольшом потоке, байт на обращение по группам."""
    import tracemalloc
    sys.path.insert(0, str(ROOT))
    from radar.config import Config
    from radar.gen import generate
    from radar.pipeline import RadarStream, _prepare
    ds = generate(seed, "mid", volume, days)
    cfg = Config()
    tickets, texts, reverse, emb, X = _prepare(ds.tickets, cfg)
    start = tickets[0].ts
    t0 = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    stream = RadarStream(ds.releases, cfg, X.shape[1], t0, start)
    for tk, tx, v in zip(tickets, texts, X):
        stream.push(tk, tx, v)
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    n = len(tickets)
    total = sum(s.size_diff for s in snap.compare_to(base, "filename"))
    by_file = {}
    for s in snap.compare_to(base, "filename")[:6]:
        by_file[os.path.relpath(s.traceback[0].filename, ROOT) if s.traceback[0].filename.startswith(str(ROOT)) else s.traceback[0].filename] = round(s.size_diff / n)
    own = {
        "vecs_float32_bytes": int(stream.vecs.nbytes),
        "tickets_objects_estimate_bytes": None,
    }
    own["vecs_bytes_per_ticket"] = round(stream.vecs.nbytes / n, 1)
    return {"tickets": n, "days": days, "volume_per_day": volume, "traced_bytes_per_ticket_total": round(total / n),
            "traced_bytes_per_ticket_by_file": by_file, "vecs": own}


def machine() -> dict:
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except Exception:   # noqa: BLE001
            return ""
    import numpy, sklearn, scipy
    return {"platform": platform.platform(), "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "cores_logical": os.cpu_count(), "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2 ** 30, 1) or None,
            "python": platform.python_version(), "numpy": numpy.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
            "threads_limit": "BLAS/OpenMP = 1, один процесс",
            "loadavg_at_start": [round(x, 2) for x in os.getloadavg()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", default="3000,30000,100000")
    ap.add_argument("--days", default=",".join(f"{k}:{v}" for k, v in DEFAULT_DAYS.items()), help="объём:дней,...")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--child", type=int, default=None)
    ap.add_argument("--child-days", type=int, default=None)
    ap.add_argument("--memprobe", action="store_true")
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.child, a.child_days, a.seed)))
        return
    if a.memprobe:
        print(json.dumps(memprobe(), ensure_ascii=False, indent=1))
        return
    days = {int(k): int(v) for k, v in (x.split(":") for x in a.days.split(","))}
    res = {"_note": "Считает mvp/eval/bench_load.py. Не править руками. Данные синтетические; меряется пропускная способность и память одного ядра, "
                    "не задержка одного обращения и не качество детекции.",
           "generated_at": datetime.now().isoformat(timespec="seconds"), "machine": machine(), "by_volume": {}}
    for v in (int(x) for x in a.volumes.split(",")):
        d = days[v]
        print(f"поток {v}/день, {d} дней ...", flush=True)
        t = time.time()
        p = subprocess.run([sys.executable, __file__, "--child", str(v), "--child-days", str(d), "--seed", str(a.seed)],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
        if p.returncode != 0:
            res["by_volume"][str(v)] = {"error": p.stderr[-2000:]}
            print("ОШИБКА", p.stderr[-500:], flush=True)
            continue
        res["by_volume"][str(v)] = json.loads(p.stdout.strip().splitlines()[-1])
        print(f"  готово за {time.time() - t:.0f} с", flush=True)
    mp = subprocess.run([sys.executable, __file__, "--memprobe"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    res["memprobe"] = json.loads(mp.stdout) if mp.returncode == 0 else {"error": mp.stderr[-500:]}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("записано", a.out)


if __name__ == "__main__":
    main()
