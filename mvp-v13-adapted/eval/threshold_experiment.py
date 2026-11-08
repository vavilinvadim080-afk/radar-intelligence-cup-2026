"""ЭКСПЕРИМЕНТ, не часть замка протокола: что даёт повышение z_threshold на большом потоке.

Меняется только z_threshold в конфиге (код детектора не трогаем), остальные параметры из tuned_params.json.
Seeds 401-405 не использовались ни для подбора, ни для замера масштаба.  Пишет ../shared/experiment-threshold-v2.json.
Запуск:  RADAR_NO_CACHE=1 python eval/threshold_experiment.py [--volume 10000] [--z 6,8,10,12] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as R                                    # noqa: E402

SEEDS = [401, 402, 403, 404, 405]
METHODS = ("poisson", "ewma")
OUT = R.ROOT / "shared" / "experiment-threshold-v2.json"
KEYS = ("n_events", "n_alerts", "caught", "precision", "recall", "ttc_median_min", "ttc_p90_min",
        "recall_within_60", "false_alarms", "false_alarms_per_week")


def build_cfgs(zs):
    best = json.loads(R.TUNED.read_text())["best"]
    return {m: [R.cfg_dict(method=m, z_threshold=z, z_version=best[m]["z_version"], windows_min=best[m]["windows_min"])
                for z in zs] for m in METHODS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", type=int, default=10000)
    ap.add_argument("--z", default="6,8,10,12")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    zs = [float(x) for x in a.z.split(",")]
    assert not set(SEEDS) & (set(R.TUNE_SEEDS) | set(R.FINAL_SEEDS) | set(R.BURNED))
    cfgs = build_cfgs(zs)
    flat = [c for m in METHODS for c in cfgs[m]]                 # один набор данных считаем один раз на все конфиги
    res = R.run_jobs([(s, n, a.volume, flat) for s in SEEDS for n in R.NOISES], a.workers)
    out = dict(kind="эксперимент, не часть замка протокола", seeds=SEEDS, volume=a.volume, z_values=zs, methods=list(METHODS),
               note="меняется только z_threshold; z_version и windows_min из tuned_params.json; код детектора не менялся",
               by_method={}, generated_at=datetime.now().isoformat(timespec="seconds"))
    for mi, m in enumerate(METHODS):
        out["by_method"][m] = {}
        for zi, z in enumerate(zs):
            d = R.by_noise(res, mi * len(zs) + zi)["all"]
            out["by_method"][m][str(z)] = {k: d[k] for k in KEYS}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("готово", OUT)


if __name__ == "__main__":
    main()
