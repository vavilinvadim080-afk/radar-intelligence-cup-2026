"""Замер масштаба: те же пороги (eval/tuned_params.json), разный поток обращений. Пороги НЕ перенастраиваются.

Запуск:  python eval/scale_check.py [--seeds 301-305] [--volumes 800,1500,3000] [--workers 4]
Пишет ../shared/metrics-scale.json и eval/REPORT-scale.md (таблица). Выводы в REPORT-scale.md дописываются руками по цифрам.
Допущение генератора (radar/gen/generator.py, scale = объём / 800): при росте потока пик регрессии и число людей в
событиях растут пропорционально, то есть ошибка задевает ту же ДОЛЮ пользователей. Фоновый шум тоже растёт.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as R                                    # noqa: E402  (ставит однопоточный BLAS и пути)

OUT = R.ROOT / "shared" / "metrics-scale.json"
REPORT = R.ROOT / "eval" / "REPORT-scale.md"
KEYS = ("n_events", "n_alerts", "caught", "precision", "recall", "ttc_median_min", "ttc_p90_min",
        "caught_within_60_share", "recall_within_60", "false_alarms", "false_alarms_per_week")


def parse_seeds(s: str) -> list:
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="301-305")
    ap.add_argument("--volumes", default="800,1500,3000")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT), help="куда писать json (по умолчанию прежний metrics-scale.json)")
    ap.add_argument("--report", default=str(REPORT), help="куда писать таблицу (по умолчанию прежний REPORT-scale.md)")
    ap.add_argument("--copy-legacy", default="", help="объёмы, которые копируются из --legacy без пересчёта, напр. 800,1500,3000")
    ap.add_argument("--legacy", default=str(OUT), help="json, откуда копируются прежние объёмы")
    return ap


def import_legacy(out: dict, legacy_path, vols) -> None:
    """Копирует by_volume и timing прежних объёмов как есть (без пересчёта) и помечает источник."""
    old = json.loads(Path(legacy_path).read_text(encoding="utf-8"))
    for v in vols:
        out["by_volume"][str(v)] = old["by_volume"][str(v)]
        out["timing"][str(v)] = old["timing"][str(v)]
    out["legacy_source"] = dict(file=Path(legacy_path).name, volumes=list(vols), generated_at=old.get("generated_at"),
                                note="значения скопированы из прежнего замера без пересчёта")


def main(argv=None):
    a = build_parser().parse_args(argv)
    out_path, report_path = Path(a.out), Path(a.report)
    seeds, new_vols = parse_seeds(a.seeds), [int(x) for x in a.volumes.split(",")]
    legacy = [int(x) for x in a.copy_legacy.split(",") if x]
    volumes = legacy + [v for v in new_vols if v not in legacy]
    tuned = json.loads(R.TUNED.read_text())
    assert not set(seeds) & (set(R.TUNE_SEEDS) | set(R.FINAL_SEEDS) | set(R.BURNED)), "нужны свежие seeds"
    methods = ("poisson", "ewma")
    cfgs = [R.cfg_dict(method=m, z_threshold=tuned["best"][m]["z_threshold"], z_version=tuned["best"][m]["z_version"],
                       windows_min=tuned["best"][m]["windows_min"]) for m in methods]
    out = dict(seeds=seeds, volumes=volumes, noise_levels=R.NOISES, params={m: tuned["best"][m] for m in methods},
               note="пороги из tuned_params.json без перенастройки; пик событий масштабируется с потоком (см. docstring)",
               by_volume={}, timing={}, generated_at=datetime.now().isoformat(timespec="seconds"))
    import_legacy(out, a.legacy, legacy) if legacy else None
    for v in new_vols:
        t0 = time.time()
        res = R.run_jobs([(s, n, v, cfgs) for s in seeds for n in R.NOISES], a.workers)
        det = {m: R.by_noise(res, i) for i, m in enumerate(methods)}
        # время до карточки по каждому событию: чтобы смотреть распределение, а не только медиану
        ttc = {m: sorted(t for r in res for t in r["runs"][i]["res"]["ttc"]) for i, m in enumerate(methods)}
        out["by_volume"][str(v)] = dict(detector={m: {k: det[m]["all"][k] for k in KEYS} | dict(
            by_noise={n: {k: det[m][n][k] for k in KEYS} for n in R.NOISES}) for m in methods}, ttc_minutes=ttc,
            mean_run_seconds=round(sum(x["seconds"] for r in res for x in r["runs"]) / (len(res) * len(cfgs)), 1))
        out["timing"][str(v)] = round(time.time() - t0)
        print(v, "готово за", out["timing"][str(v)], "с", flush=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    fmt = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"
    lines = ["# Масштаб потока: те же пороги, свежие seeds %s-%s" % (seeds[0], seeds[-1]), "",
             f"Seeds {seeds}, шум low/mid/high, пороги из `eval/tuned_params.json` без перенастройки. "
             "Цифры считает `eval/scale_check.py`, сырьё в `shared/metrics-scale.json`.", ""]
    for m in methods:
        lines += [f"## Детектор {m}", "", "| обращений/день | событий | precision | recall | медиана до карточки, мин | p90, мин | в пределах 60 мин, % событий | ложных тревог/нед |",
                  "|---|---|---|---|---|---|---|---|"]
        for v in volumes:
            d = out["by_volume"][str(v)]["detector"][m]
            lines.append(f"| {v} | {d['n_events']} | {fmt(d['precision'])} | {fmt(d['recall'])} | {fmt(d['ttc_median_min'], 0)} | "
                         f"{fmt(d['ttc_p90_min'], 0)} | {fmt(d['recall_within_60'] and d['recall_within_60'] * 100, 0)} | {fmt(d['false_alarms_per_week'])} |")
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
