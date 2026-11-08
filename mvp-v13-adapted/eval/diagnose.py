"""Диагностика детектора v1: оракул (истинные темы вместо кластеров) против кластеров TF-IDF.

Вопрос: почему recall 0.27 и precision 0.12. Варианты: виновата кластеризация (осколки), сам детектор
(пороги, окна), определение метрики или генератор (события неотличимы от шума).

Запуск:  .venv/bin/python eval/diagnose.py [--seeds 1,2,3,101,...] [--workers 7]
Пишет eval/cache/diagnose.json; выводы руками переносятся в eval/DIAGNOSIS.md (числа из этого json).
Seeds 201-210 и 301-310 сюда передавать нельзя: это свежие наборы финальной оценки.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.special import gammainc
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from radar.config import Config                      # noqa: E402
from radar.gen.generator import START                # noqa: E402
from radar.gen.topics import PRODUCTS                # noqa: E402
from radar.pipeline import group_incidents, run_full  # noqa: E402
import datacache                                      # noqa: E402

FORBIDDEN = set(range(201, 211)) | set(range(301, 311))
OUT = ROOT / "eval" / "cache" / "diagnose.json"
TOL_BEFORE, TOL_AFTER = timedelta(minutes=30), timedelta(hours=6)


def ztail(x, lam):
    if x <= 0:
        return -10.0
    p = float(gammainc(x, max(lam, 1e-9)))
    return float(norm.isf(min(max(p, 1e-300), 1 - 1e-12)))


def oracle_prepared(ds, prepared):
    """Те же обращения, но вектор = one-hot истинной темы. С порогом 0.99 онлайн-кластер = тема."""
    tickets, texts, reverse, emb, _ = prepared
    topics = sorted({ds.labels[t.id]["topic"] for t in tickets})
    pos = {tp: i for i, tp in enumerate(topics)}
    X = np.zeros((len(tickets), len(topics)), dtype=np.float32)
    for i, t in enumerate(tickets):
        X[i, pos[ds.labels[t.id]["topic"]]] = 1.0
    return tickets, texts, reverse, emb, X


def dedup_times(tks, dedup_min=60):
    last, out = {}, []
    for t in tks:
        u = t.user_id
        if u in last and (t.ts - last[u]) < timedelta(minutes=dedup_min):
            continue
        last[u] = t.ts
        out.append(t.ts)
    return out


def event_strength(ds, tickets):
    """Сила каждой регрессии: сколько людей темы в первые W минут против ожидания по тем же часам прошлых дней."""
    by_topic = defaultdict(list)
    for t in tickets:
        by_topic[(t.product, ds.labels[t.id]["topic"])].append(t)
    rows = []
    for e in ds.ground_truth["events"]:
        if e["type"] != "regression":
            continue
        onset = datetime.fromisoformat(e["onset_ts"])
        tl = by_topic[(e["product"], e["topic"])]
        times = np.array(dedup_times(tl), dtype="datetime64[s]")
        ev_times = np.array(sorted(t.ts for t in tl if ds.labels[t.id]["event"] == e["id"]), dtype="datetime64[s]")
        mobile = PRODUCTS[e["product"]][2]
        r = dict(id=e["id"], product=e["product"], mobile=mobile, peak=e["peak_per_hour"], onset_hour=onset.hour,
                 weekday=onset.weekday())
        wk = onset.weekday() >= 5
        for W in (30, 60, 120, 240):
            a, b = np.datetime64(onset), np.datetime64(onset + timedelta(minutes=W))
            x = int(((times >= a) & (times < b)).sum())
            extra = int(((ev_times >= a) & (ev_times < b)).sum())
            base = []
            for d in range(1, 15):
                o2 = onset - timedelta(days=d)
                if (o2.weekday() >= 5) != wk or o2 < START:
                    continue
                a2, b2 = np.datetime64(o2), np.datetime64(o2 + timedelta(minutes=W))
                base.append(int(((times >= a2) & (times < b2)).sum()))
                if len(base) >= 5:
                    break
            lam = float(np.mean(base)) if base else 0.0
            r[f"x{W}"], r[f"extra{W}"], r[f"lam{W}"], r[f"z{W}"] = x, extra, round(lam, 2), round(ztail(x, max(lam, 1.0)), 2)
        # сигнал версии (мобильные): обращения ТЕМЫ с версией релиза в первые 60/120 минут против остальных тем
        if mobile:
            ver = e["release_version"]
            for W in (60, 120):
                lo, hi = onset, onset + timedelta(minutes=W)
                tt = [t for t in tl if lo <= t.ts < hi and t.app_version]
                oth = [t for t in tickets if t.product == e["product"] and lo - timedelta(hours=3) <= t.ts < hi
                       and t.app_version and ds.labels[t.id]["topic"] != e["topic"]]
                q = (sum(t.app_version == ver for t in oth) + 0.5) / (len(oth) + 1)
                xv = sum(t.app_version == ver for t in tt)
                r[f"ver_x{W}"], r[f"ver_q{W}"] = xv, round(q, 4)
                # ожидание без регрессии: доля версии у других тем * обычный поток темы в это окно * доля обращений с версией
                r[f"ver_lam{W}"] = round(q * r[f"lam{W}"] * 0.8, 3)
                r[f"ver_z{W}"] = round(ztail(xv, max(q * r[f"lam{W}"] * 0.8, 0.05)), 2)
        rows.append(r)
    return rows


def micro_strength(ds, tickets):
    """Случайные микровсплески фона (в события не пишутся, тревога по ним = ложная): их сила в окне 60 минут."""
    micro = defaultdict(list)
    for t in tickets:
        lab = ds.labels[t.id]
        if lab["kind"] == "micro":
            micro[(t.product, lab["topic"], t.ts.date())].append(t.ts)
    return [len(v) for v in micro.values()]


def classify_fa(stream, ds, inc):
    """Почему ложная тревога: по меткам обращений всплесков инцидента."""
    kinds = Counter()
    for b in inc:
        idx = [i for i in stream.cluster_ticket_indexes(b) if b.started_at <= stream.tickets[i].ts <= b.detected_at]
        for i in idx:
            lab = ds.labels[stream.tickets[i].id]
            ev = lab["event"] or ""
            k = lab["kind"]
            if k == "regression":
                kinds["regression_late"] += 1   # тема регрессии, но вне допуска (повтор через часы/дни)
            elif k == "noise_spike":
                kinds["noise_spike"] += 1
            elif k == "micro":
                kinds["micro"] += 1
            elif ev:
                kinds["slow"] += 1
    if not kinds:
        return "random"
    top, n = kinds.most_common(1)[0]
    return top if n >= 2 else "random"


def fragmentation(stream, ds):
    """Сколько кластеров нужно, чтобы покрыть 80% обращений темы, и чистота кластеров (по обращениям)."""
    topic_clusters = defaultdict(Counter)
    pure = tot = 0
    for key, idx in stream.members.items():
        c = Counter(ds.labels[stream.tickets[i].id]["topic"] for i in idx)
        for tp, n in c.items():
            topic_clusters[tp][key] += n
        if len(idx) >= 20:
            pure += c.most_common(1)[0][1]
            tot += len(idx)
    frag = {}
    for tp, c in topic_clusters.items():
        if tp == "generic":
            continue
        s, acc, n = sum(c.values()), 0, 0
        for _, v in c.most_common():
            acc += v
            n += 1
            if acc >= 0.8 * s:
                break
        frag[tp] = n
    return dict(purity=pure / max(tot, 1), frag_mean=float(np.mean(list(frag.values()))), n_clusters=len(stream.members))


def job(seed, noise, volume, do_tfidf):
    import run_eval as RE
    ds, prepared = datacache.get(seed, noise, volume)
    end = max(t.ts for t in ds.tickets)
    tickets = prepared[0]
    out = dict(seed=seed, noise=noise, strength=event_strength(ds, tickets), micro=micro_strength(ds, tickets), runs={})
    orc = oracle_prepared(ds, prepared)
    variants = [("oracle_poisson", orc, dict(method="poisson", sim_threshold=0.99, use_hdbscan=False)),
                ("oracle_ewma", orc, dict(method="ewma", sim_threshold=0.99, use_hdbscan=False))]
    if do_tfidf:
        variants.append(("tfidf_poisson", prepared, dict(method="poisson")))
    for name, prep, kw in variants:
        t0 = time.time()
        cfg = Config(z_threshold=4.5, min_users=3, lambda_floor=1.0, **kw)
        stream = run_full(ds.tickets, ds.releases, cfg, prepared=prep)
        res = RE.score_run(ds, stream, end)
        incs = group_incidents(stream.bursts)
        fa_kinds = Counter()
        regs = [e for e in ds.ground_truth["events"] if e["type"] == "regression"]
        for inc in incs:
            hit = False
            for b in inc:
                tp = RE.burst_topic(stream, ds.labels, b)
                hit |= any(e["product"] == b.product and e["topic"] == tp and
                           datetime.fromisoformat(e["onset_ts"]) - TOL_BEFORE <= b.detected_at <= datetime.fromisoformat(e["onset_ts"]) + TOL_AFTER
                           for e in regs)
            if not hit:
                fa_kinds[classify_fa(stream, ds, inc)] += 1
        # пойманные события: подробности
        caught = {}
        for e in regs:
            onset = datetime.fromisoformat(e["onset_ts"])
            for b in stream.bursts:
                if b.product == e["product"] and RE.burst_topic(stream, ds.labels, b) == e["topic"] and \
                        onset - TOL_BEFORE <= b.detected_at <= onset + TOL_AFTER:
                    d = (b.detected_at - onset).total_seconds() / 60
                    caught[e["id"]] = min(caught.get(e["id"], 1e9), d)
        out["runs"][name] = dict(res=res, fa_kinds=dict(fa_kinds), caught=caught, seconds=round(time.time() - t0, 1),
                                 frag=fragmentation(stream, ds) if name.startswith("tfidf") else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5,101,102,103,104,105,106,107,108,109,110")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--no-tfidf", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    assert not FORBIDDEN & set(seeds), "свежие seeds финальной оценки в диагностике не используются"
    t0 = time.time()
    jobs = [(s, n, 800, not args.no_tfidf) for s in seeds for n in ("low", "mid", "high")]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(job, *zip(*jobs)))
    OUT.write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"diagnose: {len(res)} прогонов за {time.time() - t0:.0f} с -> {OUT}")


if __name__ == "__main__":
    main()
