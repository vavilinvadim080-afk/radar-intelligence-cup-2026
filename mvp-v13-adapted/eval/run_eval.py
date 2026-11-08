"""Слепая оценка Радара, версия 2.

Наборы seed, которые не пересекаются:
  TUNE_SEEDS  = 1..5      здесь подбираются пороги (функция tune)
  BURNED      = 101..110  тест v1: просмотрен, поэтому для финальной оценки v2 больше не годится
  FINAL_SEEDS = 201..210  свежий тест v2: только мерим с зафиксированными параметрами
  FRESH2      = 301..310  запасной свежий набор: нужен, если после просмотра 201..210 логика менялась
Защита протокола: при первом прогоне на 201..210 в eval/cache/final_lock.json пишется хэш кода radar/*.py.
Если код потом изменился, повторный прогон на 201..210 помечается «после просмотра» (в отчёте и metrics.json),
а честная оценка тогда делается на 301..310.

Запуск:
  python eval/run_eval.py                 tune (seeds 1-5) -> final (seeds 201-210) -> metrics.json и REPORT.md
  python eval/run_eval.py --stage tune    только подбор
  python eval/run_eval.py --stage final   только финальная оценка с eval/tuned_params.json
  python eval/run_eval.py --stage final --final-seeds 301-310
Цифры в отчёт попадают только из словарей, которые считает этот файл. Время каждого этапа пишется в metrics.json.
"""
from __future__ import annotations

import os

# каждый процесс считает в один поток: иначе процессы x потоки BLAS душат машину
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from radar.config import Config                      # noqa: E402
from radar.gen.generator import START               # noqa: E402
from radar.gen.topics import PRODUCTS                # noqa: E402
from radar.pipeline import group_incidents, run_full  # noqa: E402
import datacache                                      # noqa: E402

TUNE_SEEDS = [1, 2, 3, 4, 5]
BURNED = list(range(101, 111))
FINAL_SEEDS = list(range(201, 211))
FRESH2 = list(range(301, 311))
NOISES = ["low", "mid", "high"]
TOL_BEFORE = timedelta(minutes=30)     # тревога может прийти чуть раньше «официального» начала из-за округлений
TOL_AFTER = timedelta(hours=6)         # позже 6 часов после начала событие считаем пропущенным
EVAL_DAYS = 28 - 3                     # дней, на которых детектор работает (после прогрева)
OUT_METRICS = ROOT / "shared" / "metrics.json"
V1_METRICS = ROOT / "shared" / "metrics-v1.json"
REPORT = ROOT / "eval" / "REPORT.md"
TUNED = ROOT / "eval" / "tuned_params.json"
LOCK = ROOT / "eval" / "cache" / "final_lock.json"

# сетка подбора (задана до любого прогона v2 на tune-seeds; см. DIAGNOSIS.md)
GRID_METHODS = ("poisson", "ewma")
GRID_Z = (5.0, 5.5, 6.0)
GRID_ZV = (4.5, 5.0, 5.5)
GRID_WIN = ((30, 60), (30, 60, 120))


# ================================================================ разбор одного прогона
def burst_topic(stream, labels, b):
    """Главная тема всплеска по настоящим меткам генератора (метки есть только в оценке)."""
    idx = [i for i in stream.cluster_ticket_indexes(b) if b.started_at <= stream.tickets[i].ts <= b.detected_at]
    c = Counter(labels[stream.tickets[i].id]["topic"] for i in idx)
    c.pop("generic", None)
    return c.most_common(1)[0][0] if c else None


def fa_reason(stream, labels, inc):
    """Почему ложная тревога: по меткам генератора у обращений всплесков инцидента (окно начало..срабатывание)."""
    kinds = Counter()
    for b in inc:
        for i in stream.cluster_ticket_indexes(b):
            t = stream.tickets[i]
            if not (b.started_at <= t.ts <= b.detected_at):
                continue
            lab = labels[t.id]
            k = lab["kind"]
            if k == "regression":
                kinds["regression_late"] += 1
            elif k == "noise_spike":
                kinds["noise_spike"] += 1
            elif k == "micro":
                kinds["micro"] += 1
            elif lab["event"]:
                kinds["slow_growth"] += 1
    if not kinds:
        return "random"
    top, n = kinds.most_common(1)[0]
    return top if n >= 2 else "random"


def leaked(kind: str, value: str, masked: str) -> bool:
    """Осталось ли в замаскированном тексте то, что вставил генератор."""
    if kind in ("phone", "card"):
        digits = "".join(ch for ch in value if ch.isdigit())
        left = "".join(ch for ch in masked if ch.isdigit())
        return any(digits[i:i + 5] in left for i in range(len(digits) - 4))   # любой кусок из 5 цифр подряд
    if kind == "email":
        return value.split("@")[0] in masked
    if kind == "plate":
        norm = lambda s: "".join(s.split()).lower()
        return norm(value) in norm(masked)
    if kind == "addr":
        street = [w for w in value.replace(",", " ").split() if w[:1].isupper() and not w.endswith(".")]
        return any(w in masked for w in street[-1:])
    if kind == "fio":
        tokens = [t for t in value.replace(".", " ").split() if len(t) > 1]
        return any(t in masked for t in tokens)
    return False


def mask_stats(ds, tickets, texts):
    per = defaultdict(lambda: [0, 0])      # тип -> [вставлено, утекло]
    clean_changed = clean_total = 0
    for tk, masked in zip(tickets, texts):
        pii = ds.labels[tk.id]["pii"]
        for kind, value in pii:
            per[kind][0] += 1
            per[kind][1] += leaked(kind, value, masked)
        if not pii:
            clean_total += 1
            clean_changed += masked != tk.text
    return {k: v for k, v in per.items()}, clean_changed, clean_total


def score_run(ds, stream, end: datetime):
    """Сравнивает тревоги потока с ground_truth. Возвращает суммируемые счётчики.

    Тревога = инцидент: всплески с alert=True одного продукта с паузой до 2 часов (group_incidents).
    Событие поймано, если в инциденте есть всплеск по тому же продукту и теме в допуске [-30 мин, +6 ч] от начала.
    Ложная тревога = инцидент без такого всплеска. Всплески alert=False (наблюдение, продолжение) дежурного
    не будят и в precision не идут; их число в неделю пишется отдельно (watch_per_week)."""
    labels = ds.labels
    gt = ds.ground_truth["events"]
    regs = [e for e in gt if e["type"] == "regression"]
    ts_ = lambda s: datetime.fromisoformat(s)
    res = dict(n_events=len(regs), n_alerts=0, tp_alerts=0, caught=0, ttc=[], ttc_mobile=[], ttc_web=[],
               n_mobile=0, n_web=0, caught_mobile=0, caught_web=0, fa_kinds={}, caught_in_60=0,
               n_watch=sum(1 for b in stream.bursts if not b.alert and not b.alert_reason.startswith("продолжение")),
               n_continuation=sum(1 for b in stream.bursts if b.alert_reason.startswith("продолжение")),
               signal_of_first_hit=Counter(), fix=[], nofix_regression=[], nonfix=[], events=[])
    alerts = [b for b in stream.bursts if b.alert]
    incidents = group_incidents(alerts)
    res["n_alerts"] = len(incidents)
    ev_hits = defaultdict(list)
    fa_kinds = Counter()
    for inc in incidents:
        matched = []
        for b in inc:
            topic = burst_topic(stream, labels, b)
            matched += [(e, b) for e in regs if e["product"] == b.product and e["topic"] == topic
                        and ts_(e["onset_ts"]) - TOL_BEFORE <= b.detected_at <= ts_(e["onset_ts"]) + TOL_AFTER]
        if matched:
            res["tp_alerts"] += 1
            for e, b in matched:
                ev_hits[e["id"]].append(b)
        else:
            fa_kinds[fa_reason(stream, labels, inc)] += 1
    res["fa_kinds"] = dict(fa_kinds)
    for e in regs:
        hits = ev_hits.get(e["id"], [])
        onset = ts_(e["onset_ts"])
        mobile = PRODUCTS[e["product"]][2]
        res["n_mobile" if mobile else "n_web"] += 1
        row = dict(id=e["id"], product=e["product"], mobile=mobile, peak=e["peak_per_hour"], caught=bool(hits), ttc=None)
        if hits:
            first = min(hits, key=lambda b: b.detected_at)
            t = max(0.0, (first.detected_at - onset).total_seconds() / 60)
            row["ttc"], row["signal"] = t, first.signal
            res["caught"] += 1
            res["caught_mobile" if mobile else "caught_web"] += 1
            res["ttc"].append(t)
            res["ttc_mobile" if mobile else "ttc_web"].append(t)
            res["caught_in_60"] += t <= 60
            res["signal_of_first_hit"][first.signal] += 1
        res["events"].append(row)
        checks = [b.fix_check for b in sorted(hits, key=lambda b: b.detected_at) if b.fix_check]
        if e["fix_ts"]:
            res["fix"].append(dict(effective=bool(e["fix_effective"]), verdict=checks[0].verdict if checks else "not_checked",
                                   drop=checks[0].drop_percent if checks else None,
                                   residual=checks[0].residual_share if checks else None))
        elif hits:
            # регрессия без фикса: следующий обычный релиз проверяем «как будто это фикс». Подтверждать нельзя
            first = min(hits, key=lambda b: b.detected_at)
            if first.fix_check is not None:
                res["nofix_regression"].append(first.fix_check.verdict)
    res["signal_of_first_hit"] = dict(res["signal_of_first_hit"])
    # контроль: обычные релизы без фикса, проверяем 3 самых крупных кластера продукта (их тема не менялась)
    reg_topics = {(e["product"], e["topic"]) for e in regs}
    cfg = stream.cfg
    for r in ds.releases:
        if r.kind != "regular" or r.ts < START + timedelta(days=5) or r.ts + timedelta(days=4) > end:
            continue
        keys = [k for k in stream.series if k[0] == r.product]
        pre = {k: sum(1 for t in stream.vote_times.get(k, []) if r.ts - timedelta(days=3) <= t < r.ts) for k in keys}
        for k in sorted(keys, key=lambda k: -pre[k])[:3]:
            idx = stream.members[k]
            top = Counter(labels[stream.tickets[i].id]["topic"] for i in idx[-300:]).most_common(1)[0][0]
            if (r.product, top) in reg_topics:
                continue
            if cfg.verify_mode == "baseline":
                v = stream.verify_keys([k], r.ts, end, burst_start=r.ts - timedelta(days=3), release_ts=r.ts - timedelta(days=3))
            else:
                v = stream.verify_cluster(k, r.ts, end)
            res["nonfix"].append(v["verdict"])
    return res


# ================================================================ воркер (один seed + уровень шума, несколько конфигов)
def worker(seed: int, noise: str, volume: int, cfgs: list, with_mask: bool = False):
    t0 = time.time()
    ds, prepared = datacache.get(seed, noise, volume)
    t_data = time.time() - t0
    out = dict(seed=seed, noise=noise, volume=volume, runs=[], data_seconds=round(t_data, 1))
    if with_mask:
        pii, cc, ct = mask_stats(ds, prepared[0], prepared[1])
        out["mask"] = dict(per_type=pii, clean_changed=cc, clean_total=ct)
    end = max(t.ts for t in ds.tickets)
    for c in cfgs:
        t1 = time.time()
        stream = run_full(ds.tickets, ds.releases, Config(**c), prepared=prepared)
        out["runs"].append(dict(res=score_run(ds, stream, end), seconds=round(time.time() - t1, 1)))
    return out


# ================================================================ агрегация
def aggregate(results: list, n_seeds_days: float):
    """Складывает счётчики нескольких прогонов в метрики."""
    S = lambda k: sum(r[k] for r in results)
    n_events, n_alerts, tp, caught = S("n_events"), S("n_alerts"), S("tp_alerts"), S("caught")
    ttc = [t for r in results for t in r["ttc"]]
    ttc_m = [t for r in results for t in r["ttc_mobile"]]
    ttc_w = [t for r in results for t in r["ttc_web"]]
    weeks = n_seeds_days / 7
    fa = n_alerts - tp
    fa_kinds = Counter()
    sig = Counter()
    for r in results:
        fa_kinds.update(r["fa_kinds"])
        sig.update(r["signal_of_first_hit"])
    fixes = [f for r in results for f in r["fix"]]
    eff = [f for f in fixes if f["effective"]]
    ineff = [f for f in fixes if not f["effective"]]
    nonfix = [v for r in results for v in r["nonfix"]]
    nofix_reg = [v for r in results for v in r["nofix_regression"]]
    pct = lambda a, q: float(np.percentile(a, q)) if a else None
    div = lambda a, b: (a / b) if b else None
    eff_checked = [f for f in eff if f["verdict"] != "not_checked"]
    return dict(
        n_events=n_events, n_alerts=n_alerts, tp_alerts=tp, caught=caught,
        precision=div(tp, n_alerts), recall=div(caught, n_events),
        ttc_median_min=pct(ttc, 50), ttc_p90_min=pct(ttc, 90), caught_within_60_share=div(S("caught_in_60"), caught),
        recall_within_60=div(S("caught_in_60"), n_events),
        false_alarms=fa, false_alarms_per_week=fa / weeks,
        fa_noise_spike=fa_kinds.get("noise_spike", 0), fa_slow_growth=fa_kinds.get("slow_growth", 0),
        fa_other=fa - fa_kinds.get("noise_spike", 0) - fa_kinds.get("slow_growth", 0), fa_kinds=dict(fa_kinds),
        duplicate_alerts=0,
        mobile=dict(n_events=S("n_mobile"), caught=S("caught_mobile"), recall=div(S("caught_mobile"), S("n_mobile")),
                    ttc_median_min=pct(ttc_m, 50)),
        web=dict(n_events=S("n_web"), caught=S("caught_web"), recall=div(S("caught_web"), S("n_web")), ttc_median_min=pct(ttc_w, 50)),
        first_hit_signal=dict(sig),
        watch_per_week=S("n_watch") / weeks, continuation_per_week=S("n_continuation") / weeks,
        fix=dict(n_effective=len(eff), effective_confirmed=sum(f["verdict"] == "confirmed" for f in eff),
                 effective_insufficient=sum(f["verdict"] == "insufficient_data" for f in eff),
                 effective_not_checked=sum(f["verdict"] == "not_checked" for f in eff),
                 effective_confirmed_share_of_checked=div(sum(f["verdict"] == "confirmed" for f in eff_checked), len(eff_checked)),
                 n_ineffective=len(ineff), ineffective_false_confirmed=sum(f["verdict"] == "confirmed" for f in ineff),
                 ineffective_not_checked=sum(f["verdict"] == "not_checked" for f in ineff),
                 n_nonfix_checks=len(nonfix), nonfix_false_confirmed=sum(v == "confirmed" for v in nonfix),
                 nonfix_false_confirmed_share=div(sum(v == "confirmed" for v in nonfix), len(nonfix)),
                 n_nofix_regression_checks=len(nofix_reg), nofix_regression_false_confirmed=sum(v == "confirmed" for v in nofix_reg),
                 drop_median_pct_effective=pct([f["drop"] for f in eff if f["drop"] is not None], 50),
                 residual_median_effective=pct([f["residual"] for f in eff if f["residual"] is not None], 50),
                 residual_median_ineffective=pct([f["residual"] for f in ineff if f["residual"] is not None], 50)),
    )


def f1(m):
    p, r = m["precision"] or 0.0, m["recall"] or 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def run_jobs(jobs: list, workers: int):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(worker, *zip(*jobs)))


def cfg_dict(**kw):
    return Config(**kw).model_dump()


def by_noise(res: list, idx: int) -> dict:
    out = {}
    for noise in NOISES:
        runs = [r["runs"][idx]["res"] for r in res if r["noise"] == noise]
        if runs:
            out[noise] = aggregate(runs, len(runs) * EVAL_DAYS)
    allr = [r["runs"][idx]["res"] for r in res]
    out["all"] = aggregate(allr, len(allr) * EVAL_DAYS)
    return out


# ================================================================ tune: только seeds 1-5
def tune(workers: int, volume=800):
    t0 = time.time()
    print("== TUNE на seeds", TUNE_SEEDS, flush=True)
    keys, cfgs = [], []
    for m in GRID_METHODS:
        for z in GRID_Z:
            for zv in GRID_ZV:
                for win in GRID_WIN:
                    keys.append((m, z, zv, win))
                    cfgs.append(cfg_dict(method=m, z_threshold=z, z_version=zv, windows_min=list(win)))
    # абляции при значениях по умолчанию (что даёт каждая часть v2)
    abl = {"без скана подмножеств": dict(subset_scan=False), "без сигнала версии": dict(version_signal=False),
           "тревога на любой всплеск (как v1)": dict(alert_policy="all"), "проверка фикса как в v1": dict(verify_mode="ratio")}
    jobs = [(s, n, volume, cfgs) for s in TUNE_SEEDS for n in NOISES]
    res = run_jobs(jobs, workers)
    table = {k: aggregate([r["runs"][i]["res"] for r in res], len(res) * EVAL_DAYS) for i, k in enumerate(keys)}
    # критерий выбора задан заранее: максимум F1 тревог; при равенстве меньше ложных в неделю, потом быстрее
    best = {}
    for m in GRID_METHODS:
        cand = {k: v for k, v in table.items() if k[0] == m}
        k = max(cand, key=lambda k: (round(f1(cand[k]), 3), -round(cand[k]["false_alarms_per_week"], 2),
                                     -(cand[k]["ttc_median_min"] or 1e9)))
        best[m] = dict(z_threshold=k[1], z_version=k[2], windows_min=list(k[3]), f1=f1(cand[k]), precision=cand[k]["precision"],
                       recall=cand[k]["recall"], ttc_median_min=cand[k]["ttc_median_min"],
                       false_alarms_per_week=cand[k]["false_alarms_per_week"])
    t_grid = time.time() - t0
    # абляции: вокруг лучшего poisson
    bp = best["poisson"]
    acfgs = [cfg_dict(method="poisson", z_threshold=bp["z_threshold"], z_version=bp["z_version"], windows_min=bp["windows_min"], **v)
             for v in abl.values()]
    ares = run_jobs([(s, n, volume, acfgs) for s in TUNE_SEEDS for n in NOISES], workers)
    ablation = {name: aggregate([r["runs"][i]["res"] for r in ares], len(ares) * EVAL_DAYS) for i, name in enumerate(abl)}
    ablation["v2 целиком"] = table[("poisson", bp["z_threshold"], bp["z_version"], tuple(bp["windows_min"]))]
    grid_rows = [dict(method=k[0], z=k[1], z_version=k[2], windows_min=list(k[3]), precision=v["precision"], recall=v["recall"], f1=f1(v),
                      ttc_median_min=v["ttc_median_min"], fa_per_week=v["false_alarms_per_week"]) for k, v in table.items()]
    run_secs = [x["seconds"] for r in res for x in r["runs"]]
    out = dict(best=best, grid=grid_rows, ablation={k: {kk: v[kk] for kk in ("precision", "recall", "ttc_median_min",
                                                                              "false_alarms_per_week", "fix")} for k, v in ablation.items()},
               tune_seeds=TUNE_SEEDS, noise_levels=NOISES, selection="max F1 тревог; при равенстве меньше ложных/нед, потом меньше медиана",
               timing=dict(grid_seconds=round(t_grid), total_seconds=round(time.time() - t0), n_runs=len(run_secs) + len(ares) * len(acfgs),
                           mean_run_seconds=round(float(np.mean(run_secs)), 2), workers=workers),
               config_defaults=Config().model_dump(), generated_at=datetime.now().isoformat(timespec="seconds"))
    TUNED.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("tuned:", json.dumps(best, ensure_ascii=False, default=float), f"за {time.time() - t0:.0f} с", flush=True)
    return out


# ================================================================ final: свежие seeds
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "radar").rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def final(tuned: dict, seeds: list, workers: int, volume=800):
    t0 = time.time()
    assert not set(seeds) & set(TUNE_SEEDS), "финальная оценка не на tune-seeds"
    # протокол: первый просмотр набора фиксирует код; изменённый код на том же наборе помечается
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    tag = ",".join(map(str, seeds))
    h = code_hash()
    after_seen = tag in lock and lock[tag]["code"] != h
    if tag not in lock:
        lock[tag] = dict(code=h, first_run=datetime.now().isoformat(timespec="seconds"))
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1))
    print("== FINAL на seeds", seeds, "(после просмотра)" if after_seen else "", flush=True)
    methods = ("poisson", "ewma")
    pick = lambda m: dict(method=m, z_threshold=tuned["best"][m]["z_threshold"], z_version=tuned["best"][m]["z_version"],
                          windows_min=tuned["best"][m]["windows_min"])
    cfgs = [cfg_dict(**pick(m)) for m in methods]
    cfgs.append(cfg_dict(**pick("poisson"), alert_policy="all"))
    res = run_jobs([(s, n, volume, cfgs, True) for s in seeds for n in NOISES], workers)
    t_main = time.time() - t0
    det = {m: by_noise(res, i) for i, m in enumerate(methods)}
    no_gate = by_noise(res, 2)
    # маскирование
    per = defaultdict(lambda: [0, 0])
    cc = ct = 0
    for r in res:
        for k, (a, b) in r["mask"]["per_type"].items():
            per[k][0] += a
            per[k][1] += b
        cc += r["mask"]["clean_changed"]
        ct += r["mask"]["clean_total"]
    tot_a, tot_l = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    masking = dict(inserted=tot_a, leaked=tot_l, completeness=1 - tot_l / tot_a if tot_a else None,
                   per_type={k: dict(inserted=v[0], leaked=v[1], completeness=1 - v[1] / v[0]) for k, v in per.items()},
                   over_masked_clean_tickets=cc, clean_tickets=ct, over_mask_rate=cc / ct if ct else None)
    # объём 3000/день: те же параметры, 2 seed x 3 шума
    t1 = time.time()
    vol_seeds = seeds[:2]
    vres = run_jobs([(s, n, 3000, cfgs[:2]) for s in vol_seeds for n in NOISES], workers)
    vol = {m: aggregate([r["runs"][i]["res"] for r in vres], len(vres) * EVAL_DAYS) for i, m in enumerate(methods)}
    t_vol = time.time() - t1
    run_secs = [x["seconds"] for r in res for x in r["runs"]]
    timing = dict(main_seconds=round(t_main), volume_check_seconds=round(t_vol), total_seconds=round(time.time() - t0),
                  mean_run_seconds_800=round(float(np.mean(run_secs)), 2),
                  mean_run_seconds_3000=round(float(np.mean([x["seconds"] for r in vres for x in r["runs"]])), 2),
                  mean_data_seconds_800=round(float(np.mean([r["data_seconds"] for r in res])), 1), workers=workers)
    events = [dict(seed=r["seed"], noise=r["noise"], **e) for r in res for e in r["runs"][0]["res"]["events"]]
    return dict(det=det, no_gate=no_gate, masking=masking, volume=dict(volume=3000, seeds=vol_seeds, noise_levels=NOISES, detector=vol),
                timing=timing, after_seen=after_seen, code_hash=h, seeds=seeds, events=events)


# ================================================================ вывод
def fmt(v, nd=2):
    return "н/д" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def md_table(rows, header):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(r) + " |\n"
    return s


def write_outputs(tuned, F, t_tune_total):
    v1 = json.loads(V1_METRICS.read_text(encoding="utf-8")) if V1_METRICS.exists() else {}
    labels_ru = {
        "precision": "Точность: доля тревог (alert=True, склеенных в инциденты), попавших в реальную регрессию",
        "recall": "Полнота: доля регрессий, по которым пришла тревога в допуске по времени",
        "ttc_median_min": "Время до карточки, медиана, мин (от начала всплеска до срабатывания детектора)",
        "ttc_p90_min": "Время до карточки, 90-й перцентиль, мин",
        "false_alarms_per_week": "Ложных тревог в неделю",
        "watch_per_week": "Всплесков «наблюдение» в неделю (дежурного не будят, видны на дашборде)",
        "masking_completeness": "Полнота маскирования ПДн (доля вставленных ПДн, не оставшихся в тексте)",
        "fix_effective_confirmed": "Фикс сработал: сколько раз вердикт confirmed",
        "fix_ineffective_false_confirmed": "Фикс не сработал: ложных confirmed",
        "nonfix_false_confirmed": "Релизы без фикса: ложных confirmed",
    }
    tag = "после просмотра" if F["after_seen"] else "свежие, до этого не просматривались"
    metrics = dict(
        _note="Считает mvp/eval/run_eval.py. Не править руками. Данные синтетические, цифры описывают синтетику, не реальный поток.",
        version="v2", generated_at=datetime.now().isoformat(timespec="seconds"),
        labels_ru=labels_ru,
        protocol=dict(tune_seeds=TUNE_SEEDS, test_seeds=F["seeds"], test_seeds_status=tag, burned_seeds_v1=BURNED,
                      noise_levels=NOISES, daily_volume=800, days=28, warmup_days=Config().warmup_days,
                      time_tolerance_after_h=TOL_AFTER.total_seconds() / 3600, code_hash=F["code_hash"], after_seen=F["after_seen"],
                      alert_definition="тревога = всплеск alert=True (окно 48 ч после релиза продукта, не продолжение); "
                                       "инцидент = тревоги одного продукта с паузой до 2 ч",
                      description_ru=f"Параметры подбирались на seeds {TUNE_SEEDS[0]}-{TUNE_SEEDS[-1]}, оценка на seeds "
                                     f"{F['seeds'][0]}-{F['seeds'][-1]} ({tag}). Seeds 101-110 просмотрены в v1 и для оценки v2 не используются."),
        tuned_params=dict(best=tuned["best"], selection=tuned["selection"], use_hdbscan=Config().use_hdbscan,
                          sim_threshold_tfidf=Config().sim_threshold, sim_threshold_nomic=None, grid_size=len(tuned["grid"])),
        detector=F["det"], detector_without_release_gate=dict(poisson=F["no_gate"]),
        ablation_tune=tuned["ablation"],
        masking=F["masking"],
        embedder_comparison=dict(v1.get("embedder_comparison", {}), _note="не пересчитывалось в v2: взято из metrics-v1.json (seeds 101-102, nomic хуже tfidf)"),
        volume_check=F["volume"],
        timing=dict(tune=tuned["timing"], final=F["timing"], tune_plus_final_seconds=t_tune_total),
        v1_reference=dict(detector={m: {k: v1.get("detector", {}).get(m, {}).get("all", {}).get(k) for k in
                                        ("precision", "recall", "ttc_median_min", "false_alarms_per_week", "fix")}
                                    for m in ("poisson", "ewma")}, test_seeds=v1.get("protocol", {}).get("test_seeds")),
        events_poisson=F["events"],
    )
    OUT_METRICS.parent.mkdir(exist_ok=True)
    OUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_report(metrics)
    print("metrics ->", OUT_METRICS, "\nreport ->", REPORT)


def write_report(M):
    det = M["detector"]
    P = M["protocol"]
    lines = ["# Отчёт слепой оценки Радара, v2", "",
             f"Сгенерирован `eval/run_eval.py` {M['generated_at']}. Все числа ниже напечатал скрипт, руками не правились. "
             "Прошлый отчёт: `REPORT-v1.md`. Почему всё перестроено: `DIAGNOSIS.md`.",
             "Данные синтетические: цифры описывают синтетический поток, а не реальные обращения ГПН.", "",
             "## Протокол", "",
             f"- Пороги подобраны на seeds {P['tune_seeds']} (все три уровня шума), сетка {M['tuned_params']['grid_size']} вариантов, критерий: {M['tuned_params']['selection']}.",
             f"- Оценка на seeds {P['test_seeds']}: **{P['test_seeds_status']}**. Seeds 101-110 уже просмотрены в v1 и для оценки v2 не используются.",
             f"- Объём {P['daily_volume']} обращений в день, 28 дней, первые {P['warmup_days']} дня прогрев. Событие поймано, если тревога по нужному продукту и теме пришла не раньше чем за 30 минут до начала и не позже чем через {P['time_tolerance_after_h']:.0f} часов после.",
             f"- Тревога: {P['alert_definition']}. Всплески вне окна релиза идут в «наблюдение»: дежурного не будят, видны на дашборде (их число в неделю в таблице).",
             f"- Выбранные параметры: {json.dumps({m: {k: v for k, v in b.items() if k in ('z_threshold', 'z_version', 'windows_min')} for m, b in M['tuned_params']['best'].items()}, ensure_ascii=False)}; остальное по умолчанию в `radar/config.py`. Код: хэш {P['code_hash']}.",
             ""]
    v1 = M["v1_reference"]["detector"]
    rows = []
    for m in ("poisson", "ewma"):
        a, b = v1.get(m, {}), det[m]["all"]
        fv1, f2 = a.get("fix") or {}, b["fix"]
        rows.append([m, f"{fmt(a.get('precision'))} → {fmt(b['precision'])}", f"{fmt(a.get('recall'))} → {fmt(b['recall'])}",
                     f"{fmt(a.get('ttc_median_min'), 0)} → {fmt(b['ttc_median_min'], 0)}",
                     f"{fmt(a.get('false_alarms_per_week'), 1)} → {fmt(b['false_alarms_per_week'], 1)}",
                     f"{fv1.get('effective_confirmed')}/{fv1.get('n_effective')} → {f2['effective_confirmed']}/{f2['n_effective']}",
                     f"{fv1.get('nonfix_false_confirmed')}/{fv1.get('n_nonfix_checks')} → {f2['nonfix_false_confirmed']}/{f2['n_nonfix_checks']}"])
    lines += ["## Главное: v1 (seeds 101-110) → v2 (seeds " + f"{P['test_seeds'][0]}-{P['test_seeds'][-1]})", "",
              md_table(rows, ["метод", "precision", "recall", "медиана до карточки, мин", "ложных/нед", "рабочий фикс confirmed", "релизы без фикса: ложных confirmed"]),
              "Наборы разные (v1 мерили на 101-110, v2 на свежих), генератор и его параметры те же.", ""]
    for method, title in (("poisson", "Пуассон"), ("ewma", "EWMA")):
        rows = []
        for n in ("low", "mid", "high", "all"):
            d = det[method][n]
            rows.append([n if n != "all" else "все уровни", fmt(d["precision"]), fmt(d["recall"]), fmt(d["ttc_median_min"], 0),
                         fmt(d["ttc_p90_min"], 0), fmt(d["recall_within_60"]), fmt(d["false_alarms_per_week"], 1), fmt(d["watch_per_week"], 1),
                         f"{d['caught']}/{d['n_events']}", f"{d['tp_alerts']}/{d['n_alerts']}"])
        lines += [f"## Детектор v2: {title}", "",
                  md_table(rows, ["шум", "precision", "recall", "медиана до карточки, мин", "p90, мин", "поймано за 60 мин (доля всех)", "ложных тревог/нед",
                                  "наблюдение/нед", "поймано/событий", "верных тревог/всех"])]
    rows = []
    for m in ("poisson", "ewma"):
        d = det[m]["all"]
        rows.append([m, f"{d['mobile']['caught']}/{d['mobile']['n_events']}", fmt(d["mobile"]["ttc_median_min"], 0),
                     f"{d['web']['caught']}/{d['web']['n_events']}", fmt(d["web"]["ttc_median_min"], 0), json.dumps(d["first_hit_signal"], ensure_ascii=False)])
    lines += ["## Мобильные (есть версия приложения) и веб (версии нет)", "",
              md_table(rows, ["метод", "мобильные: поймано", "медиана, мин", "веб: поймано", "медиана, мин", "каким сигналом поймано первым"]),
              "У веб-продуктов нет версии в обращении, поэтому работает только сигнал темы; он медленнее (см. DIAGNOSIS.md, «физический предел»).", ""]
    rows = []
    for m in ("poisson", "ewma"):
        d = det[m]["all"]
        rows.append([m, str(d["false_alarms"])] + [str(d["fa_kinds"].get(k, 0)) for k in ("micro", "noise_spike", "regression_late", "slow_growth", "random")])
    lines += ["## Из чего состоят ложные тревоги (все уровни шума)", "",
              md_table(rows, ["метод", "всего", "случайный микровсплеск фона", "шумовой всплеск без причины", "та же регрессия позже 6 ч", "медленный рост", "прочее"]),
              "Метки причин взяты из генератора по обращениям всплеска (в работе ядра они недоступны).", ""]
    ng = M["detector_without_release_gate"]["poisson"]
    rows = [[n if n != "all" else "все", fmt(ng[n]["precision"]), fmt(ng[n]["recall"]), fmt(ng[n]["ttc_median_min"], 0), fmt(ng[n]["false_alarms_per_week"], 1)]
            for n in ("low", "mid", "high", "all")]
    lines += ["## Без фильтра по релизу (тревога на любой всплеск, Пуассон, те же пороги)", "",
              md_table(rows, ["шум", "precision", "recall", "медиана, мин", "ложных/нед"]),
              "Это ответ на вопрос «сколько даёт привязка к релизу»: детектор тот же, разница только в том, кого будить.", ""]
    rows = []
    for m in ("poisson", "ewma"):
        for n in ("low", "mid", "high", "all"):
            f = det[m][n]["fix"]
            rows.append([m, n if n != "all" else "все", f"{f['effective_confirmed']}/{f['n_effective']}", str(f["effective_not_checked"]),
                         fmt(f["effective_confirmed_share_of_checked"]), f"{f['ineffective_false_confirmed']}/{f['n_ineffective']}",
                         f"{f['nonfix_false_confirmed']}/{f['n_nonfix_checks']}", f"{f['nofix_regression_false_confirmed']}/{f['n_nofix_regression_checks']}",
                         fmt(f["residual_median_effective"]), fmt(f["residual_median_ineffective"])])
    lines += ["## Проверка фикса (v2: возврат к норме до релиза, по всем кластерам инцидента)", "",
              md_table(rows, ["метод", "шум", "рабочий фикс: confirmed", "не проверялись (не поймана)", "confirmed среди проверенных", "нерабочий фикс: ложных confirmed",
                              "релизы без фикса: ложных confirmed", "регрессия без фикса, следующий релиз: ложных confirmed", "медианный остаток (рабочие)", "медианный остаток (нерабочие)"]),
              "«Рабочий фикс» гасит тему до 6% от пика, «нерабочий» до 70%. «Релизы без фикса»: обычные релизы, три крупнейших кластера продукта, тема которых не менялась. "
              "Остаток 0 значит «тема вернулась к уровню до релиза», 1 значит «ничего не изменилось».", ""]
    ab = M["ablation_tune"]
    rows = [[k, fmt(v["precision"]), fmt(v["recall"]), fmt(v["ttc_median_min"], 0), fmt(v["false_alarms_per_week"], 1),
             f"{v['fix']['effective_confirmed']}/{v['fix']['n_effective']}", f"{v['fix']['nonfix_false_confirmed']}/{v['fix']['n_nonfix_checks']}"] for k, v in ab.items()]
    lines += ["## Что даёт каждая часть v2 (абляции на tune-seeds 1-5, Пуассон)", "",
              md_table(rows, ["вариант", "precision", "recall", "медиана, мин", "ложных/нед", "рабочий фикс confirmed", "без фикса: ложных confirmed"]),
              "Абляции считались на tune-seeds, на финальном наборе их не смотрели.", ""]
    vc = M["volume_check"]["detector"]
    rows = [[m, fmt(d["precision"]), fmt(d["recall"]), fmt(d["ttc_median_min"], 0), fmt(d["false_alarms_per_week"], 1), fmt(d["watch_per_week"], 1)] for m, d in vc.items()]
    lines += [f"## Объём 3000 обращений в день (те же пороги, seeds {M['volume_check']['seeds']}, все три шума)", "",
              md_table(rows, ["метод", "precision", "recall", "медиана, мин", "ложных/нед", "наблюдение/нед"]),
              "Пороги подобраны на 800 в день и не пересматривались.", ""]
    mk = M["masking"]
    rows = [[k, str(v["inserted"]), str(v["leaked"]), fmt(v["completeness"], 4)] for k, v in sorted(mk["per_type"].items())]
    lines += ["## Маскирование ПДн", "",
              f"Вставлено ПДн: {mk['inserted']}, осталось в тексте после маскирования: {mk['leaked']}, полнота {fmt(mk['completeness'], 4)}. "
              f"Лишнее маскирование: у {mk['over_masked_clean_tickets']} из {mk['clean_tickets']} обращений без ПДн.",
              "Маскирующие правила и генератор написаны одним автором: это проверка по своей же синтетике, на реальных текстах полнота будет ниже.", "",
              md_table(rows, ["тип", "вставлено", "утекло", "полнота"])]
    T = M["timing"]
    lines += ["## Время", "",
              f"- Подбор порогов: {T['tune']['total_seconds']} с ({T['tune']['n_runs']} прогонов, в среднем {T['tune']['mean_run_seconds']} с на прогон, {T['tune']['workers']} процессов).",
              f"- Финальная оценка: {T['final']['total_seconds']} с (основная часть {T['final']['main_seconds']} с, объём 3000/день {T['final']['volume_check_seconds']} с; прогон 800/день {T['final']['mean_run_seconds_800']} с, 3000/день {T['final']['mean_run_seconds_3000']} с).",
              "- Эмбеддинги и маскирование кэшируются по (seed, шум, объём) в `eval/cache/ds/`; HDBSCAN выключен. В v1 полный eval шёл 3,5 часа.", ""]
    lines += ["## Что эти цифры не говорят", "",
              "- Это синтетика с шаблонами: тема из 6 шаблонов, TF-IDF режет её на 3-4 кластера. Скан подмножеств собирает осколки по совместному всплеску, но не по смыслу.",
              "- Сигнал версии силён в генераторе потому, что 88% обращений регрессии приходят с новой версии с первой минуты. В жизни доля тоже высокая (баг есть только в новой версии), но объём жалоб растёт вместе с раскаткой, и в первые часы их меньше.",
              "- Тревога будит дежурного только в 48 часах после релиза продукта. Регрессия, которая проявилась позже или не связана с релизом (сбой бэкенда, партнёра), попадёт в «наблюдение», а не в тревогу.",
              "- Пороги подобраны перебором по небольшой сетке (z_threshold 5,0-6,0, z_version 4,5-5,5, окна без 120 минут или с ними), и лучший вариант лёг на верхний край сетки: z_threshold = 6,0, z_version = 5,5, окна 30/60/120. Значит, оптимум мог лежать за границей сетки (более высокие пороги, более длинные окна). Это не проверялось: подбор и финальная оценка не перезапускались, потому что протокол заморожен. Параметры минимума людей в окне (min_users = 3) вообще не входили в перебор и взяты как заданные.",
              "- Привязка к релизу это корреляция по времени и версии, не доказательство причины. Агент обязан называть её «подозреваемый релиз».", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(s: str) -> list:
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "tune", "final"], default="all")
    ap.add_argument("--final-seeds", default="201-210")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    t0 = time.time()
    if args.stage in ("all", "tune"):
        tuned = tune(args.workers)
    else:
        tuned = json.loads(TUNED.read_text(encoding="utf-8"))
    if args.stage == "tune":
        return
    seeds = parse_seeds(args.final_seeds)
    assert not set(seeds) & set(BURNED), "seeds 101-110 просмотрены в v1, для оценки v2 не годятся"
    F = final(tuned, seeds, args.workers)
    write_outputs(tuned, F, round(time.time() - t0) if args.stage == "all" else tuned["timing"]["total_seconds"] + F["timing"]["total_seconds"])
    print(f"готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
