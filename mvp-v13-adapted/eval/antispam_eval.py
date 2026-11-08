"""Замер защиты от спама и накрутки (radar/antispam.py, лимит на пользователя) -> ../shared/antispam-v1.json.

Две стадии, обе только на dev seeds 201-210 (свежие 401-410 не используются ни для подбора, ни для замера):
  --stage dev        метрики детектора на seeds 201-210 x 3 шума с защитой и без: строгая полнота/точность, «по делу», ложные будящие.
                     Режим «без проверки LLM» (judge_enabled=False, как в коде по умолчанию), все тревоги tier=page считаются будящими,
                     тревоги tier=quiet с пометкой suspicion выпадают из будящих. Критерий: строгая полнота и «по делу» в пределах 1 п.п.
  --stage scenarios  синтетическая атака поверх обычных данных (seeds 201-203, шум mid): (а) 1 пользователь x 500 обращений,
                     (б) 50-200 свежих аккаунтов с одним текстом за 20 минут, уровни мутации текста, (в) накрутка в минуты настоящей регрессии.
  --stage report     собрать shared/antispam-v1.json из сохранённых этапов
Запуск из папки mvp/:  PYTHONPATH=. .venv/bin/python eval/antispam_eval.py --stage dev --workers 6
Кэш этапов: eval/cache/antispam/*.json. Данные атак детерминированы (seed атаки = seed набора + номер сценария)."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import datacache                                                  # noqa: E402
import run_eval as RE                                             # noqa: E402
import v3_metrics as M                                            # noqa: E402
from radar import antispam                                        # noqa: E402
from radar.config import Config                                   # noqa: E402
from radar.gen import generator as G                              # noqa: E402
from radar.gen import topics as T                                 # noqa: E402
from radar.models import Ticket                                   # noqa: E402
from radar.pipeline import _prepare, group_incidents, run_full    # noqa: E402

DEV_SEEDS = list(range(201, 211))
SCEN_SEEDS = [201, 202, 203]
CACHE = ROOT / "eval" / "cache" / "antispam"
OUT = ROOT / "shared" / "antispam-v1.json"
OFF = dict(user_cap_max=0, brigade_check=False)        # защита выключена: поведение до этой работы
ON: dict = {}                                          # защита по умолчанию (значения в radar/config.py)
if os.environ.get("ANTISPAM_DUP_SIM"):                 # опыт подбора на dev: другой порог сходства (кэш пишется в dev-sim<x>.json)
    ON["brigade_dup_sim"] = float(os.environ["ANTISPAM_DUP_SIM"])
DEV_FILE = "dev.json" if not os.environ.get("ANTISPAM_DUP_SIM") else f"dev-sim{os.environ['ANTISPAM_DUP_SIM']}.json"
FILLERS = ["пожалуйста", "срочно", "ребята", "помогите", "уже", "снова", "опять", "сегодня"]


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "radar").rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ================================================================ разбор одного прогона
def burst_records(stream, ds_events, labels, data_end, attack_prefix="A") -> list:
    """По каждому тревожному всплеску: уровень, пометка накрутки, совпадение с настоящей регрессией, сколько в нём тикетов атаки."""
    mcfg = M.Cfg()
    out = []
    id2i = {t.id: i for i, t in enumerate(stream.tickets)}
    for b in stream.bursts:
        if not b.alert:
            continue
        topic = RE.burst_topic(stream, labels, b)
        flags = {"strict": [], "target": []}
        ttc = {}
        for e in ds_events:
            f = M.match_flags(e, b.detected_at, topic, b.product, data_end, mcfg)
            for k in flags:
                if f[k]:
                    flags[k].append(e["id"])
            if f["strict"]:
                ttc[e["id"]] = f["real_min"]
        idx = [i for i in stream.cluster_ticket_indexes(b) if b.started_at <= stream.tickets[i].ts <= b.detected_at]
        n_atk = sum(1 for i in idx if stream.tickets[i].id.startswith(attack_prefix))
        a = antispam.assess([(stream.tickets[i].user_id, stream.tickets[i].ts, stream.texts[i]) for i in idx], stream.cfg,
                            seen_before=lambda u, b=b: stream.user_first.get(u, b.detected_at) < b.started_at)
        out.append(dict(id=b.id, product=b.product, detected_at=b.detected_at.isoformat(timespec="minutes"), tier=b.tier, topic=topic,
                        suspicion=bool(b.suspicion), kept_page=(b.suspicion or {}).get("kept_page"), dup_share=(b.suspicion or {}).get("dup_share"),
                        dup_share_all=a["dup_share"], gap_cv_all=a["gap_cv"], fresh_share_all=a["fresh_share"], n_assessed=a["n_assessed"],
                        n_tickets=len(idx), n_attack=n_atk, strict=flags["strict"], target=flags["target"], ttc=ttc,
                        users=b.total_unique_users, count=b.count_in_window))
    return out


def to_runs(recs: list, ds, days=RE.EVAL_DAYS, seed=0, noise="mid") -> dict:
    """Запись в формате run_v3.collect_run, но только из будящих (tier=page) всплешков: так считает v3_metrics.aggregate(runs, None)."""
    page = [r for r in recs if r["tier"] == "page"]
    ids = {r["id"] for r in page}
    regs = [e for e in ds.ground_truth["events"] if e["type"] == "regression"]
    return dict(seed=seed, noise=noise, days=days,
                events=[dict(id=e["id"], product=e["product"], topic=e["topic"]) for e in regs],
                bursts=[dict(id=r["id"], share=None, tier="page", strict=r["strict"], work=r["strict"], target=r["target"],
                             ttc={eid: dict(real=v, work=v) for eid, v in r["ttc"].items()}, product=r["product"],
                             detected=r["detected_at"]) for r in page],
                incidents=_incidents(page))


def _incidents(page: list, gap_min: int = 120) -> list:
    groups, last = [], {}
    for r in sorted(page, key=lambda x: x["detected_at"]):
        t = datetime.fromisoformat(r["detected_at"])
        g = last.get(r["product"])
        if g is not None and t - g[1] <= timedelta(minutes=gap_min):
            groups[g[0]].append(r["id"])
            last[r["product"]] = (g[0], t)
        else:
            groups.append([r["id"]])
            last[r["product"]] = (len(groups) - 1, t)
    return groups


def agg_row(a: dict) -> dict:
    return dict(n_events=a["n_events"], n_page=a["n_page"], strict_tp=a["strict"]["tp"], strict_recall=a["strict"]["recall"],
                strict_precision=a["strict"]["precision"], caught=a["strict"]["caught"], ttc_median_min=a["strict"]["ttc_median_min"],
                target_tp=a["target"]["tp"], on_target_share=a["target"]["page_share"], false_pages_per_week=a["false_pages_per_week"])


# ================================================================ стадия dev
def dev_job(seed: int, noise: str) -> dict:
    ds, prep = datacache.get(seed, noise, 800)
    data_end = max(t.ts for t in ds.tickets)
    regs = [e for e in ds.ground_truth["events"] if e["type"] == "regression"]
    res = {}
    for name, cfgd in (("off", OFF), ("on", ON)):
        t0 = time.time()
        stream = run_full(ds.tickets, ds.releases, Config(**cfgd), prepared=prep)
        recs = burst_records(stream, regs, ds.labels, data_end)
        res[name] = dict(recs=recs, runs=to_runs(recs, ds, seed=seed, noise=noise), n_capped=stream.n_capped, accepted=len(stream.tickets),
                         n_bursts=len(stream.bursts), seconds=round(time.time() - t0, 1))
    per_user = defaultdict(list)
    for t in ds.tickets:
        per_user[t.user_id].append(t.ts)
    mx = 0
    for ts in per_user.values():
        ts.sort()
        j = 0
        for i in range(len(ts)):
            while ts[i] - ts[j] >= timedelta(hours=24):
                j += 1
            mx = max(mx, i - j + 1)
    res["max_tickets_per_user_24h"] = mx
    res["n_tickets"] = len(ds.tickets)
    return dict(seed=seed, noise=noise, **res)


def stage_dev(workers: int):
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(s, n) for s in DEV_SEEDS for n in RE.NOISES]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(dev_job, s, n) for s, n in jobs]
        rows = []
        for (s, n), f in zip(jobs, futs):
            r = f.result()
            print(f"  seed {s} {n}: off {len(r['off']['recs'])} тревог, on {len(r['on']['recs'])} тревог, "
                  f"отброшено лимитом {r['on']['n_capped']}, макс обращений одного пользователя за 24 ч {r['max_tickets_per_user_24h']}", flush=True)
            rows.append(r)
    (CACHE / DEV_FILE).write_text(json.dumps(rows, ensure_ascii=False))
    print(f"dev готов за {time.time() - t0:.0f} с")


def distribution(recs: list) -> dict:
    """Признаки накрутки у тревог dev: у настоящих всплесков и у ложных (метки генератора только в оценке)."""
    def pct(vals):
        v = sorted(x for x in vals if x is not None)
        if not v:
            return None
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        return dict(n=len(v), min=round(v[0], 3), median=round(q(0.5), 3), p90=round(q(0.9), 3), max=round(v[-1], 3))
    return dict(n=len(recs), dup_share=pct([x["dup_share_all"] for x in recs]), gap_cv=pct([x["gap_cv_all"] for x in recs]),
                fresh_share=pct([x["fresh_share_all"] for x in recs]), n_assessed=pct([x["n_assessed"] for x in recs]))


def dev_summary() -> dict:
    rows = json.loads((CACHE / DEV_FILE).read_text())
    out = {}
    for name in ("off", "on"):
        out[name] = agg_row(M.aggregate([r[name]["runs"] for r in rows], None))
    same = all([(x["id"], x["tier"]) for x in r["off"]["recs"]] == [(x["id"], x["tier"]) for x in r["on"]["recs"]] for r in rows)
    flagged = [x for r in rows for x in r["on"]["recs"] if x["suspicion"]]
    lost = [x for x in flagged if x["tier"] == "quiet" and (x["strict"] or x["target"])]
    v3 = json.loads((ROOT / "eval" / "cache" / "v3" / "dev-201-210-top.json").read_text())
    ref = agg_row(M.aggregate(v3, None))           # эталон: прогоны run_v3 --stage dev без проверки LLM (тот же код детектора до этой работы)
    d = {k: (None if out["on"][k] is None or out["off"][k] is None else out["on"][k] - out["off"][k]) for k in out["on"]}
    return dict(
        seeds=DEV_SEEDS, noises=RE.NOISES, runs=len(rows), tickets_total=sum(r["n_tickets"] for r in rows),
        off=out["off"], on=out["on"], delta_on_minus_off=d,
        delta_strict_recall_pp=100 * d["strict_recall"], delta_on_target_pp=100 * d["on_target_share"],
        within_1pp=bool(abs(100 * d["strict_recall"]) <= 1.0 and abs(100 * d["on_target_share"]) <= 1.0),
        alert_sets_identical=same, tickets_dropped_by_user_cap=sum(r["on"]["n_capped"] for r in rows),
        alerts_flagged_as_brigading=len(flagged), flagged_that_match_real_regression=len(lost),
        max_tickets_per_user_24h=max(r["max_tickets_per_user_24h"] for r in rows),
        real_alerts=distribution([x for r in rows for x in r["on"]["recs"] if x["strict"] or x["target"]]),
        false_alerts=distribution([x for r in rows for x in r["on"]["recs"] if not (x["strict"] or x["target"])]),
        reference_run_v3_dev_without_judge=ref,
        reference_matches_off=all(abs((ref[k] or 0) - (out["off"][k] or 0)) < 1e-9 for k in ("strict_recall", "strict_precision", "on_target_share", "n_page")))


# ================================================================ атаки
def make_text(product: str, topic: str, rng: random.Random) -> str:
    mobile = T.PRODUCTS[product][2]
    ch = "app_chat" if mobile else "web_form"
    return G.make_text(topic, product, ch, G.NOISE["low"], rng)[0]


def mutate(text: str, k: int, rng: random.Random) -> str:
    """k случайных правок слов: вставка слова-паразита, удаление слова, перестановка двух соседних букв. k=0: точная копия."""
    w = text.split()
    for _ in range(k):
        op = rng.choice(["ins", "del", "typo"])
        if op == "ins":
            w.insert(rng.randrange(len(w) + 1), rng.choice(FILLERS))
        elif op == "del" and len(w) > 4:
            del w[rng.randrange(len(w))]
        else:
            i = rng.randrange(len(w))
            if len(w[i]) >= 4:
                j = rng.randrange(len(w[i]) - 1)
                w[i] = w[i][:j] + w[i][j + 1] + w[i][j] + w[i][j + 2:]
    return " ".join(w)


def quiet_release(ds):
    """Релиз без регрессии рядом, в будни днём: туда ставим атаку (в окне релиза тревога возможна)."""
    regs = [e for e in ds.ground_truth["events"] if e["type"] == "regression"]
    for r in sorted(ds.releases, key=lambda x: x.ts):
        t = r.ts + timedelta(hours=2)
        if r.kind != "regular" or not (G.START + timedelta(days=8) <= t <= G.START + timedelta(days=20)):
            continue
        if t.weekday() >= 5 or not (9 <= t.hour <= 17):
            continue
        if any(e["product"] == r.product and abs((datetime.fromisoformat(e["onset_ts"]) - r.ts).total_seconds()) < 36 * 3600 for e in regs):
            continue
        return r
    raise RuntimeError("нет подходящего релиза")


def make_ticket(i, ts, product, user, text, releases, rng, topic_label):
    mobile = T.PRODUCTS[product][2]
    platform = rng.choices(["android", "ios"], [0.62, 0.38])[0] if mobile else "web"
    version = G.pick_version(rng, releases, product, ts) if mobile else ""
    tk = Ticket(f"A{i:06d}", ts, product, "app_chat" if mobile else "web_form", platform, version, user, text)
    return tk, {"topic": topic_label, "event": None, "pii": [], "injection": False, "kind": "attack"}


def build_attack(ds, spec: dict, seed: int):
    """-> (тикеты атаки, метки, подпись о месте и времени). Все времена и тексты детерминированы seed и spec."""
    rng = random.Random(f"{seed}-{spec['id']}")
    regs = sorted([e for e in ds.ground_truth["events"] if e["type"] == "regression"], key=lambda e: e["onset_ts"])
    where = {}
    if spec["start"] == "quiet_release":
        r = quiet_release(ds)
        product, start = r.product, r.ts + timedelta(hours=2)
        topic = rng.choice([t[0] for t in T.TOPICS if t[1] == product and t[4]])
        where = dict(product=product, start=start.isoformat(timespec="minutes"), kind="после релиза без регрессии")
    else:
        e = regs[0]
        product = e["product"]
        onset = datetime.fromisoformat(e["onset_ts"])
        start = onset + timedelta(minutes=spec["offset_min"])
        if spec["topic"] == "regression":
            topic = e["topic"]
        else:
            topic = rng.choice([t[0] for t in T.TOPICS if t[1] == product and t[0] != e["topic"]])
        where = dict(product=product, start=start.isoformat(timespec="minutes"), regression=e["id"], regression_topic=e["topic"],
                     regression_onset=e["onset_ts"], kind="вместе с настоящей регрессией")
    label_topic = topic if spec.get("label_as_topic") else "attack"
    tickets, labels = [], {}
    if spec["kind"] == "single_user":
        pool = [t for t in ds.tickets if t.product == product and t.ts < G.START + timedelta(days=7)]
        for i in range(spec["n"]):
            ts = start + timedelta(minutes=spec["minutes"] * i / spec["n"])
            text = make_text(product, topic, rng) if spec["text_mode"] == "exact_one" and i == 0 else (tickets[0].text if spec["text_mode"] == "exact_one" else rng.choice(pool).text)
            tk, lab = make_ticket(i + 1, ts, product, "u9attacker", text, ds.releases, rng, label_topic)
            tickets.append(tk)
            labels[tk.id] = lab
    else:
        base = make_text(product, topic, rng)
        for i in range(spec["n"]):
            off = spec["minutes"] * i / spec["n"] if spec["timing"] == "even" else rng.uniform(0, spec["minutes"])
            text = mutate(base, spec["mut"], rng) if spec["mut"] else base
            tk, lab = make_ticket(i + 1, start + timedelta(minutes=off), product, f"x{seed}{spec['id']}{i}", text, ds.releases, rng, label_topic)
            tickets.append(tk)
            labels[tk.id] = lab
    where["text_example"] = tickets[0].text[:120]
    return tickets, labels, where


def scenario_specs() -> list:
    s = []
    s.append(dict(id="a1", group="a", name="1 пользователь x 500 обращений, один текст, за 60 мин", kind="single_user", n=500, minutes=60,
                  text_mode="exact_one", start="quiet_release"))
    s.append(dict(id="a2", group="a", name="1 пользователь x 500 обращений, разные тексты, за 60 мин", kind="single_user", n=500, minutes=60,
                  text_mode="varied", start="quiet_release"))
    for n in (50, 100, 200):
        s.append(dict(id=f"b{n}", group="b", name=f"{n} свежих аккаунтов, один текст, за 20 мин (случайные моменты)", kind="brigade", n=n, minutes=20,
                      mut=0, timing="random", start="quiet_release"))
    s.append(dict(id="b100even", group="b", name="100 аккаунтов, один текст, за 20 мин ровно через равные паузы", kind="brigade", n=100, minutes=20,
                  mut=0, timing="even", start="quiet_release"))
    for k in (1, 2, 3, 4, 6):
        s.append(dict(id=f"m{k}", group="m", name=f"100 аккаунтов, текст с {k} случайными правками слов у каждого", kind="brigade", n=100, minutes=20,
                      mut=k, timing="random", start="quiet_release"))
    s.append(dict(id="c1", group="c", name="100 аккаунтов, чужая тема, начало через 30 мин после начала регрессии", kind="brigade", n=100, minutes=20,
                  mut=0, timing="random", start="regression", offset_min=30, topic="other"))
    s.append(dict(id="c2", group="c", name="100 аккаунтов копируют текст самой регрессии, через 30 мин после начала", kind="brigade", n=100, minutes=20,
                  mut=0, timing="random", start="regression", offset_min=30, topic="regression", label_as_topic=True))
    s.append(dict(id="c3", group="c", name="300 аккаунтов копируют текст регрессии, через 30 мин после начала", kind="brigade", n=300, minutes=20,
                  mut=0, timing="random", start="regression", offset_min=30, topic="regression", label_as_topic=True))
    s.append(dict(id="c4", group="c", name="100 аккаунтов копируют текст регрессии за 20 мин ДО её начала", kind="brigade", n=100, minutes=20,
                  mut=0, timing="random", start="regression", offset_min=-20, topic="regression", label_as_topic=True))
    return s


def scenario_job(seed: int, spec) -> dict:
    ds, _ = datacache.get(seed, "mid", 800)
    data_end = max(t.ts for t in ds.tickets)
    regs = [e for e in ds.ground_truth["events"] if e["type"] == "regression"]
    if spec is None:
        atk, alabels, where = [], {}, {}
    else:
        atk, alabels, where = build_attack(ds, spec, seed)
    labels = dict(ds.labels)
    labels.update(alabels)
    tickets = list(ds.tickets) + atk
    prep = _prepare(tickets, Config())
    prep = (prep[0], prep[1], prep[2], prep[3], prep[4].astype("float32"))
    res = {}
    for name, cfgd in (("off", OFF), ("on", ON)):
        stream = run_full(tickets, ds.releases, Config(**cfgd), prepared=prep)
        recs = burst_records(stream, regs, labels, data_end)
        cap = {k: v for k, v in stream.capped_by_user.items()}
        res[name] = dict(recs=recs, n_capped=stream.n_capped, accepted=len(stream.tickets), capped_by_user=cap,
                         attack_tickets_in_detector=sum(1 for t in stream.tickets if t.id.startswith("A")))
    return dict(seed=seed, spec=spec, where=where, n_attack=len(atk), **res)


def stage_scenarios(workers: int):
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(s, None) for s in SCEN_SEEDS] + [(s, sp) for s in SCEN_SEEDS for sp in scenario_specs()]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(scenario_job, s, sp) for s, sp in jobs]
        rows = []
        for (s, sp), f in zip(jobs, futs):
            r = f.result()
            print(f"  seed {s} {sp['id'] if sp else 'clean'} готов, {time.time() - t0:.0f} с", flush=True)
            rows.append(r)
    (CACHE / "scenarios.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(f"сценарии готовы за {time.time() - t0:.0f} с")


def outcome(rec: dict, clean_rec: dict) -> dict:
    """Что дала атака в одном режиме (защита выключена или включена)."""
    rs = rec["recs"]
    atk = [r for r in rs if r["n_attack"] > 0]
    real_ids = {e for r in clean_rec["recs"] if r["tier"] == "page" for e in r["strict"]}
    caught = {e for r in rs if r["tier"] == "page" for e in r["strict"]}
    page_atk = [r for r in atk if r["tier"] == "page"]
    false_page = [r for r in page_atk if not (r["strict"] or r["target"])]
    flagged = [r for r in atk if r["suspicion"]]
    clean_pages = {(r["product"], r["topic"]) for r in clean_rec["recs"] if r["tier"] == "page"}
    extra_pages = [r for r in rs if r["tier"] == "page" and r["n_attack"] == 0 and (r["product"], r["topic"]) not in clean_pages]
    return dict(attack_bursts=len(atk), false_alarm_page=bool(false_page), n_false_page_bursts=len(false_page),
                marked_as_brigading=bool(flagged), quiet_bursts_with_attack=sum(1 for r in atk if r["tier"] == "quiet"),
                real_regressions_caught_clean=sorted(real_ids), real_regressions_caught_attacked=sorted(caught),
                real_lost=sorted(real_ids - caught), page_bursts_total=sum(1 for r in rs if r["tier"] == "page"),
                new_page_bursts_without_attack_tickets=len(extra_pages), n_capped=rec["n_capped"], accepted=rec["accepted"],
                attack_tickets_in_detector=rec["attack_tickets_in_detector"], max_dup_share=max([r["dup_share"] or 0 for r in flagged], default=None),
                kept_page_after_recount=any(r["kept_page"] for r in flagged))


def scenario_summary() -> list:
    rows = json.loads((CACHE / "scenarios.json").read_text())
    clean = {r["seed"]: r for r in rows if r["spec"] is None}
    by_spec = defaultdict(list)
    for r in rows:
        if r["spec"] is not None:
            by_spec[r["spec"]["id"]].append(r)
    out = []
    for spec in scenario_specs():
        runs = sorted(by_spec[spec["id"]], key=lambda r: r["seed"])
        per_seed = []
        for r in runs:
            per_seed.append(dict(seed=r["seed"], where=r["where"], n_attack=r["n_attack"],
                                 off=outcome(r["off"], clean[r["seed"]]["off"]), on=outcome(r["on"], clean[r["seed"]]["on"])))
        n = len(per_seed)
        agg = {}
        for mode in ("off", "on"):
            agg[mode] = dict(
                runs=n, false_alarm_page_runs=sum(p[mode]["false_alarm_page"] for p in per_seed),
                marked_as_brigading_runs=sum(p[mode]["marked_as_brigading"] for p in per_seed),
                real_lost_runs=sum(bool(p[mode]["real_lost"]) for p in per_seed),
                real_caught_clean=sum(len(p[mode]["real_regressions_caught_clean"]) for p in per_seed),
                real_caught_attacked=sum(len(p[mode]["real_regressions_caught_attacked"]) for p in per_seed),
                attack_tickets_in_detector_mean=sum(p[mode]["attack_tickets_in_detector"] for p in per_seed) / n,
                n_capped_mean=sum(p[mode]["n_capped"] for p in per_seed) / n)
        out.append(dict(spec=spec, per_seed=per_seed, off=agg["off"], on=agg["on"]))
    return out


def sim_sensitivity() -> dict:
    """Подбор brigade_dup_sim на dev: чем ниже порог, тем больше мутаций текста ловим, но тем чаще помечаем настоящие тревоги.
    flag_rate_by_edits: доля из 200 синтетических накруток (12 людей, k правок слов у каждого), помеченных в чистой функции assess."""
    from radar import antispam as AS
    t0 = datetime(2026, 3, 10, 12)
    topics = [(t[0], t[1]) for t in T.TOPICS if t[4]]
    out = {}
    for sim in (0.8, 0.7, 0.6, 0.5):
        cfg = Config(brigade_dup_sim=sim)
        rates = {}
        for k in (0, 1, 2, 3, 4, 6):
            fl = 0
            for i in range(200):
                rng = random.Random(i * 7 + k)
                tp, pr = rng.choice(topics)
                base = make_text(pr, tp, rng)
                rows = [(f"x{j}", t0 + timedelta(minutes=rng.uniform(0, 10)), mutate(base, k, rng) if k else base) for j in range(12)]
                fl += AS.assess(rows, cfg)["suspect"]
            rates[str(k)] = fl / 200
        f = CACHE / ("dev.json" if sim == 0.8 else f"dev-sim{sim}.json")
        dev = None
        if f.exists():
            al = [x for r in json.loads(f.read_text()) for x in r["on"]["recs"]]
            dev = dict(alerts=len(al), max_dup_share=round(max(x["dup_share_all"] for x in al), 3), flagged_on_dev=sum(1 for x in al if x["suspicion"]))
        out[str(sim)] = dict(flag_rate_by_edits=rates, dev=dev)
    return out


def stage_report():
    dev = dev_summary()
    scen = scenario_summary()
    cfg = Config()
    keys = [k for k in cfg.model_dump() if k.startswith("brigade_") or k.startswith("user_cap_")]
    doc = dict(
        meta=dict(generated_at=datetime.now().isoformat(timespec="seconds"), code_hash=code_hash(), dev_seeds=DEV_SEEDS, scenario_seeds=SCEN_SEEDS,
                  fresh_seeds_401_410_used=False, judge="выключена (judge_enabled=False, как по умолчанию)",
                  protection_config={k: getattr(cfg, k) for k in keys}, off_config=OFF,
                  note="Все числа напечатаны eval/antispam_eval.py; syntetic data, dev seeds; атаки сгенерированы поверх данных набора."),
        dev=dev, scenarios=scen, dup_sim_sensitivity=sim_sensitivity())
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("записано", OUT)
    print(json.dumps(doc["dup_sim_sensitivity"], ensure_ascii=False, indent=1))
    for s in scen:
        print(s["spec"]["id"], s["spec"]["name"])
        print("   off", s["off"])
        print("   on ", s["on"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "scenarios", "report"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"dev": lambda: stage_dev(a.workers), "scenarios": lambda: stage_scenarios(a.workers), "report": stage_report}[a.stage]()


if __name__ == "__main__":
    main()
