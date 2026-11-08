"""Оценка детектора v3: два уровня тревоги (page/quiet) по второй проверке LLM и рабочие часы веб-продуктов.

Стадии:
  --stage dev     seeds 201-210 (уже просмотрены, это dev): прогон с judge_enabled, варианты «без проверки» и пороги
                  0.625/0.75/0.875, выбор порога по заранее заданному критерию -> eval/cache/v3/frozen.json
  --stage final   ОДИН прогон на свежих seeds 401-410 с замороженным конфигом и хэшем кода -> shared/metrics-v3.json,
                  eval/REPORT-v3.md, слепой файл для сильной модели eval/cache/blind-401-410/
  --stage report  пересобрать отчёт из сохранённых прогонов (LLM и конвейер не запускаются)
Повторный final на 401-410 отказывается запускаться (подбирать на них нельзя); --allow-rerun помечает результат
«после просмотра». Если код radar/ или eval/v3_metrics.py изменился после заморозки, final тоже отказывается
(--allow-code-change помечает).
Модель вызывается локально (Ollama, temperature 0); ответы кэшируются по SHA-256 в eval/cache/judge/ (JSON), поэтому
повторный запуск не ходит в модель. Запуск: nice -n 10 PYTHONPATH=. .venv/bin/python eval/run_v3.py --stage dev"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import datacache                                                  # noqa: E402
import run_eval as RE                                             # noqa: E402
import v3_metrics as M                                            # noqa: E402
from radar.config import Config                                   # noqa: E402
from radar.gen.topics import PRODUCTS                             # noqa: E402
from radar.judge import OllamaLLM, build_prompt, make_judge       # noqa: E402
from radar.pipeline import group_incidents, run_full              # noqa: E402

DEV_SEEDS = list(range(201, 211))
FINAL_SEEDS = list(range(401, 411))
THRESHOLDS = (0.625, 0.75, 0.875)
SAMPLINGS = ("top", "spread")
MODEL = "gemma4:e2b"
MAX_LOSS = 0.15
SECOND_MAX_LOSS = 0.20
CACHE = ROOT / "eval" / "cache"
JUDGE_CACHE = CACHE / "judge"
V3 = CACHE / "v3"
FROZEN = V3 / "frozen.json"
LOCK = V3 / "final_lock.json"
BLIND = CACHE / "blind-401-410"
OUT_METRICS = ROOT / "shared" / "metrics-v3.json"
OUT_REPORT = ROOT / "eval" / "REPORT-v3.md"


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "radar").rglob("*.py")) + [ROOT / "eval" / "v3_metrics.py"]:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ================================================================ один прогон (seed x шум)
def collect_run(seed: int, noise: str, volume: int, cfg: dict, model: str, cache_dir: str) -> dict:
    """Прогоняет конвейер с проверкой и возвращает запись: всё, что нужно для метрик при любом пороге."""
    ds, prep = datacache.get(seed, noise, volume)
    judge = make_judge(OllamaLLM(model), cache_dir, model=model) if model else None
    t0 = time.time()
    stream = run_full(ds.tickets, ds.releases, Config(**cfg), prepared=prep, judge=judge)
    data_end = max(t.ts for t in ds.tickets)
    regs = [e for e in ds.ground_truth["events"] if e["type"] == "regression"]
    mcfg = M.Cfg()
    id2i = {t.id: i for i, t in enumerate(stream.tickets)}
    alerts = [b for b in stream.bursts if b.alert]
    rows = []
    for b in alerts:
        topic = RE.burst_topic(stream, ds.labels, b)
        ttc, flags = {}, {"strict": [], "work": [], "target": []}
        for e in regs:
            f = M.match_flags(e, b.detected_at, topic, b.product, data_end, mcfg)
            for k in flags:
                if f[k]:
                    flags[k].append(e["id"])
            if f["work"] or f["strict"]:
                ttc[e["id"]] = dict(real=f["real_min"], work=f["work_min"])
        j = b.judge or {}
        rows.append(dict(id=b.id, product=b.product, detected_at=b.detected_at.isoformat(timespec="minutes"), topic=topic,
                         share=j.get("share"), problem=j.get("problem", ""), items=j.get("items", []), tier=b.tier,
                         judge_error=j.get("error"), alert_reason=b.alert_reason,
                         texts=[stream.texts[id2i[i]][:300] for i in j.get("example_ids", [])], example_ids=j.get("example_ids", []),
                         strict=flags["strict"], work=flags["work"], target=flags["target"], ttc=ttc))
    ids = {b.id: i for i, b in enumerate(alerts)}
    incidents = [[b.id for b in inc] for inc in group_incidents(alerts)]
    events = [dict(id=e["id"], product=e["product"], topic=e["topic"], business=e["product"] in mcfg.business_products)
              for e in regs]
    return dict(seed=seed, noise=noise, volume=volume, days=RE.EVAL_DAYS, events=events, bursts=rows, incidents=incidents,
                seconds=round(time.time() - t0, 1))


def collect(seeds: list, workers: int, cfg: dict, model: str) -> list:
    jobs = [(s, n) for s in seeds for n in RE.NOISES]
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(collect_run, s, n, 800, cfg, model, str(JUDGE_CACHE)) for s, n in jobs]
        runs = []
        for (s, n), f in zip(jobs, futs):
            r = f.result()
            print(f"  seed {s} {n}: {len(r['bursts'])} тревог-всплесков, {r['seconds']} с", flush=True)
            runs.append(r)
    return runs


def table(runs: list) -> dict:
    t = {None: M.aggregate(runs, None)}
    for thr in THRESHOLDS:
        t[thr] = M.aggregate(runs, thr)
    return t


def key(k):
    return "none" if k is None else str(k)


# ================================================================ печать и отчёт
def pct(x, nd=0):
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def num(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def rows_md(tab: dict, only=None, judge: str = "gemma") -> str:
    head = ("| вариант | будят (инцидентов) | тихих | строгая P | строгая R | медиана до карточки, мин | "
            "рабочая P | рабочая R | медиана, рабочие мин | будят по делу | ложных будят в неделю | не проверено |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = [head]
    for k, a in tab.items():
        if only is not None and k not in only:
            continue
        name = "без проверки" if k is None else f"{judge}, порог {k}"
        out.append(f"| {name} | {a['n_page']} | {a['n_quiet']} | {pct(a['strict']['precision'])} | {pct(a['strict']['recall'])} | "
                   f"{num(a['strict']['ttc_median_min'], 0)} | {pct(a['work']['precision'])} | {pct(a['work']['recall'])} | "
                   f"{num(a['work']['ttc_median_min'], 0)} | {pct(a['target']['page_share'])} | {num(a['false_pages_per_week'], 2)} | "
                   f"{a['unchecked']} |")
    return "\n".join(out)


def quiet_md(tab: dict) -> str:
    out = ["| вариант | тихих | из них по делу | доля по делу | из них строго верных |", "|---|---|---|---|---|"]
    for k, a in tab.items():
        if k is None:
            continue
        q = a["quiet"]
        out.append(f"| порог {k} | {q['n']} | {q['n_target']} | {pct(q['share_target'])} | {q['n_strict_tp']} |")
    return "\n".join(out)


def loss_md(tab: dict) -> str:
    base = tab[None]["strict"]["tp"]
    out = ["| порог | строгих верных тревог (без проверки: %d) | потеря | полнота page, строгая | полнота page, рабочая |" % base,
           "|---|---|---|---|---|"]
    for k, a in tab.items():
        if k is None:
            continue
        out.append(f"| {k} | {a['strict']['tp']} | {pct(1 - a['strict']['tp'] / base) if base else '-'} | "
                   f"{pct(a['strict']['recall'])} | {pct(a['work']['recall'])} |")
    return "\n".join(out)


DEFS = """- **строгая** оценка как в `eval/run_eval.py`: тот же продукт и тема, срабатывание в [начало - 30 мин, начало + 6 ч] по календарю. Не менялась.
- **рабочая** (вторая, дополнительная): для b2b, portal, partners допуск 6 ч считается в рабочих часах (пн-пт, 9-19); для приложений совпадает со строгой.
- **по делу**: будящая тревога (инцидент) относится к настоящей регрессии того же продукта и темы в любой момент от начала - 30 мин до конца её активности. Конец активности: у регрессии с эффективным фиксом время фикса + 12 ч, иначе конец набора данных. Тема и продукт берутся из меток генератора (только в оценке).
- **инцидент** = всплески одного продукта с паузой до 2 ч. Инцидент будит, если будит хотя бы один его всплеск (доля >= порога или проверка недоступна).
- **ложных будят в неделю** = будящие инциденты не по делу / недели наблюдения (прогонов x 25 дней / 7)."""


def render_report(dev: dict, final: list, meta: dict) -> str:
    thr = meta["threshold"]
    md = ["# Детектор v3: два уровня тревоги и рабочие часы", "",
          f"Сгенерировано `eval/run_v3.py` {meta['generated_at']}. Все числа напечатаны скриптом из `shared/metrics-v3.json`.", "",
          "## Определения", DEFS, "",
          f"## Dev: seeds {meta['dev_seeds'][0]}-{meta['dev_seeds'][-1]} (просмотрены, подбор допустим)", "",
          f"Модель проверки {meta['model']}, temperature 0, 8 примеров. Прогонов (seed x 3 шума): {meta['dev_runs']}.", "",
          ]
    for mode, tab in dev["tabs"].items():
        md += [f"### Выборка примеров: {mode}" + (" (выбрана)" if mode == meta["sampling"] else ""), "", rows_md(tab), "",
               "Потеря строгих верных:", "", loss_md(tab), "", "Тихие карточки:", "", quiet_md(tab), ""]
    md += ["### Выбор порога и способа выборки",
          f"Критерий задан заранее: максимум доли «по делу» среди будящих тревог при потере не больше {int(MAX_LOSS * 100)}% верных строгих "
          f"тревог. Пороги по способам: {meta['picks']}. Выбрано: выборка **{meta['sampling']}**, порог **{thr}**." if thr is not None else
          f"Критерий задан заранее: максимум доли «по делу» при потере не больше {int(MAX_LOSS * 100)}% верных строгих. "
          "Ни один порог не прошёл, проверку включать не рекомендуется.", ""]
    for f in final:
        v, ft = f["variant"], f["table"]
        label = "основной вариант, выбран по критерию" if v["role"] == "primary" else \
            f"второй вариант, объявлен заранее при потере до {int(v['max_loss'] * 100)}%, по основному критерию НЕ выбран"
        md += [f"## Final: свежие seeds {meta['final_seeds'][0]}-{meta['final_seeds'][-1]}, выборка {v['sampling']}, порог {v['threshold']} ({label})", "",
               f"Хэш кода `{meta['code_hash']}`, конфиг заморожен на dev до прогона. Прогонов: {meta['final_runs']}."
               + (" **ПОСЛЕ ПРОСМОТРА (повторный запуск или изменённый код).**" if meta.get("after_seen") else ""), "",
               rows_md(ft, only=[None, v["threshold"]]), "", "Остальные пороги на тех же seeds (справочно, не выбирались):", "",
               rows_md(ft, only=[t for t in THRESHOLDS if t != v["threshold"]]), "", "Потеря строгих верных:", "", loss_md(ft), "",
               "Тихие карточки:", "", quiet_md(ft), "", "По уровням шума (выбранный порог):", "",
               "| шум | будят | тихих | строгая P | строгая R | по делу | ложных будят в неделю |", "|---|---|---|---|---|---|---|"]
        for n, a_ in f["by_noise"].items():
            md.append(f"| {n} | {a_['n_page']} | {a_['n_quiet']} | {pct(a_['strict']['precision'])} | {pct(a_['strict']['recall'])} | "
                      f"{pct(a_['target']['page_share'])} | {num(a_['false_pages_per_week'], 2)} |")
        md.append("")
    if final:
        md += ["Слепая проверка сильной моделью: `eval/cache/blind-401-410/` (основной вариант) и подпапка второго варианта, подсчёт `eval/score_blind_v3.py`."]
    return "\n".join(md) + "\n"


# ================================================================ стадии
def freeze(tabs: dict, picks: dict, t0: float):
    cands = {m: tabs[m][picks[m]]["target"]["page_share"] for m in SAMPLINGS if picks[m] is not None}
    mode = max(cands, key=lambda m: (round(cands[m], 4), m == "top")) if cands else "top"
    thr = picks[mode] if cands else None
    print("выбрано:", mode, thr, f"за {time.time() - t0:.0f} с", flush=True)
    cfg = Config(judge_enabled=True, judge_sampling=mode).model_dump()
    # второй, заранее объявленный вариант для финала: другой способ выборки с лучшим порогом при ослабленном пределе потери
    # (SECOND_MAX_LOSS). Он НЕ выбран по основному критерию и в отчёте помечен как такой.
    other = [m for m in SAMPLINGS if m != mode][0]
    thr2 = M.pick_threshold(tabs[other], SECOND_MAX_LOSS)
    variants = [dict(role="primary", sampling=mode, threshold=thr)]
    if thr2 is not None:
        variants.append(dict(role="secondary", sampling=other, threshold=thr2, max_loss=SECOND_MAX_LOSS))
    frozen = dict(threshold=thr, sampling=mode, variants=variants, model=MODEL, max_loss=MAX_LOSS, code_hash=code_hash(), cfg=cfg,
                  dev_seeds=DEV_SEEDS, picks={m: picks[m] for m in SAMPLINGS}, selected_at=datetime.now().isoformat(timespec="seconds"))
    print("варианты финала:", variants, flush=True)
    FROZEN.write_text(json.dumps(frozen, ensure_ascii=False, indent=1))


def stage_dev(workers: int):
    """Dev: два способа выбора примеров (top, spread) x пороги. Выбор: у каждого способа свой порог по критерию, затем
    способ с большей долей «по делу» среди будящих (при равенстве top: он задан в постановке)."""
    t0 = time.time()
    V3.mkdir(parents=True, exist_ok=True)
    print("== DEV", DEV_SEEDS, flush=True)
    tabs, picks = {}, {}
    for mode in SAMPLINGS:
        cfg = Config(judge_enabled=True, judge_sampling=mode).model_dump()
        print(f"-- выборка {mode}", flush=True)
        runs = collect(DEV_SEEDS, workers, cfg, MODEL)
        (V3 / f"dev-201-210-{mode}.json").write_text(json.dumps(runs, ensure_ascii=False))
        tabs[mode] = table(runs)
        picks[mode] = M.pick_threshold(tabs[mode], MAX_LOSS)
        print(f"выборка {mode}, порог {picks[mode]}")
        print(rows_md(tabs[mode]))
        print(loss_md(tabs[mode]))
        print(quiet_md(tabs[mode]), flush=True)
    freeze(tabs, picks, t0)


def load_table_from(path):
    runs = json.loads(Path(path).read_text())
    return runs, table(runs)


def blind_export(runs: list, thr, meta: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    items, key_runs, gem = [], [], {}
    for r in runs:
        kr = {k: v for k, v in r.items() if k != "bursts"}
        kr["bursts"] = []
        for b in r["bursts"]:
            bid = f"{r['seed']}-{r['noise']}-{b['id']}"
            if b["texts"]:
                items.append(dict(id=bid, product=b["product"], product_name=PRODUCTS[b["product"]][0], texts=b["texts"],
                                  prompt=build_prompt(PRODUCTS[b["product"]][0], b["texts"])))
            gem[bid] = b["share"]
            kr["bursts"].append({k: v for k, v in b.items() if k not in ("texts", "problem", "items", "example_ids", "alert_reason")})
        key_runs.append(kr)
    (directory / "alerts_blind.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    (directory / "key.json").write_text(json.dumps(dict(meta=meta, threshold=thr, gemma_share=gem, runs=key_runs), ensure_ascii=False))
    print(f"слепой файл: {len(items)} тревог-всплесков -> {directory}")


def vdir(v: dict) -> Path:
    """Папка слепого файла: основной вариант в корне, второй рядом."""
    return BLIND if v["role"] == "primary" else BLIND / f"{v['role']}-{v['sampling']}"


def stage_final(workers: int, allow_rerun: bool, allow_code: bool):
    assert FROZEN.exists(), "сначала --stage dev"
    fz = json.loads(FROZEN.read_text())
    tag = "401-410"
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    h = code_hash()
    after_seen = False
    if h != fz["code_hash"] and not allow_code:
        sys.exit(f"код изменился после заморозки ({fz['code_hash']} -> {h}); конфиг заморожен. --allow-code-change пометит результат")
    if h != fz["code_hash"]:
        after_seen = True
    if tag in lock and not allow_rerun:
        sys.exit("final на 401-410 уже запускали; повторно подбирать нельзя (--allow-rerun пометит результат «после просмотра»)")
    if tag in lock:
        after_seen = True
    if tag not in lock:
        lock[tag] = dict(code=h, variants=fz["variants"], first_run=datetime.now().isoformat(timespec="seconds"))
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1))
    print("== FINAL", FINAL_SEEDS, "варианты", fz["variants"], "хэш", h, flush=True)
    t0 = time.time()
    meta = dict(model=fz["model"], code_hash=h, after_seen=after_seen, dev_seeds=DEV_SEEDS, final_seeds=FINAL_SEEDS,
                generated_at=datetime.now().isoformat(timespec="seconds"))
    for v in fz["variants"]:
        cfg = dict(fz["cfg"], judge_sampling=v["sampling"])
        if v["threshold"] is not None:
            cfg["judge_min_share"] = v["threshold"]
        print(f"-- вариант {v}", flush=True)
        runs = collect(FINAL_SEEDS, workers, cfg, fz["model"])
        (V3 / f"final-401-410-{v['sampling']}.json").write_text(json.dumps(runs, ensure_ascii=False))
        blind_export(runs, v["threshold"], dict(meta, variant=v), vdir(v))
    meta["final_seconds"] = round(time.time() - t0)
    build_outputs(meta)


def build_outputs(meta: dict):
    fz = json.loads(FROZEN.read_text())
    thr = fz["threshold"]
    dev_tabs = {}
    for mode in SAMPLINGS:
        dev_runs, dev_tabs[mode] = load_table_from(V3 / f"dev-201-210-{mode}.json")
    dev_tab = dev_tabs[fz["sampling"]]
    meta = dict(meta, dev_runs=len(dev_runs), threshold=thr, sampling=fz["sampling"], picks=fz["picks"], variants=fz["variants"])
    dev = dict(table=dev_tab, threshold=thr, tabs=dev_tabs)
    finals = []
    for v in fz["variants"]:
        f = V3 / f"final-401-410-{v['sampling']}.json"
        if not f.exists():
            continue
        fruns, ftab = load_table_from(f)
        meta["final_runs"] = len(fruns)
        by_noise = {n: M.aggregate([r for r in fruns if r["noise"] == n], v["threshold"]) for n in RE.NOISES}
        # самопроверка: уровни, которые поставил конвейер, совпадают с пересчётом по порогу
        n_stream = sum(1 for r in fruns for inc in r["incidents"] if any(b["tier"] == "page" for b in r["bursts"] if b["id"] in inc))
        assert v["threshold"] is None or n_stream == ftab[v["threshold"]]["n_page"], (n_stream, ftab[v["threshold"]]["n_page"])
        finals.append(dict(variant=v, table=ftab, by_noise=by_noise))
        meta["tiers_check_ok"] = True
    OUT_REPORT.write_text(render_report(dev, finals, meta), encoding="utf-8")
    OUT_METRICS.write_text(json.dumps(dict(meta=meta, definitions=DEFS, dev={m: {key(k): v for k, v in t.items()} for m, t in dev_tabs.items()},
                                           final=[dict(variant=f["variant"], table={key(k): v for k, v in f["table"].items()},
                                                       by_noise=f["by_noise"]) for f in finals]),
                                      ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"отчёт: {OUT_REPORT}\nметрики: {OUT_METRICS}")
    for f in finals:
        print("\nвариант", f["variant"])
        print(rows_md(f["table"]))
        print(loss_md(f["table"]))
        print(quiet_md(f["table"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "freeze", "final", "report"], required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--allow-rerun", action="store_true")
    ap.add_argument("--allow-code-change", action="store_true")
    a = ap.parse_args()
    if a.stage == "dev":
        stage_dev(a.workers)
        fz = json.loads(FROZEN.read_text())
        build_outputs(dict(model=MODEL, dev_seeds=DEV_SEEDS, final_seeds=FINAL_SEEDS, generated_at=datetime.now().isoformat(timespec="seconds"),
                           code_hash=fz["code_hash"]))
    elif a.stage == "freeze":      # заморозка из сохранённых dev-прогонов (без пересчёта)
        tabs = {m: load_table_from(V3 / f"dev-201-210-{m}.json")[1] for m in SAMPLINGS}
        freeze(tabs, {m: M.pick_threshold(tabs[m], MAX_LOSS) for m in SAMPLINGS}, time.time())
        fz = json.loads(FROZEN.read_text())
        build_outputs(dict(model=MODEL, dev_seeds=DEV_SEEDS, final_seeds=FINAL_SEEDS, generated_at=datetime.now().isoformat(timespec="seconds"),
                           code_hash=fz["code_hash"]))
    elif a.stage == "final":
        stage_final(a.workers, a.allow_rerun, a.allow_code_change)
    else:
        fz = json.loads(FROZEN.read_text())
        lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
        build_outputs(dict(model=fz["model"], dev_seeds=DEV_SEEDS, final_seeds=FINAL_SEEDS, code_hash=lock.get("401-410", {}).get("code", fz["code_hash"]),
                           generated_at=datetime.now().isoformat(timespec="seconds")))


if __name__ == "__main__":
    main()
