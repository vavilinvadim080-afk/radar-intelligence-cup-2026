"""Полнота и ложные срабатывания маскирования на размеченном наборе tests/data/mask_gold.jsonl.

Запуск из mvp/:  PYTHONPATH=. .venv/bin/python eval/mask_eval.py
Сравнивает: v1 (замороженная копия eval/baseline/mask_v1.py, маскирование до правок 2026-10-20),
v2 (radar/mask.py, регулярки), v2+ner (radar/mask.py с флагом ner=True, если natasha стоит).
Пишет ../shared/metrics-mask.json.

Правило «поймано»: после маскирования в тексте не осталось ни одного слова (2+ символа) из размеченного значения.
Это строго: если замаскирован только хвост номера, значение считается пропущенным.
Ложное срабатывание: токен, чей оригинал не пересекается ни с одним размеченным значением строки.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLD = ROOT / "tests" / "data" / "mask_gold.jsonl"
OUT = ROOT / "shared" / "metrics-mask.json"
TYPES = ["PHONE", "EMAIL", "CARD", "ACCOUNT", "CAR_PLATE", "PASSPORT", "SNILS", "PERSON", "ADDRESS"]


def load_gold() -> list:
    return [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_v1():
    spec = importlib.util.spec_from_file_location("mask_v1", ROOT / "eval" / "baseline" / "mask_v1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.mask_text


def words(s: str) -> list:
    return [w for w in re.findall(r"\w+", s) if len(w) >= 2]


def evaluate(fn, rows: list) -> dict:
    per = {t: {"total": 0, "caught": 0, "missed": []} for t in TYPES}
    fp_tokens = defaultdict(int)
    fp_lines = []
    neg = [r for r in rows if not r["pii"]]
    t0 = time.perf_counter()
    for r in rows:
        masked, mapping = fn(r["text"])
        for p in r["pii"]:
            d = per[p["type"]]
            d["total"] += 1
            left = {w.lower() for w in re.findall(r"\w+", masked)}
            if any(w.lower() in left for w in words(p["value"])):
                d["missed"].append({"value": p["value"], "masked_text": masked})
            else:
                d["caught"] += 1
        vals = [p["value"].lower() for p in r["pii"]]
        for tok, orig in mapping.items():
            o = orig.lower()
            if not any(o in v or v in o for v in vals):
                kind = re.match(r"<([A-Z_]+)_\d+>", tok).group(1)
                fp_tokens[kind] += 1
                fp_lines.append({"kind": kind, "original": orig, "text": r["text"], "line_has_pii": bool(r["pii"])})
    dt = time.perf_counter() - t0
    neg_fp = {l["text"] for l in fp_lines if not l["line_has_pii"]}
    return {
        "by_type": {t: {"total": d["total"], "caught": d["caught"],
                        "recall": round(d["caught"] / d["total"], 3) if d["total"] else None, "missed": d["missed"]}
                    for t, d in per.items()},
        "all": {"total": sum(d["total"] for d in per.values()), "caught": sum(d["caught"] for d in per.values())},
        "false_positive_tokens_by_kind": dict(fp_tokens),
        "false_positive_items": fp_lines,
        "lines_without_pii": len(neg), "lines_without_pii_with_false_mask": len(neg_fp),
        "texts_per_sec": round(len(rows) / dt, 1),
    }


SETS = {"dev": "mask_gold.jsonl", "holdout": "mask_gold_holdout.jsonl"}


def main():
    from radar import mask as M
    res = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "rule": __doc__.split("\n\n")[1].strip(),
           "sets": {"dev": "mvp/tests/data/mask_gold.jsonl: по нему правились регулярки v2, поэтому цифры v2 здесь завышены",
                    "holdout": "mvp/tests/data/mask_gold_holdout.jsonl: написан после правок, регулярки по нему не подгонялись"},
           "variants": {}}
    ner_ok = False
    try:
        from radar import mask_ner
        ner_ok = mask_ner.available()
    except Exception as e:   # noqa: BLE001
        res["ner_note"] = f"NER недоступен: {e!r}"
    fns = {"v1_before": load_v1(), "v2_regex": lambda t: M.mask_text(t),
           "v2_regex_ctx": lambda t: M.mask_text(t, ctx_names=True)}
    if ner_ok:
        fns["v2_regex_ner"] = lambda t: M.mask_text(t, ner=True)
        fns["v2_regex_ctx_ner"] = lambda t: M.mask_text(t, ner=True, ctx_names=True)
    for setname, fname in SETS.items():
        rows = [json.loads(l) for l in (ROOT / "tests" / "data" / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        res.setdefault("lines", {})[setname] = {"all": len(rows), "with_pii": sum(1 for r in rows if r["pii"]),
                                                "without_pii": sum(1 for r in rows if not r["pii"])}
        for name, fn in fns.items():
            res["variants"].setdefault(name, {})[setname] = evaluate(fn, rows)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, sets in res["variants"].items():
        for setname, v in sets.items():
            print(name, setname, v["all"], "ложных строк без ПДн:", v["lines_without_pii_with_false_mask"], "/", v["lines_without_pii"],
                  "| ", {t: f"{d['caught']}/{d['total']}" for t, d in v["by_type"].items() if d["total"]}, "| %s текстов/с" % v["texts_per_sec"])


def cascade_main(n_timing: int = 10000, seed: int = 401):
    """Каскад «регулярки -> NER только когда нужно»: три режима на dev и holdout и время на n_timing синтетических обращений.
    Дописывает блок "cascade" в ../shared/metrics-mask.json, остальные блоки не трогает.
    Запуск: nice -n 10 PYTHONPATH=. .venv/bin/python eval/mask_eval.py --cascade"""
    from radar import mask as M
    from radar import mask_ner
    if not mask_ner.available():
        sys.exit("natasha не установлена: каскад нечем сравнивать (uv pip install -r requirements-ner.txt)")
    res = json.loads(OUT.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    mask_ner._load()   # загрузка моделей один раз, в замер времени она не входит
    load_s = round(time.perf_counter() - t0, 2)
    modes = {"regex_only": False, "always_ner": True, "cascade_auto": "auto"}
    block = {"generated_at": datetime.now().isoformat(timespec="seconds"),
             "rule": "needs_ner(остаток после регулярок): заглавное слово не в начале предложения и не из словаря терминов, или триггер "
                     "(меня зовут, с уважением, ул., проспект, г., д. N, [at], это/я + Заглавное), или >=5 цифр подряд, или латинско-цифровая смесь >=5, "
                     "или латинское слово >=5 не из словаря. Пороги: Config.cascade_*",
             "ner_model_load_s": load_s, "sets": {}, "timing": {}}
    for setname, fname in SETS.items():
        rows = [json.loads(l) for l in (ROOT / "tests" / "data" / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        block["sets"][setname] = {}
        for name, flag in modes.items():
            M.cascade_stats(reset=True)
            v = evaluate(lambda t, f=flag: M.mask_text(t, ner=f), rows)
            st = M.cascade_stats()
            entry = {"caught": v["all"]["caught"], "total": v["all"]["total"],
                     "by_type": {t: f"{d['caught']}/{d['total']}" for t, d in v["by_type"].items() if d["total"]},
                     "false_positive_tokens": sum(v["false_positive_tokens_by_kind"].values()),
                     "false_positive_tokens_by_kind": v["false_positive_tokens_by_kind"],
                     "lines_without_pii_with_false_mask": v["lines_without_pii_with_false_mask"], "lines_without_pii": v["lines_without_pii"]}
            if flag == "auto":
                entry["texts_with_ner_call"] = st["ner_calls"]
                entry["texts"] = st["texts"]
                entry["ner_share"] = round(st["ner_calls"] / st["texts"], 3) if st["texts"] else None
            block["sets"][setname][name] = entry
        a, c = block["sets"][setname]["always_ner"]["caught"], block["sets"][setname]["cascade_auto"]["caught"]
        block["sets"][setname]["criterion_cascade_ge_always_minus_2"] = c >= a - 2
    from radar.gen import generate
    tickets = []
    day_volume = 800
    while len(tickets) < n_timing:
        tickets += generate(seed + len(tickets), "mid", day_volume).tickets
    texts = [t.text for t in tickets[:n_timing]]
    block["timing"]["texts"] = len(texts)
    block["timing"]["source"] = f"синтетика radar.gen, seeds начиная с {seed}, шум mid, {day_volume} в день; тексты те же для трёх режимов"
    for name, flag in modes.items():
        M.cascade_stats(reset=True)
        t0 = time.perf_counter()
        for tx in texts:
            M.mask_text(tx, ner=flag)
        dt = time.perf_counter() - t0
        st = M.cascade_stats()
        block["timing"][name] = {"seconds": round(dt, 2), "texts_per_sec": round(len(texts) / dt, 1)}
        if flag == "auto":
            block["timing"][name]["ner_calls"] = st["ner_calls"]
            block["timing"][name]["ner_share"] = round(st["ner_calls"] / st["texts"], 3)
    tm = block["timing"]
    block["timing"]["cascade_vs_always_speedup_x"] = round(tm["always_ner"]["seconds"] / tm["cascade_auto"]["seconds"], 2)
    block["timing"]["cascade_vs_regex_slowdown_x"] = round(tm["cascade_auto"]["seconds"] / tm["regex_only"]["seconds"], 2)
    res["cascade"] = block
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in block.items() if k != "rule"}, ensure_ascii=False, indent=1))


def _has_ner_flag(M) -> bool:
    import inspect
    return "ner" in inspect.signature(M.mask_text).parameters


if __name__ == "__main__":
    if "--cascade" in sys.argv:
        cascade_main()
    else:
        main()
