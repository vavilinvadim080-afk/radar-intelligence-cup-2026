"""Метрики детектора v3: чистые функции без запуска конвейера (их тестирует tests/test_eval_v3.py).

Три способа сказать «тревога верная»:
  strict  как в eval/run_eval.py: тот же продукт и тема, срабатывание в [начало - 30 мин, начало + 6 ч] календарных.
  work    то же, но для веб-продуктов рабочего времени (b2b, portal, partners) допуск «6 часов» считается рабочими
          часами (пн-пт 9-19). Регрессия, начавшаяся в 18:00 в пятницу, иначе «не успевает» до утра понедельника.
          Для приложений (azs, bonus) work совпадает со strict.
  target  («по делу») тревога относится к настоящей регрессии того же продукта и темы в любой момент от
          (начало - 30 мин) до конца её активности. Конец активности по ground truth: если был эффективный фикс,
          то время фикса + 12 ч (дальше жалобы падают до 6% пика); иначе до конца набора данных
          (регрессия без фикса или с неэффективным фиксом тянется неделями). Тема и продукт определяются по меткам
          генератора, как в строгой оценке (burst_topic).
Инцидент = группа всплесков одного продукта с паузой до 2 часов (group_incidents). Инцидент «будит» (page), если
будит хотя бы один его всплеск; иначе quiet."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar.judge import tier_for              # noqa: E402
from radar.workhours import work_minutes      # noqa: E402


@dataclass
class Cfg:
    tol_before: timedelta = timedelta(minutes=30)
    tol_after: timedelta = timedelta(hours=6)
    work_tol_min: float = 6 * 60               # допуск в рабочих минутах
    business_products: tuple = ("b2b", "portal", "partners")
    hours: tuple = (9, 19)
    fix_grace: timedelta = timedelta(hours=12)


def active_end(e: dict, data_end: datetime, cfg: Cfg) -> datetime:
    if e.get("fix_ts") and e.get("fix_effective"):
        return min(data_end, datetime.fromisoformat(e["fix_ts"]) + cfg.fix_grace)
    return data_end


def match_flags(e: dict, detected: datetime, topic: Optional[str], product: str, data_end: datetime, cfg: Cfg) -> dict:
    """Как тревога (всплеск) detected с темой topic по продукту product соотносится с регрессией e."""
    out = dict(strict=False, work=False, target=False, real_min=None, work_min=None)
    if e["product"] != product or e["topic"] != topic:
        return out
    onset = datetime.fromisoformat(e["onset_ts"])
    real = (detected - onset).total_seconds() / 60
    biz = product in cfg.business_products
    wm = work_minutes(onset, detected, *cfg.hours) if biz else max(real, 0.0)
    out["real_min"], out["work_min"] = real, wm
    if detected < onset - cfg.tol_before:
        return out
    out["strict"] = detected <= onset + cfg.tol_after
    out["work"] = (wm <= cfg.work_tol_min) if biz else out["strict"]
    out["target"] = detected <= active_end(e, data_end, cfg)
    return out


# ---------------------------------------------------------------- агрегация
def _med(a):
    return float(np.median(a)) if a else None


def _div(a, b):
    return (a / b) if b else None


def aggregate(runs: list, thr: Optional[float], override: Optional[dict] = None) -> dict:
    """Метрики по запускам (seed x шум) при пороге доли thr. thr=None: проверки нет, все тревоги будят.
    override: {"seed-noise-burstid": доля} подменяет ответы модели (слепая проверка сильной моделью)."""
    override = override or {}
    n_events = n_page = n_quiet = unchecked = 0
    tp = {"strict": 0, "work": 0, "target": 0}
    q = dict(n=0, n_target=0, n_strict_tp=0, n_work_tp=0)
    caught = {"strict": 0, "work": 0, "target": 0}
    ttc_real, ttc_work = [], []
    days = 0.0
    for r in runs:
        days += r["days"]
        n_events += len(r["events"])
        bursts = {b["id"]: b for b in r["bursts"]}
        hit = {"strict": {}, "work": {}, "target": {}}     # событие -> лучшие (самые ранние) минуты среди будящих всплесков
        for inc in r["incidents"]:
            bs = [bursts[i] for i in inc]
            shares = [override.get(f"{r['seed']}-{r['noise']}-{b['id']}", b["share"]) for b in bs]
            unchecked += sum(1 for s in shares if s is None) if thr is not None else 0
            page = thr is None or any(tier_for(s, thr) == "page" for s in shares)
            flags = {k: any(b[k] for b in bs) for k in tp}
            if page:
                n_page += 1
                for k in tp:
                    tp[k] += flags[k]
                for b in bs:
                    for k in hit:
                        for eid in b[k]:
                            t = b["ttc"].get(eid, {})
                            h = hit[k].setdefault(eid, [None, None])
                            for j, key in enumerate(("real", "work")):
                                if t.get(key) is not None and (h[j] is None or t[key] < h[j]):
                                    h[j] = t[key]
            else:
                n_quiet += 1
                q["n"] += 1
                q["n_target"] += flags["target"]
                q["n_strict_tp"] += flags["strict"]
                q["n_work_tp"] += flags["work"]
        for k in caught:
            caught[k] += len(hit[k])
        ttc_real += [max(0.0, v[0]) for v in hit["strict"].values() if v[0] is not None]
        ttc_work += [max(0.0, v[1]) for v in hit["work"].values() if v[1] is not None]
    weeks = days / 7
    return dict(
        thr=thr, n_events=n_events, n_alerts=n_page + n_quiet, n_page=n_page, n_quiet=n_quiet, unchecked=unchecked, weeks=weeks,
        strict=dict(tp=tp["strict"], precision=_div(tp["strict"], n_page), caught=caught["strict"],
                    recall=_div(caught["strict"], n_events), ttc_median_min=_med(ttc_real)),
        work=dict(tp=tp["work"], precision=_div(tp["work"], n_page), caught=caught["work"],
                  recall=_div(caught["work"], n_events), ttc_median_min=_med(ttc_work)),
        target=dict(tp=tp["target"], page_share=_div(tp["target"], n_page), recall_events=_div(caught["target"], n_events)),
        quiet=dict(n=q["n"], n_target=q["n_target"], share_target=_div(q["n_target"], q["n"]),
                   n_strict_tp=q["n_strict_tp"], n_work_tp=q["n_work_tp"]),
        false_pages_per_week=_div(n_page - tp["target"], weeks),
        strict_false_pages_per_week=_div(n_page - tp["strict"], weeks),
        pages_per_week=_div(n_page, weeks),
    )


def pick_threshold(table: dict, max_loss: float = 0.15) -> Optional[float]:
    """Критерий выбора порога (задан до просмотра): максимум доли «по делу» среди будящих тревог при потере
    не больше max_loss строгих верных тревог относительно варианта без проверки (ключ None). При равенстве меньше
    ложных будящих в неделю. Если порогов, прошедших ограничение, нет: None (проверку не включать)."""
    base = table[None]["strict"]["tp"]
    ok = {t: v for t, v in table.items() if t is not None and base and (1 - v["strict"]["tp"] / base) <= max_loss}
    if not ok:
        return None
    return max(ok, key=lambda t: (round(ok[t]["target"]["page_share"] or 0, 4), -ok[t]["false_pages_per_week"]))


def shares_from_verdicts(verdicts: dict, blind_items: list) -> dict:
    """Вердикты сильной модели {id: {problem, items}} -> {id: доля}. Для каждого id слепого файла: доля = среднее items,
    если items список булевых длины, равной числу текстов; иначе None («не проверено», тревога будит)."""
    out = {}
    for it in blind_items:
        v = verdicts.get(it["id"])
        items = v.get("items") if isinstance(v, dict) else None
        ok = isinstance(items, list) and len(items) == len(it["texts"]) and all(isinstance(x, bool) for x in items)
        out[it["id"]] = (sum(items) / len(items)) if ok else None
    return out
