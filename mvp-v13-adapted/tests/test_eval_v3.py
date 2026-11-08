"""Метрики оценки v3 (eval/v3_metrics.py): совпадения тревог с регрессиями, уровни, агрегация, выбор порога."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import v3_metrics as M  # noqa: E402

D = datetime


def ev(eid="E1", product="b2b", onset=D(2026, 1, 5, 16, 0), fix_ts=None, fix_effective=None, topic="t1"):
    return dict(id=eid, product=product, topic=topic, onset_ts=onset.isoformat(timespec="minutes"),
                fix_ts=fix_ts.isoformat(timespec="minutes") if fix_ts else None, fix_effective=fix_effective)


def test_match_flags_strict_work_target_for_late_web_alert():
    e = ev()                                          # пн 16:00, веб
    det = D(2026, 1, 6, 9, 30)                        # вт 9:30: через 17.5 ч, рабочих 3 ч + 30 мин
    f = M.match_flags(e, det, "t1", "b2b", data_end=D(2026, 1, 28), cfg=M.Cfg())
    assert not f["strict"] and f["work"] and f["target"]
    assert abs(f["real_min"] - 17.5 * 60) < 1e-6 and abs(f["work_min"] - 210) < 1e-6


def test_match_flags_mobile_work_equals_real():
    e = ev(product="azs")
    det = D(2026, 1, 6, 9, 30)
    f = M.match_flags(e, det, "t1", "azs", data_end=D(2026, 1, 28), cfg=M.Cfg())
    assert not f["strict"] and not f["work"] and f["target"]            # у приложения рабочие часы не применяются
    assert f["work_min"] == f["real_min"]


def test_match_flags_wrong_topic_or_product_or_too_early():
    e = ev()
    kw = dict(data_end=D(2026, 1, 28), cfg=M.Cfg())
    assert not any(M.match_flags(e, D(2026, 1, 5, 16, 30), "other", "b2b", **kw)[k] for k in ("strict", "work", "target"))
    assert not any(M.match_flags(e, D(2026, 1, 5, 16, 30), "t1", "portal", **kw)[k] for k in ("strict", "work", "target"))
    assert not M.match_flags(e, D(2026, 1, 5, 15, 0), "t1", "b2b", **kw)["target"]      # раньше -30 мин
    assert M.match_flags(e, D(2026, 1, 5, 15, 40), "t1", "b2b", **kw)["strict"]          # допуск -30 мин


def test_target_ends_with_effective_fix_plus_12h_but_not_with_ineffective():
    fix = D(2026, 1, 7, 12, 0)
    late = D(2026, 1, 8, 10, 0)                       # через 22 ч после фикса
    kw = dict(data_end=D(2026, 1, 28), cfg=M.Cfg())
    assert not M.match_flags(ev(fix_ts=fix, fix_effective=True), late, "t1", "b2b", **kw)["target"]
    assert M.match_flags(ev(fix_ts=fix, fix_effective=False), late, "t1", "b2b", **kw)["target"]
    assert M.match_flags(ev(fix_ts=fix, fix_effective=True), fix + timedelta(hours=11), "t1", "b2b", **kw)["target"]


def burst(bid, share, strict=(), work=(), target=(), ttc=None):
    return dict(id=bid, share=share, strict=list(strict), work=list(work or strict), target=list(target or work or strict),
                ttc=ttc or {})


def run_of(bursts, incidents, events=("E1", "E2")):
    return dict(seed=1, noise="mid", days=7.0, events=[dict(id=e, product="b2b", mobile=False) for e in events],
                bursts=bursts, incidents=incidents)


def sample_run():
    return run_of([
        burst("B1", 0.9, strict=["E1"], ttc={"E1": dict(real=30.0, work=30.0)}),                   # верная, сильный консенсус
        burst("B2", 0.5, strict=["E2"], ttc={"E2": dict(real=60.0, work=60.0)}),                   # верная, но модель сомневается
        burst("B3", 0.4),                                                                          # ложная, глушится
        burst("B4", None),                                                                         # ложная, не проверена -> будит
        burst("B5", 0.5, work=["E1"], target=["E1"], ttc={"E1": dict(real=1000.0, work=100.0)}),   # поздняя настоящая
    ], [["B1"], ["B2"], ["B3"], ["B4"], ["B5"]])


def test_aggregate_no_judge_counts_everything_as_page():
    a = M.aggregate([sample_run()], thr=None)
    assert a["n_page"] == 5 and a["n_quiet"] == 0 and a["strict"]["tp"] == 2
    assert a["strict"]["precision"] == 0.4 and a["strict"]["recall"] == 1.0
    assert a["target"]["page_share"] == 3 / 5 and abs(a["false_pages_per_week"] - 2.0) < 1e-9


def test_aggregate_threshold_splits_tiers_and_fail_open():
    a = M.aggregate([sample_run()], thr=0.75)
    assert a["n_page"] == 2 and a["n_quiet"] == 3           # B1 (0.9) и B4 (не проверена) будят
    assert a["unchecked"] == 1
    assert a["strict"]["tp"] == 1 and a["strict"]["recall"] == 0.5
    assert a["quiet"]["n"] == 3 and a["quiet"]["n_target"] == 2 and a["quiet"]["n_strict_tp"] == 1
    assert a["target"]["page_share"] == 0.5


def test_aggregate_incident_wakes_if_any_burst_page():
    r = run_of([burst("B1", 0.4), burst("B2", 0.9, strict=["E1"], ttc={"E1": dict(real=10.0, work=10.0)})], [["B1", "B2"]])
    a = M.aggregate([r], thr=0.75)
    assert a["n_page"] == 1 and a["strict"]["tp"] == 1


def test_work_metrics_and_strict_untouched():
    a = M.aggregate([sample_run()], thr=None)
    assert a["work"]["tp"] == 3 and a["work"]["recall"] == 1.0
    assert a["strict"]["ttc_median_min"] == 45.0 and a["work"]["ttc_median_min"] == 45.0     # E1: 30, E2: 60; у E1 берём самую раннюю


def test_share_override_replaces_judge_answers():
    a = M.aggregate([sample_run()], thr=0.75, override={"1-mid-B3": 1.0, "1-mid-B1": 0.0})
    assert a["n_page"] == 2 and a["strict"]["tp"] == 0     # будят B3 (1.0) и B4 (не проверена); B1 (0.0) заглушена


def test_pick_threshold_max_target_precision_with_loss_cap():
    base = dict(strict=dict(tp=100), target=dict(page_share=0.5), false_pages_per_week=10.0)
    mk = lambda tp, share, fp: dict(strict=dict(tp=tp), target=dict(page_share=share), false_pages_per_week=fp)
    table = {None: base, 0.625: mk(95, 0.6, 8.0), 0.75: mk(86, 0.7, 5.0), 0.875: mk(60, 0.9, 2.0)}
    assert M.pick_threshold(table, max_loss=0.15) == 0.75
    assert M.pick_threshold({None: base, 0.875: mk(60, 0.9, 2.0)}, max_loss=0.15) is None   # никто не прошёл: без проверки


def test_shares_from_verdicts():
    items = [dict(id="a", texts=["1", "2", "3", "4"]), dict(id="b", texts=["1", "2"]), dict(id="c", texts=["1", "2"])]
    v = {"a": dict(problem="p", items=[True, True, True, False]), "b": dict(problem="p", items=[True]), "x": {}}
    assert M.shares_from_verdicts(v, items) == {"a": 0.75, "b": None, "c": None}
