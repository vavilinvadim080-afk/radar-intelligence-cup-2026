from datetime import datetime, timedelta

import numpy as np

from radar.verify import compare_rates, verify_fix

FIX = datetime(2026, 3, 16, 14, 0)


def _times(rate_before, rate_after, ctrl=False, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for h in range(-72, 12 + 72 + 12):
        r = rate_before if h < 0 else rate_after
        out += [FIX + timedelta(hours=h, minutes=float(rng.uniform(0, 60))) for _ in range(rng.poisson(r))]
    return sorted(out)


def test_hand_example():
    r = compare_rates(120, 18, 1000, 1000)
    assert r["verdict"] == "confirmed"
    assert abs(r["ratio"] - 0.15) < 1e-9 and abs(r["drop_percent"] - 85.0) < 1e-9


def test_no_change_not_confirmed_and_too_few_insufficient():
    assert compare_rates(50, 48, 1000, 1000)["verdict"] == "not_confirmed"
    assert compare_rates(6, 0, 1000, 1000)["verdict"] == "insufficient_data"


def test_exposure_matters():
    # после фикса окно вдвое «толще» по общему потоку: 100 против 80 это падение, а не рост
    r = compare_rates(100, 80, 1000, 2000)
    assert r["ratio"] == 0.4


def test_verify_fix_confirmed_with_control_and_not_without_drop():
    topic = _times(2.0, 0.1)
    control = _times(20.0, 20.0, seed=1)
    now = FIX + timedelta(days=5)
    r = verify_fix(topic, FIX, control_times=control, now=now)
    assert r["verdict"] == "confirmed" and r["drop_percent"] > 80
    flat = _times(2.0, 2.0, seed=2)
    assert verify_fix(flat, FIX, control_times=control, now=now)["verdict"] == "not_confirmed"


def test_weekend_effect_cancelled_by_control():
    """Тема и контроль одинаково упали из-за выходных: без падения самой темы фикс подтверждать нельзя."""
    topic = _times(2.0, 1.0, seed=3)      # упала вдвое...
    control = _times(20.0, 10.0, seed=4)   # ...но и весь продукт упал вдвое
    r = verify_fix(topic, FIX, control_times=control, now=FIX + timedelta(days=5))
    assert r["verdict"] == "not_confirmed" and abs(r["drop_percent"]) < 25


def test_not_enough_time_after_fix():
    topic = _times(2.0, 0.1)
    r = verify_fix(topic, FIX, control_times=_times(20, 20, seed=1), now=FIX + timedelta(days=1))
    assert r["verdict"] == "insufficient_data"
