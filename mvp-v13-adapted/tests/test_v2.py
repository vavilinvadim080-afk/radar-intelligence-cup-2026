"""Тесты частей v2: скан подмножеств, доля новой версии, проверка фикса «возврат к норме», тревога только после релиза."""
from datetime import datetime, timedelta

import numpy as np

from radar.config import Config
from radar.detect import adoption_share, cap_outliers, ltss_scan, ptail_z
from radar.models import Release, Ticket
from radar.pipeline import RadarStream
from radar.verify import residual_test


def test_subset_scan_sums_fragments():
    # тема разрезана на 3 осколка: в каждом 4 человека при ожидании 1. По отдельности z 2.1, вместе 12 против 3: z 3.8
    rows = [("a", 4, 1.0, 1.0), ("b", 4, 1.0, 1.0), ("c", 4, 1.0, 1.0), ("d", 1, 1.0, 1.0)]
    z, C, B, keys = ltss_scan(rows, floor=1.0, min_member=2, member_z=1.5)
    assert set(keys) == {"a", "b", "c"} and C == 12 and B == 3.0
    assert abs(ptail_z(4, 1.0) - 2.08) < 0.01 and abs(z - 3.80) < 0.01


def test_subset_scan_ignores_unelevated_clusters():
    rows = [("a", 9, 1.0, 1.0), ("b", 2, 3.0, 1.0)]        # b не повышен: в сумму не идёт
    z, C, B, keys = ltss_scan(rows, floor=1.0, min_member=2, member_z=1.5)
    assert keys == ["a"] and C == 9


def test_adoption_share_excludes_regression_cluster():
    # 200 обращений фона, из них 4 на новой версии (2%); кластер регрессии: 10 из 12 на новой версии
    per_key = {"bg%d" % i: (20, 0) for i in range(10)}
    per_key["bg0"] = (20, 4)
    per_key["reg"] = (12, 10)
    q = adoption_share(per_key)
    assert q < 0.03                                     # сырая доля была бы 14/212 = 0.066


def test_cap_outliers_keeps_normal_hours():
    counts, g = [2, 3, 2, 30, 2, None], [1.0] * 6
    out = cap_outliers(counts, g)
    assert out[0] == 2 and out[5] is None and out[3] < 30


def test_residual_test_hand_example():
    # норма 0.01 на обращение контроля, всплеск 0.06, после фикса 0.0073: остаток около -0.05, подтверждено
    r = residual_test(100, 10000, 120, 2000, 22, 3000)
    assert r["verdict"] == "confirmed" and abs(r["residual"] - (-0.0533)) < 1e-3
    # после фикса 0.03: осталось 40% всплеска, не подтверждено
    assert residual_test(100, 10000, 120, 2000, 90, 3000)["verdict"] == "not_confirmed"
    # всплеска над нормой не было: проверять нечего
    assert residual_test(100, 10000, 25, 2000, 20, 3000)["verdict"] == "insufficient_data"


def _stream(release_at_day5: bool, hours_before: float = 2):
    """Одна тема, 2 человека в час; на 6-й день 25 человек за 25 минут. Релиз продукта за 2 часа до всплеска или нет."""
    rng = np.random.default_rng(0)
    t0 = datetime(2026, 3, 2)
    b0 = t0 + timedelta(days=5, hours=12)
    rel = [Release("azs", "4.2.0", b0 - timedelta(hours=hours_before), "regular")] if release_at_day5 else []
    s = RadarStream(rel, Config(warmup_days=3, min_users=6), dim=4, t0=t0, start_ts=t0)
    vec = np.array([1.0, 0, 0, 0], dtype=np.float32)
    times = [t0 + timedelta(hours=h, minutes=float(rng.uniform(0, 60))) for h in range(7 * 24) for _ in range(rng.poisson(2))]
    times += [b0 + timedelta(minutes=float(rng.uniform(0, 25))) for _ in range(25)]
    times.sort()
    for i, ts in enumerate(times):
        s.push(Ticket(f"T{i}", ts, "azs", "app_chat", "android", "4.1.0", f"u{i}", "x"), "x", vec)
    s.finish(times[-1])
    return s


def test_alert_only_after_release():
    with_rel = _stream(True)
    assert any(b.alert for b in with_rel.bursts)
    without = _stream(False)
    assert without.bursts and not any(b.alert for b in without.bursts)   # всплеск есть, но это «наблюдение»


# ---------------------------------------------------------------- сигнал версии, проверка фикса, воспроизводимость
def _version_stream(burst_version: str, seed: int = 0):
    """Фон: 2 человека/час, 5% обращений на новой версии 4.2.0 (релиз за 2 часа до всплеска). Всплеск 8 человек за 10 минут
    на версии burst_version: мало людей для темы (z темы ниже порога 6), но много для сигнала версии (ожидание 0.1)."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 3, 2)
    b0 = t0 + timedelta(days=5, hours=12)
    rel = [Release("azs", "4.2.0", b0 - timedelta(hours=2), "regular")]
    s = RadarStream(rel, Config(warmup_days=3, min_users=3), dim=4, t0=t0, start_ts=t0)
    vecs = np.eye(4, dtype=np.float32)          # 4 ортогональных темы: тема 0 всплеск, темы 1-3 фон (общая доля версии считается по ним)
    rows = []
    for h in range(7 * 24):
        for _ in range(rng.poisson(6)):
            ts = t0 + timedelta(hours=h, minutes=float(rng.uniform(0, 60)))
            ver = "4.2.0" if (ts >= rel[0].ts and rng.random() < 0.05) else "4.1.0"
            rows.append((ts, ver, int(rng.integers(1, 4))))
    rows += [(b0 + timedelta(minutes=float(rng.uniform(0, 10))), burst_version, 0) for _ in range(8)]
    rows.sort()
    for i, (ts, ver, k) in enumerate(rows):
        s.push(Ticket(f"T{i}", ts, "azs", "app_chat", "android", ver, f"u{i}", "x"), "x", vecs[k])
    s.finish(rows[-1][0])
    return s


def test_version_signal_only_on_new_version():
    on_new = _version_stream("4.2.0")
    assert any(b.alert and b.signal == "version" for b in on_new.bursts)
    on_old = _version_stream("4.1.0")                      # те же 8 человек, но на старой версии: сигнала версии нет
    assert not any(b.signal == "version" for b in on_old.bursts)
    assert not any(b.alert for b in on_old.bursts)         # 8 человек за 10 минут для темы не всплеск (z ниже порога)


def _times(start, n, span_h):
    return [start + timedelta(hours=span_h * i / n) for i in range(n)]


def test_verify_fix_with_and_without_control():
    fix = datetime(2026, 3, 10, 12)
    pre = _times(fix - timedelta(days=3), 90, 72)                  # 90 обращений темы до фикса
    post_start = fix + timedelta(hours=12)
    post_down = _times(post_start, 9, 72)                          # после: 9 (падение в 10 раз)
    post_same = _times(post_start, 88, 72)                         # после: 88 (не изменилось)
    ctrl = _times(fix - timedelta(days=3), 3000, 72 + 12 + 72)     # контроль идёт равномерно весь период
    from radar.verify import verify_fix
    ok = verify_fix(pre + post_down, fix, control_times=sorted(ctrl))
    assert ok["verdict"] == "confirmed" and ok["used_control"]
    bad = verify_fix(pre + post_same, fix, control_times=sorted(ctrl))
    assert bad["verdict"] == "not_confirmed"
    flat = verify_fix(pre + post_down, fix)                        # без контроля: плоский недельный профиль
    assert not flat["used_control"] and flat["verdict"] == "confirmed"
    few = verify_fix(pre[:5] + post_down, fix, control_times=sorted(ctrl))
    assert few["verdict"] == "insufficient_data"                   # меньше 10 событий до фикса


def test_verify_fix_baseline_returns_to_norm():
    from radar.verify import verify_fix_baseline
    rel = datetime(2026, 3, 2)
    burst = rel + timedelta(days=1)
    fix = burst + timedelta(days=3)
    # норма: 7 дней до релиза, 7 обращений темы на 7000 контроля (0.001); всплеск 3 дня: 180 на 3000 (0.06)
    base = _times(rel - timedelta(days=7), 7, 168)
    spike = _times(burst, 180, 72)
    ctrl = sorted(_times(rel - timedelta(days=7), 7000, 168) + _times(burst, 3000, 72) + _times(fix, 3000, 84))
    post_ok = _times(fix + timedelta(hours=12), 3, 72)             # вернулась к норме
    post_bad = _times(fix + timedelta(hours=12), 90, 72)           # осталась половина всплеска
    args = dict(fix_ts=fix, burst_start=burst, release_ts=rel)
    ok = verify_fix_baseline(sorted(base + spike + post_ok), ctrl, **args)
    bad = verify_fix_baseline(sorted(base + spike + post_bad), ctrl, **args)
    assert ok["verdict"] == "confirmed" and ok["residual"] < 0.1
    assert bad["verdict"] == "not_confirmed" and bad["residual"] > 0.35
    early = verify_fix_baseline(sorted(base + spike), ctrl, now=fix + timedelta(hours=13), **args)
    assert early["verdict"] == "insufficient_data"                 # после фикса данных ещё нет


def test_reproducible_by_seed():
    a, b = _stream(True), _stream(True)
    key = lambda s: [(x.detected_at, x.signal, x.z, x.alert) for x in s.bursts]
    assert key(a) == key(b)
    r1 = residual_test(100, 10000, 120, 2000, 22, 3000)
    r2 = residual_test(100, 10000, 120, 2000, 22, 3000)
    assert r1 == r2 and residual_test(100, 10000, 120, 2000, 22, 3000, seed=1)["residual_ci_high"] != r1["residual_ci_high"]
    from radar.gen import generate
    d1, d2 = generate(7, "low", 150), generate(7, "low", 150)
    assert [(t.id, t.ts, t.text) for t in d1.tickets] == [(t.id, t.ts, t.text) for t in d2.tickets]


def test_alert_window_48h():
    near = _stream(True, hours_before=40)          # релиз за 40 ч до всплеска: внутри окна 48 ч, тревога
    far = _stream(True, hours_before=60)           # за 60 ч: окно закрыто, всплеск только в наблюдение
    assert any(b.alert for b in near.bursts)
    assert far.bursts and not any(b.alert for b in far.bursts)
