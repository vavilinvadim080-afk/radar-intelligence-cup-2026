import numpy as np
from datetime import datetime, timedelta

from radar.config import Config
from radar.detect import baseline_ewma, baseline_poisson, zscore
from radar.models import Ticket
from radar.pipeline import RadarStream


def test_poisson_z_quiet_vs_burst():
    rng = np.random.default_rng(0)
    counts = list(rng.poisson(5, 168))          # неделя часов со ставкой 5 в час
    g = [1.0] * 168
    rate = baseline_poisson(counts, g)
    assert 4.5 < rate < 5.5
    z_quiet, lam = zscore(5, rate, 1.0, 60)
    z_burst, _ = zscore(30, rate, 1.0, 60)
    assert z_quiet < 1.5 and z_burst > 6
    assert abs(lam - rate) < 1e-9


def test_zscore_is_exact_tail_not_normal_approximation():
    # ожидали 4 человека, пришло 14. Нормальное приближение даёт 5 сигм, точный хвост Пуассона P(X>=14)=7.6e-5
    z, lam = zscore(14, 4.0, 1.0, 60)
    assert lam == 4.0 and 3.6 < z < 4.0
    # малая ставка: ждали 1, пришло 7. Нормальное приближение 6 сигм, на деле около 3.8
    z_small, _ = zscore(7, 1.0, 1.0, 60)
    assert 3.5 < z_small < 4.1
    # большая ставка: приближения сходятся
    z_big, _ = zscore(160, 100.0, 1.0, 60)
    assert abs(z_big - 6.0) < 0.8


def test_ewma_weights_recent_hours_more():
    counts = [2.0] * 100 + [10.0] * 24             # ставка выросла в последние сутки
    g = [1.0] * 124
    m_fast, _ = baseline_ewma(counts, g, 6.0)
    m_slow, _ = baseline_ewma(counts, g, 200.0)
    assert m_fast > m_slow and m_fast > 8


def test_missing_hours_are_skipped():
    counts = [5, None, 5, None]
    assert baseline_poisson(counts, [1, 1, 1, 1]) == 5.0


def _stream_with_series(burst: bool, seed=0):
    """Искусственный поток: одна тема (один вектор), ставка 2 человека в час, на 6-й день 25 человек за 25 минут."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 3, 2)
    cfg = Config(warmup_days=3, min_users=6, z_threshold=4.5, use_hdbscan=False)
    s = RadarStream([], cfg, dim=4, t0=t0, start_ts=t0)
    vec = np.array([1.0, 0, 0, 0], dtype=np.float32)
    times = []
    for h in range(7 * 24):
        for _ in range(rng.poisson(2)):
            times.append(t0 + timedelta(hours=h, minutes=float(rng.uniform(0, 60))))
    if burst:
        b0 = t0 + timedelta(days=5, hours=12)
        times += [b0 + timedelta(minutes=float(rng.uniform(0, 25))) for _ in range(25)]
    times.sort()
    for i, ts in enumerate(times):
        s.push(Ticket(f"T{i}", ts, "azs", "app_chat", "android", "4.1.0", f"u{i}", "не работает оплата"), "не работает оплата", vec)
    s.finish(times[-1])
    return s


def test_stream_detects_burst_and_stays_quiet_without_it():
    with_burst = _stream_with_series(True)
    assert len(with_burst.bursts) >= 1
    b = with_burst.bursts[0]
    assert datetime(2026, 3, 7, 12, 0) <= b.detected_at <= datetime(2026, 3, 7, 13, 30)
    quiet = _stream_with_series(False)
    assert len(quiet.bursts) == 0


def test_one_user_is_one_vote():
    cfg = Config(warmup_days=0, use_hdbscan=False)
    t0 = datetime(2026, 3, 2)
    s = RadarStream([], cfg, dim=2, t0=t0, start_ts=t0)
    vec = np.array([1.0, 0], dtype=np.float32)
    for i in range(10):    # один и тот же человек пишет 10 раз за 10 минут
        s.push(Ticket(f"T{i}", t0 + timedelta(minutes=i), "azs", "app_chat", "ios", "", "u1", "x"), "x", vec)
    key = next(iter(s.series))
    assert len(s.vote_times[key]) == 1
