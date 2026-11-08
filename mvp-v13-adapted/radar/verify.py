"""Проверка фикса: упала ли интенсивность темы после релиза-фикса.

Математика (простыми словами, число для калькулятора в WALKTHROUGH.md):
 * считаем обращения темы за N дней ДО фикса (n_до) и N дней ПОСЛЕ (n_после, после паузы lag на раскатку версии);
 * «экспозиция» окна: сколько вообще обращений ждали бы в это окно, если бы тема не менялась.
   С контролем: экспозиция = число обращений того же продукта БЕЗ этой темы за то же окно.
   Так выходные и сезонность гасятся: они трогают и тему, и контроль одинаково.
   Без контроля: экспозиция = сумма недельного профиля продукта по часам окна (hourly_profile).
 * интенсивность = n / экспозиция, отношение R = (n_после/э_после) / (n_до/э_до). R=1 значит «ничего не изменилось».
 * тест Пуассона на равенство интенсивностей сводится к биномиальному: при равных интенсивностях каждое из
   n=n_до+n_после обращений попадает в «после» с вероятностью p0 = э_после/(э_до+э_после).
   Односторонний p-value = P(X <= n_после), X ~ Binomial(n, p0).
 * доверительный интервал для R: берём точный интервал Клоппера-Пирсона для доли q = n_после/n и переводим
   R = q/(1-q) * э_до/э_после.
 * вердикт: confirmed, если p < alpha И верхняя граница R не выше max_ratio (падение не меньше 20% с запасом);
   insufficient_data, если n_до < min_events (тему нечем сравнивать); иначе not_confirmed.
"""
from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta
from typing import Optional

from scipy.stats import binomtest


def compare_rates(n_pre: int, n_post: int, e_pre: float, e_post: float, alpha: float = 0.01,
                  max_ratio: float = 0.8, min_events: int = 10, ci_level: float = 0.95) -> dict:
    """Чистая математика без дат. Возвращает словарь с вердиктом и числами."""
    out = dict(verdict="insufficient_data", ratio=None, ci_low=None, ci_high=None, drop_percent=None,
               p_value=None, n_pre=n_pre, n_post=n_post)
    n = n_pre + n_post
    if n_pre < min_events or n == 0 or e_pre <= 0 or e_post <= 0:
        return out
    p0 = e_post / (e_pre + e_post)
    test = binomtest(n_post, n, p0, alternative="less")
    ci = binomtest(n_post, n, p0).proportion_ci(confidence_level=ci_level, method="exact")   # двусторонний интервал для q
    k = e_pre / e_post

    def to_ratio(q):
        return float("inf") if q >= 1 else q / (1 - q) * k

    ratio = (n_post / n_pre) * k if n_pre else float("inf")
    out.update(ratio=ratio, ci_low=float(to_ratio(ci.low)), ci_high=float(to_ratio(ci.high)), p_value=float(test.pvalue),
               drop_percent=(1 - ratio) * 100)
    out["verdict"] = "confirmed" if (test.pvalue < alpha and out["ci_high"] <= max_ratio) else "not_confirmed"
    return out


def weekly_profile(times: list[datetime]) -> list[float]:
    """Недельный профиль: среднее число обращений за час для каждого из 168 часов недели (пн 0:00 = 0)."""
    if not times:
        return [1.0] * 168
    sums = [0.0] * 168
    for t in times:
        sums[t.weekday() * 24 + t.hour] += 1
    first, last = min(times).date(), max(times).date()
    per_wd = [0] * 7   # сколько раз каждый день недели встретился в периоде
    for i in range((last - first).days + 1):
        per_wd[(first + timedelta(days=i)).weekday()] += 1
    return [max(sums[wd * 24 + h] / max(per_wd[wd], 1), 1e-6) for wd in range(7) for h in range(24)]


def exposure_from_profile(start: datetime, end: datetime, profile: list[float]) -> float:
    """Сумма профиля по часам окна [start, end): сколько обращений «положено» в это окно."""
    total, t = 0.0, start
    while t < end:
        nxt = min(end, t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
        total += profile[t.weekday() * 24 + t.hour] * (nxt - t).total_seconds() / 3600
        t = nxt
    return total


def _count(times: list[datetime], a: datetime, b: datetime) -> int:
    """Сколько элементов отсортированного списка в [a, b)."""
    return bisect_left(times, b) - bisect_left(times, a)


def verify_fix(topic_times: list[datetime], fix_ts: datetime, control_times: Optional[list[datetime]] = None,
               hourly_profile: Optional[list[float]] = None, days: int = 3, lag_hours: int = 12,
               pre_from: Optional[datetime] = None, now: Optional[datetime] = None, alpha: float = 0.01,
               max_ratio: float = 0.8, min_events: int = 10) -> dict:
    """topic_times, control_times: отсортированные времена обращений (темы и контроля).
    pre_from: не заходить в «до» глубже этого момента (например, начала всплеска).
    Если данных после фикса меньше нужных days дней (now - post_start < days), вердикт insufficient_data."""
    post_start = fix_ts + timedelta(hours=lag_hours)
    post_end = post_start + timedelta(days=days)
    pre_end = fix_ts
    pre_start = fix_ts - timedelta(days=days)
    if pre_from is not None:
        pre_start = max(pre_start, pre_from)
    res = dict(verdict="insufficient_data", ratio=None, ci_low=None, ci_high=None, drop_percent=None, p_value=None,
               n_pre=0, n_post=0, used_control=control_times is not None)
    if now is not None and now < post_end:
        return res
    if pre_end <= pre_start:
        return res
    n_pre, n_post = _count(topic_times, pre_start, pre_end), _count(topic_times, post_start, post_end)
    if control_times is not None:
        e_pre, e_post = float(_count(control_times, pre_start, pre_end)), float(_count(control_times, post_start, post_end))
    else:
        prof = hourly_profile or [1.0] * 168
        e_pre, e_post = exposure_from_profile(pre_start, pre_end, prof), exposure_from_profile(post_start, post_end, prof)
    out = compare_rates(n_pre, n_post, e_pre, e_post, alpha, max_ratio, min_events)
    out["used_control"] = control_times is not None
    return out


# ---------------------------------------------------------------- v2: возврат к норме
def residual_test(n_base: int, e_base: float, n_pre: int, e_pre: float, n_post: int, e_post: float,
                  alpha: float = 0.01, max_residual: float = 0.35, min_events: int = 10,
                  n_draws: int = 4000, seed: int = 0) -> dict:
    """Чистая математика проверки фикса v2 (числовой пример в WALKTHROUGH.md).

    Три окна одной темы (одного набора кластеров): норма (дни до релиза), всплеск (от начала всплеска до фикса),
    после фикса. Интенсивность r = n / e, где e это экспозиция (обращения контроля: остальной продукт).
      остаток R = (r_после - r_норма) / (r_всплеск - r_норма)
    R = 0: тема вернулась к норме, фикс сработал; R = 1: ничего не изменилось. В отличие от v1 (отношение
    «после/до») здесь норма темы вычитается: у темы с большим фоном падение на 20% не выдаётся за успех,
    а у темы, где регрессия просто затухает, остаток видно.
    Проверки:
      1) был ли всплеск: односторонний биномиальный тест n_всплеск против нормы (как в compare_rates), p < alpha;
      2) интервал для R: интенсивности разыгрываются из гамма-апостериорных Gamma(n + 0.5, 1/e)
         (априорное Джеффриса), R считается на каждой из n_draws выборок, берём 2.5% и 97.5% квантили.
    Вердикт: confirmed, если всплеск был и верхняя граница R не выше max_residual; insufficient_data,
    если обращений в окне всплеска меньше min_events или всплеска относительно нормы не видно; иначе not_confirmed."""
    out = dict(verdict="insufficient_data", residual=None, residual_ci_low=None, residual_ci_high=None,
               ratio=None, ci_low=None, ci_high=None, drop_percent=None, p_value=None,
               n_pre=n_pre, n_post=n_post, n_base=n_base)
    if n_pre < min_events or min(e_base, e_pre, e_post) <= 0:
        return out
    # 1) всплеск над нормой: при равных интенсивностях доля «всплеска» в n_base + n_pre равна e_pre / (e_base + e_pre)
    p0 = e_pre / (e_base + e_pre)
    p_elev = float(binomtest(n_pre, n_base + n_pre, p0, alternative="greater").pvalue)
    # 2) остаток и его интервал
    import numpy as np
    rng = np.random.default_rng(seed)
    rb = rng.gamma(n_base + 0.5, 1.0 / e_base, n_draws)
    rp = rng.gamma(n_pre + 0.5, 1.0 / e_pre, n_draws)
    ra = rng.gamma(n_post + 0.5, 1.0 / e_post, n_draws)
    ex = rp - rb
    R = np.where(ex > 0, (ra - rb) / np.where(ex > 0, ex, 1.0), 1e9)   # нет всплеска в выборке: остаток «бесконечный»
    r_b, r_p, r_a = n_base / e_base, n_pre / e_pre, n_post / e_post
    resid = (r_a - r_b) / (r_p - r_b) if r_p > r_b else None
    lo, hi = (float(x) for x in np.quantile(R, [0.025, 0.975]))
    ratio = r_a / r_p if r_p > 0 else None
    rq = np.quantile(ra / rp, [0.025, 0.975])
    out.update(residual=resid, residual_ci_low=lo, residual_ci_high=hi, ratio=ratio, ci_low=float(rq[0]),
               ci_high=float(rq[1]), drop_percent=None if ratio is None else (1 - ratio) * 100, p_value=p_elev)
    if p_elev >= alpha or resid is None:
        out["verdict"] = "insufficient_data"      # всплеска над нормой не видно: проверять нечего
    elif hi <= max_residual:
        out["verdict"] = "confirmed"
    else:
        out["verdict"] = "not_confirmed"
    return out


def verify_fix_baseline(topic_times: list[datetime], control_times: list[datetime], fix_ts: datetime,
                        burst_start: datetime, release_ts: datetime, days: int = 3, lag_hours: int = 12,
                        baseline_days: int = 7, now: Optional[datetime] = None, alpha: float = 0.01,
                        max_residual: float = 0.35, min_events: int = 10) -> dict:
    """Окна: норма [release_ts - baseline_days, release_ts), всплеск [max(burst_start, fix_ts - days), fix_ts),
    после [fix_ts + lag, fix_ts + lag + days). Времена отсортированы. Экспозиция = число обращений контроля."""
    post_start = fix_ts + timedelta(hours=lag_hours)
    post_end = post_start + timedelta(days=days)
    pre_start, pre_end = max(burst_start, fix_ts - timedelta(days=days)), fix_ts
    base_start, base_end = release_ts - timedelta(days=baseline_days), min(release_ts, burst_start)
    empty = dict(verdict="insufficient_data", residual=None, residual_ci_low=None, residual_ci_high=None, ratio=None,
                 ci_low=None, ci_high=None, drop_percent=None, p_value=None, n_pre=0, n_post=0, n_base=0, used_control=True)
    if (now is not None and now < post_end) or pre_end <= pre_start or base_end <= base_start:
        return empty
    n = lambda T, a, b: _count(T, a, b)
    out = residual_test(n(topic_times, base_start, base_end), float(n(control_times, base_start, base_end)),
                        n(topic_times, pre_start, pre_end), float(n(control_times, pre_start, pre_end)),
                        n(topic_times, post_start, post_end), float(n(control_times, post_start, post_end)),
                        alpha, max_residual, min_events)
    out["used_control"] = True
    return out
