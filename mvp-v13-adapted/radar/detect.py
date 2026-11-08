"""Детектор всплесков: сравнивает число людей в окне 30/60 минут с ожидаемым по истории.

Вход функций: числа, не объекты. Поэтому их легко проверить руками (см. tests/test_detect.py и WALKTHROUGH.md).

Общая схема для одного кластера в момент now:
  x       = сколько РАЗНЫХ людей написали в этот кластер за последние W минут
  g       = сезонный множитель на этот час (суточный и недельный ритм продукта, средний будний час = 1)
  lambda  = сколько людей ждали в окне по истории = (ставка кластера в «средний» час) * g * W/60
  z       = число сигм: насколько редко видеть x человек при ожидании lambda (точный хвост Пуассона, см. zscore)
Всплеск, если z >= порога И x >= min_users.

Два способа оценить ставку и дисперсию:
  poisson: ставка = (сумма обращений за baseline_days) / (сумма g за те же часы), дисперсия = lambda
           (счётчики редких событий ведут себя как Пуассон: дисперсия равна среднему)
  ewma:    ставка = экспоненциально взвешенное среднее (недавние часы весят больше, период полураспада
           ewma_halflife_h), избыточная дисперсия берётся из того же взвешенного разброса.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from scipy.special import gammainc, ndtri
from scipy.stats import binom


# ---------------------------------------------------------------- чистые функции
def baseline_poisson(counts: list[float], g: list[float]) -> float:
    """Ставка кластера на единицу сезонного множителя: sum(counts) / sum(g). None-часы пропускаем."""
    num = sum(c for c, gg in zip(counts, g) if c is not None)
    den = sum(gg for c, gg in zip(counts, g) if c is not None)
    return num / den if den > 0 else 0.0


def baseline_ewma(counts: list[float], g: list[float], halflife_h: float) -> tuple[float, float]:
    """Взвешенная ставка m и «дисперсия на единицу экспозиции» v. counts[-1] самый свежий час.

    Вес часа с возрастом a (в часах): 0.5 ** (a / halflife). m = sum(w*c) / sum(w*g).
    v = sum(w * (c - m*g)^2) / sum(w*g): для чистого Пуассона v ~ m."""
    n = len(counts)
    sw_c = sw_g = 0.0
    for i, (c, gg) in enumerate(zip(counts, g)):
        if c is None:
            continue
        w = 0.5 ** ((n - 1 - i) / halflife_h)
        sw_c += w * c
        sw_g += w * gg
    if sw_g <= 0:
        return 0.0, 0.0
    m = sw_c / sw_g
    sv = 0.0
    for i, (c, gg) in enumerate(zip(counts, g)):
        if c is None:
            continue
        w = 0.5 ** ((n - 1 - i) / halflife_h)
        sv += w * (c - m * gg) ** 2
    return m, sv / sw_g


def zscore(x: float, rate: float, g_now: float, window_min: int, var_unit: float | None = None,
           lambda_floor: float = 0.3) -> tuple[float, float]:
    """Возвращает (z, lambda).

    lambda = ставка * экспозиция окна, не ниже lambda_floor (новые кластеры без истории).
    Простая формула (x - lambda) / sqrt(lambda) на малых lambda врёт: при lambda=1 и x=7 она даёт z=6, а точная
    вероятность увидеть 7 и больше равна 8,3e-5 (точно 8.32e-5), то есть реальные «6 сигм» это примерно 3.8. Поэтому z здесь
    «z-эквивалент точного хвоста»: p = P(X >= x) при X ~ Пуассон(lambda), z = такое число сигм нормального
    распределения, у которого правый хвост равен p. На больших lambda результат близок к (x-lambda)/sqrt(lambda).
    var_unit=None: чистый Пуассон. Иначе (EWMA) счётчики могут быть «рваннее» Пуассона: считаем коэффициент
    избыточной дисперсии phi = max(1, var_unit*экспозиция / lambda) и берём тот же хвост от (x/phi, lambda/phi).
    Хвост считается через gammainc: P(X >= k) = gammainc(k, lambda) и для нецелых k."""
    exposure = g_now * window_min / 60.0
    lam = max(rate * exposure, lambda_floor)
    phi = 1.0 if var_unit is None else max(1.0, var_unit * exposure / lam)
    if x <= 0:
        return -10.0, lam
    p = float(gammainc(x / phi, lam / phi))      # P(X >= x) в масштабе phi
    return float(-ndtri(min(max(p, 1e-300), 1.0 - 1e-12))), lam   # -ndtri(p) = norm.isf(p), но в 80 раз быстрее


def cap_outliers(counts: list, g: list) -> list:
    """Срезает часы-выбросы перед расчётом базовой линии (в v1 для этого «замораживались» часы всплесков).

    Ставка первого прохода r0 = sum(c)/sum(g); час с c > r0*g + 3*sqrt(r0*g) + 2 считаем выбросом и
    обрезаем до этой границы. Не зависит от решений детектора, поэтому прогон с любыми порогами даёт
    одну и ту же базовую линию (и подбор порогов честен)."""
    r0 = baseline_poisson(counts, g)
    out = []
    for c, gg in zip(counts, g):
        if c is None:
            out.append(None)
            continue
        mu = r0 * gg
        out.append(min(c, mu + 3.0 * mu ** 0.5 + 2.0))
    return out


def ltss_scan(rows: list, floor: float, min_member: int = 2, max_k: int = 8, member_z: float = -10.0) -> tuple:
    """Сканирование подмножеств кластеров одного продукта (Neill, «fast subset scan», свойство LTSS).

    rows: [(key, x, mu, phi)]: x людей в окне, mu ожидание без всплеска, phi коэффициент избыточной
    дисперсии (1 для Пуассона, у EWMA из разброса истории). Одна тема режется на несколько кластеров-осколков,
    всплеск делится между ними, и каждый осколок по отдельности слабый. Для статистики «сумма x против суммы
    ожиданий» лучшее подмножество всегда среди первых k кластеров, упорядоченных по x/mu (LTSS), поэтому
    перебор линейный, а не 2^n. Одиночный кластер (k=1) тоже кандидат: так скан не хуже детектора v1.
    В сумму (k >= 2) идут только кластеры, где x >= min_member и собственный z >= member_z: иначе в сумму
    затягиваются случайные кластеры с парой обращений, и всплеск «размазывается» по чужим темам.
    Возвращает (z, C, B, [keys]) лучшего подмножества."""
    best = (-10.0, 0, 0.0, [])
    if not rows:
        return best
    lam = [max(r[2], floor) for r in rows]
    zs = []
    for r, l in zip(rows, lam):                     # одиночные кластеры
        z = ptail_z(r[1] / r[3], l / r[3])
        zs.append(z)
        if z > best[0]:
            best = (z, r[1], l, [r[0]])
    order = sorted(range(len(rows)), key=lambda i: -rows[i][1] / lam[i])
    C, B, BP, keys = 0, 0.0, 0.0, []
    for i in order:
        if len(keys) >= max_k:
            break
        key, x, _, phi = rows[i]
        if x < min_member or zs[i] < member_z:
            continue
        C += x
        B += lam[i]
        BP += lam[i] * phi
        keys.append(key)
        if len(keys) >= 2:
            phi_s = BP / B
            z = ptail_z(C / phi_s, B / phi_s)
            if z > best[0]:
                best = (z, C, B, list(keys))
    return best


def ptail_z(x: float, lam: float) -> float:
    """z-эквивалент точного хвоста Пуассона P(X >= x), см. zscore."""
    if x <= 0:
        return -10.0
    p = float(gammainc(x, max(lam, 1e-9)))
    return float(-ndtri(min(max(p, 1e-300), 1.0 - 1e-12)))


def adoption_share(per_key: dict, min_v: int = 2, p_cut: float = 1e-3) -> float:
    """Доля новой версии в потоке продукта без кластеров, где она подозрительно высока.

    per_key: {key: (n_всего, n_с_новой_версией)} за последние часы. Если считать долю по всему продукту,
    сама регрессия её раздувает (её обращения почти все на новой версии) и ожидание растёт вместе с сигналом.
    Поэтому выкидываем кластеры, где новой версии слишком много для общей доли q_raw: биномиальный хвост
    P(X >= n_v | n, q_raw) < p_cut и n_v >= min_v. Сглаживание +0.5/+1, нижняя граница 0.005."""
    N = sum(n for n, _ in per_key.values())
    V = sum(v for _, v in per_key.values())
    if N == 0:
        return 0.005
    q_raw = (V + 0.5) / (N + 1.0)
    for n, v in per_key.values():
        if v >= min_v and binom.sf(v - 1, n, q_raw) < p_cut:
            N -= n
            V -= v
    return max((V + 0.5) / (N + 1.0), 0.005)


# ---------------------------------------------------------------- сезонность
class SeasonalProfile:
    """Сезонный множитель g(час) для каждого продукта: час суток x (будни/выходной), средний будний час = 1.

    Учим по почасовым итогам продукта за последние 14 дней и пересчитываем раз в сутки. Только прошлое.
    Часы индексируются от понедельника 00:00 (t0): час h = (ts - t0) в часах; день недели = (h // 24) % 7."""

    def __init__(self, t0: datetime, history_days: int = 14):
        self.t0, self.history_days = t0, history_days
        self.hour_totals: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._cache: dict[str, tuple[int, dict]] = {}

    def hour_index(self, ts: datetime) -> int:
        return int((ts - self.t0).total_seconds() // 3600)

    def add(self, product: str, ts: datetime):
        self.hour_totals[product][self.hour_index(ts)] += 1

    @staticmethod
    def slot(h: int) -> tuple[int, int]:
        return (1 if (h // 24) % 7 >= 5 else 0), h % 24

    def factors(self, product: str, now_h: int) -> dict:
        day = now_h // 24
        cached = self._cache.get(product)
        if cached and cached[0] == day:
            return cached[1]
        lo, hi = max(0, day * 24 - self.history_days * 24), day * 24     # только полные прошлые сутки
        sums, cnts = defaultdict(float), defaultdict(int)
        for h in range(lo, hi):
            s = self.slot(h)
            sums[s] += self.hour_totals[product].get(h, 0)
            cnts[s] += 1
        mean = {s: sums[s] / cnts[s] for s in sums}
        if not mean or sum(mean.values()) == 0:
            f = {}
        else:
            # сглаживаем соседние часы (0.25, 0.5, 0.25): профиль малых продуктов шумный
            sm = {}
            for wk in (0, 1):
                base = {hh: mean.get((wk, hh), mean.get((0, hh))) for hh in range(24)}
                if any(v is None for v in base.values()):
                    continue
                for hh in range(24):
                    sm[(wk, hh)] = 0.25 * base[(hh - 1) % 24] + 0.5 * base[hh] + 0.25 * base[(hh + 1) % 24]
            avg = sum(sm.values()) / len(sm) if sm else 1.0
            overall = sum(sm.get((0, hh), 0) for hh in range(24)) / 24 or avg
            f = {s: max(v / overall, 0.05) for s, v in sm.items()}
        self._cache[product] = (day, f)
        return f

    def g(self, product: str, h: int, now_h: int) -> float:
        f = self.factors(product, now_h)
        s = self.slot(h)
        return f.get(s, f.get((0, s[1]), 1.0)) if f else 1.0


# ---------------------------------------------------------------- ряд одного кластера
class ClusterSeries:
    """Что детектор помнит про один кластер: почасовые голоса и недавние голоса (для окон 30/60 минут)."""

    def __init__(self, first_seen: datetime, first_hour: int):
        self.first_seen = first_seen
        self.first_hour = first_hour                       # часы раньше этого в базовую линию не идут
        self.hourly: dict[int, int] = defaultdict(int)     # час -> число голосов
        self.frozen: set[int] = set()                      # часы, когда шёл всплеск: в базовую линию не берём
        self.recent: list[tuple] = []                      # (время, user_id, версия) за последние часы
        self.last_vote: dict[str, datetime] = {}           # user_id -> время его последнего голоса (для дедупа)

    def add_vote(self, ts: datetime, user: str, hour_idx: int, dedup_min: int, version: str = "") -> bool:
        """Один user_id в окне dedup_min даёт один голос. Возвращает True, если голос засчитан."""
        last = self.last_vote.get(user)
        if last is not None and (ts - last) < timedelta(minutes=dedup_min):
            return False
        self.last_vote[user] = ts
        self.hourly[hour_idx] += 1
        self.recent.append((ts, user, version))
        return True

    def window_users(self, now: datetime, window_min: int) -> tuple[int, datetime | None]:
        """(сколько разных людей в окне, время самого раннего обращения в окне)."""
        lo = now - timedelta(minutes=window_min)
        users, first = set(), None
        for ts, u, _ in self.recent:
            if ts > lo:
                users.add(u)
                if first is None or ts < first:
                    first = ts
        return len(users), first

    def window_counts(self, now: datetime, window_min: int, version: str | None) -> tuple:
        """(разных людей в окне, из них с версией version, время первого голоса в окне). recent отсортирован."""
        lo = now - timedelta(minutes=window_min)
        users, vusers, first = set(), set(), None
        for ts, u, v in reversed(self.recent):
            if ts <= lo:
                break
            users.add(u)
            if version and v == version:
                vusers.add(u)
            first = ts
        return len(users), len(vusers), first

    def purge(self, now: datetime, keep_min: int = 180):
        lo = now - timedelta(minutes=keep_min)
        if self.recent and self.recent[0][0] <= lo:
            self.recent = [r for r in self.recent if r[0] > lo]
        # last_vote чистим редко: словарь ограничен числом людей, это дёшево

    def history(self, now_h: int, days: int, guard_h: int, gfun) -> tuple[list, list]:
        """Почасовые счётчики [now_h - days*24 - guard .. now_h - guard) и сезонные g. Часы до рождения
        кластера и «замороженные» часы всплесков отдаются как None (в среднее не идут)."""
        counts, gs = [], []
        for h in range(now_h - days * 24 - guard_h, now_h - guard_h):
            counts.append(None if (h in self.frozen or h < self.first_hour) else self.hourly.get(h, 0))
            gs.append(gfun(h))
        return counts, gs

    def merge_from(self, other: "ClusterSeries"):
        self.first_seen = min(self.first_seen, other.first_seen)
        self.first_hour = min(self.first_hour, other.first_hour)
        for h, c in other.hourly.items():
            self.hourly[h] += c
        self.frozen |= other.frozen
        self.recent = sorted(self.recent + other.recent)
        for u, t in other.last_vote.items():
            if u not in self.last_vote or t > self.last_vote[u]:
                self.last_vote[u] = t
