"""Конвейер целиком: маскирование -> эмбеддинг -> кластеры -> детектор -> релиз -> цена -> проверка фикса.

Главные функции:
  run(tickets, releases, config)       -> list[Burst]            все всплески за период
  iter_run(tickets, releases, config)  -> генератор событий      как будто обращения идут по времени (демо)
  run_full(...)                        -> RadarStream            то же, но возвращает и состояние (для оценки и дашборда)
  card_input(stream, burst)            -> SuspectedCardInput     то, что получит агент следующей волны

Время потоковое: RadarStream.push(ticket) принимает обращения строго по возрастанию времени, а раз в
step_min минут (по времени обращений, а не по часам компьютера) делает шаг детектора.

Что изменилось в v2 (почему, с числами: eval/DIAGNOSIS.md):
  * детектор смотрит не на один кластер, а на подмножество кластеров продукта (скан LTSS): тема режется
    на 3-4 осколка, и всплеск, поделённый между ними, раньше терялся;
  * второй сигнал у мобильных продуктов: люди темы на версии последнего релиза против ожидания по доле этой
    версии в остальном потоке продукта;
  * тревога (alert) только в окне release_max_hours после релиза продукта и не по кластеру, по которому уже
    была тревога за realert_days; остальное идёт в «наблюдение» (alert=False) и дежурного не будит;
  * базовая линия не зависит от решений детектора (выбросы обрезаются), поэтому прогон детерминирован и честен
    при подборе порогов;
  * проверка фикса по всем кластерам инцидента и против нормы темы до релиза (verify_mode="baseline").
"""
from __future__ import annotations

import copy
import dataclasses
from bisect import bisect_right
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from typing import Iterator, Optional

import numpy as np

from . import antispam
from .cluster import OnlineClusterer, hdbscan_groups
from .config import Config
from .judge import pick_examples, tier_for
from .cost import load_params, unique_users, weekly_cost_rub
from .detect import (ClusterSeries, SeasonalProfile, adoption_share, baseline_ewma, baseline_poisson,
                     cap_outliers, ltss_scan, ptail_z)
from .embed import make_embedder
from .mask import mask_tickets
from .models import Burst, FixCheck, Release, SuspectedCardInput, Ticket
from .release import find_suspect
from .verify import verify_fix, verify_fix_baseline
from .gen.topics import PRODUCTS


class RadarStream:
    def __init__(self, releases: list[Release], cfg: Config, dim: int, t0: datetime, start_ts: datetime):
        self.cfg, self.releases = cfg, releases
        self.rel_by_product: dict[str, list] = defaultdict(list)
        for r in sorted(releases, key=lambda r: r.ts):
            self.rel_by_product[r.product].append(r)
        self.rel_ts = {p: [r.ts for r in rs] for p, rs in self.rel_by_product.items()}
        self.t0 = t0                                    # понедельник 00:00 недели, где начались данные
        self.season = SeasonalProfile(t0)
        self.warmup_end = start_ts + timedelta(days=cfg.warmup_days)
        self.dim = dim
        self.clusterers: dict[str, OnlineClusterer] = {}
        self.series: dict[tuple, ClusterSeries] = {}    # (product, cluster) -> ряд
        self.members: dict[tuple, list] = defaultdict(list)       # -> индексы обращений кластера
        self.vote_times: dict[tuple, list] = defaultdict(list)    # -> времена засчитанных голосов
        self.tickets: list[Ticket] = []
        self.texts: list[str] = []                      # ЗАМАСКИРОВАННЫЕ тексты, единственное, что видят дальше
        self.vecs = np.zeros((1024, dim), dtype=np.float32)
        self.cluster_of: list[int] = []                 # онлайн-кластер обращения на момент прихода
        self.product_idx: dict[str, list] = defaultdict(list)
        self.active: dict[str, set] = defaultdict(set)  # продукт -> кластеры, где недавно были голоса
        self.open: dict[str, Burst] = {}                # id -> открытый всплеск
        self.busy: dict[tuple, str] = {}                # кластер -> id открытого всплеска, в который он входит
        self.alerted: dict[tuple, datetime] = {}        # кластер -> когда по нему последний раз была тревога
        self.prod_recent: dict[str, deque] = defaultdict(deque)   # продукт -> (время, версия, кластер) за adoption_hours
        self.has_versions: set = set()                  # продукты, у обращений которых бывает версия
        self.bstate: dict[str, dict] = {}
        self.bursts: list[Burst] = []
        self.events: list[tuple] = []                   # (время, "opened"|"closed"|"fix_checked", burst_id)
        self.next_step = start_ts.replace(second=0, microsecond=0)
        self.last_day = None
        self.cost_params = load_params("base")
        self.n_merges = 0
        self._rate_cache: dict[tuple, tuple] = {}
        self._g_cache: dict[str, tuple] = {}
        self.keep_min = max(cfg.windows_min) + cfg.step_min
        self.ids: set = set()                           # id принятых обращений: повтор не засчитывается второй раз
        self.last_ts: Optional[datetime] = None         # самое позднее время принятого обращения
        self.judge = None                               # judge(product, texts) -> {problem, items, share, ok}; нужен при cfg.judge_enabled
        self.rejected: list[tuple] = []                 # (id, причина, подробности): что push не принял; читает журнал
        # --- защита от спама и накрутки ---
        self.user_hits: dict[str, deque] = defaultdict(deque)   # user_id -> времена принятых обращений в окне user_cap_window_min
        self.capped_by_user: dict[str, int] = {}        # user_id -> сколько его обращений отброшено лимитом
        self.n_capped = 0                               # всего отброшено лимитом на пользователя
        self.user_first: dict[str, datetime] = {}       # user_id -> время первого обращения в потоке (диагностика «свежих» аккаунтов)
        self.on_antispam = None                         # on_antispam(kind, payload): сюда идут срабатывания защиты (журнал аудита); None: молча

    # ------------------------------------------------------------ вспомогательное
    def hour(self, ts: datetime) -> int:
        return self.season.hour_index(ts)

    def resolve(self, product: str, cid: int) -> tuple:
        return (product, self.clusterers[product].resolve(cid))

    def _key_of_ticket(self, idx: int) -> tuple:
        p = self.tickets[idx].product
        return self.resolve(p, self.cluster_of[idx])

    def latest_release(self, product: str, ts: datetime) -> Optional[Release]:
        i = bisect_right(self.rel_ts.get(product, []), ts)
        return self.rel_by_product[product][i - 1] if i else None

    def burst_keys(self, b: Burst) -> list:
        return list(self.bstate[b.id]["keys"])

    # ------------------------------------------------------------ приём обращений
    def admit(self, tk: Ticket) -> tuple:
        """Проверка перед приёмом: (вердикт, обращение). Вердикты: ok, duplicate, late_clamped, too_late, user_capped.
        Повтор id не принимается. Больше user_cap_max обращений одного user_id за user_cap_window_min минут: user_capped.
        Обращение старше последнего принятого не больше чем на late_tolerance_min минут
        принимается со временем прихода (ts заменяется на last_ts: окна и сезонность считаются по порядку прихода),
        старше этого отбрасывается. При подаче по порядку времени (весь замер качества) вердикт всегда ok."""
        if tk.id in self.ids:
            return "duplicate", tk
        if self.last_ts is not None and tk.ts < self.last_ts:
            lag = (self.last_ts - tk.ts).total_seconds() / 60.0
            if lag > self.cfg.late_tolerance_min:
                return "too_late", tk
            tk = dataclasses.replace(tk, ts=self.last_ts)
            return ("user_capped", tk) if self._user_capped(tk) else ("late_clamped", tk)
        return ("user_capped", tk) if self._user_capped(tk) else ("ok", tk)

    def _user_capped(self, tk: Ticket) -> bool:
        """Лимит на пользователя: в окне user_cap_window_min уже принято user_cap_max его обращений. Окно скользит по времени обращений."""
        cap = self.cfg.user_cap_max
        if cap <= 0:
            return False
        dq = self.user_hits.get(tk.user_id)
        if not dq:
            return False
        lo = tk.ts - timedelta(minutes=self.cfg.user_cap_window_min)
        while dq and dq[0] <= lo:
            dq.popleft()
        return len(dq) >= cap

    def _antispam(self, kind: str, payload: dict):
        if self.on_antispam is not None:
            self.on_antispam(kind, payload)

    def push(self, tk: Ticket, masked_text: str, vec: np.ndarray) -> str:
        """Принимает обращение, возвращает вердикт admit. Подавать лучше по возрастанию времени; отклонения см. admit."""
        verdict, tk2 = self.admit(tk)
        if verdict in ("duplicate", "too_late"):
            self.rejected.append((tk.id, verdict, tk.ts.isoformat(timespec="seconds")))
            return verdict
        if verdict == "user_capped":
            n = self.capped_by_user[tk.user_id] = self.capped_by_user.get(tk.user_id, 0) + 1
            self.n_capped += 1
            if n == 1 or n % 100 == 0:       # в журнал и в rejected: первое отбрасывание и каждое сотое, иначе флуд забил бы сам журнал
                self.rejected.append((tk.id, verdict, f"user_id={tk.user_id}, отброшено {n}"))
                self._antispam("antispam_user_capped", {"user_id": tk.user_id, "dropped": n, "cap": self.cfg.user_cap_max,
                                                        "window_min": self.cfg.user_cap_window_min, "ticket_id": tk.id,
                                                        "ts": tk.ts.isoformat(timespec="seconds")})
            return verdict
        if verdict == "late_clamped":
            self.rejected.append((tk.id, verdict, f"{tk.ts.isoformat(timespec='seconds')} -> {tk2.ts.isoformat(timespec='seconds')}"))
        tk = tk2
        self.ids.add(tk.id)
        self.last_ts = tk.ts
        self.user_hits[tk.user_id].append(tk.ts)
        self.user_first.setdefault(tk.user_id, tk.ts)
        self._push(tk, masked_text, vec)
        return verdict

    def _push(self, tk: Ticket, masked_text: str, vec: np.ndarray):
        self.advance_to(tk.ts)
        idx = len(self.tickets)
        if idx == len(self.vecs):
            self.vecs = np.vstack([self.vecs, np.zeros_like(self.vecs)])
        self.vecs[idx] = vec
        self.tickets.append(tk)
        self.texts.append(masked_text)
        cl = self.clusterers.setdefault(tk.product, OnlineClusterer(self.dim, self.cfg.sim_threshold))
        cid = cl.assign(vec.astype(np.float64))
        self.cluster_of.append(cid)
        key = (tk.product, cid)
        h = self.hour(tk.ts)
        if key not in self.series:
            self.series[key] = ClusterSeries(tk.ts, h)
        self.members[key].append(idx)
        self.product_idx[tk.product].append(idx)
        self.season.add(tk.product, tk.ts)
        if tk.app_version:
            self.has_versions.add(tk.product)
        self.prod_recent[tk.product].append((tk.ts, tk.app_version, key))
        if self.series[key].add_vote(tk.ts, tk.user_id, h, self.cfg.dedup_min, tk.app_version):
            self.vote_times[key].append(tk.ts)
            self.active[tk.product].add(key)
        bid = self.busy.get(key)
        if bid is not None:   # идёт всплеск: накапливаем людей, платформы, версии
            b = self.open[bid]
            st = self.bstate[bid]
            st["users"].add(tk.user_id)
            st["platforms"][tk.platform] += 1
            if tk.app_version:
                st["versions"][tk.app_version] += 1
            b.last_seen = tk.ts

    def advance_to(self, ts: datetime):
        """Делает все шаги детектора, назначенные до момента ts."""
        step = timedelta(minutes=self.cfg.step_min)
        while self.next_step < ts:
            self._step(self.next_step)
            self.next_step += step

    # ------------------------------------------------------------ базовая линия
    def _gvec(self, product: str, now_h: int) -> list:
        """Сезонные множители g для часов истории [now_h - days*24 - guard, now_h - guard). Кэш на час."""
        c = self._g_cache.get(product)
        if c and c[0] == now_h:
            return c[1]
        cfg = self.cfg
        lo, hi = now_h - cfg.baseline_days * 24 - cfg.guard_hours, now_h - cfg.guard_hours
        gs = [self.season.g(product, h, now_h) for h in range(lo, hi)]
        self._g_cache[product] = (now_h, gs)
        return gs

    def _rate(self, key: tuple, now_h: int) -> tuple:
        """(ставка на единицу g, var_unit или None) кластера. Считается раз в час потокового времени.
        Часы до рождения кластера не идут в среднее, часы-выбросы обрезаются (cap_outliers)."""
        c = self._rate_cache.get(key)
        if c and c[0] == now_h:
            return c[1], c[2]
        cfg = self.cfg
        s = self.series[key]
        lo, hi = now_h - cfg.baseline_days * 24 - cfg.guard_hours, now_h - cfg.guard_hours
        gs = self._gvec(key[0], now_h)
        hourly, first = s.hourly, s.first_hour
        counts = [None if h < first else hourly.get(h, 0) for h in range(lo, hi)]
        counts = cap_outliers(counts, gs)
        if cfg.method == "ewma":
            m, v = baseline_ewma(counts, gs, cfg.ewma_halflife_h)
        else:
            m, v = baseline_poisson(counts, gs), None
        self._rate_cache[key] = (now_h, m, v)
        return m, v

    def _adoption(self, product: str, now: datetime, version: str) -> float:
        """Доля версии version в потоке продукта за adoption_hours без кластеров с подозрительно высокой долей."""
        dq = self.prod_recent[product]
        lo = now - timedelta(hours=self.cfg.adoption_hours)
        while dq and dq[0][0] < lo:
            dq.popleft()
        per_key: dict = {}
        for _, v, k in dq:
            n, nv = per_key.get(k, (0, 0))
            per_key[k] = (n + 1, nv + (v == version))
        return adoption_share(per_key)

    # ------------------------------------------------------------ шаг детектора
    def _step(self, now: datetime):
        cfg = self.cfg
        day = now.date()
        if self.last_day is None:
            self.last_day = day
        elif day != self.last_day:
            self.last_day = day
            if now >= self.warmup_end:
                self._daily(now)
        if now < self.warmup_end:
            for dq in self.prod_recent.values():   # прогрев: только не даём очереди расти
                lo = now - timedelta(hours=cfg.adoption_hours)
                while dq and dq[0][0] < lo:
                    dq.popleft()
            return
        now_h = self.hour(now)
        for product in sorted(self.active):   # порядок фиксируем: результат не должен зависеть от хэшей
            self._step_product(product, now, now_h)

    def _step_product(self, product: str, now: datetime, now_h: int):
        cfg = self.cfg
        keys = self.active[product]
        rel = self.latest_release(product, now)
        in_window = rel is not None and (now - rel.ts) <= timedelta(hours=cfg.release_max_hours)
        V = rel.version if (in_window and cfg.version_signal and product in self.has_versions) else None
        q = self._adoption(product, now, V) if V else None
        if q is not None and q > cfg.version_q_max:
            q = None
        g_now = self.season.g(product, now_h, now_h)
        stats = {}
        for key in sorted(keys):
            s = self.series[key]
            s.purge(now, self.keep_min)
            if not s.recent:
                continue
            stats[key] = {w: s.window_counts(now, w, V) for w in cfg.windows_min}
        for key in [k for k in keys if k not in stats and k not in self.busy]:
            keys.discard(key)
        # открытые всплески продукта: живы ли ещё
        for bid in [i for i, b in self.open.items() if b.product == product]:
            self._update_burst(self.open[bid], now, now_h, stats, g_now, q)
        thr, min_users = cfg.for_product(product)
        z_need = thr if (in_window or cfg.alert_policy == "all") else max(thr, cfg.z_watch)
        floor_member = min(cfg.scan_min_member, min_users)
        best = None   # (запас над порогом, сигнал, окно, z, C, B, ключи, zv, CV, BV)
        for w in cfg.windows_min:
            rows, vrows = [], []
            expo = g_now * w / 60.0
            for key, st in stats.items():
                if key in self.busy:
                    continue
                x, xv, _ = st[w]
                if x < floor_member:
                    continue
                rate, var = self._rate(key, now_h)
                mu = rate * expo
                phi = 1.0 if var is None else max(1.0, var * expo / max(mu, cfg.lambda_floor))
                rows.append((key, x, mu, phi))
                if q is not None and xv > 0:
                    vrows.append((key, xv, mu * q, 1.0))
            if not rows:
                continue
            if cfg.subset_scan:
                z, C, B, ks = ltss_scan(rows, cfg.lambda_floor, cfg.scan_min_member, cfg.scan_max_k, cfg.scan_member_z)
            else:
                z, C, B, ks = ltss_scan(rows, cfg.lambda_floor, 10 ** 9, 1)
            if z >= z_need and C >= min_users:
                cand = (z - z_need, "topic", w, z, C, B, ks, None, 0, 0.0)
                if best is None or cand[0] > best[0]:
                    best = cand
            if vrows:
                if cfg.subset_scan:
                    zv, CV, BV, kv = ltss_scan(vrows, cfg.version_floor, cfg.scan_min_member, cfg.scan_max_k, cfg.scan_member_z)
                else:
                    zv, CV, BV, kv = ltss_scan(vrows, cfg.version_floor, 10 ** 9, 1)
                if zv >= cfg.z_version and CV >= cfg.min_users_version:
                    zt, Ct, Bt = self._sum_z(kv, stats, w, now_h, expo)
                    cand = (zv - cfg.z_version, "version", w, zt, Ct, Bt, kv, zv, CV, BV)
                    if best is None or cand[0] > best[0]:
                        best = cand
        if best is not None:
            self._open_burst(product, now, best, stats, in_window, rel)

    def _sum_z(self, ks: list, stats: dict, w: int, now_h: int, expo: float) -> tuple:
        """z темы по набору кластеров ks в окне w: (z, сумма людей, сумма ожиданий)."""
        C, B = 0, 0.0
        for k in ks:
            st = stats.get(k)
            C += st[w][0] if st else 0
            rate, _ = self._rate(k, now_h)
            B += max(rate * expo, self.cfg.lambda_floor)
        return ptail_z(C, B), C, B

    def _update_burst(self, b: Burst, now: datetime, now_h: int, stats: dict, g_now: float, q: Optional[float]):
        """Закрытие: z по кластерам всплеска упал ниже порога*clear_ratio и держится clear_hold_min минут."""
        cfg = self.cfg
        st = self.bstate[b.id]
        thr, _ = cfg.for_product(b.product)
        alive = False
        for w in cfg.windows_min:
            z, C, _ = self._sum_z(st["keys"], stats, w, now_h, g_now * w / 60.0)
            if z >= thr * cfg.clear_ratio and C >= 1:
                alive = True
                break
        if alive:
            st["below_since"] = None
            for k in st["keys"]:
                if k in self.alerted:          # тема ещё горит: тревога по ней не повторяется
                    self.alerted[k] = now
        elif st["below_since"] is None:
            st["below_since"] = now
        elif now - st["below_since"] >= timedelta(minutes=cfg.clear_hold_min):
            self._close_burst(b, now)

    # ------------------------------------------------------------ жизнь всплеска
    def _open_burst(self, product: str, now: datetime, best: tuple, stats: dict, in_window: bool, rel):
        cfg = self.cfg
        _, signal, w, z, C, B, ks, zv, CV, BV = best
        firsts = [stats[k][w][2] for k in ks if k in stats and stats[k][w][2] is not None]
        started = min(firsts) if firsts else now
        n = len(self.bursts) + 1
        lead = ks[0]
        b = Burst(id=f"B{n:04d}", product=product, cluster_id=f"{product}:{lead[1]}", method=cfg.method,
                  window_min=w, started_at=started, detected_at=now, last_seen=now, status="open", z=round(float(z), 2),
                  count_in_window=int(C), baseline_in_window=round(float(B), 2), total_unique_users=0,
                  cluster_size=sum(len(self.members[k]) for k in ks))
        b.cluster_ids = [f"{product}:{k[1]}" for k in ks]
        b.signal = signal
        if zv is not None:
            b.z_version, b.version_count, b.version_baseline = round(float(zv), 2), int(CV), round(float(BV), 3)
        cont = [k for k in ks if k in self.alerted and now - self.alerted[k] <= timedelta(days=cfg.realert_days)]
        same_rel = []
        if cfg.release_incident_hours > 0 and in_window:
            same_rel = [x for x in self.bursts if x.alert and x.product == product and self.bstate[x.id].get("rel") == rel.version
                        and now - x.detected_at <= timedelta(hours=cfg.release_incident_hours)]
        if same_rel and not cont:
            b.alert, b.alert_reason = False, f"продолжение {same_rel[0].id} (тот же релиз {rel.version})"
            cont = list(ks)
        elif cont:
            prev = [x.id for x in self.bursts if x.alert and set(self.bstate[x.id]["keys"]) & set(cont)]
            b.alert, b.alert_reason = False, "продолжение " + (prev[-1] if prev else "прошлой тревоги")
        elif cfg.alert_policy == "all" or in_window:
            b.alert = True
            b.alert_reason = (f"через {(now - rel.ts).total_seconds() / 3600:.1f} ч после релиза {rel.version}, сигнал {signal}"
                              if in_window else f"всплеск без релиза, сигнал {signal}")
        else:
            b.alert, b.alert_reason = False, "наблюдение: нет релиза продукта за последние " \
                                             f"{cfg.release_max_hours:.0f} ч"
        quiet_susp = False
        if b.alert and cfg.brigade_check:
            quiet_susp = self._screen_brigade(b, best, stats, now, in_window)
        if (b.alert and not quiet_susp) or cont:   # подозрительная тревога тему не «занимает»: настоящая регрессия в ней позже должна будить
            for k in ks:
                self.alerted[k] = now
        users, plats, vers = set(), Counter(), Counter()
        for i in self._union_members(ks, started):
            t = self.tickets[i]
            users.add(t.user_id)
            plats[t.platform] += 1
            if t.app_version:
                vers[t.app_version] += 1
        self.bstate[b.id] = dict(users=users, platforms=plats, versions=vers, below_since=None, keys=list(ks), key=lead,
                                 rel=rel.version if (in_window and rel is not None) else None)
        if quiet_susp:
            b.tier = "quiet"                       # накрутка: тихая карточка, проверку моделью не тратим
        elif b.alert:
            self._tier(b, now, w, stats)
        for k in ks:
            self.busy[k] = b.id
        self.open[b.id] = b
        self.bursts.append(b)
        self._refresh(b, now)
        self.events.append((now, "opened", b.id))

    def _screen_brigade(self, b: Burst, best: tuple, stats: dict, now: datetime, in_window: bool) -> bool:
        """Признак накрутки (radar/antispam.py) по людям окна срабатывания. True: тревогу не будить (tier quiet).
        Перед решением копии группы вычитаются (остаётся один голос на группу): если остаток людей всё ещё даёт тревогу, настоящая
        регрессия под накруткой не прячется, tier остаётся прежним, а suspicion пишется пометкой (kept_page)."""
        cfg = self.cfg
        _, signal, w, z, C, B, ks, zv, CV, BV = best
        started = b.started_at
        idx = [i for i in self._union_members(ks, started) if self.tickets[i].ts <= now]
        res = antispam.assess([(self.tickets[i].user_id, self.tickets[i].ts, self.texts[i]) for i in idx], cfg,
                              seen_before=lambda u: self.user_first.get(u, now) < started)
        if not res["suspect"]:
            return False
        group = set(res["group_users"])
        thr, min_users = cfg.for_product(b.product)
        z_need = thr if (in_window or cfg.alert_policy == "all") else max(thr, cfg.z_watch)
        c_rest = antispam.users_without_duplicates(int(C), len(group))
        z_rest = ptail_z(c_rest, B)
        ok = z_rest >= z_need and c_rest >= min_users
        if signal == "version":
            vnew = self.latest_release(b.product, now)
            vers = {self.tickets[i].user_id for i in idx if vnew is not None and self.tickets[i].app_version == vnew.version}
            gv = len(group & vers)
            cv_rest = int(CV) - gv + (1 if gv else 0)
            zv_rest = ptail_z(cv_rest, BV)
            ok = ok and zv_rest >= cfg.z_version and cv_rest >= cfg.min_users_version
        info = {k: res[k] for k in ("reason", "n_users", "n_assessed", "dup_users", "dup_share", "gap_cv", "fresh_share")}
        info.update(kept_page=bool(ok), users_after_recount=int(c_rest), z_after_recount=round(float(z_rest), 2))
        b.suspicion = info
        if ok:
            b.alert_reason += f" | {res['reason']}; копии вычтены, остаток {c_rest} чел. (z {z_rest:.1f}) всё ещё тревога, будим"
        else:
            b.alert_reason += f" | {res['reason']}; тихая карточка, дежурного не будим, автоответ не включается"
        self._antispam("antispam_brigading", {"burst_id": b.id, "product": b.product, "kept_page": bool(ok), "ts": now.isoformat(timespec="minutes"),
                                              **{k: info[k] for k in ("n_users", "dup_users", "dup_share", "gap_cv", "fresh_share",
                                                                      "users_after_recount", "z_after_recount")}})
        return not ok

    def _tier(self, b: Burst, now: datetime, w: int, stats: dict):
        """v3: уровень тревоги. Без проверки (judge_enabled=False) или если проверить нельзя, тревога будит (fail-open)."""
        cfg = self.cfg
        b.tier = "page"
        if not cfg.judge_enabled:
            return
        ks = self.bstate[b.id]["keys"]
        now_h = self.hour(now)
        expo = self.season.g(b.product, now_h, now_h) * w / 60.0
        excess = {k: (stats[k][w][0] if k in stats else 0) - self._rate(k, now_h)[0] * expo for k in ks}
        cands = []
        for k in ks:
            for i in reversed(self.members.get(k, [])):
                if self.tickets[i].ts < b.started_at:
                    break
                cands.append((i, self.tickets[i].ts, self.tickets[i].user_id, k))
        idxs = pick_examples(cands, excess, cfg.judge_examples, cfg.judge_sampling)
        res = None
        if self.judge is not None and idxs:
            try:
                res = self.judge(b.product, [self.texts[i][:300] for i in idxs])
            except Exception:                       # проверка не должна ронять детектор
                res = None
        if not isinstance(res, dict):
            res = {"problem": "", "items": [], "share": None, "ok": False, "error": "проверка недоступна"}
        share = res.get("share")
        b.judge = {**res, "example_ids": [self.tickets[i].id for i in idxs]}   # id примеров пишем и когда проверка не удалась
        b.tier = tier_for(share, cfg.judge_min_share)
        b.alert_reason += (f" | проверка: доля {share:.2f}, уровень {b.tier}" if share is not None
                           else " | проверка: не проверено, будим")

    def _close_burst(self, b: Burst, now: datetime):
        self.open.pop(b.id, None)
        for k in self.bstate[b.id]["keys"]:
            if self.busy.get(k) == b.id:
                del self.busy[k]
        b.status = "closed"
        self._refresh(b, now)
        self.events.append((now, "closed", b.id))

    def _union_members(self, keys: list, since: Optional[datetime] = None) -> list:
        """Индексы обращений всех кластеров keys (по времени), не раньше since."""
        out = []
        for k in keys:
            m = self.members.get(k, [])
            if since is not None:
                # members отсортированы по индексу, а индекс растёт со временем: идём с конца
                j = len(m)
                while j > 0 and self.tickets[m[j - 1]].ts >= since:
                    j -= 1
                m = m[j:]
            out.extend(m)
        out.sort()
        return out

    def _refresh(self, b: Burst, now: datetime):
        """Пересчитывает то, что зависит от накопленного: люди, платформы, примеры, цена, подозреваемый релиз."""
        cfg = self.cfg
        st = self.bstate[b.id]
        keys = st["keys"]
        b.total_unique_users = len(st["users"])
        b.platforms, b.versions = dict(st["platforms"]), dict(st["versions"])
        b.cluster_size = sum(len(self.members[k]) for k in keys)
        # примеры: ближайшие к центру обращения всплеска (по косинусу), без повторов текста
        idxs = self._union_members(keys, b.started_at)
        if idxs:
            V = self.vecs[idxs]
            c = V.mean(axis=0)
            order = np.argsort(-(V @ c), kind="stable")
            seen, ids = set(), []
            for j in order:
                i = idxs[j]
                if self.texts[i] in seen:
                    continue
                seen.add(self.texts[i])
                ids.append(i)
                if len(ids) >= cfg.n_examples:
                    break
            b.example_ids = [self.tickets[i].id for i in ids]
            b.examples = [{"id": self.tickets[i].id, "text": self.texts[i]} for i in ids]
        week_ago = now - timedelta(days=7)
        recent = [self.tickets[i] for i in self._union_members(keys, week_ago)]
        b.weekly_cost_rub = weekly_cost_rub(unique_users(recent), self.cost_params["t"], self.cost_params["C1"])
        # релиз
        in_burst = [self.tickets[i] for i in idxs]
        prod_recent = []
        for i in reversed(self.product_idx[b.product]):
            if self.tickets[i].ts < b.started_at - timedelta(hours=3):
                break
            prod_recent.append(self.tickets[i].app_version)
        first_seen = min(self.series[k].first_seen for k in keys)
        b.suspected_release = find_suspect(self.releases, b.product, b.started_at, first_seen,
                                           [t.app_version for t in in_burst], prod_recent,
                                           cfg.release_max_hours, cfg.new_version_share_min)

    # ------------------------------------------------------------ раз в сутки
    def _daily(self, now: datetime):
        if self.cfg.use_hdbscan:
            self._rebuild(now)
        self._check_fixes(now)

    def _rebuild(self, now: datetime):
        lo = now - timedelta(days=7)
        for product, cl in self.clusterers.items():
            idx = []
            for i in reversed(self.product_idx[product]):
                if self.tickets[i].ts < lo:
                    break
                idx.append(i)
            if len(idx) < 100:
                continue
            ids = np.array([cl.resolve(self.cluster_of[i]) for i in idx])
            groups = hdbscan_groups(self.vecs[idx].astype(np.float64), ids, self.cfg.hdbscan_min_cluster_size,
                                    self.cfg.hdbscan_merge_share, max_points=2500)
            for g in groups:
                keep = min(g)
                for drop in g:
                    if drop != keep and (product, drop) in self.series and (product, keep) in self.series:
                        self._merge(product, keep, drop)

    def _merge(self, product: str, keep: int, drop: int):
        """Вливает кластер drop в keep: ряды, участники, голоса; всплески, где был drop, теперь указывают на keep."""
        kk, dk = (product, keep), (product, drop)
        self.clusterers[product].merge(keep, drop)
        self.series[kk].merge_from(self.series.pop(dk))
        self.members[kk] = sorted(self.members[kk] + self.members.pop(dk))
        self.vote_times[kk] = sorted(self.vote_times[kk] + self.vote_times.pop(dk, []))
        self._rate_cache.pop(kk, None)
        self._rate_cache.pop(dk, None)
        if dk in self.active[product]:
            self.active[product].discard(dk)
            self.active[product].add(kk)
        if dk in self.alerted:
            self.alerted[kk] = max(self.alerted.pop(dk), self.alerted.get(kk, datetime.min))
        if dk in self.busy:
            bid = self.busy.pop(dk)
            self.busy.setdefault(kk, bid)
        for b in self.bursts:
            st = self.bstate[b.id]
            if dk in st["keys"]:
                st["keys"] = [k for k in dict.fromkeys(kk if k == dk else k for k in st["keys"])]
                st["key"] = st["keys"][0]
                b.cluster_ids = [f"{product}:{k[1]}" for k in st["keys"]]
                b.cluster_id = b.cluster_ids[0]
        self.n_merges += 1

    # ------------------------------------------------------------ проверка фикса
    def fix_candidate(self, b: Burst) -> Optional[Release]:
        later = [r for r in self.releases if r.product == b.product and b.detected_at < r.ts <= b.detected_at + timedelta(days=7)]
        if not later:
            return None
        fixes = [r for r in later if r.kind == "fix"]
        return min(fixes or later, key=lambda r: r.ts)

    def _control_times(self, product: str, keys) -> list:
        mem = set()
        for k in keys:
            mem.update(self.members.get(k, []))
        return [self.tickets[i].ts for i in self.product_idx[product] if i not in mem]

    def verify_cluster(self, key: tuple, fix_ts: datetime, now: datetime, pre_from: Optional[datetime] = None) -> dict:
        """Проверка фикса v1 для одного кластера (отношение «после/до» с контролем). Для инструмента агента."""
        cfg = self.cfg
        topic_times = self.vote_times.get(key, [])
        control = self._control_times(key[0], [key]) if cfg.verify_use_control else None
        return verify_fix(topic_times, fix_ts, control_times=control, days=cfg.verify_days, lag_hours=cfg.verify_lag_hours,
                          pre_from=pre_from, now=now, alpha=cfg.verify_alpha, max_ratio=cfg.verify_max_ratio,
                          min_events=cfg.verify_min_events)

    def verify_keys(self, keys: list, fix_ts: datetime, now: datetime, burst_start: datetime,
                    release_ts: datetime) -> dict:
        """Проверка фикса v2 по набору кластеров: вернулась ли тема к норме до релиза (verify_fix_baseline)."""
        cfg = self.cfg
        topic_times = sorted(t for k in keys for t in self.vote_times.get(k, []))
        control = self._control_times(keys[0][0], keys)
        return verify_fix_baseline(topic_times, control, fix_ts, burst_start, release_ts, days=cfg.verify_days,
                                   lag_hours=cfg.verify_lag_hours, baseline_days=cfg.verify_baseline_days, now=now,
                                   alpha=cfg.verify_alpha, max_residual=cfg.verify_max_residual,
                                   min_events=cfg.verify_min_events)

    def incident_keys(self, b: Burst) -> tuple:
        """Кластеры инцидента для проверки фикса: все тревоги того же продукта по тому же подозреваемому релизу
        (осколки одной темы ловятся разными всплесками). Возвращает (ключи, самое раннее начало)."""
        keys, start = list(self.bstate[b.id]["keys"]), b.started_at
        sr = b.suspected_release
        if sr is not None:
            for x in self.bursts:
                if x.id != b.id and x.product == b.product and x.alert and x.suspected_release is not None \
                        and x.suspected_release.version == sr.version and abs(x.detected_at - b.detected_at) <= timedelta(hours=24):
                    keys += [k for k in self.bstate[x.id]["keys"] if k not in keys]
                    start = min(start, x.started_at)
        return keys, start

    def _check_fixes(self, now: datetime):
        cfg = self.cfg
        for b in self.bursts:
            if b.fix_check is not None:
                continue
            fix = self.fix_candidate(b)
            if fix is None:
                continue
            ready = fix.ts + timedelta(hours=cfg.verify_lag_hours, days=cfg.verify_days)
            if now < ready:
                continue
            if cfg.verify_mode == "baseline":
                keys, start = self.incident_keys(b)
                rel_ts = b.suspected_release.ts if b.suspected_release is not None else start
                r = self.verify_keys(keys, fix.ts, now, start, rel_ts)
                b.fix_check = FixCheck(fix.version, fix.ts, r["verdict"], None if r["drop_percent"] is None else round(r["drop_percent"], 1),
                                       r["ratio"], r["ci_low"], r["ci_high"], r["p_value"], r["n_pre"], r["n_post"],
                                       True, now, mode="baseline",
                                       residual_share=None if r["residual"] is None else round(r["residual"], 3),
                                       residual_ci_high=None if r["residual_ci_high"] is None else round(r["residual_ci_high"], 3),
                                       n_baseline=r["n_base"], clusters=[f"{k[0]}:{k[1]}" for k in keys])
            else:
                key = self.bstate[b.id]["key"]
                r = self.verify_cluster(key, fix.ts, now, pre_from=b.started_at)
                b.fix_check = FixCheck(fix.version, fix.ts, r["verdict"], None if r["drop_percent"] is None else round(r["drop_percent"], 1),
                                       r["ratio"], r["ci_low"], r["ci_high"], r["p_value"], r["n_pre"], r["n_post"],
                                       r["used_control"], now)
            self.events.append((now, "fix_checked", b.id))

    # ------------------------------------------------------------ конец потока
    def finish(self, now: datetime):
        self.advance_to(now)
        self._check_fixes(now)
        for b in self.bursts:
            self._refresh(b, now)
        return self.bursts

    def cluster_ticket_indexes(self, b: Burst) -> list:
        """Индексы обращений всех кластеров всплеска (v1: одного кластера)."""
        return self._union_members(self.bstate[b.id]["keys"])


# ---------------------------------------------------------------- публичные функции
def _prepare(tickets: list[Ticket], cfg: Config):
    tickets = sorted(tickets, key=lambda t: t.ts)
    start = tickets[0].ts
    warm_end = start + timedelta(days=cfg.warmup_days)
    texts, reverse = mask_tickets(tickets, ner=cfg.mask_ner, ctx_names=cfg.mask_ctx_names)
    emb = make_embedder(cfg.embedder, cfg.svd_dim)
    warm = [tx for t, tx in zip(tickets, texts) if t.ts < warm_end]
    emb.fit(warm)
    X = emb.transform(texts)
    return tickets, texts, reverse, emb, X


def run_full(tickets: list[Ticket], releases: list[Release], config: Optional[Config] = None,
             until: Optional[datetime] = None, prepared=None, judge=None) -> RadarStream:
    """Прогоняет всё и возвращает состояние. prepared: результат _prepare (чтобы не считать эмбеддинг заново)."""
    cfg = config or Config()
    tickets, texts, reverse, emb, X = prepared or _prepare(tickets, cfg)
    start = tickets[0].ts
    t0 = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    stream = RadarStream(releases, cfg, X.shape[1], t0, start)
    stream.embedder_name = emb.name
    stream.judge = judge            # v3: проверка тревоги (radar.judge.make_judge); используется при config.judge_enabled
    stream.reverse_maps = reverse   # словарь обратной подстановки: для человека, дальше по конвейеру не идёт
    last = start
    for tk, tx, v in zip(tickets, texts, X):
        if until is not None and tk.ts > until:
            break
        stream.push(tk, tx, v)
        last = tk.ts
    stream.finish(until or last)
    return stream


def run(tickets: list[Ticket], releases: list[Release], config: Optional[Config] = None,
        until: Optional[datetime] = None) -> list[Burst]:
    """Главный вход: список всплесков (Burst). until: остановиться на этом моменте (перемотка)."""
    return run_full(tickets, releases, config, until).bursts


def iter_run(tickets: list[Ticket], releases: list[Release], config: Optional[Config] = None,
             skip_until: Optional[datetime] = None, judge=None) -> Iterator[tuple]:
    """Потоковый режим для демо. Подаёт обращения по времени и отдаёт события по мере появления:
        (момент, "opened"|"closed"|"fix_checked", копия Burst на этот момент).
    skip_until: перемотка. Всё до этого момента обрабатывается молча (состояние копится), события после него отдаются."""
    cfg = config or Config()
    tickets, texts, reverse, emb, X = _prepare(tickets, cfg)
    start = tickets[0].ts
    t0 = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    stream = RadarStream(releases, cfg, X.shape[1], t0, start)
    stream.judge = judge
    seen = 0
    for tk, tx, v in zip(tickets, texts, X):
        stream.push(tk, tx, v)
        while seen < len(stream.events):
            ts, kind, bid = stream.events[seen]
            seen += 1
            if skip_until is None or ts >= skip_until:
                b = next(x for x in stream.bursts if x.id == bid)
                yield ts, kind, copy.deepcopy(b)
    stream.finish(tickets[-1].ts)
    while seen < len(stream.events):
        ts, kind, bid = stream.events[seen]
        seen += 1
        if skip_until is None or ts >= skip_until:
            yield ts, kind, copy.deepcopy(next(x for x in stream.bursts if x.id == bid))


def group_incidents(bursts: list[Burst], gap_min: int = 120) -> list[list[Burst]]:
    """Один инцидент = всплески одного продукта, сработавшие с паузой не больше gap_min.
    Тема режется на несколько кластеров, и без склейки команда получила бы по карточке на каждый кусок."""
    groups: list[list[Burst]] = []
    last: dict[str, tuple] = {}   # продукт -> (индекс группы, время последнего срабатывания)
    for b in sorted(bursts, key=lambda x: x.detected_at):
        g = last.get(b.product)
        if g is not None and b.detected_at - g[1] <= timedelta(minutes=gap_min):
            groups[g[0]].append(b)
            last[b.product] = (g[0], b.detected_at)
        else:
            groups.append([b])
            last[b.product] = (len(groups) - 1, b.detected_at)
    return groups


def card_input(stream: RadarStream, b: Burst) -> SuspectedCardInput:
    """Собирает вход для агента: только замаскированные тексты и числа, посчитанные кодом."""
    name = PRODUCTS[b.product][0]
    return SuspectedCardInput(
        burst_id=b.id, product=b.product, product_name=name, detected_at=b.detected_at, started_at=b.started_at,
        unique_users=b.total_unique_users, window_min=b.window_min, z=b.z, platforms=b.platforms, versions=b.versions,
        suspected_release=b.suspected_release, weekly_cost_rub=b.weekly_cost_rub, evidence=b.examples,
        allowed_ids=list(b.example_ids))
