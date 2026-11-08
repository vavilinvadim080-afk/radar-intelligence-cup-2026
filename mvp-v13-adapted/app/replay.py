"""Подготовка воспроизведения: один раз прогоняем конвейер ядра в потоковом режиме и запоминаем всё,
что нужно дашборду и агенту. Дальше сервер только «проигрывает» запись с нужной скоростью.

Прогон повторяет radar.pipeline.iter_run, но держит поток (RadarStream) в руках: нужен card_input и ряды по кластерам.
Результат кэшируется в app/cache/replay-<seed>-<noise>-<volume>.json вместе с SHA-256 (удалить файл = пересчитать, около 30 секунд).
Формат JSON, не pickle: чужой файл в app/cache не исполняет код. Старые *.pkl не читаются.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from radar.config import Config
from radar.gen import generate
from radar.models import Burst, Cluster, FixCheck, Release, SuspectedCardInput, SuspectedRelease, to_plain
from radar.pipeline import RadarStream, _prepare, card_input
from radar.gen.topics import PRODUCTS

CACHE = Path(__file__).parent / "cache"
BIN_MIN = 30
DEMO_SEED = 777


@dataclass
class Replay:
    seed: int
    start: datetime
    end: datetime
    tickets: list                 # [(id, ts, product, platform, version, channel, original, masked, pii_flag)]
    bins_total: dict              # product -> [число обращений за 30 мин]
    burst_bins: dict              # burst_id -> {"bins": [...], "times": [iso ...]} люди кластера по 30-минуткам
    events: list                  # [{"ts", "kind", "burst": Burst, "ci": SuspectedCardInput|None}] по времени
    releases: list                # [{"product","version","ts","kind","changelog"}]
    ground_truth: dict
    injections: list = field(default_factory=list)   # события «пришло обращение с инструкцией»: {"ts","ci"}
    embedder: str = ""
    history_members: dict = field(default_factory=dict)  # burst_id -> [{product,id,observed_at}], реальные индексы потока
    history_authors: dict = field(default_factory=dict)  # product -> ticket_id -> исходный user_id, не имя


class _HistoryCapture:
    """Read-only projection of membership observed after each push/finish.

    A final cluster can contain old tickets added by a later merge. Date each new
    edge at observation time, never at that old ticket's timestamp. Normal pushes
    inspect only new accepted indexes; full scans occur on burst birth or merges.
    """
    def __init__(self):
        self.members, self._seen, self._by_key = {}, {}, {}
        self._indexed = self._bursts = self._merges = 0

    def _add(self, stream, burst, index, observed):
        if index in self._seen[burst.id]:
            return
        tk = stream.tickets[index]  # indexes address accepted stream input, not build's input list
        self._seen[burst.id].add(index)
        self.members[burst.id].append(dict(product=tk.product, id=tk.id,
            observed_at=max(tk.ts, burst.detected_at, observed)))

    def observe(self, stream, observed):
        merged = stream.n_merges != self._merges
        added = stream.bursts[self._bursts:]
        if merged:
            self._by_key.clear()
        for burst in stream.bursts if merged else added:
            self.members.setdefault(burst.id, [])
            self._seen.setdefault(burst.id, set())
            for key in stream.burst_keys(burst):
                self._by_key.setdefault(stream.resolve(*key), {})[burst.id] = burst
            for index in stream.cluster_ticket_indexes(burst):
                self._add(stream, burst, index, observed)
        for index in range(self._indexed, len(stream.tickets)):
            tk = stream.tickets[index]
            key = stream.resolve(tk.product, stream.cluster_of[index])
            for burst in self._by_key.get(key, {}).values():
                self._add(stream, burst, index, observed)
        self._indexed, self._bursts, self._merges = len(stream.tickets), len(stream.bursts), stream.n_merges


def _bins(times: list, start: datetime, n: int) -> list:
    out = [0] * n
    for t in times:
        i = int((t - start).total_seconds() // (BIN_MIN * 60))
        if 0 <= i < n:
            out[i] += 1
    return out


def _injection_inputs(ds, tickets, texts, ci_template_for) -> list:
    """Для каждого обращения с инструкцией собираем маленький «вход агента»: само обращение и до 4 соседей той же темы
    (темы берём из меток генератора: это построение демо-набора, а не подсказка агенту)."""
    labels = ds.labels
    by_id = {t.id: (t, tx) for t, tx in zip(tickets, texts)}
    res = []
    for e in ds.ground_truth["events"]:
        if e["type"] != "prompt_injection":
            continue
        for tid in e["ticket_ids"]:
            tk, tx = by_id[tid]
            neigh = [t for t in tickets if t.product == tk.product and labels[t.id]["topic"] == labels[tid]["topic"]
                     and not labels[t.id]["injection"] and abs((t.ts - tk.ts).total_seconds()) < 6 * 3600 and t.id != tid]
            neigh = neigh[:4]
            ev = [{"id": tid, "text": tx}] + [{"id": t.id, "text": by_id[t.id][1]} for t in neigh]
            at = tk.ts + timedelta(minutes=5)
            ci = SuspectedCardInput(
                burst_id=f"W-{tid}", product=tk.product, product_name=PRODUCTS[tk.product][0], detected_at=at, started_at=tk.ts,
                unique_users=len({tk.user_id} | {t.user_id for t in neigh}), window_min=360, z=0.0,
                platforms={}, versions={}, suspected_release=None, weekly_cost_rub=0.0, evidence=ev,
                allowed_ids=[x["id"] for x in ev], min_users_for_status=6)
            res.append({"ts": at, "ci": ci, "ticket_id": tid})
    return res


def build(seed: int = DEMO_SEED, noise: str = "mid", volume: int = 800, *, cfg=None) -> Replay:
    ds = generate(seed, noise=noise, daily_volume=volume)
    cfg = cfg or Config()
    tickets, texts, reverse, emb, X = _prepare(ds.tickets, cfg)
    start = tickets[0].ts
    t0 = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    stream = RadarStream(ds.releases, cfg, X.shape[1], t0, start)
    events, seen = [], 0
    history, authors = _HistoryCapture(), {}

    def drain():
        nonlocal seen
        while seen < len(stream.events):
            ts, kind, bid = stream.events[seen]
            seen += 1
            b = copy.deepcopy(next(x for x in stream.bursts if x.id == bid))
            events.append({"ts": ts, "kind": kind, "burst": b, "ci": card_input(stream, b) if kind == "opened" else None})

    for tk, tx, v in zip(tickets, texts, X):
        authors.setdefault(tk.product, {}).setdefault(tk.id, tk.user_id)
        stream.push(tk, tx, v)
        history.observe(stream, max(tk.ts, stream.last_ts or tk.ts))
        drain()
    stream.finish(tickets[-1].ts)
    history.observe(stream, max(tickets[-1].ts, stream.last_ts or tickets[-1].ts))
    drain()

    n_bins = int((tickets[-1].ts - start).total_seconds() // (BIN_MIN * 60)) + 1
    bins_total = {p: _bins([t.ts for t in tickets if t.product == p], start, n_bins) for p in PRODUCTS}
    burst_bins = {}
    for b in stream.bursts:
        idx = stream.cluster_ticket_indexes(b)
        burst_bins[b.id] = _bins([tickets[i].ts for i in idx], start, n_bins)
    rows = []
    for tk, tx in zip(tickets, texts):
        rows.append((tk.id, tk.ts, tk.product, tk.platform, tk.app_version, tk.channel, tk.text, tx, tx != tk.text))
    rels = [{"product": r.product, "version": r.version, "ts": r.ts, "kind": r.kind, "changelog": r.changelog} for r in ds.releases]
    inj = _injection_inputs(ds, tickets, texts, None)
    return Replay(seed, start, tickets[-1].ts, rows, bins_total, burst_bins, events, rels, ds.ground_truth, inj, emb.name,
                  history_members=history.members, history_authors=authors)


# ---------------------------------------------------------------- безопасный кэш: JSON + SHA-256 вместо pickle
class CacheIntegrityError(Exception):
    """Файл кэша повреждён, подменён или не в нашем формате."""


CACHE_FORMAT = "radar-replay-json-v1"


def _allowed_classes() -> dict:
    return {c.__name__: c for c in (Replay, Burst, SuspectedRelease, FixCheck, SuspectedCardInput, Cluster, Release)}


def _enc(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        if type(o).__name__ not in _allowed_classes():
            raise TypeError(f"класс {type(o).__name__} не разрешён в кэше")
        return {"__dc__": type(o).__name__, "fields": {f.name: _enc(getattr(o, f.name)) for f in dataclasses.fields(o)}}
    if isinstance(o, datetime):
        return {"__dt__": o.isoformat()}
    if isinstance(o, tuple):
        return {"__tuple__": [_enc(x) for x in o]}
    if isinstance(o, (list, set)):
        return [_enc(x) for x in o]
    if isinstance(o, dict):
        bad = [k for k in o if not isinstance(k, str)]
        if bad:
            raise TypeError(f"ключ словаря не строка: {bad[0]!r}")
        return {k: _enc(v) for k, v in o.items()}
    if hasattr(o, "item") and hasattr(o, "dtype"):   # скаляр numpy
        return o.item()
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    raise TypeError(f"тип {type(o).__name__} не умеем писать в кэш")


def _dec(o):
    if isinstance(o, list):
        return [_dec(x) for x in o]
    if isinstance(o, dict):
        if "__dc__" in o:
            cls = _allowed_classes().get(o["__dc__"])
            if cls is None or not isinstance(o.get("fields"), dict):
                raise CacheIntegrityError(f"в кэше неразрешённый класс {o.get('__dc__')!r}")
            return cls(**{k: _dec(v) for k, v in o["fields"].items()})
        if "__dt__" in o and len(o) == 1:
            return datetime.fromisoformat(o["__dt__"])
        if "__tuple__" in o and len(o) == 1:
            return tuple(_dec(x) for x in o["__tuple__"])
        return {k: _dec(v) for k, v in o.items()}
    return o


def digest(data) -> str:
    """SHA-256 от канонической записи JSON (ключи по порядку, без пробелов)."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def write_cache(rep: Replay, path: Path) -> None:
    path = Path(path)
    data = _enc(rep)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"format": CACHE_FORMAT, "sha256": digest(data), "data": data}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_cache(path: Path) -> Replay:
    """Читает кэш, сверив SHA-256. Любое отклонение: CacheIntegrityError. Класс объекта берётся только из белого списка."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as e:   # не JSON (в том числе pickle под видом json) или не UTF-8
        raise CacheIntegrityError(f"{Path(path).name}: файл не JSON ({type(e).__name__}); pickle не читаем") from None
    if not isinstance(doc, dict) or doc.get("format") != CACHE_FORMAT or "data" not in doc:
        raise CacheIntegrityError(f"{Path(path).name}: неизвестный формат кэша")
    if not isinstance(doc.get("sha256"), str) or digest(doc["data"]) != doc["sha256"]:
        raise CacheIntegrityError(f"{Path(path).name}: контрольная сумма SHA-256 не сходится, файл изменён")
    rep = _dec(doc["data"])
    if not isinstance(rep, Replay):
        raise CacheIntegrityError(f"{Path(path).name}: внутри не запись прогона")
    return rep


def load_or_build(seed: int = DEMO_SEED, noise: str = "mid", volume: int = 800, rebuild: bool = False,
                  *, embedder: str | None = None, sim_threshold: float | None = None) -> Replay:
    CACHE.mkdir(exist_ok=True)
    requested=embedder or os.environ.get('RADAR_EMBEDDER')
    if requested in {'local-e5','local-qwen'} and sim_threshold is None:
        supplied=os.environ.get('RADAR_SIM_THRESHOLD')
        if supplied is None:raise ValueError('A local encoder requires an explicit clustering threshold')
        try:sim_threshold=float(supplied)
        except ValueError:raise ValueError('Invalid cluster similarity threshold') from None
    cfg=None;suffix=''
    if requested is not None or sim_threshold is not None:
        from radar.embed import embedding_identity
        requested=requested or 'tfidf';cfg=Config(embedder=requested)
        if sim_threshold is not None:
            if isinstance(sim_threshold,bool) or not isinstance(sim_threshold,(int,float)) or not 0<sim_threshold<=1:
                raise ValueError('Invalid cluster similarity threshold')
            cfg.sim_threshold=sim_threshold
        body=dict(config=cfg.model_dump(),embedding=embedding_identity(requested))
        suffix='-'+hashlib.sha256(json.dumps(body,sort_keys=True,default=str).encode()).hexdigest()[:20]
    path = CACHE / f"replay-{seed}-{noise}-{volume}{suffix}.json"
    if path.exists() and not rebuild:
        try:
            return read_cache(path)
        except CacheIntegrityError as e:
            warnings.warn(f"кэш прогона отклонён ({e}); пересобираю", RuntimeWarning)
    r = build(seed, noise, volume, cfg=cfg) if cfg else build(seed, noise, volume)
    write_cache(r, path)
    return r


def cluster_source_for(rep: Replay):
    """get_cluster для инструмента агента: примеры, динамика за 12 часов (по 30 минут), платформы, версии."""
    last = {}
    for e in rep.events:
        last[e["burst"].id] = e["burst"]

    def source(burst_id: str, limit: int = 8):
        b = last.get(burst_id)
        if b is None:
            return {"examples": []}
        i_end = int((b.detected_at - rep.start).total_seconds() // (BIN_MIN * 60))
        dyn = rep.burst_bins.get(burst_id, [])[max(0, i_end - 24):i_end + 1]
        return {"burst_id": burst_id, "examples": b.examples[:limit], "platforms": b.platforms, "versions": b.versions,
                "people_per_30min_last_12h": dyn}
    return source


def card_plan(rep: Replay, merge_gap_min: int = 120) -> list:
    """Какие карточки будут созданы: тревоги одного продукта с паузой до 2 часов склеиваются в одну карточку
    (та же идея, что у radar.pipeline.group_incidents, но по потоку событий).
    Возвращает [{"card_id", "ci", "burst_ids", "ts", "kind"}], первая тревога группы даёт вход для агента.
    Плюс карточки-наблюдения по обращениям с инструкцией (kind = "watch")."""
    plan, last = [], {}
    for e in rep.events:
        b = e["burst"]
        if e["kind"] != "opened" or not b.alert:
            continue
        side = (b.product, bool(getattr(b, "suspicion", None)) and b.tier == "quiet")   # тревога «подозрение на накрутку» в одну карточку с настоящей не склеивается
        g = last.get(side)
        if g is not None and e["ts"] - g["last"] <= timedelta(minutes=merge_gap_min):
            g["plan"]["burst_ids"].append(b.id)
            g["last"] = e["ts"]
            continue
        item = {"card_id": b.id, "ci": e["ci"], "burst_ids": [b.id], "ts": e["ts"], "kind": "alert"}
        plan.append(item)
        last[side] = {"plan": item, "last": e["ts"]}
    for inj in rep.injections:
        plan.append({"card_id": inj["ci"].burst_id, "ci": inj["ci"], "burst_ids": [inj["ci"].burst_id], "ts": inj["ts"], "kind": "watch"})
    plan.sort(key=lambda x: x["ts"])
    return plan
