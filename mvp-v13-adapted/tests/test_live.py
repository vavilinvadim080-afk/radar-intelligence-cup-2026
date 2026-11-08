"""Контрольные точки и устойчивость приёма: radar/live.py и RadarStream.admit/push."""
import json
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.audit import Audit
from radar.config import Config
from radar.gen import generate
from radar.live import FORMAT_VERSION, LiveRadar, _checksum, atomic_write_json


def sig(bursts):
    return [(b.id, b.product, tuple(b.cluster_ids), b.detected_at, b.started_at, b.z, b.count_in_window, b.alert,
             b.total_unique_users, b.status) for b in bursts]


@pytest.fixture(scope="module")
def ds():
    return generate(904, "mid", 500, 21)


@pytest.fixture(scope="module")
def tickets(ds):
    return sorted(ds.tickets, key=lambda t: t.ts)


@pytest.fixture(scope="module")
def baseline(ds, tickets):
    """Поток без перезапуска: эталон для сравнения."""
    live = LiveRadar(ds.releases, Config())
    live.ingest_many(tickets)
    assert any(b.alert for b in live.bursts), "в наборе должна быть хотя бы одна тревога, иначе сравнение пустое"
    return live


def split_index(baseline, tickets):
    """Точка сохранения: через 20 минут после первого срабатывания (всплеск обычно ещё открыт)."""
    b0 = baseline.bursts[0]
    return next(i for i, t in enumerate(tickets) if t.ts >= b0.detected_at + timedelta(minutes=20))


def test_a_restart_gives_same_alerts(ds, tickets, baseline, tmp_path):
    k = split_index(baseline, tickets)
    first = LiveRadar(ds.releases, Config())
    first.ingest_many(tickets[:k])
    assert first.stream.open, "на момент сохранения должен идти открытый всплеск, иначе проверка слабая"
    path = str(tmp_path / "ckpt.json")
    first.save_checkpoint(path)
    del first                                           # «перезапуск»
    events = []
    second = LiveRadar.load_checkpoint(path, ds.releases, Config(), log=lambda kind, p: events.append(kind))
    assert "checkpoint_loaded" in events and "checkpoint_summary_differs" not in events
    assert second.stream.open, "открытый всплеск должен пережить перезапуск"
    second.ingest_many(tickets[k:])
    assert sig(second.bursts) == sig(baseline.bursts)
    assert second.n_accepted() == baseline.n_accepted()


def test_a_checkpoint_has_no_raw_pii(ds, tickets, tmp_path):
    live = LiveRadar(ds.releases, Config())
    live.ingest_many(tickets[:4000])
    path = tmp_path / "ckpt.json"
    live.save_checkpoint(str(path))
    blob = path.read_text(encoding="utf-8")
    emails = [v for t in tickets[:4000] for k, v in ds.labels[t.id]["pii"] if k == "email" and "@" in v]
    assert emails and all(e not in blob for e in emails)


def test_a_save_during_warmup_restores(ds, tickets, tmp_path):
    live = LiveRadar(ds.releases, Config())
    live.ingest_many(tickets[:500])
    assert live.stream is None
    path = str(tmp_path / "w.json")
    live.save_checkpoint(path)
    again = LiveRadar.load_checkpoint(path, ds.releases, Config())
    assert again.stream is None and again.n_accepted() == 500
    for t in tickets[500:520]:
        assert again.ingest(t) == "ok"


def test_a_save_is_atomic_old_file_survives_failed_write(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.json"
    atomic_write_json(str(path), {"v": 1})
    import radar.live as L

    def boom(*a, **k):
        raise OSError("диск полон")
    monkeypatch.setattr(L.json, "dump", boom)
    with pytest.raises(OSError):
        atomic_write_json(str(path), {"v": 2})
    assert json.loads(path.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.json"], "временный файл не должен оставаться"


def _corrupt(kind, good: str) -> str:
    body = json.loads(good)
    if kind == "empty":
        return ""
    if kind == "garbage":
        return "\x00\x01not json"
    if kind == "truncated":
        return good[: len(good) // 2]
    if kind == "wrong_version":
        body["format_version"] = FORMAT_VERSION + 1
        body["checksum"] = _checksum(body)           # сумма верна, отвергнуть должна именно версия
        return json.dumps(body)
    if kind == "bad_checksum":
        body["tickets"][0]["text"] = "подмена"
        return json.dumps(body)
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["empty", "garbage", "truncated", "wrong_version", "bad_checksum", "missing"])
def test_b_bad_checkpoint_gives_clean_start_and_audit_record(kind, ds, tickets, tmp_path):
    good = LiveRadar(ds.releases, Config())
    good.ingest_many(tickets[:300])
    gpath = tmp_path / "good.json"
    good.save_checkpoint(str(gpath))
    path = tmp_path / "bad.json"
    if kind != "missing":
        path.write_text(_corrupt(kind, gpath.read_text(encoding="utf-8")), encoding="utf-8")
    audit = Audit()
    live = LiveRadar.load_checkpoint(str(path), ds.releases, Config(), log=lambda k, p: audit.log("system", "code", {"msg": k, **p}))
    assert live.n_accepted() == 0 and live.stream is None
    rows = [r for r in audit.rows() if r["payload"]["msg"] == "checkpoint_rejected"]
    assert len(rows) == 1 and rows[0]["payload"]["reason"]
    assert live.ingest(tickets[0]) == "ok"             # после чистого старта система принимает обращения


def test_c_duplicates_do_not_double_counters(ds, tickets, baseline):
    events = []
    live = LiveRadar(ds.releases, Config(), log=lambda k, p: events.append(k))
    with_dups = []
    for i, t in enumerate(tickets):
        with_dups.append(t)
        if i % 5 == 0:
            with_dups.append(t)                          # тот же id второй раз сразу после первого
    dup_count = len(with_dups) - len(tickets)
    res = live.ingest_many(with_dups)
    assert res["duplicate"] == dup_count
    assert live.n_accepted() == len(tickets)
    assert events.count("ticket_duplicate") == dup_count
    st, st0 = live.stream, baseline.stream
    assert sum(len(v) for v in st.members.values()) == sum(len(v) for v in st0.members.values())
    assert {p: sum(h.values()) for p, h in st.season.hour_totals.items()} == {p: sum(h.values()) for p, h in st0.season.hour_totals.items()}
    assert sig(live.bursts) == sig(baseline.bursts)


def test_c_legacy_push_without_guard_double_counts(ds, tickets):
    """Демонстрация проблемы: до правки повтор id удваивал счётчики. _push это прежнее тело push без проверки."""
    live = LiveRadar(ds.releases, Config())
    live.ingest_many(tickets[:4000])
    st = live.stream
    t = tickets[3999]
    n_before, total_before = len(st.tickets), sum(sum(h.values()) for h in st.season.hour_totals.values())
    st._push(t, st.texts[-1], st.vecs[len(st.tickets) - 1])
    assert len(st.tickets) == n_before + 1
    assert sum(sum(h.values()) for h in st.season.hour_totals.values()) == total_before + 1
    # а штатный push повтор отклоняет
    st2_before = len(st.tickets)
    assert st.push(t, st.texts[-1], st.vecs[len(st.tickets) - 1]) == "duplicate"
    assert len(st.tickets) == st2_before


def jitter(tickets, minutes):
    """Соседние обращения в пределах `minutes` меняются местами: порядок прихода не равен порядку времени."""
    out = list(tickets)
    i = 0
    while i < len(out) - 1:
        if out[i + 1].ts - out[i].ts <= timedelta(minutes=minutes) and i % 7 == 0:
            out[i], out[i + 1] = out[i + 1], out[i]
            i += 2
        else:
            i += 1
    return out


def test_d_late_within_window_does_not_break_detector(ds, tickets, baseline):
    shuffled = jitter(tickets, 10)
    assert shuffled != tickets
    events = []
    live = LiveRadar(ds.releases, Config(), log=lambda k, p: events.append(k))
    res = live.ingest_many(shuffled)
    assert res.get("late_clamped", 0) > 100 and "too_late" not in res
    assert live.n_accepted() == len(tickets)
    # тревоги те же: продукт, признак тревоги, время срабатывания не сдвинулось больше чем на два шага детектора
    a = [(b.product, b.alert) for b in baseline.bursts if b.alert]
    b_ = [(b.product, b.alert) for b in live.bursts if b.alert]
    assert a == b_
    for x, y in zip([b for b in baseline.bursts if b.alert], [b for b in live.bursts if b.alert]):
        assert abs((x.detected_at - y.detected_at).total_seconds()) <= 2 * 60 * Config().step_min
    stamps = [t.ts for t in live.stream.tickets]
    assert stamps == sorted(stamps), "после приёма времена обращений в потоке не убывают"
    for p, s in live.stream.series.items():           # главное: recent остаётся упорядоченным, окна не слепнут
        times = [r[0] for r in s.recent]
        assert times == sorted(times)


def test_d_too_late_is_dropped_and_logged(ds, tickets):
    events = []
    live = LiveRadar(ds.releases, Config(), log=lambda k, p: events.append((k, p)))
    live.ingest_many(tickets[:4000])
    n = live.n_accepted()
    old = replace(tickets[100], id="LATE-1")
    assert live.ingest(old) == "too_late"
    assert live.n_accepted() == n
    assert ("ticket_too_late", {"ticket_id": "LATE-1", "ts": old.ts.isoformat(timespec="seconds")}) in events
    boundary = replace(tickets[3999], id="LATE-2", ts=tickets[3999].ts - timedelta(minutes=Config().late_tolerance_min))
    assert live.ingest(boundary) == "late_clamped"


def test_d_legacy_push_with_old_timestamp_blinds_windows(ds, tickets):
    """Демонстрация проблемы: опоздавший голос в конце recent останавливает обратный просмотр окна."""
    live = LiveRadar(ds.releases, Config())
    live.ingest_many(tickets[:6000])
    st = live.stream
    now = st.last_ts
    key = max(st.series, key=lambda k: len(st.series[k].recent))
    s = st.series[key]
    before = s.window_counts(now, 120, None)[0]
    assert before >= 2
    s.add_vote(now - timedelta(hours=5), "u-late", st.hour(now), 60, "")   # прежнее поведение: голос в конец списка без проверки
    assert s.window_counts(now, 120, None)[0] < before
