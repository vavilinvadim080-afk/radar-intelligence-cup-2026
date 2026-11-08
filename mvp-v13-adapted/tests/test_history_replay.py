"""Saved stream membership, not reconstructed labels or final-cluster lookahead."""
from datetime import datetime, timedelta
import json
from types import SimpleNamespace
import threading

import numpy as np

from app import replay
from app.history import HistoryReader, LegacyHistory
from radar.config import Config
from radar.models import Ticket
from test_history import legacy


AT = datetime(2026, 3, 10, 12)


def ticket(ident, minute=0, author='source-user'):
    return Ticket(ident, AT+timedelta(minutes=minute), 'bonus', 'app_chat', 'android', '1', author, 'Не работает вход')


class ObservedStream:
    """Only the public membership operations needed by the capture."""
    def __init__(self):
        self.tickets = [ticket('OTHER'), ticket('MATCH', 1)]
        self.cluster_of = [1, 2]
        self.bursts = [SimpleNamespace(id='B1', product='bonus', detected_at=AT+timedelta(minutes=2))]
        self.n_merges = self.scans = 0

    def resolve(self, product, cid):
        return product, 2 if self.n_merges else cid

    def burst_keys(self, burst):
        return [('bonus', 2)]

    def cluster_ticket_indexes(self, burst):
        self.scans += 1
        return [i for i, cid in enumerate(self.cluster_of) if self.resolve('bonus', cid) == ('bonus', 2)]


def test_capture_is_incremental_and_does_not_backdate_members_added_by_a_merge():
    assert hasattr(replay, '_HistoryCapture'), 'Replay does not record observed stream membership'
    stream = ObservedStream(); capture = replay._HistoryCapture()
    capture.observe(stream, AT+timedelta(minutes=2))
    stream.tickets.append(ticket('NEW', 3)); stream.cluster_of.append(2)
    capture.observe(stream, AT+timedelta(minutes=3))
    capture.observe(stream, AT+timedelta(minutes=3))  # rejected input changes no accepted index
    assert stream.scans == 1, 'Ordinary arrivals must not rescan the entire cluster'
    stream.n_merges += 1
    capture.observe(stream, AT+timedelta(minutes=4))
    saved = {r['id']: r for r in capture.members['B1']}
    assert set(saved) == {'MATCH', 'NEW', 'OTHER'}
    assert saved['MATCH']['observed_at'] == AT+timedelta(minutes=2)
    assert saved['NEW']['observed_at'] == AT+timedelta(minutes=3)
    assert saved['OTHER']['observed_at'] == AT+timedelta(minutes=4)
    assert stream.scans == 2


def tiny_build(monkeypatch, user_cap_max=0):
    tickets = [ticket(f'T{i:03}', i, f'author-{i//2}') for i in range(12)]
    class Dataset:
        releases = []
        ground_truth = {}  # build's existing output field; never used by history capture
        @property
        def labels(self):
            raise AssertionError('History must not use labels')
    ds = Dataset(); ds.tickets = tickets
    monkeypatch.setattr(replay, 'generate', lambda *a, **k: ds)
    # Synthetic fixed vectors replace preparation only; the real RadarStream runs.
    monkeypatch.setattr(replay, '_prepare', lambda ts, cfg:
        (ts, [t.text for t in ts], {}, SimpleNamespace(name='synthetic-fixed-vectors'),
         np.tile(np.array([[1., 0.]], dtype=np.float32), (len(ts), 1))))
    # This independent legacy demo generator reads labels; it is outside history.
    monkeypatch.setattr(replay, '_injection_inputs', lambda *a: [])
    cfg = Config(warmup_days=0, use_hdbscan=False, z_threshold=1, min_users=3,
                 alert_policy='all', version_signal=False, subset_scan=False, mask_ner=False, user_cap_max=user_cap_max)
    return replay.build(seed=1, volume=12, cfg=cfg), tickets


def test_tiny_real_stream_build_records_all_members_and_native_author_ids_without_nn(monkeypatch):
    rep, tickets = tiny_build(monkeypatch)
    assert getattr(rep, 'history_members', {}), 'build still saves samples only'
    opened = [e for e in rep.events if e['kind'] == 'opened']
    assert opened
    members = rep.history_members[opened[0]['burst'].id]
    assert {r['id'] for r in members} == {t.id for t in tickets}
    assert all(r['observed_at'] >= opened[0]['ts'] for r in members)
    assert rep.history_authors == {'bonus': {t.id: t.user_id for t in tickets}}
    assert rep.embedder == 'synthetic-fixed-vectors'
    plan = replay.card_plan(rep)
    assert plan
    radar = SimpleNamespace(rep=rep, now=rep.end, mode='template', lock=threading.RLock(), world=object(),
        plan=plan, autoreply=SimpleNamespace(incidents={}, queue=[], operator=[]),
        cards={p['card_id']: dict(id=p['card_id'], product=p['ci'].product,
               created_sim=p['ts'], burst_ids=p['burst_ids']) for p in plan})
    page = HistoryReader().page(LegacyHistory(radar), product='bonus', topic=plan[0]['card_id'])
    assert page['total'] == len(tickets) and page['unique_authors'] == 6
    assert page['coverage'] == 'saved_stream_membership'
    assert {row['author']['id'] for row in page['items']} == {tk.user_id for tk in tickets}


def test_build_uses_accepted_stream_indexes_after_user_capping(monkeypatch):
    rep, tickets = tiny_build(monkeypatch, user_cap_max=1)
    assert rep.history_members
    saved_ids = {row['id'] for members in rep.history_members.values() for row in members}
    assert saved_ids == {tk.id for tk in tickets[::2]}
    # Rejected messages remain in the source history, with their source author,
    # but do not acquire a fabricated detector membership.
    assert set(rep.history_authors['bonus']) == {tk.id for tk in tickets}


def test_new_cache_roundtrips_and_old_cache_keeps_default_empty_fields_without_rebuilding(tmp_path, monkeypatch):
    rep, _ = tiny_build(monkeypatch)
    assert hasattr(rep, 'history_members'), 'Replay cache has no optional history fields'
    path = tmp_path/'replay-1-mid-12.json'; replay.write_cache(rep, path)
    loaded = replay.read_cache(path)
    assert loaded.history_members == rep.history_members
    assert loaded.history_authors == rep.history_authors
    doc = json.loads(path.read_text())
    for name in ('history_members', 'history_authors'):
        del doc['data']['fields'][name]
    doc['sha256'] = replay.digest(doc['data'])
    path.write_text(json.dumps(doc)); before = path.read_bytes()
    monkeypatch.setattr(replay, 'CACHE', tmp_path)
    monkeypatch.delenv('RADAR_EMBEDDER', raising=False)
    def no_build(*a, **k):
        raise AssertionError('A valid old cache must not be regenerated')
    monkeypatch.setattr(replay, 'build', no_build)
    old = replay.load_or_build(seed=1, volume=12)
    assert old.history_members == old.history_authors == {}
    assert path.read_bytes() == before


def test_recorded_full_membership_paginates_authors_and_hides_future_merge_edges():
    r = legacy(125); r.autoreply.incidents.clear()
    r.rep.history_authors = {'bonus': {f'T{i:04}': f'author-{i//3}' for i in range(125)}}
    r.rep.history_members = {'B1': [dict(id=f'T{i:04}', product='bonus', observed_at=AT) for i in range(124)] +
                                [dict(id='T0124', product='bonus', observed_at=AT+timedelta(hours=2))]}
    source, reader = LegacyHistory(r), HistoryReader(secret=b'saved-membership-test')
    first = reader.page(source, product='bonus', topic='B1')
    assert first['total'] == 124, 'History still has samples only, or exposes a future merge'
    assert first['coverage'] == 'saved_stream_membership'
    assert first['unique_authors'] == 42 and first['messages_without_author'] == 0
    assert all(x['author']['kind'] == 'source' and x['author']['name'] is None for x in first['items'])
    r.now = AT+timedelta(hours=3)
    ids = [x['id'] for x in first['items']]; page = first
    while page['has_next']:
        page = reader.page(source, product='bonus', topic='B1', cursor=page['next_cursor'])
        ids.extend(x['id'] for x in page['items'])
    assert len(ids) == len(set(ids)) == 124 and 'T0124' not in ids
    assert reader.page(source, product='bonus', topic='B1')['total'] == 125
    assert reader.page(source, product='bonus', topic='B1', cursor=first['current_cursor'])['items'] == first['items']
    # A watch card without a detector record remains explicitly partial.
    r.cards['WATCH'] = dict(id='WATCH', product='bonus', created_sim=AT, burst_ids=['W1'])
    watch = reader.page(source, product='bonus', topic='WATCH')
    assert watch['coverage'] == 'confirmed_ids_only' and 'неполная' in watch['notice'].lower()
