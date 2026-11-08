"""History reads saved results; it neither classifies nor calls a model."""
from datetime import datetime, timedelta
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from radar_app.core import Radar


def module():
    assert importlib.util.find_spec('app.history'), 'Unified history reader is not implemented'
    return importlib.import_module('app.history')


@pytest.fixture
def needs(tmp_path):
    calls = []
    def analyze(text, catalog):
        calls.append(text)
        return dict(topic_key='export-orders', label='Выгрузка заказов', kind='wish', quote=text, confidence=.9)
    radar = Radar(tmp_path/'needs.db', secret='history-test-secret-32-characters',
                  now='2026-11-01T12:00:00Z', ner=False, analyzer=analyze)
    yield radar, calls
    radar.close()


def ingest(radar, n=125):
    for i in range(n):
        radar.ingest(dict(product='webneft', id=f'T{i:04}', author=f'author-{i//3}',
                          created_at='2026-10-20T10:00:00Z', text=f'Выгрузка заказов, сообщение {i}'))


def test_history_reads_saved_analysis_with_a_bounded_first_page(needs):
    h = module(); radar, calls = needs; ingest(radar)
    before = len(calls)
    reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.NeedsHistory(radar)
    page = reader.page(source, product='webneft', topic='wish:export-orders')
    assert len(page['items']) == 50 and page['has_next']
    assert page['total'] == 125 and page['unique_authors'] == 42
    assert len(calls) == before
    assert page['items'][0]['author']['kind'] == 'hmac'
    assert len(page['items'][0]['author']['id']) == 64
    assert page['items'][0]['author']['name'] is None
    assert page['items'][0]['original_text'] is None
    assert page['items'][0]['source_url'] is None


def test_keyset_ties_and_backfilled_arrivals_do_not_change_the_snapshot(needs):
    h = module(); radar, _ = needs; ingest(radar)
    reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.NeedsHistory(radar)
    first = reader.page(source, product='webneft', topic='wish:export-orders')
    for ident, at in [('NEW-LATE', '2026-10-31T12:00:00Z'), ('NEW-BACKFILL', '2026-10-19T10:00:00Z')]:
        radar.ingest(dict(product='webneft', id=ident, author=ident, created_at=at, text='Выгрузка заказов'))
    ids = [r['id'] for r in first['items']]; page = first
    while page['has_next']:
        page = reader.page(source, product='webneft', topic='wish:export-orders', cursor=page['next_cursor'])
        ids.extend(r['id'] for r in page['items'])
    assert len(ids) == len(set(ids)) == 125
    assert set(ids) == {f'T{i:04}' for i in range(125)}
    back = reader.page(source, product='webneft', topic='wish:export-orders', cursor=first['current_cursor'])
    assert back['items'] == first['items']
    assert reader.page(source, product='webneft')['total'] == 127


def test_cursor_is_signed_and_bound_to_product_topic_channel_and_limit(needs):
    h = module(); radar, _ = needs; ingest(radar, 60)
    reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.NeedsHistory(radar)
    page = reader.page(source, product='webneft', topic='wish:export-orders')
    for changes in [dict(product='portal'), dict(topic='__unassigned__'), dict(channel='audio'), dict(limit=10)]:
        args = dict(product='webneft', topic='wish:export-orders', cursor=page['current_cursor']); args.update(changes)
        with pytest.raises(h.InvalidCursor): reader.page(source, **args)
    with pytest.raises(h.InvalidCursor): reader.page(source, product='webneft', cursor=page['current_cursor']+'broken')
    with pytest.raises(ValueError): reader.page(source, product='webneft', limit=101)


def test_filters_unknown_empty_and_microsecond_order_are_explicit(needs):
    h = module(); radar, _ = needs
    for ident, at in [('A', '2026-10-20T10:00:00Z'), ('B', '2026-10-20T10:00:00.000001Z')]:
        radar.ingest(dict(product='webneft', id=ident, author=ident, created_at=at, text='Выгрузка заказов'))
    radar.ingest(dict(product='webneft', id='AUDIO', author='audio-author', created_at='2026-10-21T10:00:00Z', channel='audio'))
    reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.NeedsHistory(radar)
    assert [r['id'] for r in reader.page(source, product='webneft', channel='text')['items']] == ['B', 'A']
    unknown = reader.page(source, product='webneft', topic='__unassigned__')
    assert unknown['items'][0]['status'] == 'pending_media'
    assert unknown['items'][0]['text'] == '' and unknown['items'][0]['original_text'] is None
    assert reader.page(source, product='webneft', start='2026-10-20', end='2026-10-20')['total'] == 2
    assert reader.page(source, product='portal')['items'] == []
    with pytest.raises(h.UnknownTopic): reader.page(source, product='webneft', topic="wish:missing' OR 1=1 --")


def test_replaced_source_record_expires_a_snapshot_instead_of_silently_skipping_it(needs):
    h = module(); radar, _ = needs; ingest(radar, 60)
    reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.NeedsHistory(radar)
    first = reader.page(source, product='webneft')
    radar.ingest(dict(product='webneft', id='T0001', author='author-0', created_at='2026-10-20T10:00:00Z', text='Изменённая выгрузка заказов'))
    with pytest.raises(h.SnapshotExpired): reader.page(source, product='webneft', cursor=first['next_cursor'])


def test_removing_a_backfilled_highest_rowid_cannot_silently_shorten_a_snapshot(needs):
    h = module(); radar, _ = needs; ingest(radar, 60)
    radar.ingest(dict(product='webneft', id='OLDEST', author='old-author', created_at='2026-10-01T10:00:00Z', text='Выгрузка заказов'))
    reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.NeedsHistory(radar)
    first = reader.page(source, product='webneft')
    with radar.db:
        radar.db.execute("DELETE FROM tickets WHERE product='webneft' AND id='OLDEST'")
    with pytest.raises(h.SnapshotExpired): reader.page(source, product='webneft', cursor=first['next_cursor'])


def test_a_duplicate_source_delivery_does_not_expire_or_duplicate_the_snapshot(needs):
    h = module(); radar, _ = needs; ingest(radar, 60)
    reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.NeedsHistory(radar)
    first = reader.page(source, product='webneft')
    result = radar.ingest(dict(product='webneft', id='T0001', author='author-0', created_at='2026-10-20T10:00:00Z', text='Выгрузка заказов, сообщение 1'))
    assert result['status'] == 'duplicate'
    second = reader.page(source, product='webneft', cursor=first['next_cursor'])
    assert len(first['items']) + len(second['items']) == 60
    assert set(r['id'] for r in first['items']).isdisjoint(r['id'] for r in second['items'])


class ReplayWithoutGold:
    @property
    def ground_truth(self): raise AssertionError('History must not read gold')
    @property
    def labels(self): raise AssertionError('History must not classify from labels')


def legacy(n=125):
    rep = ReplayWithoutGold(); at = datetime(2026, 3, 10, 12)
    rep.tickets = [(f'T{i:04}', at, 'bonus', 'web', '1', 'chat', f'Источник {i}', f'Маска {i}', False) for i in range(n)]
    rep.bins_total = {'bonus': [], 'portal': []}
    ci = SimpleNamespace(allowed_ids=['T0000', 'REL-1'], evidence=[{'id': 'T0000'}])
    rep.events = [dict(ts=at, burst=SimpleNamespace(id='B1', product='bonus', examples=[{'id': 'T0000'}]), ci=ci)]
    r = SimpleNamespace(rep=rep, now=at+timedelta(hours=1), mode='template', lock=threading.RLock(), world=object(),
        cards={'B1': dict(id='B1', product='bonus', created_sim=at.isoformat(), burst_ids=['B1'], agent={'card': {'title': 'Не работает вход'}})},
        plan=[dict(card_id='B1', burst_ids=['B1'], ci=ci, ts=at)],
        autoreply=SimpleNamespace(incidents={'B1': {'writers': [f'T{i:04}' for i in range(n)]}}, queue=[], operator=[]))
    r.cluster_source = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Do not run a tool on history open'))
    return r


def test_legacy_indexes_only_existing_ids_and_reports_missing_authors():
    h = module(); radar = legacy(); reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.LegacyHistory(radar)
    page = reader.page(source, product='bonus', topic='B1')
    assert page['total'] == 125 and len(page['items']) == 50
    assert page['coverage'] == 'confirmed_ids_only'
    assert page['unique_authors'] is None
    assert all(r['author']['id'] is None and r['author']['name'] is None for r in page['items'])
    assert page['items'][0]['original_text'].startswith('Источник ')
    assert 'неполная' in page['notice'].lower()
    assert reader.page(source, product='portal')['total'] == 0


def test_new_legacy_associations_do_not_reorder_an_existing_snapshot():
    h = module(); radar = legacy(); radar.autoreply.incidents['B1']['writers'] = [f'T{i:04}' for i in range(100)]
    reader, source = h.HistoryReader(secret=b'history-cursor-secret'), h.LegacyHistory(radar)
    first = reader.page(source, product='bonus', topic='B1')
    radar.autoreply.incidents['B1']['writers'].extend(f'T{i:04}' for i in range(100, 125))
    second = reader.page(source, product='bonus', topic='B1', cursor=first['next_cursor'])
    assert first['total'] == second['total'] == 100
    assert set(r['id'] for r in first['items']).isdisjoint(r['id'] for r in second['items'])
    assert reader.page(source, product='bonus', topic='B1')['total'] == 125


def test_readonly_needs_history_does_not_initialize_an_analyzer(tmp_path):
    h = module(); path = tmp_path/'not-created.db'
    source = h.NeedsHistory.from_path(path, mode='internal')
    page = h.HistoryReader(secret=b'history-cursor-secret').page(source, product='webneft')
    assert page['total'] == 0 and not path.exists()
    source.close()


def test_existing_needs_database_is_opened_readonly_without_core_initialization(needs, tmp_path):
    import sqlite3
    h = module(); radar, calls = needs; ingest(radar, 5)
    source = h.NeedsHistory.from_path(tmp_path/'needs.db', mode='internal')
    try:
        assert h.HistoryReader(secret=b'history-cursor-secret').page(source, product='webneft')['total'] == 5
        assert len(calls) == 5
        with pytest.raises(sqlite3.OperationalError): source.db.execute('DELETE FROM tickets')
    finally:
        source.close()


def test_a_future_saved_example_cannot_hide_an_already_recorded_routing():
    h = module(); radar = legacy(3)
    radar.autoreply.incidents['B1']['writers'] = ['T0000', 'T0002']
    radar.rep.events.append(dict(ts=radar.now+timedelta(hours=1), burst=SimpleNamespace(id='B1', product='bonus', examples=[{'id':'T0002'}]), ci=None))
    page = h.HistoryReader(secret=b'history-cursor-secret').page(h.LegacyHistory(radar), product='bonus', topic='B1')
    assert {r['id'] for r in page['items']} == {'T0000','T0002'}
