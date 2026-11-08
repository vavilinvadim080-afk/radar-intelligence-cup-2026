"""Real HTTP contract on isolated TestClient sources, without server lifespan."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from radar_app.core import Radar
from app import server
import app.needs as needs_router

AUTH={'Authorization':'Bearer history-duty'}
VIEW={'Authorization':'Bearer history-view'}


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('RADAR_API_TOKEN','history-duty')
    monkeypatch.setenv('RADAR_TOKEN_VIEW','history-view')
    monkeypatch.setenv('RADAR_NEEDS_PRODUCTS','webneft')
    monkeypatch.setattr(needs_router,'RUNTIME',tmp_path/'runtime')
    def no_model(*a,**k):raise AssertionError('History must not initialize models or replay')
    monkeypatch.setattr(server,'load_or_build',no_model)
    monkeypatch.setattr(needs_router,'private_config',no_model)
    radar=Radar(tmp_path/'needs.db',secret='history-http-secret-32-characters',now='2026-11-01T12:00:00Z',ner=False)
    for i in range(125):
        radar.ingest(dict(product='webneft',id=f'T{i:04}',author=f'a{i}',created_at='2026-10-20T10:00:00Z',text='Нужна выгрузка заказов'))
    monkeypatch.setattr(server.app.state,'needs_radar',radar,raising=False)
    # Existing saved source, not generation, fitting the history adapter.
    at=datetime(2026,3,10,12)
    rows=[(f'L{i:04}',at,'bonus','web','1','chat','Текст источника','Маскированный текст',False) for i in range(70)]
    legacy=SimpleNamespace(rep=SimpleNamespace(tickets=rows,bins_total={'bonus':[]},events=[]),now=at,mode='template',
        lock=threading.RLock(),world=object(),cards={'B1':dict(id='B1',product='bonus',created_sim=at.isoformat(),burst_ids=['B1'])},
        plan=[],autoreply=SimpleNamespace(incidents={'B1':{'writers':[r[0] for r in rows]}},queue=[],operator=[]))
    monkeypatch.setattr(server,'R',legacy)
    yield TestClient(server.app),radar,legacy
    radar.close()


def test_auth_scope_limits_and_no_cross_mode_cursor(client):
    c,_,_=client
    assert c.get('/api/history?mode=needs&product=webneft').status_code==401
    assert c.get('/api/history?mode=needs&product=other',headers=VIEW).status_code==403
    assert c.get('/api/history?mode=incidents&product=webneft',headers=VIEW).status_code==403
    assert c.get('/api/history?mode=other&product=webneft',headers=VIEW).status_code==422
    assert c.get('/api/history?mode=needs&product=webneft&limit=101',headers=VIEW).status_code==422
    first=c.get('/api/history?mode=needs&product=webneft',headers=VIEW).json()
    assert len(first['items'])==50 and first['total']==125
    assert c.get('/api/history',params=dict(mode='incidents',product='bonus',cursor=first['current_cursor']),headers=VIEW).status_code==400
    assert c.get('/api/history',params=dict(mode='needs',product='webneft',cursor='bad'),headers=VIEW).status_code==400


def test_both_card_topics_have_real_server_pagination(client):
    c,_,_=client
    for mode,product,topic,count in [('needs','webneft','wish:export-orders',125),('incidents','bonus','B1',70)]:
        params=dict(mode=mode,product=product,topic=topic)
        first=c.get('/api/history',params=params,headers=VIEW)
        assert first.status_code==200
        data=first.json();ids=[r['id'] for r in data['items']]
        while data['has_next']:
            data=c.get('/api/history',params={**params,'cursor':data['next_cursor']},headers=VIEW).json()
            ids.extend(r['id'] for r in data['items'])
        assert len(ids)==len(set(ids))==count
        assert c.get('/api/history',params={**params,'cursor':first.json()['current_cursor']},headers=VIEW).json()['items']==first.json()['items']


def test_missing_topic_and_incompatible_snapshot_are_explicit(client):
    c,r,_=client
    assert c.get('/api/history?mode=needs&product=webneft&topic=wish:missing',headers=AUTH).status_code==404
    first=c.get('/api/history?mode=needs&product=webneft',headers=AUTH).json()
    r.ingest(dict(product='webneft',id='T0001',author='changed',created_at='2026-10-20T10:00:00Z',text='Иная выгрузка заказов'))
    assert c.get('/api/history',params=dict(mode='needs',product='webneft',cursor=first['next_cursor']),headers=AUTH).status_code==409


def test_cold_history_does_not_create_a_database_or_load_analyzer(client,monkeypatch,tmp_path):
    c,_,_=client
    monkeypatch.setattr(server.app.state,'needs_radar',None)
    monkeypatch.setattr(server,'R',None)
    assert c.get('/api/history/context?mode=incidents',headers=AUTH).json()['products']==[]
    context=c.get('/api/history/context?mode=needs',headers=AUTH)
    assert context.status_code==200
    page=c.get('/api/history?mode=needs&product=webneft',headers=AUTH)
    assert page.status_code==200 and page.json()['items']==[]
    assert not (tmp_path/'runtime').exists()


def test_history_reset_invalidates_old_legacy_snapshot(client):
    c,_,r=client
    first=c.get('/api/history?mode=incidents&product=bonus&topic=B1',headers=AUTH).json()
    r.world=object()
    response=c.get('/api/history',params=dict(mode='incidents',product='bonus',topic='B1',cursor=first['next_cursor']),headers=AUTH)
    assert response.status_code==409


def test_static_history_assets_are_local(client):
    c,_,_=client
    page=c.get('/').text
    assert '/history.js' in page and '/history.css' in page
    assert c.get('/history.js').status_code==200
    assert c.get('/history.css').status_code==200


def test_new_recorded_membership_and_authors_are_available_through_the_same_api(client):
    c,_,r=client
    r.autoreply.incidents.clear()
    r.rep.history_members={'B1':[dict(product='bonus',id=row[0],observed_at=r.now) for row in r.rep.tickets]}
    r.rep.history_authors={'bonus':{row[0]:f'source-{i//2}' for i,row in enumerate(r.rep.tickets)}}
    page=c.get('/api/history?mode=incidents&product=bonus&topic=B1',headers=AUTH).json()
    assert page['coverage']=='saved_stream_membership' and page['total']==70 and page['unique_authors']==35
    assert all(row['author']['kind']=='source' and row['author']['name'] is None for row in page['items'])
    context=c.get('/api/history/context?mode=incidents',headers=AUTH).json()
    assert 'неполная' not in context['notice'].lower()
