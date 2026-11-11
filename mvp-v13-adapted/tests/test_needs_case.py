"""Contract of the new needs mode inside the original application's API."""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from radar_app.core import Radar
from app import server
from app.needs import install_needs

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('RADAR_API_TOKEN','test-duty-key')
    monkeypatch.setenv('RADAR_TOKEN_VIEW','test-view-key')
    monkeypatch.setenv('RADAR_NEEDS_PRODUCTS','webneft')
    radar=Radar(tmp_path/'case.db',secret='test-key-at-least-16-characters',now='2026-11-01T12:00:00Z',ner=False)
    server.app.state.needs_radar=radar
    c=TestClient(server.app)
    yield c
    radar.close()
    server.app.state.needs_radar=None

AUTH={'Authorization':'Bearer test-duty-key'}
VIEW={'Authorization':'Bearer test-view-key'}

def test_new_mode_keeps_old_routes_and_dashboard(client):
    assert {'/api/cards','/api/control','/api/needs/topics','/api/needs/demo'}<={r.path for r in server.app.routes}
    page=client.get('/').text
    assert 'id="chart"' in page and 'id="cards"' in page
    assert 'id="needs-view"' in page
    assert 'id="needs-tab"' in page


def test_auth_scope_and_viewer_do_not_mutate(client):
    assert client.get('/api/needs/topics?product=webneft').status_code==401
    assert client.get('/api/needs/topics?product=other',headers=AUTH).status_code==403
    assert client.post('/api/needs/demo',headers=VIEW,json={}).status_code==403
    assert client.get('/api/needs/topics?product=webneft',headers=VIEW).status_code==200


def test_demo_decision_draft_and_no_double_count(client):
    assert client.post('/api/needs/demo',headers=AUTH,json={}).status_code==200
    first=client.get('/api/needs/topics?product=webneft',headers=AUTH).json()['cards']
    assert first and all(c['unique_users']>=3 and c['evidence'] for c in first)
    assert client.post('/api/needs/demo',headers=AUTH,json={}).status_code==200
    assert first==client.get('/api/needs/topics?product=webneft',headers=AUTH).json()['cards']
    card=first[0]
    data={'product':'webneft','topic':card['id'],'status':'accepted'}
    assert client.post('/api/needs/decision',headers=AUTH,json=data).status_code==200
    out=client.post('/api/needs/tool',headers=AUTH,json={'product':'webneft','name':'queue_draft','args':{'topic':card['id']}})
    assert out.status_code==200 and out.json()['status']=='pending_delivery'
    assert client.post('/api/needs/tool',headers=AUTH,json={'product':'webneft','name':'delete','args':{}}).status_code==422
    assert client.post('/api/needs/decision',headers=AUTH,json={**data,'product':'other'}).status_code==403


def test_cross_origin_and_invalid_ticket(client):
    assert client.post('/api/needs/demo',headers={**AUTH,'Origin':'https://untrusted.invalid'},json={}).status_code==403
    assert client.post('/api/needs/ingest',headers=AUTH,json={'product':'webneft'}).status_code==422
    assert client.post('/api/needs/decision',headers=AUTH,json={'product':'webneft','topic':'missing','status':'accepted'}).status_code==404


@pytest.mark.parametrize('version',['v5b','v5c'])
def test_new_semantic_experiment_requires_explicit_opt_in(client,tmp_path,monkeypatch,version):
    import app.needs as router
    monkeypatch.setattr(router,'RUNTIME',tmp_path/'isolated-runtime')
    monkeypatch.setenv('RADAR_NEEDS_MODE','semantic-'+version);monkeypatch.delenv('RADAR_ALLOW_EXPERIMENTAL',raising=False)
    server.app.state.needs_radar=None
    assert client.get('/api/needs/status',headers=AUTH).status_code==409


@pytest.mark.parametrize('version,expected_mode',[
    ('v5b','local-calibrated-v5b-experiment'),('v5c','local-verifier-v5c-experiment')])
def test_semantic_experiment_is_connected_to_the_needs_api_without_touching_the_default_database(client,tmp_path,monkeypatch,version,expected_mode):
    import app.needs as router
    import importlib
    local=importlib.import_module('radar_app.local_selection_runtime_'+version)
    class Analyzer:
        mode=expected_mode;minimum_confidence=0;supports_examples=True
        def __call__(self,text,catalog):raise AssertionError('Status must not run inference')
    monkeypatch.setattr(router,'RUNTIME',tmp_path/'isolated-runtime')
    monkeypatch.setattr(local,'experimental_analyzer',lambda:Analyzer())
    monkeypatch.setenv('RADAR_NEEDS_MODE','semantic-'+version);server.app.state.needs_radar=None
    try:
        response=client.get('/api/needs/status',headers=AUTH)
        assert response.status_code==200 and response.json()['mode']==expected_mode
        assert (tmp_path/'isolated-runtime'/('semantic-'+version+'-needs.db')).exists()
    finally:
        if server.app.state.needs_radar is not None:server.app.state.needs_radar.close()
        server.app.state.needs_radar=None
