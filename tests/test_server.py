import pytest
from fastapi.testclient import TestClient
from radar_app.core import Radar
from radar_app.server import create_app

@pytest.fixture
def client(tmp_path):
    radar=Radar(tmp_path/'api.db',secret='long-test-secret-key',now='2026-11-01T12:00:00Z')
    app=create_app(radar,tokens={'edit':{'role':'editor','products':['webneft']},'read':{'role':'viewer','products':['webneft']},'load':{'role':'ingest','products':['webneft']}},allowed_hosts=['testserver'])
    with TestClient(app) as c:
        yield c
    radar.close()

def test_token_and_product_scope_required(client):
    assert client.get('/api/cards/webneft').status_code==401
    assert client.get('/api/cards/portal',headers={'Authorization':'Bearer read'}).status_code==403
    assert client.get('/api/cards/webneft',headers={'Authorization':'Bearer read'}).status_code==200

def test_viewer_cannot_import_or_decide(client):
    assert client.post('/api/import',json={'tickets':[]},headers={'Authorization':'Bearer read'}).status_code==403
    assert client.post('/api/decision/webneft',json={'topic':'x','status':'accepted'},headers={'Authorization':'Bearer read'}).status_code==403

def test_host_and_cross_site_mutation_blocked(client):
    assert client.get('/',headers={'Host':'evil.org'}).status_code==400
    assert client.post('/api/import',json={'tickets':[]},headers={'Authorization':'Bearer load','Origin':'https://evil.org'}).status_code==403

def test_input_body_and_batch_limits(client):
    h={'Authorization':'Bearer load'}
    assert client.post('/api/import',content='x'*2100000,headers=h).status_code==413
    assert client.post('/api/import',json={'tickets':[{}]*101},headers=h).status_code==422

def test_end_to_end_import_evidence_and_feedback(client):
    rows=[dict(id=str(i),author=str(i),product='webneft',created_at='2026-10-15T10:00:00Z',text='Хотим Excel для заказов') for i in range(3)]
    r=client.post('/api/import',json={'tickets':rows},headers={'Authorization':'Bearer load'})
    assert r.status_code==200
    assert r.json()['statuses']=={'accepted':3}
    card=client.get('/api/cards/webneft',headers={'Authorization':'Bearer read'}).json()['cards'][0]
    assert card['unique_users']==3
    assert len(card['evidence'])==3
    d=client.post('/api/decision/webneft',json={'topic':card['id'],'status':'accepted'},headers={'Authorization':'Bearer edit'})
    assert d.status_code==200
    assert client.get('/api/cards/webneft',headers={'Authorization':'Bearer read'}).json()['cards'][0]['status']=='accepted'

def test_no_token_or_sensitive_data_in_page(client):
    r=client.get('/')
    assert r.status_code==200
    assert 'Bearer edit' not in r.text

def test_invalid_ticket_is_rejected_not_500(client):
    r=client.post('/api/import',json={'tickets':[{'id':'broken'}]},headers={'Authorization':'Bearer load'})
    assert r.status_code==422
