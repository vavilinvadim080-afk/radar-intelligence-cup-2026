import pytest
from radar_app.core import Radar
from radar_app.tools import ProductTools

def make(tmp_path):
    r=Radar(tmp_path/'tools.db',secret='stable-long-test-key',now='2026-11-01T12:00:00Z')
    for i in range(3):r.ingest(dict(id=str(i),product='webneft',author=str(i),created_at='2026-10-15T10:00:00Z',text='Хотим Excel для заказов'))
    return r,ProductTools(r,'webneft')

def test_unsupported_tool_cannot_write_or_contact_user(tmp_path):
    r,t=make(tmp_path)
    for name in ['delete_ticket','notify_user','http_request']:
        with pytest.raises(ValueError):t.call(name,{})

def test_draft_requires_product_decision_and_is_idempotent(tmp_path):
    r,t=make(tmp_path);key=r.digest('webneft')[0]['id']
    with pytest.raises(ValueError):t.call('queue_draft',{'topic':key})
    r.decide('webneft',key,'accepted')
    first=t.call('queue_draft',{'topic':key})
    second=t.call('queue_draft',{'topic':key})
    assert first['id']==second['id']
    assert first['status']=='pending_delivery'
    assert r.db.execute('SELECT count(*) FROM outbox').fetchone()[0]==1

def test_digest_is_queued_not_sent_and_repeated_call_is_same(tmp_path):
    r,t=make(tmp_path)
    a=t.call('queue_digest',{})
    assert a['status']=='pending_delivery'
    assert a['id']==t.call('queue_digest',{})['id']

def test_tool_rejects_cross_product_argument(tmp_path):
    r,t=make(tmp_path)
    with pytest.raises(ValueError):t.call('list_topics',{'product':'portal'})
    assert len(t.call('list_topics',{}))==1
