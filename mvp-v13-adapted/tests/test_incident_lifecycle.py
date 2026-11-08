"""Incident lifecycle: replay restores fix evidence and late human decisions keep it."""
from datetime import datetime
import pytest
from app.replay import load_or_build
from app.state import Radar

@pytest.fixture(scope='module')
def replay():return load_or_build()

@pytest.fixture
def radar(replay):return Radar(replay,mode='template',sync_agent=True)

def test_seek_past_fix_keeps_confirmation(radar):
    radar.reset(datetime(2026,3,17,0,10))
    card=radar.cards['B0003']
    assert card['fix'] is not None, 'Seeking to completed verification must restore recorded fix evidence'
    assert card['fix']['verdict']=='confirmed'
    assert card['close_proposal'] is None, 'A task requires an explicit human decision first'

def test_late_task_decision_proposes_close(radar):
    radar.reset(datetime(2026,3,9,18,30))
    radar.advance(datetime(2026,3,17,0,10))
    assert radar.cards['B0003']['fix']['verdict']=='confirmed'
    radar.decide('B0003','our_bug')
    proposal=radar.cards['B0003']['close_proposal']
    assert proposal and proposal['state']=='ждёт человека'
    task=radar.world.task(proposal['task_key'])
    assert task['status']!='done'
    radar.confirm_close('B0003')
    assert radar.world.task(proposal['task_key'])['status']=='done'

def test_late_autoreply_approval_keeps_fixed_proposal(radar):
    radar.reset(datetime(2026,3,9,18,30))
    radar.advance(datetime(2026,3,17,0,10))
    radar.approve_autoreply('B0003')
    view=radar.autoreply.view('B0003')
    assert view['proposal'] and view['proposal']['state']=='ждёт человека'
    assert radar.autoreply.mailings==[], 'Fix mailing remains human-approved'

def test_stream_emits_reply_update_when_counters_change(radar):
    radar.reset(datetime(2026,3,9,18,30));radar.approve_autoreply('B0003')
    events=[];radar.listeners.append(events.append)
    radar.advance(datetime(2026,3,9,21))
    assert radar.autoreply.stats()['autoreplied']>0
    assert any(e['type']=='autoreply_updated' for e in events), 'The browser needs a signal for changed counters'
