"""Isolated synthetic UI fixture. Run only on 127.0.0.1:8798, no models/cache."""
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['RADAR_API_TOKEN'] = 'history-browser-fixture'
os.environ['RADAR_NEEDS_PRODUCTS'] = 'webneft,portal'
os.environ['RADAR_LAZY_INCIDENTS'] = '1'
from radar_app.core import Radar
from app import server
import app.needs as needs


def forbidden(*a, **k):
    raise AssertionError('Fixture must not build replay, initialize analyzer or call a model')


def main():
    with tempfile.TemporaryDirectory(prefix='radar-history-ui-') as directory:
        def analyze(text, catalog):
            return dict(topic_key='export-orders', label='Выгрузка заказов', kind='wish', quote=text, confidence=.9)
        radar = Radar(Path(directory)/'needs.db', secret='synthetic-history-browser-secret',
                      now='2026-11-01T12:00:00Z', ner=False, analyzer=analyze)
        def ingest(ident, at='2026-10-20T10:00:00Z'):
            radar.ingest(dict(product='webneft', id=ident, author='author-'+ident[-3:], created_at=at,
                              text='Нужна выгрузка заказов, синтетическое сообщение '+ident))
        for i in range(125):
            ingest(f'N{i:04}')
        radar.ingest(dict(product='webneft', id='MEDIA', author='media-author', created_at='2026-10-21T10:00:00Z', channel='audio'))
        server.app.state.needs_radar = radar
        server.load_or_build = needs.private_config = forbidden
        server.agent_mod.OllamaClient.available = lambda self: False
        at = datetime(2026, 3, 10, 12)
        rows = [(f'L{i:04}', at, 'bonus', 'web', '1', 'chat', 'Текст источника '+str(i), 'Маскированный текст '+str(i), False) for i in range(125)]
        card = dict(id='B1', product='bonus', product_name='Бонусы', kind='alert', tier='page', created_sim=at.isoformat(),
                    burst_ids=['B1'], pending=False, decision=None, close_proposal=None, suspicion=None,
                    facts=dict(unique_users=42, weekly_cost_rub=0, z=6, suspected_release=None),
                    agent=dict(mode='template', tools_used=[], steps=0, latency_s=0,
                               card=dict(title='Не работает вход · искусственный пример', status='новая', severity='medium',
                                         claims=[], repro_steps_draft=[], existing_ticket=None)))
        rep = NS(tickets=rows, bins_total={'bonus':[125], 'portal':[0]}, events=[], seed=1, embedder='synthetic-fixture',
                 start=at, end=at+timedelta(hours=1), history_authors={'bonus':{r[0]:f'source-author-{i//3}' for i,r in enumerate(rows)}},
                 history_members={'B1':[dict(product='bonus', id=r[0], observed_at=at) for r in rows]})
        server.R = NS(rep=rep, now=at, mode='template', lock=threading.RLock(), world=NS(tasks=lambda:[],drafts=lambda:[],outbox=lambda:[]),
                      cards={'B1':card}, plan=[], recorded={}, overrides={}, threshold_log=[], suppressed=[], events_log=[], listeners=[],
                      autoreply=NS(incidents={},queue=[],operator=[],view=lambda *a:None),
                      audit=NS(rows=lambda *a:[],verify_chain=lambda:dict(ok=True,count=0,head=None)),
                      series=lambda *a:dict(now=at.isoformat(),t0=at.isoformat(),bin_min=30,products=rep.bins_total,bursts={},releases=[],alerts=[],thresholds={}),
                      tickets_feed=lambda *a:dict(with_pii=[],latest=[]), fix_status=lambda:[],
                      autoreply_view=lambda:dict(stats=dict(autoreplied=0,to_operator=0,mailings=0,
                          assumed_minutes_saved=dict(low=0,high=0,source='synthetic fixture')),queue=[],operator=[],mailings=[]))
        @server.app.post('/_fixture/arrivals')
        def arrivals():
            ingest('NEW-LATE', '2026-10-31T10:00:00Z'); ingest('NEW-BACKFILL', '2026-10-19T10:00:00Z')
            return {'ok':True}
        @server.app.get('/preview-history')
        def preview():
            from fastapi.responses import HTMLResponse
            page=(server.STATIC/'dashboard.html').read_text()
            bootstrap='''<script>
sessionStorage.setItem('radar_token','history-browser-fixture');
window.addEventListener('load',()=>document.querySelector('#history-tab')?.click());
</script><style>.history-heading h2::after{content:" · тестовые обращения";color:var(--warn);font-size:.7em}</style>'''
            page=page.replace('<head>','<head>'+bootstrap,1)
            page=page.replace('<title>Радар: дашборд</title>','<title>Радар: история, тестовые обращения</title>',1)
            page=page.replace('Радар · единый поток обращений','Превью истории · вымышленные обращения',1)
            return HTMLResponse(page)
        import uvicorn
        try:
            uvicorn.run(server.app, host='127.0.0.1', port=8798, lifespan='off', access_log=False, log_level='warning')
        finally:
            radar.close()


if __name__ == '__main__':
    main()
