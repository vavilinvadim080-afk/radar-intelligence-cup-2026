"""Official-case needs mode added to the original incident application.

This router reuses the persistent ingestion/validation library. It neither sends
messages nor replaces the legacy detector and its API.
"""
from __future__ import annotations
import hmac
import json
import os
import re
from pathlib import Path
import secrets
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict
from radar_app.core import Radar
from radar_app.tools import ProductTools

ROOT=Path(__file__).resolve().parents[2]
RUNTIME=Path(__file__).resolve().parents[1]/'runtime'

class ProductBody(BaseModel):
    model_config=ConfigDict(extra='forbid')
    product:str=Field(min_length=1,max_length=64,pattern=r'^[A-Za-z0-9_-]+$')
class DecisionBody(ProductBody):
    topic:str=Field(min_length=1,max_length=200)
    status:str
class ToolBody(ProductBody):
    name:str
    args:dict=Field(default_factory=dict)
class ImportBody(ProductBody):
    id:str
    author:str
    created_at:str
    text:str=Field(default='',max_length=4000)
    channel:str='text'
    attachments:list[dict]=Field(default_factory=list,max_length=5)


def private_config():
    RUNTIME.mkdir(parents=True,exist_ok=True)
    file=RUNTIME/'.secrets.json'
    if not file.exists():
        fd=os.open(file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump({'hmac':secrets.token_hex(32),'token':secrets.token_urlsafe(32)},f)
    return json.loads(file.read_text())


def install_needs(app):
    def history_source():
        current=getattr(app.state,'needs_radar',None)
        if current is not None:return current
        # Read an existing DB without initializing an analyzer or writing keys.
        configured=os.environ.get('RADAR_NEEDS_MODE','demo')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',configured):
            raise HTTPException(409,'Неизвестный источник истории')
        return RUNTIME/(configured+'-needs.db'),configured

    def get():
        if getattr(app.state,'needs_radar',None) is None:
            mode=os.environ.get('RADAR_NEEDS_MODE','demo')
            config=private_config();kw={}
            if mode=='semantic-v5b':
                from radar_app.local_selection_runtime_v5b import experimental_analyzer
                try:analyzer=experimental_analyzer()
                except ValueError:raise HTTPException(409,'Локальный смысловой эксперимент ещё не подготовлен или не разрешён')
                kw={'analyzer':analyzer,'now':'2026-11-01T12:00:00Z'}
            elif mode=='semantic':
                from radar_app.semantic_runtime import components
                analyzer,_=components()
                kw={'analyzer':analyzer,'now':'2026-11-01T12:00:00Z'}
            elif mode=='internal':
                from radar_app.gateway import InternalGateway
                gateway=InternalGateway(os.environ['RADAR_MODEL_URL'],allowed_hosts=os.environ['RADAR_MODEL_IPS'].split(','),
                   model=os.environ.get('RADAR_MODEL_NAME','Qwen3.8'),asr_model=os.environ.get('RADAR_ASR_NAME','Qwen-ASR'),
                   vl_model=os.environ.get('RADAR_VL_NAME','Qwen-VL'),token=os.environ.get('RADAR_MODEL_TOKEN',''))
                kw={'analyzer':gateway.analyze,'extractor':gateway.extract}
            elif mode=='demo':kw={'now':'2026-11-01T12:00:00Z'}
            else:raise HTTPException(409,'Неизвестный режим анализа')
            app.state.needs_radar=Radar(RUNTIME/(mode+'-needs.db'),secret=config['hmac'],**kw)
        return app.state.needs_radar

    def principal(request,product=None,write=False):
        supplied=request.headers.get('authorization','').removeprefix('Bearer ')
        duty=[os.environ.get(k,'') for k in ('RADAR_API_TOKEN','RADAR_TOKEN_DUTY')]
        view=os.environ.get('RADAR_TOKEN_VIEW','')
        is_duty=any(v and hmac.compare_digest(v,supplied) for v in duty)
        is_view=bool(view and hmac.compare_digest(view,supplied))
        if not(is_duty or is_view):raise HTTPException(401,'Требуется ключ доступа')
        if write and not is_duty:raise HTTPException(403,'Только чтение')
        # Dedicated configured product scope; UI arguments cannot expand it.
        products=[p.strip() for p in os.environ.get('RADAR_NEEDS_PRODUCTS','webneft,portal').split(',') if p.strip()]
        if product and product not in products:raise HTTPException(403,'Продукт недоступен')
        origin=request.headers.get('origin')
        if write and origin and origin!=str(request.base_url).rstrip('/'):
            raise HTTPException(403,'Origin denied')
        return products

    @app.get('/api/needs/status')
    def status(request:Request):
        products=principal(request);r=get()
        return {'mode':r.mode,'products':products,'window_days':r.window_days,'min_users':r.min_users,'stats':r.stats()}

    @app.get('/api/needs/topics')
    def topics(request:Request,product:str):
        principal(request,product);r=get()
        return {'cards':r.digest(product),'mode':r.mode,'product':product}

    @app.get('/api/needs/context')
    def context(request:Request,product:str):
        principal(request,product);r=get()
        ProductTools(r,product)
        with r.lock:
            tickets=[dict(x) for x in r.db.execute('SELECT id,created,status,channel,text FROM tickets WHERE product=? ORDER BY created DESC LIMIT 100',(product,))]
            events=[dict(x) for x in r.db.execute('SELECT seq,at,kind,ref FROM events WHERE product=? ORDER BY seq DESC LIMIT 30',(product,))]
            outbox=[dict(x) for x in r.db.execute('SELECT id,kind,status,created FROM outbox WHERE product=? ORDER BY created DESC LIMIT 20',(product,))]
        return {'tickets':tickets,'events':events,'outbox':outbox}

    @app.post('/api/needs/demo')
    def demo(request:Request):
        products=principal(request,write=True);r=get()
        if r.mode not in {'demo-rules','local-semantic-prototype','local-calibrated-v5b-experiment'}:raise HTTPException(409,'Пример доступен только в локальном режиме')
        rows=json.loads((ROOT/'data/demo.json').read_text())
        results=[r.ingest(x) for x in rows if x['product'] in products]
        return {'results':results,'stats':r.stats()}

    @app.post('/api/needs/ingest')
    def ingest(body:ImportBody,request:Request):
        principal(request,body.product,True)
        try:return get().ingest(body.model_dump())
        except (ValueError,KeyError,TypeError):raise HTTPException(422,'Неверное обращение')

    @app.post('/api/needs/decision')
    def decide(body:DecisionBody,request:Request):
        principal(request,body.product,True)
        try:get().decide(body.product,body.topic,body.status)
        except ValueError:raise HTTPException(422,'Неверный статус')
        except KeyError:raise HTTPException(404,'Тема не найдена')
        return {'status':body.status}

    @app.post('/api/needs/tool')
    def tool(body:ToolBody,request:Request):
        write=body.name in {'queue_draft','queue_digest'}
        principal(request,body.product,write)
        try:return ProductTools(get(),body.product).call(body.name,body.args)
        except (ValueError,KeyError,TypeError):raise HTTPException(422,'Инструмент или аргументы отклонены')

    return {'history_source':history_source,'principal':principal}
