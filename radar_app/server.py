"""Local demonstration API; production ingress and SSO are deployment integrations."""
from collections import Counter, defaultdict, deque
import hmac
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from .economics import SCENARIOS, calculate

ROOT=Path(__file__).resolve().parent.parent


class Boundary:
    def __init__(self,app,hosts):
        self.app=app;self.hosts=set(hosts)
    async def __call__(self,scope,receive,send):
        if scope["type"]!="http":
            return await self.app(scope,receive,send)
        headers={k.decode().lower():v.decode() for k,v in scope.get("headers",[])}
        host=headers.get("host",""); hostname=urlsplit("//"+host).hostname
        if hostname not in self.hosts:
            return await JSONResponse({"detail":"Host denied"},400)(scope,receive,send)
        origin=headers.get("origin")
        if scope["method"] not in {"GET","HEAD"} and origin and origin!=scope.get("scheme","http")+"://"+host:
            return await JSONResponse({"detail":"Origin denied"},403)(scope,receive,send)
        chunks=[];size=0
        while True:
            message=await receive()
            if message["type"]=="http.disconnect":return
            body=message.get("body",b"");size+=len(body)
            if size>2000000:
                return await JSONResponse({"detail":"Body limit: 2 MB"},413)(scope,receive,send)
            chunks.append(body)
            if not message.get("more_body",False):break
        body=b"".join(chunks);sent=False
        async def replay():
            nonlocal sent
            if not sent:
                sent=True;return {"type":"http.request","body":body,"more_body":False}
            return await receive()
        async def safe_send(message):
            if message["type"]=="http.response.start":
                message.setdefault("headers",[]).extend([(b"x-content-type-options",b"nosniff"),(b"referrer-policy",b"no-referrer"),(b"cache-control",b"no-store"),
                    (b"content-security-policy",b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")])
            await send(message)
        await self.app(scope,replay,safe_send)


def create_app(radar,*,tokens,allowed_hosts=("127.0.0.1","localhost")):
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(Boundary,hosts=allowed_hosts)
    requests=defaultdict(deque)

    def principal(request,roles=None,product=None):
        supplied=request.headers.get("authorization","").removeprefix("Bearer ")
        entry=next((v for k,v in tokens.items() if hmac.compare_digest(k,supplied)),None)
        if entry is None:raise HTTPException(401,"Требуется ключ доступа")
        if roles and entry["role"] not in roles:raise HTTPException(403,"Недостаточно прав")
        if product and product not in entry["products"]:raise HTTPException(403,"Продукт недоступен")
        now=time.monotonic();q=requests[supplied]
        while q and q[0]<now-60:q.popleft()
        if len(q)>=120:raise HTTPException(429,"Лимит 120 запросов в минуту",headers={"Retry-After":"60"})
        q.append(now)
        return entry

    @app.get("/",response_class=HTMLResponse)
    def index():
        p=ROOT/"radar_app"/"static"/"index.html"
        return p.read_text() if p.exists() else "<h1>Радар</h1><p>Интерфейс готовится.</p>"

    @app.get("/app.js")
    def js():
        from fastapi.responses import Response
        return Response((ROOT/"radar_app"/"static"/"app.js").read_text(),media_type="application/javascript")

    @app.get("/api/cards/{product}")
    def cards(product:str,request:Request):
        principal(request,product=product)
        return {"cards":radar.digest(product),"mode":radar.mode,"product":product}

    @app.get("/api/queue/{product}")
    def queue(product:str,request:Request):
        principal(request,product=product)
        with radar.lock:
            rows=radar.db.execute("SELECT id,status,channel,created FROM tickets WHERE product=? ORDER BY created DESC LIMIT 100",(product,)).fetchall()
        return {"rows":[dict(r) for r in rows]}

    @app.get("/api/economics")
    def economics(request:Request):
        principal(request)
        return {k:{"inputs":v,"results":calculate(v)} for k,v in SCENARIOS.items()}

    @app.post("/api/demo")
    async def demo(request:Request):
        user=principal(request,{"admin"})
        if radar.mode!="demo-rules":raise HTTPException(409,"Демо доступно только в демонстрационном режиме")
        rows=json.loads((ROOT/"data"/"demo.json").read_text())
        if any(t['product'] not in user['products'] for t in rows):raise HTTPException(403,"Продукт недоступен")
        result=[]
        for row in rows:result.append(await run_in_threadpool(radar.ingest,row))
        return {"statuses":dict(Counter(r['status'] for r in result)),"synthetic":True}

    @app.post("/api/import")
    async def ingest(request:Request):
        user=principal(request,{"ingest","admin"})
        try:
            body=await request.json(); rows=body["tickets"]
            if not isinstance(rows,list) or not 1<=len(rows)<=100:raise ValueError()
            if any(not isinstance(t,dict) or not isinstance(t.get("product"),str) for t in rows):
                raise ValueError()
            if any(t["product"] not in user["products"] for t in rows):
                raise HTTPException(403,"Продукт недоступен")
            results=[]
            for row in rows:
                try: results.append(await run_in_threadpool(radar.ingest,row))
                except (ValueError,KeyError,TypeError):
                    # Per-row result: successful rows stay committed and retry is idempotent.
                    results.append({"status":"invalid"})
            if all(r["status"]=="invalid" for r in results):raise ValueError()
            return {"results":results,"statuses":dict(Counter(r["status"] for r in results))}
        except HTTPException:raise
        except (ValueError,KeyError,TypeError):raise HTTPException(422,"Проверьте поля, дату, канал и размер пакета") from None

    @app.post("/api/decision/{product}")
    async def decision(product:str,request:Request):
        principal(request,{"editor","admin"},product)
        try:
            b=await request.json()
            radar.decide(product,b["topic"],b["status"])
        except KeyError:raise HTTPException(404,"Карточка не найдена") from None
        except (ValueError,TypeError):raise HTTPException(422,"Недопустимое решение") from None
        return {"ok":True}

    return app
