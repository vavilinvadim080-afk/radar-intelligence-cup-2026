"""FastAPI-сервер демо. Запуск: `make demo` (или `uvicorn app.server:app --port 8777`).

Поток событий: GET /api/stream (SSE, text/event-stream). Данные страницы: GET-эндпоинты ниже.
Время воспроизведения: скорость в минутах потокового времени за одну секунду реального (по умолчанию 10).
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import agent as agent_mod
from . import security
from .replay import load_or_build
from .state import SCENES, Radar

ROOT = Path(__file__).resolve().parents[1]          # Intelligence-Cup-2026/
STATIC = Path(__file__).parent / "static"
TICK_S = 0.25                                        # шаг проигрывателя в реальных секундах

MAX_BODY = 16 * 1024            # байт; все тела запросов этого API короче сотни байт
MAX_STREAMS = 20                # одновременных SSE-подключений
# заголовок Host: защита от DNS rebinding (чужая страница, привязанная к 127.0.0.1). За шлюзом: RADAR_ALLOWED_HOSTS=radar.company.ru
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("RADAR_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if h.strip()]

# лимит частоты POST (скользящее окно в памяти, на клиента по IP): RADAR_RATE_POST_MAX запросов за RADAR_RATE_WINDOW_S секунд
rate_limiter = security.SlidingWindowLimiter(int(os.environ.get("RADAR_RATE_POST_MAX", "120")), float(os.environ.get("RADAR_RATE_WINDOW_S", "60")))

# ведра токенов (token bucket) на клиента по IP: кнопки решений строже, чтение мягче. Исчерпано: 429 и заголовок Retry-After.
# RADAR_RATE_BUTTON_PER_S / RADAR_RATE_BUTTON_BURST, RADAR_RATE_READ_PER_S / RADAR_RATE_READ_BURST. За обратным прокси адрес клиента один.
button_limiter = security.TokenBucketLimiter(float(os.environ.get("RADAR_RATE_BUTTON_PER_S", "1")), int(os.environ.get("RADAR_RATE_BUTTON_BURST", "20")))
read_limiter = security.TokenBucketLimiter(float(os.environ.get("RADAR_RATE_READ_PER_S", "30")), int(os.environ.get("RADAR_RATE_READ_BURST", "200")))
BUTTON_PATHS = re.compile(r"^/api/(cards/[^/]+/(decision|confirm-close)|outbox/[^/]+/approve|incidents/[^/]+/(autoreply/(approve|disable)|fixed/approve))$")
READ_EXEMPT = ("/api/status", "/api/stream")        # страница спрашивает статус всегда; поток ограничен MAX_STREAMS
AUDIT_EVERY_S = 60.0                                # одно и то же срабатывание лимита от одного клиента пишется в журнал не чаще раза в минуту
_audit_last: dict = {}

R: Radar | None = None
player = {"speed": 10.0, "playing": False}          # speed: минут потокового времени в секунду


def get_radar() -> Radar:
    global R
    if R is None:
        rep = load_or_build()
        R = Radar(rep, mode=os.environ.get("RADAR_MODE", "replay"))
        history_service = getattr(app.state, 'history_service', None)
        if history_service is not None:
            history_service.source('incidents', R)
    return R


def _loop():
    """Фоновый поток: двигает потоковое время. Остановлен, пока playing = False."""
    last = time.time()
    while True:
        time.sleep(TICK_S)
        dt, last = time.time() - last, time.time()
        if not player["playing"] or R is None:
            continue
        with R.lock:
            end = R.rep.end
            nxt = min(R.now + timedelta(minutes=player["speed"] * dt), end)
            R.advance(nxt)
            R.emit({"type": "tick", "now": R.now.isoformat(timespec="minutes")})
            if nxt >= end:
                player["playing"] = False


@asynccontextmanager
async def lifespan(app):
    if os.environ.get("RADAR_LAZY_INCIDENTS") != "1":
        get_radar()
    threading.Thread(target=_loop, daemon=True).start()
    yield


class BodyLimit:
    """ASGI-обёртка: тело запроса больше MAX_BODY байт получает 413 (по Content-Length и по фактически прочитанному)."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.inner(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length")
        too_big = JSONResponse({"detail": f"тело запроса больше {MAX_BODY} байт"}, status_code=413)
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY:
            return await too_big(scope, receive, send)
        seen = 0

        async def counted():
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > MAX_BODY:
                    raise _TooBig()
            return msg

        try:
            await self.inner(scope, counted, send)
        except _TooBig:
            await too_big(scope, receive, send)


class _TooBig(Exception):
    pass


class AccessGuard:
    """ASGI-обёртка: вход по токену, роли и лимит частоты POST. Токены читаются из окружения на каждый запрос.
    Вход выключен (демо на 127.0.0.1), пока не задан ни один из RADAR_API_TOKEN / RADAR_TOKEN_VIEW / RADAR_TOKEN_DUTY.
    /api/status открыт всегда (страница спрашивает по нему, нужен ли токен), страница `/` отдаётся без токена.
    Роль «наблюдатель» (viewer) читает, «дежурный» (duty) ещё и нажимает кнопки (любой POST)."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            return await self.inner(scope, receive, send)
        method, path = scope["method"], scope["path"]
        roles = security.token_roles()
        role = security.DUTY
        if roles and path != "/api/status":
            auth = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
            role = security.role_of(auth, roles)
            if role is None:
                return await JSONResponse({"detail": "нужен токен: заголовок Authorization: Bearer <токен>"}, status_code=401,
                                          headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
            if method not in ("GET", "HEAD", "OPTIONS") and role != security.DUTY:
                return await JSONResponse({"detail": "роль «наблюдатель»: только чтение, кнопки решений доступны дежурному"},
                                          status_code=403)(scope, receive, send)
        scope["radar_role"] = role
        client = (scope.get("client") or ("?", 0))[0]
        if method == "POST":
            if not rate_limiter.allow(client):
                return await self._deny(scope, receive, send, client, path, "rate_post", float(rate_limiter.window_s),
                                        "слишком частые запросы: превышен лимит частоты POST, повторите позже")
            if BUTTON_PATHS.match(path):
                ok, retry = button_limiter.allow(client)
                if not ok:
                    return await self._deny(scope, receive, send, client, path, "rate_buttons", retry,
                                            "слишком частые нажатия кнопок: лимит исчерпан, повторите позже")
        elif method in ("GET", "HEAD") and path not in READ_EXEMPT:
            ok, retry = read_limiter.allow(client)
            if not ok:
                return await self._deny(scope, receive, send, client, path, "rate_read", retry,
                                        "слишком частые запросы на чтение: лимит исчерпан, повторите позже")
        await self.inner(scope, receive, send)

    async def _deny(self, scope, receive, send, client: str, path: str, rule: str, retry: float, detail: str):
        """429 с Retry-After и запись в журнал аудита (не чаще раза в AUDIT_EVERY_S на пару клиент/правило: флуд запросов не должен флудить журнал)."""
        wait = max(1, int(-(-retry // 1)))
        now = time.monotonic()
        if len(_audit_last) > 1000:
            _audit_last.clear()
        key = (id(R.audit) if R is not None else 0, client, rule)     # журнал пересоздаётся при сбросе демо: у каждого свой счётчик
        if R is not None and now - _audit_last.get(key, -1e9) >= AUDIT_EVERY_S:
            _audit_last[key] = now
            R.audit.log("antispam", "code", {"rule": rule, "client": client, "path": path, "retry_after_s": wait}, None, R.now)
        return await JSONResponse({"detail": detail}, status_code=429, headers={"Retry-After": str(wait)})(scope, receive, send)


app = FastAPI(title="Радар: демо", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(BodyLimit)
app.add_middleware(AccessGuard)   # снаружи всех: сначала вход и лимит частоты, потом размер тела


# ---------------------------------------------------------------- страница и поток
@app.get("/")
def index():
    return FileResponse(STATIC / "dashboard.html")


@app.get("/api/stream")
async def stream():
    r = get_radar()
    if len(r.listeners) >= MAX_STREAMS:
        raise HTTPException(503, "слишком много подключений к потоку")
    q: queue.Queue = queue.Queue(maxsize=200)
    r.listeners.append(q.put_nowait)

    async def gen():
        try:
            yield "data: " + json.dumps({"type": "hello", "now": r.now.isoformat(timespec="minutes")}) + "\n\n"
            while True:
                try:
                    msg = q.get_nowait()
                    yield "data: " + json.dumps(msg, ensure_ascii=False) + "\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.2)
        finally:
            r.listeners.remove(q.put_nowait) if q.put_nowait in r.listeners else None

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- состояние
@app.get("/api/status")
def status():
    r = get_radar()
    ollama = agent_mod.OllamaClient().available()
    return {"now": r.now.isoformat(timespec="minutes"), "speed": player["speed"], "playing": player["playing"], "mode": r.mode,
            "ollama": ollama, "recorded": bool(r.recorded.get("results")), "seed": r.rep.seed, "embedder": r.rep.embedder,
            "end": r.rep.end.isoformat(timespec="minutes"), "start": r.rep.start.isoformat(timespec="minutes"),
            "label": "синтетические данные, демо-seed %d" % r.rep.seed, "auth_required": bool(security.token_roles()),
            "scenes": {k: v.isoformat(timespec="minutes") for k, v in SCENES.items()}}


@app.get("/api/series")
def series(hours: int = Query(72, ge=1, le=720)):
    return get_radar().series(hours)


@app.get("/api/cards")
def cards():
    r = get_radar()
    with r.lock:
        out = sorted(r.cards.values(), key=lambda c: c["created_sim"], reverse=True)
        out = [{**c, "autoreply": r.autoreply.view(c["id"], r.now)} for c in out]
        return {"cards": json.loads(json.dumps(out, default=str)), "thresholds": r.overrides, "threshold_log": r.threshold_log,
                "suppressed": r.suppressed, "events": r.events_log[-20:]}


@app.get("/api/tracker")
def tracker():
    r = get_radar()
    return {"tasks": r.world.tasks()[:40], "drafts": r.world.drafts(), "outbox": r.world.outbox()}


@app.get("/api/audit")
def audit(limit: int = Query(80, ge=1, le=500), card_id: str | None = Query(None, max_length=64)):
    return {"rows": get_radar().audit.rows(card_id, limit)}


@app.get("/api/audit/verify")
def audit_verify():
    """Проверка цепочки хэшей журнала: ok, число строк, первая порванная запись, head (последний хэш, его стоит хранить снаружи)."""
    return get_radar().audit.verify_chain()


@app.get("/api/whoami")
def whoami(request: Request):
    """Роль по токену: viewer (наблюдатель) или duty (дежурный). Вход выключен: duty."""
    return {"role": request.scope.get("radar_role", security.DUTY), "auth_required": bool(security.token_roles())}


@app.get("/api/tickets")
def tickets(limit: int = Query(8, ge=1, le=50)):
    return get_radar().tickets_feed(limit)


@app.get("/api/fix")
def fix():
    return {"items": get_radar().fix_status()}


@app.get("/api/metrics")
def metrics():
    """Метрики качества на слепых seeds (не демо-seed). Читаются из shared/, здесь не пересчитываются."""
    res = {}
    for name, f in (("detector", "metrics.json"), ("scale", "metrics-scale.json"), ("agent", "metrics-agent.json")):
        p = ROOT / "shared" / f
        res[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    d, a = res["detector"], res["agent"]
    summary = {"note": "слепые seeds, синтетика, не демо-seed"}
    if d:
        e = d["detector"]["ewma"]["all"]
        summary["detector"] = {k: e[k] for k in ("precision", "recall", "ttc_median_min", "false_alarms_per_week")}
        summary["detector"]["seeds"] = d["protocol"]["test_seeds"]
        summary["masking_completeness"] = d["masking"]["completeness"]
    if a:
        summary["agent"] = a.get("summary", a)
    # v12: основные числа панели «Качество на слепых seeds»: финальный слепой прогон детектора v3 (seeds 401-410, режим «без проверки»,
    # как в коде по умолчанию) и агент v2 + строгая схема. Старые ключи выше оставлены для совместимости, панель их не показывает.
    v3p, ag2p = ROOT / "shared" / "metrics-v3.json", ROOT / "shared" / "metrics-agent-v2-schema.json"
    if v3p.exists():
        v3 = json.loads(v3p.read_text(encoding="utf-8")); t = v3["final"][0]["table"]["none"]
        summary["detector_v3"] = {"strict_precision": t["strict"]["precision"], "strict_recall": t["strict"]["recall"], "work_recall": t["work"]["recall"],
                                  "on_target_share": t["target"]["page_share"], "false_pages_per_week": t["false_pages_per_week"],
                                  "ttc_median_min": t["strict"]["ttc_median_min"], "seeds": v3["meta"]["final_seeds"], "mode": "без проверки (judge_enabled=False)"}
    if ag2p.exists():
        a2 = json.loads(ag2p.read_text(encoding="utf-8"))
        summary["agent_v2"] = {k: a2[k] for k in ("variant", "n_cards", "model", "first_try_valid_share", "claims_valid_share", "template_fallback_share", "steps_mean", "latency_s_median", "n_injections", "injection_resisted_share")}
    return {"summary": summary, **res}


# ---------------------------------------------------------------- управление
class Control(BaseModel):
    speed: float | None = None
    playing: bool | None = None


class Jump(BaseModel):
    scene: int | None = None
    to: str | None = Field(None, max_length=32)
    reset: bool = False

    @model_validator(mode="after")
    def _check(self):
        if self.scene is None and self.to is None:
            raise ValueError("нужна сцена или время")
        if self.scene is not None and self.scene not in SCENES:
            raise ValueError("сцены нет, есть " + ", ".join(map(str, SCENES)))
        if self.scene is None:
            try:
                datetime.fromisoformat(self.to)
            except ValueError:
                raise ValueError("время в формате ГГГГ-ММ-ДДTЧЧ:ММ")
        return self


class Mode(BaseModel):
    mode: str = Field(max_length=16)


class Decision(BaseModel):
    action: str = Field(max_length=32)
    task_key: str | None = Field(None, max_length=64)


@app.post("/api/control")
def control(c: Control):
    if c.speed is not None:
        player["speed"] = max(0.5, min(c.speed, 600.0))
    if c.playing is not None:
        player["playing"] = c.playing
    get_radar().emit({"type": "control", **player})
    return player


@app.post("/api/jump")
def jump(j: Jump):
    r = get_radar()
    to = SCENES[j.scene] if j.scene is not None else datetime.fromisoformat(j.to)
    if j.reset:
        r.reset(to)
    else:
        r.jump(to)
    return {"now": r.now.isoformat(timespec="minutes")}


@app.post("/api/mode")
def set_mode(m: Mode):
    if m.mode not in ("replay", "live", "template"):
        raise HTTPException(400, "режим: replay, live или template")
    r = get_radar()
    r.mode = m.mode
    r.emit({"type": "refresh"})
    return {"mode": r.mode}


@app.post("/api/cards/{card_id}/decision")
def decision(card_id: str, d: Decision):
    r = get_radar()
    if card_id not in r.cards:
        raise HTTPException(404, "нет такой карточки")
    try:
        return r.decide(card_id, d.action, d.task_key)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/cards/{card_id}/confirm-close")
def confirm_close(card_id: str):
    r = get_radar()
    if card_id not in r.cards:
        raise HTTPException(404, "нет такой карточки")
    try:
        return r.confirm_close(card_id)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/outbox/{oid}/approve")
def approve(oid: str):
    try:
        return get_radar().approve_outbox(oid)
    except StopIteration:
        raise HTTPException(404, "нет такого сообщения")


# ---------------------------------------------------------------- автоответ на повторные обращения
@app.get("/api/autoreply")
def autoreply_state():
    """Метрика «закрыто автоответом», состояние по инцидентам, мок-очереди ответов и рассылок. Контактов и текстов обращений здесь нет."""
    return get_radar().autoreply_view()


def _autoreply_call(card_id: str, fn):
    r = get_radar()
    if card_id not in r.cards:
        raise HTTPException(404, "нет такого инцидента")
    try:
        return fn()
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/incidents/{card_id}/autoreply/approve")
def autoreply_approve(card_id: str):
    return _autoreply_call(card_id, lambda: get_radar().approve_autoreply(card_id))


@app.post("/api/incidents/{card_id}/autoreply/disable")
def autoreply_disable(card_id: str):
    return _autoreply_call(card_id, lambda: get_radar().disable_autoreply(card_id))


@app.post("/api/incidents/{card_id}/fixed/approve")
def autoreply_fixed(card_id: str):
    return _autoreply_call(card_id, lambda: get_radar().approve_fixed_mailing(card_id))


# Official-case mode; the legacy incident endpoints above remain available.
from .needs import install_needs
from .history import install_history
_needs_access = install_needs(app)
install_history(app, lambda: R, _needs_access['history_source'], _needs_access['principal'])

@app.get("/needs.js")
def needs_script():
    return FileResponse(STATIC / "needs.js",media_type="application/javascript")

@app.get('/history.js')
def history_script():
    return FileResponse(STATIC/'history.js',media_type='application/javascript')

@app.get('/history.css')
def history_style():
    return FileResponse(STATIC/'history.css',media_type='text/css')
