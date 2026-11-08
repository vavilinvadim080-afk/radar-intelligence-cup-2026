"""Защита приложения от злоупотреблений (app/security.py, app/state.py, app/autoreply.py, app/server.py): частота запросов,
бюджет LLM, карточка «подозрение на накрутку», глобальный лимит автоответа, журнал аудита. Сначала красные тесты, потом код."""
import os
from datetime import datetime, timedelta
from types import SimpleNamespace

os.environ["RADAR_MODE"] = "template"

import pytest
from fastapi.testclient import TestClient

from app import agent as agent_mod
from app import security, server, state
from app.audit import Audit
from app.autoreply import AutoReply
from app.replay import card_plan
from radar.models import Burst


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- token bucket
def test_token_bucket_allows_burst_then_refills():
    clk = Clock()
    b = security.TokenBucketLimiter(rate_per_s=1.0, burst=3, clock=clk)
    assert [b.allow("a")[0] for _ in range(4)] == [True, True, True, False]
    ok, retry = b.allow("a")
    assert not ok and 0 < retry <= 1.0
    clk.t += 1.0
    assert b.allow("a")[0] and not b.allow("a")[0]
    assert b.allow("other")[0]                                # другой клиент: свой запас


def test_token_bucket_retry_after_is_time_until_next_token():
    clk = Clock()
    b = security.TokenBucketLimiter(rate_per_s=0.5, burst=1, clock=clk)
    b.allow("a")
    ok, retry = b.allow("a")
    assert not ok and abs(retry - 2.0) < 1e-6


def test_token_bucket_memory_is_bounded():
    clk = Clock()
    b = security.TokenBucketLimiter(rate_per_s=1.0, burst=2, clock=clk, max_keys=100)
    for i in range(1000):
        b.allow(f"c{i}")
        clk.t += 0.01
    assert len(b._state) <= 150


# ---------------------------------------------------------------- бюджет LLM
def test_llm_budget_limits_per_product_and_total_per_hour():
    clk = Clock()
    bud = security.LlmBudget(per_product=2, total=3, clock=clk)
    assert bud.try_acquire("azs")[0] and bud.try_acquire("azs")[0]
    ok, why = bud.try_acquire("azs")
    assert not ok and "продукт" in why
    assert bud.try_acquire("bonus")[0]
    ok, why = bud.try_acquire("portal")
    assert not ok and "общий" in why
    clk.t += 3601
    assert bud.try_acquire("azs")[0]                            # час прошёл, бюджет вернулся


# ---------------------------------------------------------------- HTTP: частота и журнал
@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


@pytest.fixture(autouse=True)
def fresh(client):
    server.player["playing"] = False
    client.post("/api/jump", json={"to": "2026-03-09T18:20", "reset": True})
    for lim in (server.button_limiter, server.read_limiter):
        lim.reset()


def antispam_rows(client):
    return [r for r in client.get("/api/audit?limit=500").json()["rows"] if r["kind"] == "antispam"]


def test_buttons_get_429_with_retry_after_and_audit_row(client, monkeypatch):
    monkeypatch.setattr(server, "button_limiter", security.TokenBucketLimiter(rate_per_s=0.01, burst=2))
    codes = [client.post("/api/cards/B0003/decision", json={"action": "known"}).status_code for _ in range(4)]
    assert 429 in codes and codes[0] != 429
    r = client.post("/api/cards/B0003/decision", json={"action": "known"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    rows = antispam_rows(client)
    assert rows and rows[-1]["payload"]["rule"] == "rate_buttons" and rows[-1]["actor"] == "code"


def test_audit_rows_for_rate_limit_are_throttled(client, monkeypatch):
    monkeypatch.setattr(server, "button_limiter", security.TokenBucketLimiter(rate_per_s=0.001, burst=1))
    for _ in range(60):
        client.post("/api/cards/B0003/decision", json={"action": "known"})
    assert len(antispam_rows(client)) <= 2                       # флуд запросов не должен флудить журнал


def test_reads_are_rate_limited_per_client(client, monkeypatch):
    monkeypatch.setattr(server, "read_limiter", security.TokenBucketLimiter(rate_per_s=0.01, burst=5))
    codes = [client.get("/api/cards").status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]
    assert client.get("/api/status").status_code == 200           # статус страница спрашивает всегда: не режем


def test_status_endpoint_and_stream_not_throttled_by_read_bucket(client, monkeypatch):
    monkeypatch.setattr(server, "read_limiter", security.TokenBucketLimiter(rate_per_s=0.001, burst=1))
    assert all(client.get("/api/status").status_code == 200 for _ in range(10))


def test_post_sliding_window_429_is_also_audited(client, monkeypatch):
    monkeypatch.setattr(server, "rate_limiter", security.SlidingWindowLimiter(2, 60))
    codes = [client.post("/api/control", json={"speed": 10}).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[2] == 429
    assert any(r["payload"]["rule"] == "rate_post" for r in antispam_rows(client))


# ---------------------------------------------------------------- LLM вызывается на всплеск, не на обращение
class SpyLLM:
    def available(self):
        return True


def make_live_radar(monkeypatch, per_product=100, total=1000, rep=None):
    calls = []

    def fake_write_card(ci, world, audit, cluster_source=None, llm=None, max_attempts=2, max_steps=6, force_template=False):
        calls.append({"burst": ci.burst_id, "product": ci.product, "force_template": force_template})
        return {"facts": {"z": ci.z, "started_at": ci.started_at.isoformat()}, "card": {"title": "t", "claims": [], "severity": "medium"},
                "mode": "шаблон, без LLM" if force_template else "LLM", "fallback_reason": "режим без LLM включён вручную" if force_template else None,
                "tool_calls": [], "validator": [], "steps": 0, "attempts": 0, "latency_s": 0.0, "tools_used": []}

    monkeypatch.setattr(agent_mod, "write_card", fake_write_card)
    rep = rep or server.load_or_build()
    r = state.Radar(rep, mode="live", llm=SpyLLM(), sync_agent=True)
    r.llm_budget = security.LlmBudget(per_product=per_product, total=total)
    return r, calls


def test_agent_is_called_per_burst_not_per_ticket(monkeypatch):
    r, calls = make_live_radar(monkeypatch)
    r.reset(state.SCENES[1])
    calls.clear()
    r.advance(r.rep.end)
    n_tickets = len(r.rep.tickets)
    n_alert_cards = len([c for c in r.cards.values() if c["kind"] == "alert"])
    assert n_tickets > 5000 and n_alert_cards >= 3
    assert len(calls) <= len([p for p in r.plan]) and len(calls) < n_tickets / 100      # вызовов столько же, сколько карточек, а не обращений
    assert len({c["burst"] for c in calls}) == len(calls)                               # на каждый всплеск не больше одного вызова


def test_llm_budget_exhausted_falls_back_to_template_card_and_audits(monkeypatch):
    r, calls = make_live_radar(monkeypatch, per_product=0, total=0)
    r.reset(state.SCENES[1])
    calls.clear()
    r.advance(r.rep.end)
    live_calls = [c for c in calls if not c["force_template"]]
    assert calls and not live_calls                                                      # модель не вызывалась ни разу
    alert_cards = [c for c in r.cards.values() if c["kind"] == "alert"]
    assert alert_cards and all("бюджет" in c["agent"]["fallback_reason"] for c in alert_cards if c["agent"]["mode"] != "LLM")
    rows = [x for x in r.audit.rows(None, 500) if x["kind"] == "antispam"]
    assert rows and rows[-1]["payload"]["rule"] == "llm_budget"


def test_llm_budget_counts_only_real_model_calls(monkeypatch):
    r, calls = make_live_radar(monkeypatch, per_product=100, total=1)
    r.reset(state.SCENES[1])
    calls.clear()
    r.advance(r.rep.end)
    assert len([c for c in calls if not c["force_template"]]) == 1                      # общий бюджет 1 вызов в час


# ---------------------------------------------------------------- подозрение на накрутку в карточке
def burst(i, product, minutes, suspicion=None, tier="page"):
    t = datetime(2026, 3, 9, 12, 0) + timedelta(minutes=minutes)
    b = Burst(f"B{i:04d}", product, "c", "ewma", 30, t, t, t, "open", 7.0, 12, 1.0, 12, 30)
    b.tier, b.suspicion = tier, suspicion
    return b


def test_card_plan_does_not_merge_real_alert_into_quiet_suspicion_card():
    susp = {"reason": "подозрение на накрутку", "kept_page": False}
    ev = [{"ts": b.detected_at, "kind": "opened", "burst": b, "ci": object()}
          for b in (burst(1, "azs", 0, susp, "quiet"), burst(2, "azs", 30), burst(3, "azs", 40, susp, "quiet"))]
    plan = card_plan(SimpleNamespace(events=ev, injections=[]))
    ids = {p["card_id"]: p["burst_ids"] for p in plan}
    assert ids == {"B0001": ["B0001", "B0003"], "B0002": ["B0002"]}


def test_suspicion_card_is_quiet_has_marker_and_blocks_autoreply(monkeypatch):
    rep = server.load_or_build()
    target = next(e["burst"] for e in rep.events if e["kind"] == "opened" and e["burst"].alert)
    saved = (target.tier, target.suspicion)
    target.tier, target.suspicion = "quiet", {"reason": "подозрение на накрутку: 30 из 32", "kept_page": False, "dup_share": 0.94}
    try:
        r, calls = make_live_radar(monkeypatch, rep=rep)
        r.reset(rep.end)
        card = r.cards[target.id]
        assert card["tier"] == "quiet" and card["suspicion"]["dup_share"] == 0.94
        assert card["agent"]["mode"] != "LLM" and "накрутк" in card["agent"]["fallback_reason"]
        with pytest.raises(ValueError, match="накрутк"):
            r.approve_autoreply(target.id)
        rows = [x for x in r.audit.rows(None, 500) if x["kind"] == "antispam"]
        assert any(x["payload"]["rule"] == "autoreply_blocked_suspicion" for x in rows)
    finally:
        target.tier, target.suspicion = saved


# ---------------------------------------------------------------- глобальный дневной лимит автоответа
def autoreply_with_two_incidents(global_limit):
    audit = Audit()
    ar = AutoReply(audit, daily_limit=100, min_sim=0.0, global_daily_limit=global_limit)
    t = datetime(2026, 3, 9, 12, 0)
    for inc in ("B1", "B2"):
        ar.approve(inc, "azs", "АЗС", t, t)
    return ar, audit, t


def test_autoreply_global_daily_limit_across_incidents():
    ar, audit, t = autoreply_with_two_incidents(3)
    out = [ar.handle_ticket("B1" if i % 2 else "B2", f"T{i}", "приложение азс не открывается вход", t + timedelta(minutes=i), sim=0.9) for i in range(6)]
    assert [o["action"] for o in out] == ["autoreply"] * 3 + ["operator"] * 3
    assert out[3]["reason"] == "глобальный дневной лимит"
    rows = [r for r in audit.rows(None, 200) if r["kind"] == "antispam"]
    assert len(rows) == 1 and rows[0]["payload"]["rule"] == "autoreply_global_limit"           # одна запись на сутки, не по записи на обращение


def test_autoreply_global_limit_resets_next_day():
    ar, _, t = autoreply_with_two_incidents(1)
    assert ar.handle_ticket("B1", "T1", "приложение азс не открывается вход", t, sim=0.9)["action"] == "autoreply"
    assert ar.handle_ticket("B1", "T2", "приложение азс не открывается вход", t + timedelta(minutes=1), sim=0.9)["action"] == "operator"
    assert ar.handle_ticket("B1", "T3", "приложение азс не открывается вход", t + timedelta(days=1), sim=0.9)["action"] == "autoreply"


def test_autoreply_global_limit_default_and_env(monkeypatch):
    assert AutoReply(Audit()).global_daily_limit >= AutoReply.DEFAULT_DAILY_LIMIT
    monkeypatch.setenv("RADAR_AUTOREPLY_GLOBAL_DAILY_LIMIT", "7")
    assert AutoReply(Audit()).global_daily_limit == 7
