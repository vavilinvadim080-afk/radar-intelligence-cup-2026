"""Автоответ в приложении: эндпоинты, проводка обращений из записи, связь с проверкой фикса (набор seed 777, режим шаблона)."""
import json
import os

os.environ["RADAR_MODE"] = "template"

import pytest
from fastapi.testclient import TestClient

from app import server


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


@pytest.fixture(autouse=True)
def fresh(client):
    server.player["playing"] = False
    client.post("/api/jump", json={"to": "2026-03-09T18:20", "reset": True})   # карточка B0003 (bonus) уже есть


def approve(client, cid="B0003"):
    return client.post(f"/api/incidents/{cid}/autoreply/approve")


def test_nothing_goes_out_before_click(client):
    client.post("/api/jump", json={"to": "2026-03-09T23:00"})
    s = client.get("/api/autoreply").json()
    assert s["stats"]["autoreplied"] == 0 and s["queue"] == [] and s["incidents"]["B0003"]["state"] == "не включён"


def test_approve_then_repeated_tickets_get_template_reply(client):
    r = approve(client)
    assert r.status_code == 200 and r.json()["state"] == "включён"
    client.post("/api/jump", json={"to": "2026-03-10T06:00"})
    s = client.get("/api/autoreply").json()
    assert s["stats"]["autoreplied"] > 0 and s["incidents"]["B0003"]["replied"] == s["stats"]["autoreplied"]
    first = s["queue"][0]
    assert first["incident"] == "B0003" and "B0003" in first["text"] and first["status"].startswith("в очереди")
    assert set(first) == {"id", "incident", "ticket_id", "text", "status", "ts"}          # контактов и текста обращения нет
    # метрика на странице: сколько закрыто автоответом
    assert client.get("/api/cards").json()["cards"][0]["autoreply"]["replied"] == s["stats"]["autoreplied"]


def test_reply_text_is_never_a_user_ticket_text(client):
    approve(client)
    client.post("/api/jump", json={"to": "2026-03-10T06:00"})
    s = client.get("/api/autoreply").json()
    rep = server.get_radar().rep
    texts = [row[6] for row in rep.tickets] + [row[7] for row in rep.tickets]
    sample = {t.strip() for t in texts if len(t.strip()) > 25}
    rows = [r for r in client.get("/api/audit?limit=500").json()["rows"] if r["kind"].startswith("autoreply_")]
    assert rows                                                    # (в журнале есть и agent_input с примерами карточки: он не про автоответ)
    blob = json.dumps([s["queue"], rows], ensure_ascii=False)
    assert not any(t in blob for t in sample)


def test_disable_stops_counter(client):
    approve(client)
    client.post("/api/jump", json={"to": "2026-03-09T21:00"})
    n = client.get("/api/autoreply").json()["stats"]["autoreplied"]
    assert client.post("/api/incidents/B0003/autoreply/disable").json()["state"] == "выключен"
    client.post("/api/jump", json={"to": "2026-03-10T09:00"})
    assert client.get("/api/autoreply").json()["stats"]["autoreplied"] == n


def test_unknown_noise_and_watch_cards_rejected(client):
    assert approve(client, "B9999").status_code == 404
    assert client.post("/api/incidents/B0003/autoreply/disable").status_code == 409        # не включали
    client.post("/api/cards/B0003/decision", json={"action": "noise"})
    assert approve(client).status_code == 409                                              # «шум»: автоответ не нужен


def test_fix_confirmation_proposes_mailing_but_does_not_send(client):
    approve(client)
    client.post("/api/jump", json={"to": "2026-03-17T00:10"})
    s = client.get("/api/autoreply").json()
    assert s["incidents"]["B0003"]["proposal"]["state"] == "ждёт человека" and s["mailings"] == []
    r = client.post("/api/incidents/B0003/fixed/approve")
    assert r.status_code == 200 and r.json()["ticket_ids"] and r.json()["status"].startswith("в очереди")
    assert "contacts" not in r.json() and "email" not in json.dumps(r.json()).lower()
    assert client.post("/api/incidents/B0003/fixed/approve").status_code == 409            # второй раз нельзя
    assert len(client.get("/api/autoreply").json()["mailings"]) == 1


def test_human_can_press_fixed_without_verify(client):
    approve(client)
    r = client.post("/api/incidents/B0003/fixed/approve")
    assert r.status_code == 200 and r.json()["source"] == "человек"


def test_fixed_without_autoreply_is_409(client):
    assert client.post("/api/incidents/B0003/fixed/approve").status_code == 409


def test_actions_are_audited(client):
    approve(client)
    client.post("/api/jump", json={"to": "2026-03-09T21:00"})
    kinds = {r["kind"] for r in client.get("/api/audit?limit=500&card_id=B0003").json()["rows"]}
    assert {"autoreply_approved", "autoreply_sent"} <= kinds
    assert client.get("/api/audit/verify").json()["ok"]


def test_viewer_cannot_approve(client, monkeypatch):
    monkeypatch.setenv("RADAR_TOKEN_VIEW", "v-token")
    monkeypatch.setenv("RADAR_TOKEN_DUTY", "d-token")
    assert client.post("/api/incidents/B0003/autoreply/approve", headers={"Authorization": "Bearer v-token"}).status_code == 403
    assert client.post("/api/incidents/B0003/autoreply/approve", headers={"Authorization": "Bearer d-token"}).status_code == 200


def test_no_autoreply_after_incident_bursts_closed(client):
    """Вспышки B0003 и B0004 закрылись 2026-03-10 02:57: поток той же темы после этого уже не инцидент, ответов нет."""
    approve(client)
    client.post("/api/jump", json={"to": "2026-03-10T03:30"})
    n = client.get("/api/autoreply").json()["stats"]["autoreplied"]
    assert n > 0
    client.post("/api/jump", json={"to": "2026-03-12T12:00"})
    assert client.get("/api/autoreply").json()["stats"]["autoreplied"] == n
