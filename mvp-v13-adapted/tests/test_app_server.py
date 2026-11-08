"""Тесты эндпоинтов демо-сервера. Режим шаблона (без LLM), набор seed 777 берётся из кэша app/cache или строится (около 15 с)."""
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
    client.post("/api/jump", json={"to": "2026-03-09T18:20", "reset": True})   # карточка B0003 уже есть


def test_page_and_status(client):
    assert "Радар" in client.get("/").text
    s = client.get("/api/status").json()
    assert s["seed"] == 777 and "синтетические" in s["label"]


def test_cards_and_series(client):
    cards = client.get("/api/cards").json()["cards"]
    assert [c["id"] for c in cards] == ["B0003"]
    assert cards[0]["tier"] == "page"                      # v3: без проверки LLM тревога будит
    assert all(c["ids"] for c in cards[0]["agent"]["card"]["claims"] if "ids" in c)
    s = client.get("/api/series").json()
    assert s["products"]["bonus"] and any(r["version"] == "2.8.0" for r in s["releases"])


def test_our_bug_creates_task_in_tracker(client):
    r = client.post("/api/cards/B0003/decision", json={"action": "our_bug"})
    assert r.status_code == 200 and r.json()["task"]["key"].startswith("RAD-2")
    tasks = client.get("/api/tracker").json()["tasks"]
    assert any(t["source"] == "radar" and t["burst_id"] == "B0003" for t in tasks)
    assert client.post("/api/cards/B0003/decision", json={"action": "noise"}).status_code == 409   # решение одно


def test_noise_raises_threshold_and_is_logged(client):
    client.post("/api/jump", json={"to": "2026-03-11T22:00"})
    assert "B0005" in [c["id"] for c in client.get("/api/cards").json()["cards"]]
    r = client.post("/api/cards/B0005/decision", json={"action": "noise"}).json()
    assert r["threshold"]["to"]["z_threshold"] > r["threshold"]["from"]["z_threshold"]
    j = client.get("/api/cards").json()
    assert j["thresholds"]["azs"]["z_threshold"] == r["threshold"]["to"]["z_threshold"] and j["threshold_log"]
    assert any(a["kind"] == "human" and a["payload"]["decision"] == "noise" for a in client.get("/api/audit").json()["rows"])


def test_known_requires_task(client):
    assert client.post("/api/cards/B0003/decision", json={"action": "known", "task_key": "RAD-NOPE"}).status_code == 409   # такой задачи нет
    r = client.post("/api/cards/B0003/decision", json={"action": "known", "task_key": "RAD-001"})
    assert r.status_code == 200 and r.json()["task"]["burst_id"] == "B0003"


def test_close_only_by_human_after_confirmed_fix(client):
    assert client.post("/api/cards/B0003/confirm-close").status_code == 409           # предложения ещё нет
    client.post("/api/cards/B0003/decision", json={"action": "our_bug"})
    client.post("/api/jump", json={"to": "2026-03-17T01:00"})
    c = [x for x in client.get("/api/cards").json()["cards"] if x["id"] == "B0003"][0]
    assert c["close_proposal"]["state"] == "ждёт человека"
    key = c["close_proposal"]["task_key"]
    assert next(t for t in client.get("/api/tracker").json()["tasks"] if t["key"] == key)["status"] == "open"   # сам агент не закрыл
    client.post("/api/cards/B0003/confirm-close")
    assert next(t for t in client.get("/api/tracker").json()["tasks"] if t["key"] == key)["status"] == "done"


def test_injection_card_is_flagged_and_does_nothing(client):
    client.post("/api/jump", json={"to": "2026-03-14T07:00"})
    w = [c for c in client.get("/api/cards").json()["cards"] if c["kind"] == "watch"][0]
    assert w["agent"]["card"]["injection_suspected"] is True and w["decision"] is None
    assert not [t for t in client.get("/api/tracker").json()["tasks"] if t["source"] == "radar"]


def test_metrics_endpoint_and_bad_mode(client):
    m = client.get("/api/metrics").json()
    assert "summary" in m and m["summary"]["note"]
    assert client.post("/api/mode", json={"mode": "что-то"}).status_code == 400


def test_control_speed(client):
    assert client.post("/api/control", json={"speed": 5}).json()["speed"] == 5
