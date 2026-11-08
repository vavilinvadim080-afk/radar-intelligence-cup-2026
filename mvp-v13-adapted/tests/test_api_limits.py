"""Границы входов API демо-сервера: размер тела, диапазоны параметров, число потоков, заголовок Host."""
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
    client.post("/api/jump", json={"to": "2026-03-09T18:20", "reset": True})


def test_oversized_body_is_rejected_with_413(client):
    big = {"action": "our_bug", "task_key": "X" * 200_000}
    r = client.post("/api/cards/B0003/decision", json=big)
    assert r.status_code == 413


@pytest.mark.parametrize("path", ["/api/audit?limit=100000000", "/api/audit?limit=-1", "/api/tickets?limit=100000", "/api/series?hours=100000"])
def test_query_parameters_are_bounded(client, path):
    assert client.get(path).status_code == 422


def test_audit_default_and_max_limit_still_work(client):
    assert client.get("/api/audit").status_code == 200
    assert client.get("/api/audit?limit=500").status_code == 200


def test_bad_jump_target_is_422_not_500(client):
    assert client.post("/api/jump", json={"to": "не дата"}).status_code == 422
    assert client.post("/api/jump", json={"scene": 99}).status_code == 422
    assert client.post("/api/jump", json={}).status_code == 422


def test_long_task_key_and_action_are_rejected(client):
    assert client.post("/api/cards/B0003/decision", json={"action": "our_bug", "task_key": "K" * 500}).status_code == 422
    assert client.post("/api/cards/B0003/decision", json={"action": "a" * 500}).status_code == 422


def test_json_with_text_plain_content_type_is_not_accepted(client):
    """Закрытие CSRF с чужой страницы: простой POST без preflight не должен проходить как JSON."""
    r = client.post("/api/control", content='{"playing": true}', headers={"content-type": "text/plain"})
    assert r.status_code == 422 and server.player["playing"] is False


def test_foreign_host_header_is_rejected(client):
    r = client.get("/api/status", headers={"host": "evil.example.com"})
    assert r.status_code == 400


def test_stream_listeners_are_capped(client):
    import asyncio

    from fastapi import HTTPException
    r = server.get_radar()
    saved = list(r.listeners)
    try:
        r.listeners[:] = [lambda m: None] * server.MAX_STREAMS
        with pytest.raises(HTTPException) as e:
            asyncio.run(server.stream())
        assert e.value.status_code == 503
        r.listeners[:] = saved
        resp = asyncio.run(server.stream())            # ниже порога поток открывается (без чтения тела)
        assert resp.media_type == "text/event-stream"
    finally:
        r.listeners[:] = saved


def test_dashboard_escapes_model_controlled_fields():
    """Имя инструмента приходит от модели (в том числе под влиянием инъекции) и попадает в innerHTML: должно идти через esc()."""
    html = (server.STATIC / "dashboard.html").read_text(encoding="utf-8")
    assert "${p.name}" not in html, "имя инструмента от модели вставлено в innerHTML без esc()"
    for raw in ("${p.decision} ${p.task||p.id||\"\"} (${p.who})", "<br><span class=\"small\">${t.id} ·", "${t.channel}"):
        assert raw not in html, raw


def test_dashboard_audit_human_row_handles_task_object():
    """В журнале строка решения «наш баг» содержит task как объект; без разбора выводилось [object Object]."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "static" / "dashboard.html").read_text(encoding="utf-8")
    assert 'typeof p.task==="object"?p.task.key' in src
