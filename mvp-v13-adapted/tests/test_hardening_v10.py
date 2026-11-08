"""Блок B (2026-10-27): закрытые дыры из THREAT-MODEL. Сначала красные тесты на отказ, потом код.
1 безопасный кэш прогона; 2 HMAC на demo/replay.json; 3 цепочка хэшей в журнале; 4 токены и роли; 5 лимит частоты POST."""
import json
import os
import pickle
import sqlite3
from datetime import datetime

os.environ["RADAR_MODE"] = "template"

import pytest
from fastapi.testclient import TestClient

from app import replay as replay_mod
from app import security
from app import server
from app.audit import Audit
from radar.models import Burst, SuspectedCardInput, SuspectedRelease


# ---------------------------------------------------------------- 1. кэш прогона: JSON + SHA-256, pickle не читаем
def tiny_replay():
    sr = SuspectedRelease("2.8.0", datetime(2026, 3, 9, 14, 45), 2.8, True, 0.8, 0.1, "high")
    b = Burst("B0001", "bonus", "C1", "ewma", 30, datetime(2026, 3, 9, 17, 32, 11), datetime(2026, 3, 9, 18, 2), datetime(2026, 3, 9, 18, 5),
              "open", 5.1, 13, 2.0, 13, 40, platforms={"android": 11}, versions={"2.8.0": 8}, examples=[{"id": "T1", "text": "t"}], suspected_release=sr)
    ci = SuspectedCardInput("B0001", "bonus", "Бонус", datetime(2026, 3, 9, 18, 2), datetime(2026, 3, 9, 17, 32), 13, 30, 5.1,
                            {"android": 11}, {"2.8.0": 8}, sr, 21200.0, [{"id": "T1", "text": "t"}], ["T1"])
    return replay_mod.Replay(
        777, datetime(2026, 3, 9), datetime(2026, 3, 10), [("T1", datetime(2026, 3, 9, 1, 2), "bonus", "ios", "2.8.0", "app_chat", "o", "m", False)],
        {"bonus": [1, 2]}, {"B0001": [0, 1]}, [{"ts": datetime(2026, 3, 9, 18, 2), "kind": "opened", "burst": b, "ci": ci}],
        [{"product": "bonus", "version": "2.8.0", "ts": datetime(2026, 3, 9, 14, 45), "kind": "regular", "changelog": "x"}], {"events": []}, [], "tfidf")


def test_cache_roundtrip_json_keeps_types(tmp_path):
    p = tmp_path / "r.json"
    rep = tiny_replay()
    replay_mod.write_cache(rep, p)
    back = replay_mod.read_cache(p)
    assert back == rep and isinstance(back.tickets[0], tuple) and isinstance(back.events[0]["burst"], Burst)
    assert json.loads(p.read_text(encoding="utf-8"))["sha256"]      # это JSON, а не pickle


def test_cache_with_changed_byte_is_rejected(tmp_path):
    p = tmp_path / "r.json"
    replay_mod.write_cache(tiny_replay(), p)
    p.write_text(p.read_text(encoding="utf-8").replace('"T1"', '"T9"', 1), encoding="utf-8")
    with pytest.raises(replay_mod.CacheIntegrityError, match="SHA-256"):
        replay_mod.read_cache(p)


def test_cache_unknown_class_is_rejected_even_with_valid_hash(tmp_path):
    p = tmp_path / "r.json"
    replay_mod.write_cache(tiny_replay(), p)
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["data"]["fields"]["tickets"] = {"__dc__": "os.system", "x": 1}
    doc["sha256"] = replay_mod.digest(doc["data"])
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(replay_mod.CacheIntegrityError):
        replay_mod.read_cache(p)


def test_pickle_cache_is_never_read(tmp_path, monkeypatch):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    (tmp_path / "replay-1-mid-800.pkl").write_bytes(pickle.dumps(Evil()))
    (tmp_path / "replay-1-mid-800.json").write_bytes(pickle.dumps(Evil()))   # и под видом json
    monkeypatch.setattr(replay_mod, "CACHE", tmp_path)
    built = []
    monkeypatch.setattr(replay_mod, "build", lambda *a, **k: built.append(1) or tiny_replay())
    rep = replay_mod.load_or_build(1, "mid", 800)
    assert not marker.exists() and built == [1] and rep.seed == 777
    assert replay_mod.read_cache(tmp_path / "replay-1-mid-800.json") == tiny_replay()    # пересобран в безопасном формате


# ---------------------------------------------------------------- 2. HMAC-подпись demo/replay.json
def test_replay_signature_accepts_untouched_file(tmp_path):
    f = tmp_path / "replay.json"
    f.write_text('{"results": {}}', encoding="utf-8")
    security.sign_file(f, b"k1")
    assert security.verify_file(f, b"k1") is None


def test_replay_with_changed_content_is_refused_with_clear_message(tmp_path):
    f = tmp_path / "replay.json"
    f.write_text('{"results": {"B1": {"x": 1}}}', encoding="utf-8")
    security.sign_file(f, b"k1")
    f.write_text('{"results": {"B1": {"x": 2}}}', encoding="utf-8")
    with pytest.raises(security.ReplayTamperedError, match="подпись"):
        security.verify_file(f, b"k1")


def test_replay_signed_with_other_key_is_refused(tmp_path):
    f = tmp_path / "replay.json"
    f.write_text("{}", encoding="utf-8")
    security.sign_file(f, b"k1")
    with pytest.raises(security.ReplayTamperedError):
        security.verify_file(f, b"other")


def test_load_recorded_generates_key_signs_on_first_run_then_checks(tmp_path, monkeypatch):
    monkeypatch.delenv("RADAR_HMAC_KEY", raising=False)
    f, keyf = tmp_path / "replay.json", tmp_path / "hmac.key"
    f.write_text('{"results": {"A": 1}}', encoding="utf-8")
    assert security.load_recorded(f, keyf) == {"results": {"A": 1}}
    assert keyf.exists() and (tmp_path / "replay.json.sig").exists()
    assert security.load_recorded(f, keyf) == {"results": {"A": 1}}
    f.write_text('{"results": {"A": 2}}', encoding="utf-8")
    with pytest.raises(security.ReplayTamperedError):
        security.load_recorded(f, keyf)


def test_missing_signature_is_refused_when_key_already_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_HMAC_KEY", "from-env")
    f = tmp_path / "replay.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(security.ReplayTamperedError, match="подпис"):
        security.load_recorded(f, tmp_path / "hmac.key")


def test_radar_refuses_tampered_recorded_replay(tmp_path, monkeypatch):
    from app import state
    f = tmp_path / "replay.json"
    f.write_text('{"results": {}}', encoding="utf-8")
    monkeypatch.setenv("RADAR_HMAC_KEY", "k")
    security.sign_file(f, b"k")
    f.write_text('{"results": {"B0003": {"facts": {}}}}', encoding="utf-8")
    monkeypatch.setattr(state, "RECORDED", f)
    with pytest.raises(security.ReplayTamperedError):
        state.Radar(server.load_or_build(), mode="template")


# ---------------------------------------------------------------- 3. цепочка хэшей журнала
def filled(n=5, path=":memory:"):
    a = Audit(path)
    for i in range(n):
        a.log("system", "code", {"i": i}, "B1")
    return a


def test_chain_ok_on_untouched_log():
    v = filled().verify_chain()
    assert v["ok"] is True and v["rows"] == 5 and v["broken_at"] is None and len(v["head"]) == 64


def test_chain_detects_edited_payload(tmp_path):
    p = str(tmp_path / "a.db")
    a = filled(5, p)
    sqlite3.connect(p).execute("UPDATE audit SET payload='{\"i\": 99}' WHERE id=3").connection.commit()
    v = Audit(p).verify_chain()
    assert v["ok"] is False and v["broken_at"] == 3


def test_chain_detects_deleted_middle_row(tmp_path):
    p = str(tmp_path / "a.db")
    filled(5, p)
    c = sqlite3.connect(p)
    c.execute("DELETE FROM audit WHERE id=2")
    c.commit()
    v = Audit(p).verify_chain()
    assert v["ok"] is False and v["broken_at"] == 3


def test_each_row_stores_prev_hash_and_chain_survives_reopen(tmp_path):
    p = str(tmp_path / "a.db")
    filled(3, p)
    a2 = Audit(p)
    a2.log("human", "human", {"d": 1}, "B2")
    rows = sqlite3.connect(p).execute("SELECT prev_hash, row_hash FROM audit ORDER BY id").fetchall()
    assert rows[0][0] == "0" * 64 and all(rows[i][0] == rows[i - 1][1] for i in range(1, 4))
    assert a2.verify_chain()["ok"] is True


def test_legacy_rows_without_hashes_are_counted_not_failed(tmp_path):
    p = str(tmp_path / "old.db")
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE audit(id INTEGER PRIMARY KEY AUTOINCREMENT, wall_ts TEXT, sim_ts TEXT, burst_id TEXT, kind TEXT, actor TEXT, payload TEXT)")
    c.execute("INSERT INTO audit(wall_ts,sim_ts,burst_id,kind,actor,payload) VALUES('t',NULL,NULL,'system','code','{}')")
    c.commit()
    a = Audit(p)
    a.log("system", "code", {"n": 1})
    v = a.verify_chain()
    assert v["ok"] is True and v["legacy_rows"] == 1 and v["rows"] == 1


# ---------------------------------------------------------------- HTTP: 3 (verify), 4 (токены), 5 (лимит)
@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


@pytest.fixture(autouse=True)
def fresh(client, monkeypatch):
    for k in ("RADAR_API_TOKEN", "RADAR_TOKEN_VIEW", "RADAR_TOKEN_DUTY"):
        monkeypatch.delenv(k, raising=False)
    server.player["playing"] = False
    server.rate_limiter.reset()
    client.post("/api/jump", json={"to": "2026-03-09T18:20", "reset": True})


def test_audit_verify_endpoint(client):
    j = client.get("/api/audit/verify").json()
    assert j["ok"] is True and j["rows"] >= 1


def test_audit_verify_endpoint_reports_break(client):
    r = server.get_radar()
    r.audit.db.execute("UPDATE audit SET payload='{\"x\": 1}' WHERE id=1")
    r.audit.db.commit()
    j = client.get("/api/audit/verify").json()
    assert j["ok"] is False and j["broken_at"] == 1


def bearer(t):
    return {"Authorization": f"Bearer {t}"}


def test_no_tokens_configured_means_open_demo(client):
    assert client.get("/api/cards").status_code == 200


def test_api_token_required_for_everything_but_status(client, monkeypatch):
    monkeypatch.setenv("RADAR_API_TOKEN", "s3cret")
    assert client.get("/api/status").status_code == 200
    assert client.get("/").status_code == 200
    for path in ("/api/cards", "/api/audit", "/api/audit/verify", "/api/series", "/api/tickets", "/api/metrics", "/api/tracker", "/api/fix"):
        r = client.get(path)
        assert r.status_code == 401, path
        assert r.headers["www-authenticate"].startswith("Bearer")
    assert client.get("/api/cards", headers=bearer("wrong")).status_code == 401
    assert client.get("/api/cards", headers={"Authorization": "Basic s3cret"}).status_code == 401
    assert client.get("/api/cards", headers=bearer("s3cret")).status_code == 200
    assert client.post("/api/control", json={"speed": 5}).status_code == 401
    assert client.post("/api/control", json={"speed": 5}, headers=bearer("s3cret")).status_code == 200


def test_stream_requires_token_too(client, monkeypatch):
    monkeypatch.setenv("RADAR_API_TOKEN", "s3cret")
    assert client.get("/api/stream").status_code == 401


def test_viewer_can_read_but_not_press_buttons(client, monkeypatch):
    monkeypatch.setenv("RADAR_TOKEN_VIEW", "v-token")
    monkeypatch.setenv("RADAR_TOKEN_DUTY", "d-token")
    assert client.get("/api/cards", headers=bearer("v-token")).status_code == 200
    r = client.post("/api/cards/B0003/decision", json={"action": "our_bug"}, headers=bearer("v-token"))
    assert r.status_code == 403 and "наблюдател" in r.json()["detail"]
    assert client.post("/api/control", json={"speed": 5}, headers=bearer("v-token")).status_code == 403
    assert client.post("/api/cards/B0003/decision", json={"action": "our_bug"}, headers=bearer("d-token")).status_code == 200
    assert client.post("/api/cards/B0003/decision", json={"action": "noise"}).status_code == 401


def test_status_reports_role_requirement_without_leaking_tokens(client, monkeypatch):
    monkeypatch.setenv("RADAR_API_TOKEN", "s3cret")
    s = client.get("/api/status").json()
    assert s["auth_required"] is True and "s3cret" not in json.dumps(s)
    monkeypatch.delenv("RADAR_API_TOKEN")
    assert client.get("/api/status").json()["auth_required"] is False


def test_whoami_returns_role(client, monkeypatch):
    monkeypatch.setenv("RADAR_TOKEN_VIEW", "v")
    monkeypatch.setenv("RADAR_TOKEN_DUTY", "d")
    assert client.get("/api/whoami", headers=bearer("v")).json()["role"] == "viewer"
    assert client.get("/api/whoami", headers=bearer("d")).json()["role"] == "duty"


def test_sliding_window_unit():
    t = [0.0]
    lim = security.SlidingWindowLimiter(3, 10.0, clock=lambda: t[0])
    assert [lim.allow("a") for _ in range(4)] == [True, True, True, False]
    assert lim.allow("b") is True                    # другой клиент считается отдельно
    t[0] = 10.5
    assert lim.allow("a") is True                    # окно сдвинулось


def test_post_rate_limit_returns_429_and_get_is_not_limited(client, monkeypatch):
    monkeypatch.setattr(server, "rate_limiter", security.SlidingWindowLimiter(3, 60.0))
    codes = [client.post("/api/control", json={"speed": 5}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = client.post("/api/control", json={"speed": 5})
    assert "Retry-After" in r.headers and "частот" in r.json()["detail"]
    assert all(client.get("/api/status").status_code == 200 for _ in range(10))
