"""Якорь head: отдельный файл только для дописывания (имитация WORM). verify_chain сверяет цепочку с якорями."""
import json

import pytest

from app.audit import Audit, row_digest


def make(tmp_path, n=6):
    a = Audit(str(tmp_path / "audit.db"))
    for i in range(n):
        a.log("system", "code", {"i": i}, "B1")
    return a, str(tmp_path / "anchor" / "head.anchor")


def test_export_anchor_writes_head_with_timestamp(tmp_path):
    a, anc = make(tmp_path)
    rec = a.export_anchor(anc)
    assert rec["rows"] == 6 and rec["head"] == a.verify_chain()["head"] and rec["ts"]
    lines = open(anc, encoding="utf-8").read().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["head"] == rec["head"]


def test_anchor_file_is_append_only(tmp_path):
    a, anc = make(tmp_path)
    a.export_anchor(anc)
    a.log("system", "code", {"i": 99})
    a.export_anchor(anc)
    assert len(open(anc, encoding="utf-8").read().splitlines()) == 2


def test_intact_chain_passes_with_anchor(tmp_path):
    a, anc = make(tmp_path)
    a.export_anchor(anc)
    a.log("system", "code", {"after": 1})            # строки после якоря: цепочка растёт, это нормально
    r = a.verify_chain(anc)
    assert r["ok"] and r["anchor"]["ok"] and r["anchor"]["anchors"] == 1 and r["anchor"]["unanchored_rows"] == 1


def test_verify_without_anchor_file_is_as_before(tmp_path):
    a, _ = make(tmp_path)
    r = a.verify_chain()
    assert r["ok"] and "anchor" not in r


def test_rewritten_row_before_anchor_is_caught(tmp_path):
    a, anc = make(tmp_path)
    a.export_anchor(anc)
    a.backend.raw_execute("UPDATE audit SET payload=? WHERE id=2", ('{"i":"подмена"}',))
    r = a.verify_chain(anc)
    assert not r["ok"] and r["broken_at"] == 2


def test_full_consistent_rewrite_before_anchor_is_caught(tmp_path):
    """Владелец файла пересчитал всю цепочку (умеет SHA-256): цепочка сама целая, но head уже не тот, что в якоре."""
    a, anc = make(tmp_path)
    a.export_anchor(anc)
    rows = list(a.backend.all_ordered())
    prev = "0" * 64
    for (rid, wall, sim, burst, kind, actor, payload, _p, _h) in rows:
        payload = '{"i":"чужое"}' if rid == 3 else payload
        h = row_digest(prev, wall, sim, burst, kind, actor, payload)
        a.backend.raw_execute("UPDATE audit SET payload=?, prev_hash=?, row_hash=? WHERE id=?", (payload, prev, h, rid))
        prev = h
    r = a.verify_chain(anc)
    assert not r["ok"] and r["anchor"]["ok"] is False and "якор" in r["anchor"]["error"]


def test_truncated_chain_is_caught(tmp_path):
    a, anc = make(tmp_path)
    a.export_anchor(anc)
    a.backend.raw_execute("DELETE FROM audit WHERE id>3")
    assert a.verify_chain()["ok"]                     # без якоря усечение не видно: ровно то, что закрывает якорь
    r = a.verify_chain(anc)
    assert not r["ok"] and r["anchor"]["ok"] is False and "короче" in r["anchor"]["error"]


def test_truncate_then_append_new_tail_is_caught(tmp_path):
    a, anc = make(tmp_path)
    a.export_anchor(anc)
    a.backend.raw_execute("DELETE FROM audit WHERE id>4")
    for i in range(4):
        a.log("system", "code", {"forged": i})      # хвост дописан заново: длина прежняя, хэш на позиции якоря другой
    r = a.verify_chain(anc)
    assert not r["ok"] and r["anchor"]["ok"] is False


def test_edited_anchor_file_is_caught(tmp_path):
    a, anc = make(tmp_path)
    a.export_anchor(anc)
    a.log("system", "code", {"i": 7})
    a.export_anchor(anc)
    lines = open(anc, encoding="utf-8").read().splitlines()
    rec = json.loads(lines[0]); rec["head"] = "f" * 64
    open(anc, "w", encoding="utf-8").write(json.dumps(rec) + "\n" + lines[1] + "\n")
    r = a.verify_chain(anc)
    assert not r["ok"] and r["anchor"]["ok"] is False


def test_hmac_key_signs_anchor_and_wrong_key_fails(tmp_path):
    a, anc = make(tmp_path)
    rec = a.export_anchor(anc, key=b"k1")
    assert rec["hmac"]
    assert a.verify_chain(anc, key=b"k1")["ok"]
    r = a.verify_chain(anc, key=b"k2")
    assert not r["ok"] and r["anchor"]["ok"] is False


def test_refuses_to_anchor_broken_chain(tmp_path):
    a, anc = make(tmp_path)
    a.backend.raw_execute("UPDATE audit SET payload='{}' WHERE id=2")
    with pytest.raises(ValueError):
        a.export_anchor(anc)


def test_empty_log_not_anchored(tmp_path):
    a = Audit()
    with pytest.raises(ValueError):
        a.export_anchor(str(tmp_path / "x.anchor"))


def test_anchor_path_from_env_used_by_verify(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_AUDIT_ANCHOR", str(tmp_path / "env.anchor"))
    a = Audit(str(tmp_path / "a.db"))
    a.log("system", "code", {"i": 1})
    a.export_anchor()
    a.backend.raw_execute("DELETE FROM audit")
    assert not a.verify_chain()["ok"]
