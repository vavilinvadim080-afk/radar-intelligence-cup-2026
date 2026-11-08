"""Журнал аудита должен переживать перезапуск и сброс демо, если задан файл (RADAR_AUDIT_DB / audit_path)."""
import os

os.environ["RADAR_MODE"] = "template"

from app import server
from app.audit import Audit
from app.state import Radar


def test_audit_survives_reset_and_new_process(tmp_path):
    rep = server.load_or_build()
    path = str(tmp_path / "audit.db")
    r1 = Radar(rep, mode="template", audit_path=path)
    r1.audit.log("human", "human", {"decision": "our_bug", "who": "дежурный"}, "B0003")
    n_before = len(r1.audit.rows(limit=500))
    r1.reset(r1.now)                                     # кнопка сброса не должна стирать журнал
    rows = r1.audit.rows(limit=500)
    assert any(r["kind"] == "human" for r in rows) and len(rows) > n_before
    r2 = Radar(rep, mode="template", audit_path=path)    # «перезапуск»
    assert any(r["kind"] == "human" and r["payload"]["decision"] == "our_bug" for r in r2.audit.rows(limit=500))


def test_audit_default_is_memory_and_resets():
    rep = server.load_or_build()
    r = Radar(rep, mode="template")
    r.audit.log("human", "human", {"decision": "x"}, "B1")
    r.reset(r.now)
    assert not any(row["kind"] == "human" for row in r.audit.rows(limit=500))


def test_audit_rows_limit_is_never_unbounded():
    a = Audit()
    for i in range(600):
        a.log("system", "code", {"i": i})
    assert len(a.rows(limit=-1)) <= 500 and len(a.rows(limit=10**9)) <= 500 and len(a.rows(limit=10)) == 10
