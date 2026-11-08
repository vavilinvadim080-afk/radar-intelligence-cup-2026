"""Журнал аудита: один набор тестов на оба бэкенда (SQLite всегда, PostgreSQL если задан RADAR_TEST_PG_DSN) плюс проверка PG-бэкенда на поддельном psycopg.

Тест на настоящем PostgreSQL: RADAR_TEST_PG_DSN="postgresql://владелец:пароль@localhost/radar_test" (роль с правом создавать таблицы).
Он создаёт таблицу audit_test_<случайное> и в конце её удаляет. Без переменной тесты PG пропускаются.
"""
import os
import sys
import types
import uuid

import pytest

from app.audit import Audit, GENESIS, row_digest

PG_DSN = os.environ.get("RADAR_TEST_PG_DSN")


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.skipif(not PG_DSN, reason="нет RADAR_TEST_PG_DSN: PostgreSQL в этой среде не поднимаем"))])
def audit(request, tmp_path):
    if request.param == "sqlite":
        yield Audit(str(tmp_path / "a.db"))
        return
    from app.audit_pg import PgBackend
    table = "audit_test_" + uuid.uuid4().hex[:8]
    PgBackend.init_schema(PG_DSN, table)
    a = Audit(dsn=PG_DSN, table=table)
    yield a
    a.backend.raw_execute(f"DROP TABLE {table}")


def fill(a, n=5):
    for i in range(n):
        a.log("system", "code", {"i": i}, "B1")


def test_log_and_rows_same_interface(audit):
    fill(audit, 3)
    rows = audit.rows(limit=10)
    assert [r["payload"]["i"] for r in rows] == [0, 1, 2] and rows[0]["kind"] == "system" and rows[0]["burst_id"] == "B1"
    assert len(audit.rows("B1", 2)) == 2 and audit.rows("нет", 5) == []


def test_rows_limit_is_bounded(audit):
    fill(audit, 3)
    assert len(audit.rows(limit=-1)) == 1 and len(audit.rows(limit=10**9)) <= 500


def test_chain_ok_and_head(audit):
    fill(audit)
    v = audit.verify_chain()
    assert v["ok"] and v["rows"] == 5 and len(v["head"]) == 64 and v["head"] != GENESIS


def test_edit_breaks_chain(audit):
    fill(audit)
    audit.backend.raw_execute("UPDATE audit SET payload='{\"i\": 99}' WHERE id=3".replace("audit", audit.backend.table))
    v = audit.verify_chain()
    assert not v["ok"] and v["broken_at"] == 3


def test_delete_breaks_chain(audit):
    fill(audit)
    audit.backend.raw_execute(f"DELETE FROM {audit.backend.table} WHERE id=2")
    assert audit.verify_chain()["broken_at"] == 3


def test_unicode_payload_hash_is_exact(audit):
    audit.log("human", "human", {"who": "дежурный", "text": "Ёж"}, "B2")
    assert audit.verify_chain()["ok"]
    assert audit.rows()[0]["payload"]["who"] == "дежурный"


# ---------------------------------------------------------------- PgBackend на поддельном psycopg (настоящий PostgreSQL не нужен)
class FakeConn:
    """Подделка соединения psycopg: пишет все SQL-запросы и отвечает на те, что нужны бэкенду."""

    def __init__(self, log):
        self.log, self.last = log, ""

    def execute(self, sql, params=None):
        self.last = " ".join(sql.split())
        self.log.append((self.last, params))
        return self

    def fetchone(self):
        if "to_regclass" in self.last:
            return ("audit",)
        if "RETURNING id" in self.last:
            return (7,)
        return None

    def fetchall(self):
        return []

    def __iter__(self):
        return iter([])

    def close(self):
        pass


class Tx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def fake_psycopg(monkeypatch):
    calls = []
    conn = FakeConn(calls)
    conn.transaction = lambda: Tx()
    mod = types.ModuleType("psycopg")
    mod.connect = lambda dsn, **kw: calls.append(("connect", dsn, kw)) or conn
    monkeypatch.setitem(sys.modules, "psycopg", mod)
    return calls, conn


def test_pg_backend_app_role_uses_only_insert_and_select(fake_psycopg):
    from app.audit_pg import PgBackend
    calls, conn = fake_psycopg
    b = PgBackend("postgresql://app@h/radar")
    b.append("w", None, "B1", "system", "code", "{}")
    b.tail(None, 5)
    list(b.all_ordered())
    sql = " ".join(c[0].upper() for c in calls if isinstance(c[0], str) and c[0] != "connect")
    for forbidden in ("UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE"):
        assert forbidden not in sql, forbidden
    assert "INSERT INTO AUDIT" in sql and "PG_ADVISORY_XACT_LOCK" in sql    # цепочка пишется под замком, иначе два процесса порвут её


def test_pg_backend_does_not_create_tables_by_itself(fake_psycopg):
    from app.audit_pg import PgBackend
    calls, _ = fake_psycopg
    PgBackend("postgresql://app@h/radar")
    assert not any("CREATE" in str(c[0]).upper() for c in calls)


def test_pg_init_schema_creates_table_and_is_owner_only(fake_psycopg):
    from app.audit_pg import PgBackend
    calls, _ = fake_psycopg
    PgBackend.init_schema("postgresql://owner@h/radar", "audit")
    text = " ".join(str(c[0]) for c in calls)
    assert "CREATE TABLE IF NOT EXISTS audit" in text and "row_hash" in text and "prev_hash" in text


def test_pg_table_name_is_validated(fake_psycopg):
    from app.audit_pg import PgBackend
    with pytest.raises(ValueError):
        PgBackend("postgresql://x", table="audit; DROP TABLE users")


def test_missing_psycopg_gives_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    from app.audit_pg import PgBackend
    with pytest.raises(RuntimeError, match="requirements-pg.txt"):
        PgBackend("postgresql://x")


def test_radar_picks_postgres_when_dsn_set(monkeypatch, fake_psycopg):
    from app import audit as audit_mod
    monkeypatch.setenv("RADAR_AUDIT_DSN", "postgresql://app@h/radar")
    a = audit_mod.audit_from_env(None)
    assert type(a.backend).__name__ == "PgBackend"
    monkeypatch.delenv("RADAR_AUDIT_DSN")
    assert type(audit_mod.audit_from_env(None).backend).__name__ == "SqliteBackend"
