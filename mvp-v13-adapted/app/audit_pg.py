"""Бэкенд журнала аудита на PostgreSQL (psycopg 3, необязательная зависимость: requirements-pg.txt).

Выбирается переменной RADAR_AUDIT_DSN (app/audit.py::audit_from_env). Интерфейс тот же, что у SQLite-бэкенда: append / tail / all_ordered.

Две роли (README, раздел «Журнал аудита в PostgreSQL»):
  * владелец (radar_owner) создаёт таблицу один раз: `python -m app.audit_pg init "<DSN владельца>"`;
  * роль приложения (radar_app) имеет только INSERT и SELECT на таблицу и USAGE на последовательность. UPDATE и DELETE ей не выданы,
    поэтому приложение, даже взломанное, переписать журнал не может. Этот модуль под ролью приложения таблиц не создаёт.
Колонка payload это TEXT, а не JSONB: JSONB переупорядочивает ключи и ломает точный пересчёт хэша.
Две копии приложения пишут в одну цепочку по очереди: запись идёт в транзакции под pg_advisory_xact_lock.
"""
from __future__ import annotations

import re

from .audit import GENESIS, row_digest

LOCK_KEY = 7_203_117_001     # произвольная константа для advisory lock (общая для всех копий приложения)
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _check_table(table: str) -> str:
    if not _IDENT.match(table):
        raise ValueError(f"имя таблицы {table!r} не годится: строчные латинские буквы, цифры, подчёркивание")
    return table


def _psycopg():
    try:
        import psycopg
    except ImportError:
        raise RuntimeError("RADAR_AUDIT_DSN задан, но пакет psycopg не установлен: uv pip install -r requirements-pg.txt") from None
    return psycopg


class PgBackend:
    def __init__(self, dsn: str, table: str = "audit"):
        self.table = _check_table(table)
        self.conn = _psycopg().connect(dsn, autocommit=True)
        row = self.conn.execute("SELECT to_regclass(%s)", (self.table,)).fetchone()
        if not row or row[0] is None:
            raise RuntimeError(f"таблицы {self.table} в базе нет: владелец выполняет python -m app.audit_pg init \"<DSN владельца>\"")

    @classmethod
    def init_schema(cls, owner_dsn: str, table: str = "audit") -> None:
        t = _check_table(table)
        conn = _psycopg().connect(owner_dsn, autocommit=True)
        try:
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {t}(
                id BIGSERIAL PRIMARY KEY, wall_ts TEXT NOT NULL, sim_ts TEXT, burst_id TEXT,
                kind TEXT NOT NULL, actor TEXT NOT NULL, payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL UNIQUE)""")
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_burst_idx ON {t}(burst_id, id)")
        finally:
            conn.close()

    def append(self, wall_ts, sim_ts, burst_id, kind, actor, payload) -> int:
        t = self.table
        with self.conn.transaction():
            self.conn.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
            r = self.conn.execute(f"SELECT row_hash FROM {t} ORDER BY id DESC LIMIT 1").fetchone()
            prev = r[0] if r else GENESIS
            new_id = self.conn.execute(
                f"INSERT INTO {t}(wall_ts,sim_ts,burst_id,kind,actor,payload,prev_hash,row_hash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (wall_ts, sim_ts, burst_id, kind, actor, payload, prev, row_digest(prev, wall_ts, sim_ts, burst_id, kind, actor, payload))).fetchone()
        return new_id[0]

    def tail(self, burst_id, limit) -> list:
        t = self.table
        q = f"SELECT id,wall_ts,sim_ts,burst_id,kind,actor,payload FROM {t}" + (" WHERE burst_id=%s" if burst_id else "") + " ORDER BY id DESC LIMIT %s"
        return list(self.conn.execute(q, ([burst_id] if burst_id else []) + [limit]).fetchall())

    def all_ordered(self):
        yield from self.conn.execute(f"SELECT id,wall_ts,sim_ts,burst_id,kind,actor,payload,prev_hash,row_hash FROM {self.table} ORDER BY id")

    def raw_execute(self, sql: str, params=None):
        """Только для тестов (имитация порчи журнала владельцем базы). Приложение этот метод не вызывает."""
        return self.conn.execute(sql, params)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "init":
        PgBackend.init_schema(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "audit")
        print("таблица журнала создана (или уже была)")
    else:
        print('использование: python -m app.audit_pg init "postgresql://radar_owner:...@host/radar" [имя_таблицы]')
