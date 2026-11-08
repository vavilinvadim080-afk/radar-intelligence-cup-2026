"""Журнал аудита: каждый вызов агента, инструмента и решение человека. Только добавление, цепочка хэшей.

Бэкенд подключаемый, интерфейс один (log / rows / verify_chain):
  * SQLite (по умолчанию; ":memory:" или файл RADAR_AUDIT_DB) - демо без настройки;
  * PostgreSQL через RADAR_AUDIT_DSN (psycopg 3, requirements-pg.txt) - см. README, раздел «Журнал аудита в PostgreSQL».

Цепочка: у каждой строки prev_hash (row_hash предыдущей; у первой 64 нуля) и row_hash = SHA-256(prev_hash и все поля строки).
Правка поля или удаление строки из середины рвёт цепочку, verify_chain() называет первый id, где она порвана.
Чего цепочка НЕ ловит: усечение хвоста (удалены последние строки) и полную перепись всей цепочки тем, кто умеет считать SHA-256.
Против этого head (последний row_hash) нужно хранить снаружи (SIEM, отдельная таблица с другими правами), он отдаётся в verify_chain().

Якорь head (v3, имитация WORM): Audit.export_anchor(путь) дописывает в отдельный файл строку JSON {rows, head, ts, prev, ahash[, hmac]};
verify_chain(путь) сверяет цепочку со ВСЕМИ якорями: на позиции rows хэш должен совпасть с head. Ловит усечение (строк меньше, чем в якоре)
и любую правку или полную пересчитанную перепись того, что было до якоря. Строки ПОСЛЕ последнего якоря ничем не прикрыты (их число
в anchor.unanchored_rows) и подмену этого хвоста ловит только следующий якорь. Локальный файл не настоящий WORM: кто может править
и файл журнала, и файл якоря, перепишет оба; в проде якорь уходит в хранилище с блокировкой версий (S3 Object Lock) или SIEM.
Файл якорей сам связан цепочкой (prev/ahash) и при ключе подписан HMAC, но хвостовые строки файла якорей без внешней копии удалить можно.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
from datetime import datetime

GENESIS = "0" * 64
COLS = ["id", "wall_ts", "sim_ts", "burst_id", "kind", "actor", "payload"]


def row_digest(prev_hash: str, wall_ts, sim_ts, burst_id, kind, actor, payload: str) -> str:
    """SHA-256 строки журнала. payload берётся как хранимая строка JSON (не как объект), чтобы пересчёт был точным."""
    blob = json.dumps([prev_hash, wall_ts, sim_ts, burst_id, kind, actor, payload], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SqliteBackend:
    table = "audit"

    def raw_execute(self, sql: str, params=()):
        """Только для тестов (имитация порчи журнала владельцем файла). Приложение этот метод не вызывает."""
        cur = self.db.execute(sql, params)
        self.db.commit()
        return cur

    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS audit(
            id INTEGER PRIMARY KEY AUTOINCREMENT, wall_ts TEXT, sim_ts TEXT, burst_id TEXT,
            kind TEXT, actor TEXT, payload TEXT, prev_hash TEXT, row_hash TEXT)""")
        have = {r[1] for r in self.db.execute("PRAGMA table_info(audit)")}
        for col in ("prev_hash", "row_hash"):   # журнал, созданный до цепочки хэшей: добавляем столбцы, старые строки остаются без хэша
            if col not in have:
                self.db.execute(f"ALTER TABLE audit ADD COLUMN {col} TEXT")
        self.db.commit()

    def append(self, wall_ts, sim_ts, burst_id, kind, actor, payload) -> int:
        """Добавляет строку и считает её хэш от хэша предыдущей. Вызывается под замком Audit.lock (процесс один)."""
        r = self.db.execute("SELECT row_hash FROM audit WHERE row_hash IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
        prev = r[0] if r else GENESIS
        cur = self.db.execute(
            "INSERT INTO audit(wall_ts,sim_ts,burst_id,kind,actor,payload,prev_hash,row_hash) VALUES(?,?,?,?,?,?,?,?)",
            (wall_ts, sim_ts, burst_id, kind, actor, payload, prev, row_digest(prev, wall_ts, sim_ts, burst_id, kind, actor, payload)))
        self.db.commit()
        return cur.lastrowid

    def tail(self, burst_id, limit) -> list:
        q = "SELECT id,wall_ts,sim_ts,burst_id,kind,actor,payload FROM audit" + (" WHERE burst_id=?" if burst_id else "") + " ORDER BY id DESC LIMIT ?"
        return self.db.execute(q, ([burst_id] if burst_id else []) + [limit]).fetchall()

    def all_ordered(self):
        yield from self.db.execute("SELECT id,wall_ts,sim_ts,burst_id,kind,actor,payload,prev_hash,row_hash FROM audit ORDER BY id")


class Audit:
    def __init__(self, path: str = ":memory:", dsn: str | None = None, table: str = "audit", anchor_path: str | None = None):
        """dsn задан: PostgreSQL (app/audit_pg.py), иначе SQLite по пути path. anchor_path: файл якорей (иначе RADAR_AUDIT_ANCHOR)."""
        self.anchor_path = anchor_path
        if dsn:
            from .audit_pg import PgBackend
            self.backend = PgBackend(dsn, table)
        else:
            self.backend = SqliteBackend(path)
        self.lock = threading.Lock()

    @property
    def db(self):   # прежнее имя (тесты и отладка ходят прямо в соединение SQLite)
        return self.backend.db

    def log(self, kind: str, actor: str, payload: dict, burst_id: str | None = None, sim_ts=None) -> int:
        """kind: agent_input | tool_call | tool_denied | model_reply | validator | human | system.
        actor: agent | code | human. payload: любой словарь (пишется как JSON)."""
        wall = datetime.now().isoformat(timespec="seconds")
        sim = sim_ts.isoformat(timespec="minutes") if hasattr(sim_ts, "isoformat") else sim_ts
        body = json.dumps(payload, ensure_ascii=False, default=str)
        with self.lock:
            return self.backend.append(wall, sim, burst_id, kind, actor, body)

    def rows(self, burst_id: str | None = None, limit: int = 200) -> list:
        limit = max(1, min(int(limit), 500))      # верхняя граница: журнал не отдаётся целиком одним запросом
        with self.lock:
            out = self.backend.tail(burst_id, limit)
        res = []
        for r in out:
            d = dict(zip(COLS, r))
            d["payload"] = json.loads(d["payload"])
            res.append(d)
        return list(reversed(res))

    def verify_chain(self, anchor_path: str | bool | None = None, key: bytes | None = None) -> dict:
        """Проверка цепочки по всему журналу. Строки без хэша (журнал создан до цепочки) считаются отдельно и не ломают проверку.
        Есть файл якорей (аргумент, Audit(anchor_path=...) или RADAR_AUDIT_ANCHOR): в ответ добавляется ключ anchor, и ok требует его ok."""
        if anchor_path is None:
            anchor_path = self._anchor_path()      # False: якоря не проверять
        anchors, anchor_err = None, None
        if anchor_path:
            if os.path.exists(anchor_path):
                anchors, anchor_err = _read_anchors(anchor_path, key or _env_key())
            else:
                anchors, anchor_err = [], "файла якорей нет: " + anchor_path
        want = {a["rows"] for a in anchors or []}
        prev, n, legacy, broken, at = GENESIS, 0, 0, None, {}
        with self.lock:
            for (rid, wall, sim, burst, kind, actor, payload, p_hash, r_hash) in self.backend.all_ordered():
                if r_hash is None and n == 0:
                    legacy += 1
                    continue
                n += 1
                if broken is None and (p_hash != prev or r_hash != row_digest(prev, wall, sim, burst, kind, actor, payload)):
                    broken = rid
                prev = r_hash or prev
                if n in want:
                    at[n] = prev
        res = {"ok": broken is None, "rows": n, "legacy_rows": legacy, "broken_at": broken, "head": prev}
        if anchors is not None:
            err = anchor_err
            for a in anchors if err is None else []:
                if a["rows"] > n:
                    err = f"цепочка короче якоря: в журнале {n} строк, якорь от {a['ts']} на {a['rows']} (усечение хвоста)"
                    break
                if at.get(a["rows"]) != a["head"]:
                    err = f"хэш строки {a['rows']} не совпал с якорем от {a['ts']}: журнал переписан до якоря"
                    break
            last = anchors[-1]["rows"] if anchors else 0
            res["anchor"] = {"ok": err is None, "anchors": len(anchors), "last_rows": last, "unanchored_rows": max(0, n - last), "error": err}
            res["ok"] = res["ok"] and err is None
        return res

    def _anchor_path(self) -> str | None:
        return self.anchor_path or os.environ.get("RADAR_AUDIT_ANCHOR") or None

    def export_anchor(self, anchor_path: str | None = None, key: bytes | None = None) -> dict:
        """Дописывает в файл якорей текущий head с меткой времени. Пустой или порванный журнал не якорим (ValueError):
        якорь на порванную цепочку закрепил бы подмену. key (или RADAR_AUDIT_ANCHOR_KEY): подпись HMAC-SHA256."""
        path = anchor_path or self._anchor_path()
        if not path:
            raise ValueError("не задан файл якорей (аргумент, anchor_path или RADAR_AUDIT_ANCHOR)")
        v = self.verify_chain(anchor_path=False)   # без якорей: сверяем только саму цепочку
        if not v["ok"] or v["rows"] == 0:
            raise ValueError("журнал пуст или цепочка порвана (broken_at=%s): якорь не пишем" % v["broken_at"])
        key = key or _env_key()
        prev_ahash = GENESIS
        if os.path.exists(path):
            recs, err = _read_anchors(path, key)
            if err:
                raise ValueError("файл якорей повреждён, дописывать нельзя: " + err)
            prev_ahash = recs[-1]["ahash"] if recs else GENESIS
        rec = {"rows": v["rows"], "head": v["head"], "ts": datetime.now().isoformat(timespec="seconds"), "prev": prev_ahash}
        rec["ahash"] = _anchor_hash(rec)
        if key:
            rec["hmac"] = hmac.new(key, rec["ahash"].encode(), "sha256").hexdigest()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:   # режим «a»: только дописывание
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec


def _env_key() -> bytes | None:
    k = os.environ.get("RADAR_AUDIT_ANCHOR_KEY")
    return k.encode() if k else None


def _anchor_hash(rec: dict) -> str:
    return hashlib.sha256(f"{rec['prev']}|{rec['rows']}|{rec['head']}|{rec['ts']}".encode()).hexdigest()


def _read_anchors(path: str, key: bytes | None):
    """([записи], None) или ([], текст_ошибки): цепочка prev/ahash файла якорей и, при ключе, подпись каждой записи."""
    out, prev = [], GENESIS
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                ok = rec["prev"] == prev and rec["ahash"] == _anchor_hash(rec) and isinstance(rec["rows"], int)
            except (ValueError, KeyError, TypeError):
                ok = False
            if not ok:
                return [], f"якорь №{i} в файле изменён или порван"
            if key and not hmac.compare_digest(rec.get("hmac", ""), hmac.new(key, rec["ahash"].encode(), "sha256").hexdigest()):
                return [], f"подпись якоря №{i} не сходится (чужой ключ или правка)"
            prev = rec["ahash"]
            out.append(rec)
    return out, None


def audit_from_env(path: str | None) -> Audit:
    """PostgreSQL, если задан RADAR_AUDIT_DSN; иначе SQLite: файл path (или RADAR_AUDIT_DB), иначе память."""
    return Audit(path or ":memory:", dsn=os.environ.get("RADAR_AUDIT_DSN") or None)
