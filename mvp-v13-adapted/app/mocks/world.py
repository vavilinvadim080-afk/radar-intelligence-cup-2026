"""Мок-системы вокруг Радара: релизы, инциденты мониторинга, трекер, база знаний.

Всё в одной SQLite (файл или :memory:). В боевой версии вместо этого модуля стоят адаптеры под
ваши системы; интерфейс функций тот же. Данные условные и заполняются из демо-набора генератора.

Правило времени: каждая выборка принимает `now` (потоковое время) и не показывает «будущее».
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from datetime import datetime, timedelta

from radar.gen import topics as T
from .kb_articles import KB_ARTICLES

SCHEMA = """
CREATE TABLE IF NOT EXISTS releases(id TEXT PRIMARY KEY, product TEXT, version TEXT, ts TEXT, kind TEXT, changelog TEXT);
CREATE TABLE IF NOT EXISTS incidents(id TEXT PRIMARY KEY, product TEXT, started_at TEXT, ended_at TEXT, title TEXT, severity TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS tracker(key TEXT PRIMARY KEY, product TEXT, title TEXT, status TEXT, created_at TEXT,
                                   description TEXT, source TEXT, burst_id TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS kb(id TEXT PRIMARY KEY, product TEXT, title TEXT, body TEXT);
CREATE TABLE IF NOT EXISTS drafts(id TEXT PRIMARY KEY, burst_id TEXT, product TEXT, title TEXT, description TEXT,
                                  severity TEXT, status TEXT, created_at TEXT, tracker_key TEXT);
CREATE TABLE IF NOT EXISTS outbox(id TEXT PRIMARY KEY, burst_id TEXT, product TEXT, message TEXT, status TEXT,
                                  created_at TEXT, sent_via TEXT);
"""


def _tokens(text: str) -> set:
    """Слова в нижнем регистре, обрезанные до 5 букв: грубая замена стеммера («баллы»~«баллов»)."""
    return {w[:5] for w in re.findall(r"[а-яёa-z0-9]{3,}", (text or "").lower())}


def release_id(product: str, version: str) -> str:
    return f"REL-{product}-{version}"


class MockWorld:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._next_key = 200   # ключи задач RAD-201, RAD-202 ... создаёт человек через Радар

    # ---------- заполнение
    def add_release(self, r) -> None:
        self.db.execute("INSERT OR REPLACE INTO releases VALUES(?,?,?,?,?,?)",
                        (release_id(r.product, r.version), r.product, r.version, r.ts.isoformat(timespec="minutes"),
                         r.kind, r.changelog))

    def add_incident(self, id_, product, start, end, title, severity="minor", source="мониторинг"):
        self.db.execute("INSERT OR REPLACE INTO incidents VALUES(?,?,?,?,?,?,?)",
                        (id_, product, start.isoformat(timespec="minutes"),
                         end.isoformat(timespec="minutes") if end else None, title, severity, source))

    def add_task(self, key, product, title, status, created_at, description="", source="seed", burst_id=None):
        self.db.execute("INSERT OR REPLACE INTO tracker VALUES(?,?,?,?,?,?,?,?,?)",
                        (key, product, title, status, created_at.isoformat(timespec="minutes"), description, source,
                         burst_id, created_at.isoformat(timespec="minutes")))

    def commit(self):
        self.db.commit()

    # ---------- чтение (то, что видят инструменты агента)
    def get_releases(self, product: str, since: datetime | None, now: datetime) -> list:
        q = "SELECT * FROM releases WHERE product=? AND ts<=?"
        args = [product, now.isoformat(timespec="minutes")]
        if since:
            q += " AND ts>=?"
            args.append(since.isoformat(timespec="minutes"))
        return [dict(r) for r in self.db.execute(q + " ORDER BY ts DESC LIMIT 10", args)]

    def get_incidents(self, product: str, since: datetime | None, now: datetime) -> list:
        q = "SELECT * FROM incidents WHERE product=? AND started_at<=?"
        args = [product, now.isoformat(timespec="minutes")]
        if since:
            q += " AND (ended_at IS NULL OR ended_at>=?)"
            args.append(since.isoformat(timespec="minutes"))
        return [dict(r) for r in self.db.execute(q + " ORDER BY started_at DESC LIMIT 10", args)]

    def search_tracker(self, query: str, product: str | None, now: datetime, limit: int = 5) -> list:
        qt = _tokens(query)
        rows = self.db.execute("SELECT * FROM tracker WHERE created_at<=?" + (" AND product=?" if product else ""),
                               [now.isoformat(timespec="minutes")] + ([product] if product else [])).fetchall()
        scored = []
        for r in rows:
            common = qt & _tokens(r["title"] + " " + r["description"])
            if common:
                scored.append((len(common) / max(len(qt), 1), dict(r)))
        scored.sort(key=lambda x: -x[0])
        return [{**r, "score": round(s, 2)} for s, r in scored[:limit] if s >= 0.25]

    def task(self, key: str) -> dict | None:
        r = self.db.execute("SELECT * FROM tracker WHERE key=?", (key,)).fetchone()
        return dict(r) if r else None

    def tasks(self) -> list:
        return [dict(r) for r in self.db.execute("SELECT * FROM tracker ORDER BY created_at DESC")]

    def search_kb(self, query: str, product: str | None, limit: int = 3) -> list:
        qt = _tokens(query)
        out = []
        for r in self.db.execute("SELECT * FROM kb" + (" WHERE product=?" if product else ""),
                                 [product] if product else []):
            common = qt & _tokens(r["title"] + " " + r["body"])
            if common:
                out.append((len(common) / max(len(qt), 1), dict(r)))
        out.sort(key=lambda x: -x[0])
        return [{**r, "score": round(s, 2)} for s, r in out[:limit] if s >= 0.2]

    def id_exists(self, ref: str) -> bool:
        """Для валидатора: существует ли такой id релиза, инцидента или задачи."""
        for table, col in (("releases", "id"), ("incidents", "id"), ("tracker", "key")):
            if self.db.execute(f"SELECT 1 FROM {table} WHERE {col}=?", (ref,)).fetchone():
                return True
        return False

    # ---------- запись (только через человека, кроме черновиков и очереди)
    def new_draft(self, burst_id, product, title, description, severity, now) -> dict:
        n = self.db.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] + 1
        d = dict(id=f"DR-{n:03d}", burst_id=burst_id, product=product, title=title, description=description,
                 severity=severity, status="ждёт подтверждения", created_at=now.isoformat(timespec="minutes"), tracker_key=None)
        self.db.execute("INSERT INTO drafts VALUES(:id,:burst_id,:product,:title,:description,:severity,:status,:created_at,:tracker_key)", d)
        self.db.commit()
        return d

    def new_outbox(self, burst_id, product, message, now) -> dict:
        n = self.db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] + 1
        o = dict(id=f"OUT-{n:03d}", burst_id=burst_id, product=product, message=message, status="ждёт подтверждения",
                 created_at=now.isoformat(timespec="minutes"), sent_via=None)
        self.db.execute("INSERT INTO outbox VALUES(:id,:burst_id,:product,:message,:status,:created_at,:sent_via)", o)
        self.db.commit()
        return o

    def drafts(self) -> list:
        return [dict(r) for r in self.db.execute("SELECT * FROM drafts ORDER BY id")]

    def outbox(self) -> list:
        return [dict(r) for r in self.db.execute("SELECT * FROM outbox ORDER BY id")]

    def create_task_from_draft(self, draft_id: str, now: datetime) -> dict:
        """Действие ЧЕЛОВЕКА (кнопка «наш баг»): черновик становится задачей в трекере."""
        d = self.db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
        if d is None:
            raise KeyError(draft_id)
        self._next_key += 1
        key = f"RAD-{self._next_key}"
        self.add_task(key, d["product"], d["title"], "open", now, d["description"], source="radar", burst_id=d["burst_id"])
        self.db.execute("UPDATE drafts SET status='создана задача', tracker_key=? WHERE id=?", (key, draft_id))
        self.db.commit()
        return self.task(key)

    def set_task_status(self, key: str, status: str, now: datetime) -> dict:
        """Действие ЧЕЛОВЕКА (подтвердил «предлагаю закрыть»). Агент до этой функции не дотягивается."""
        self.db.execute("UPDATE tracker SET status=?, updated_at=? WHERE key=?", (status, now.isoformat(timespec="minutes"), key))
        self.db.commit()
        return self.task(key)

    def link_task(self, key: str, burst_id: str) -> dict:
        """Действие ЧЕЛОВЕКА («известно, уже чиним»): кластер привязывается к существующей задаче."""
        self.db.execute("UPDATE tracker SET burst_id=? WHERE key=?", (burst_id, key))
        self.db.commit()
        return self.task(key)

    def set_outbox_status(self, oid: str, status: str, via: str | None = None):
        self.db.execute("UPDATE outbox SET status=?, sent_via=? WHERE id=?", (status, via, oid))
        self.db.commit()


# ---------------------------------------------------------------- заполнение из набора генератора
def build_world(releases: list, ground_truth: dict, seed: int, path: str = ":memory:",
                known_topics: list | None = None) -> MockWorld:
    """Строит мок-системы из календаря релизов и ground_truth генератора.

    Что кладём (все числа и названия условные):
      * релизы 1:1 из календаря;
      * инциденты: фоновые (плановые работы, медленный сервис) плюс у части регрессий (~40%) инцидент мониторинга;
      * трекер: ~30 задач по темам продуктов; у ~30% регрессий уже есть задача «в работе» (создана через 1-3 часа после начала);
      * база знаний: 15 статей.
    known_topics: принудительно завести задачу «в работе» для этих тем (для демо-сцены «известно, уже чиним»).
    """
    w = MockWorld(path)
    rng = random.Random(seed * 7919 + 13)
    for r in releases:
        w.add_release(r)
    start = datetime.fromisoformat(ground_truth["start"])
    events = ground_truth["events"]
    topic_by_id = {t[0]: t for t in T.TOPICS}

    # фоновые инциденты
    n = 0
    for product in T.PRODUCTS:
        for _ in range(3):
            n += 1
            s = start + timedelta(days=rng.uniform(1, 26), hours=rng.uniform(0, 8))
            title = rng.choice(["Плановые работы на сервере базы данных", "Повышенное время ответа внешнего платёжного шлюза",
                                "Кратковременная недоступность CDN", "Обновление сертификатов"])
            w.add_incident(f"INC-{n:03d}", product, s, s + timedelta(minutes=rng.randint(20, 90)), title, "minor")

    # регрессии: инцидент и задача с вероятностью
    rk = 0
    for e in events:
        if e["type"] != "regression":
            continue
        onset = datetime.fromisoformat(e["onset_ts"])
        topic = topic_by_id[e["topic"]]
        if rng.random() < 0.4:
            n += 1
            w.add_incident(f"INC-{n:03d}", e["product"], onset + timedelta(minutes=rng.randint(5, 30)), None,
                           f"Рост ошибок сервиса: {topic[2].lower()}", "major")
        if rng.random() < 0.3 or (known_topics and e["topic"] in known_topics):
            rk += 1
            w.add_task(f"RAD-{100 + rk}", e["product"], f"{topic[2]} (сообщения от пользователей)", "in_progress",
                       onset + timedelta(hours=rng.uniform(1, 3)), "Заведено дежурным, есть воспроизведение, исправление в работе.")
    # обычные задачи бэклога
    k = 0
    for product in T.PRODUCTS:
        own = [t for t in T.TOPICS if t[1] == product]
        for t in rng.sample(own, min(len(own), 6)):
            k += 1
            status = rng.choice(["open", "in_progress", "done", "done", "wontfix"])
            created = start + timedelta(days=rng.uniform(-40, 20))
            suffix = rng.choice(["у части пользователей", "после обновления", "на старых устройствах", "по обращениям поддержки"])
            w.add_task(f"RAD-{k:03d}", product, f"{t[2]} {suffix}", status, created, f"Тема: {t[2]}.")
    for tid, product, title, body in KB_ARTICLES:
        w.db.execute("INSERT OR REPLACE INTO kb VALUES(?,?,?,?)", (f"KB-{tid}", product, title, body))
    w.commit()
    return w
