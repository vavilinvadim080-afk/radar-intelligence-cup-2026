"""Инструменты агента. У каждого: описание, JSON-схема аргументов, ограничение.

Чего у агента НЕТ вообще (и не появится): закрыть или удалить задачу, написать пользователю, изменить порог.
Если модель просит такое действие, `Toolbox.run` отвечает отказом и пишет попытку в журнал (`tool_denied`).
create_ticket_draft делает только черновик, notify_team только кладёт сообщение в очередь подтверждения.
Реальная отправка в Telegram происходит ПОСЛЕ кнопки человека и только при заданных TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

import jsonschema

# имена, которые модель может «выдумать» под влиянием инъекции; все запрещены
FORBIDDEN_HINTS = ("close", "delete", "remove", "drop", "set_status", "resolve", "send_to_user", "message_user",
                   "email_user", "reply_user", "write_user", "update_threshold", "set_threshold", "mark_noise", "merge")


# Объяснения для модели. Поведение и права инструментов они не меняют: только текст в промпте.
# Формат: аргументы / когда звать / что вернёт / права.
TOOL_GUIDES = {
    "get_cluster": "get_cluster(burst_id, limit=8). Когда: нужно больше примеров обращений, чем во входе. "
                   "Вернёт: примеры обращений (id и замаскированный текст), платформы, версии, число людей за каждые 30 минут последних 12 часов. Права: чтение, только текущая карточка.",
    "get_releases": "get_releases(product, days=7). Когда: есть подозреваемый релиз или нужно понять, что менялось перед всплеском. "
                    "Вернёт: список релизов с id вида REL-..., версией, временем выхода и changelog. Права: чтение.",
    "get_incidents": "get_incidents(product, days=2). Когда: жалобы похожи на сбой инфраструктуры (не работает всё сразу, на всех версиях). "
                     "Вернёт: инциденты мониторинга с id вида INC-... и временем. Права: чтение.",
    "search_tracker": "search_tracker(query). Когда: ВСЕГДА первым шагом, с коротким описанием симптома. "
                      "Вернёт: похожие задачи (key вида RAD-..., title, status, score от 0 до 1); score от 0.4 и status open или in_progress "
                      "значит, что задача уже есть. Права: чтение.",
    "search_kb": "search_kb(query). Когда: нужно проверить, известна ли проблема и есть ли обходной путь. "
                 "Вернёт: статьи базы знаний (id вида KB-..., title, body). Права: чтение.",
    "create_ticket_draft": "create_ticket_draft(title, description, severity). Когда: только если уверен, что задачи ещё нет и она нужна. "
                           "Вернёт: номер черновика со статусом «ждёт подтверждения». Права: черновик, задачу создаёт только человек.",
    "notify_team": "notify_team(message). Когда: нужно коротко предупредить команду продукта. "
                   "Вернёт: номер сообщения в очереди. Права: очередь, отправляет только человек; пользователям писать нельзя.",
}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    restriction: str
    func: Callable
    guide: str = ""      # объяснение для модели: когда звать и что вернёт (попадает в промпт v2)


class Toolbox:
    """Набор инструментов для одной карточки. ctx хранит то, что агент менять не может."""

    def __init__(self, world, audit, now: datetime, burst_id: str, product: str, allowed_ids: list,
                 cluster_source: Callable | None = None):
        self.world, self.audit, self.now = world, audit, now
        self.burst_id, self.product = burst_id, product
        self.allowed_ids = set(allowed_ids)      # id обращений, на которые карточка вправе ссылаться
        self.seen_ids: set = set()               # id, которые инструменты реально вернули (для валидатора)
        self.seen_texts: dict = {}               # id обращения -> замаскированный текст, который вернул get_cluster (проверка покрытия утверждений)
        self.cluster_source = cluster_source
        self.calls: list = []
        self.tools = {t.name: t for t in self._make()}
        for t in self.tools.values():
            t.guide = TOOL_GUIDES.get(t.name, t.description)

    def guide(self) -> str:
        """Список инструментов для промпта: по строке на инструмент, в конце что запрещено."""
        lines = [f"- {t.guide}" for t in self.tools.values()]
        lines.append("Запрещено и не существует: закрыть или удалить задачу, написать пользователю, менять пороги.")
        return "\n".join(lines)

    # ---------- реализация
    def _get_cluster(self, burst_id: str, limit: int = 8):
        if burst_id != self.burst_id:
            return {"error": "доступен только кластер текущей карточки"}
        data = self.cluster_source(burst_id, limit) if self.cluster_source else {"examples": []}
        for e in data.get("examples", []):
            self.seen_ids.add(e["id"])
            if e.get("text"):
                self.seen_texts[e["id"]] = e["text"]
        return data

    def _get_releases(self, product: str, days: int = 7):
        rows = self.world.get_releases(product, self.now - timedelta(days=days), self.now)
        for r in rows:
            self.seen_ids.add(r["id"])
        return {"releases": rows}

    def _get_incidents(self, product: str, days: int = 2):
        rows = self.world.get_incidents(product, self.now - timedelta(days=days), self.now)
        for r in rows:
            self.seen_ids.add(r["id"])
        return {"incidents": rows}

    def _search_tracker(self, query: str, product: str | None = None):
        rows = self.world.search_tracker(query, product or self.product, self.now)
        for r in rows:
            self.seen_ids.add(r["key"])
        return {"tasks": [{k: r[k] for k in ("key", "title", "status", "created_at", "score")} for r in rows]}

    def _search_kb(self, query: str):
        rows = self.world.search_kb(query, self.product)
        for r in rows:
            self.seen_ids.add(r["id"])
        return {"articles": [{k: r[k] for k in ("id", "title", "body")} for r in rows]}

    def _create_ticket_draft(self, title: str, description: str, severity: str = "medium"):
        d = self.world.new_draft(self.burst_id, self.product, title[:200], description[:2000], severity, self.now)
        return {"draft_id": d["id"], "status": d["status"], "note": "задача не создана, ждёт подтверждения человека"}

    def _notify_team(self, message: str):
        o = self.world.new_outbox(self.burst_id, self.product, message[:1000], self.now)
        return {"outbox_id": o["id"], "status": o["status"], "note": "сообщение в очереди, отправит человек"}

    def _make(self) -> list:
        S = lambda props, req: {"type": "object", "properties": props, "required": req, "additionalProperties": False}
        return [
            Tool("get_cluster", "Обращения кластера текущей карточки (замаскированные), динамика по часам, платформы, версии.",
                 S({"burst_id": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["burst_id"]),
                 "только текущая карточка, только замаскированный текст", self._get_cluster),
            Tool("get_releases", "Релизы продукта за последние дни с changelog.",
                 S({"product": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 30}}, ["product"]),
                 "только чтение", self._get_releases),
            Tool("get_incidents", "Инциденты мониторинга по продукту за последние дни.",
                 S({"product": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 14}}, ["product"]),
                 "только чтение", self._get_incidents),
            Tool("search_tracker", "Поиск существующих задач в трекере по тексту. Вызывай до создания черновика.",
                 S({"query": {"type": "string", "maxLength": 300}, "product": {"type": "string"}}, ["query"]),
                 "только чтение", self._search_tracker),
            Tool("search_kb", "Поиск в базе знаний: известные проблемы и обходные пути.",
                 S({"query": {"type": "string", "maxLength": 300}}, ["query"]),
                 "только чтение", self._search_kb),
            Tool("create_ticket_draft", "Черновик задачи. Ничего не создаёт: черновик подтверждает человек кнопкой.",
                 S({"title": {"type": "string", "maxLength": 200}, "description": {"type": "string", "maxLength": 2000},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]}}, ["title", "description"]),
                 "создаёт только черновик со статусом «ждёт подтверждения»", self._create_ticket_draft),
            Tool("notify_team", "Сообщение команде продукта. Попадает в очередь подтверждения, пользователям написать нельзя.",
                 S({"message": {"type": "string", "maxLength": 1000}}, ["message"]),
                 "адресат один: команда продукта; отправка только после кнопки человека", self._notify_team),
        ]

    # ---------- выполнение
    def specs(self) -> list:
        """Описание для промпта и для function calling Ollama."""
        return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
                for t in self.tools.values()]

    def run(self, name: str, args: dict) -> dict:
        """Единственный вход для выполнения. Неизвестное или запрещённое имя: отказ, запись в журнал."""
        if name not in self.tools:
            forbidden = any(h in str(name).lower() for h in FORBIDDEN_HINTS)
            self.audit.log("tool_denied", "agent", {"name": name, "args": args, "forbidden_action": forbidden},
                           self.burst_id, self.now)
            self.calls.append({"name": name, "denied": True})
            return {"error": f"инструмента {name!r} нет; закрывать, удалять и писать пользователям агенту запрещено"}
        tool = self.tools[name]
        try:
            jsonschema.validate(args, tool.schema)
        except jsonschema.ValidationError as e:
            self.audit.log("tool_denied", "agent", {"name": name, "args": args, "schema_error": e.message[:200]},
                           self.burst_id, self.now)
            self.calls.append({"name": name, "denied": True})
            return {"error": f"аргументы не прошли схему: {e.message[:120]}"}
        result = tool.func(**args)
        self.audit.log("tool_call", "agent", {"name": name, "args": args, "result": result}, self.burst_id, self.now)
        self.calls.append({"name": name, "denied": False})
        return result


def telegram_enabled() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send_telegram(text: str) -> str:
    """Реальная отправка. Вызывается только из обработчика кнопки человека и только при заданных env."""
    if not telegram_enabled():
        return "выключено (нет TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)"
    import httpx
    url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage"
    r = httpx.post(url, json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text[:4000]}, timeout=15)
    return "telegram" if r.status_code == 200 else f"ошибка {r.status_code}"
