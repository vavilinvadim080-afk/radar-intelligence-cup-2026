"""Автоответ на повторные обращения по известному инциденту.

Решение: один клик человека на инцидент, дальше шаблонные ответы без людей. Без Flask/FastAPI и без LLM, чтобы тесты
вызывали методы напрямую (как app/state.py).

Что делает код:
  * approve(): дежурный нажал «включить автоответ» на карточке инцидента. До этого клика ответы не уходят;
  * handle_ticket(): новое обращение, которое пайплайн отнёс к инциденту. Либо шаблонный ответ (app/replies.py), либо
    «оператору» с причиной: инцидент не подтверждён, автоответ выключен, инцидент закрыт, маркер инъекции, низкое
    сходство с кластером, дневной лимит;
  * propose_fixed() / approve_fixed(): после подтверждения фикса (radar/verify.py) или кнопки «починили» формируется
    рассылка «починили» всем, кто писал по инциденту. Уходит в мок-очередь, только id обращений, контактов в Радаре нет;
    сама рассылка формируется только после клика человека.
Что НЕ делает: не читает контакты, не отправляет сообщения (очередь забирает внешняя система), не использует текст
обращения в ответе (текст нужен только для проверок: инъекция и сходство; в очередь и в аудит он не попадает).
Каждое действие пишется в аудит (app/audit.py).
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

from . import replies, textsim
from .agent import injection_in

ASSUMPTIONS = Path(__file__).resolve().parents[1] / "shared" / "autoreply-assumptions.json"
Q_STATUS = "в очереди на отправку вне Радара"


def _env_num(name: str, default, cast):
    try:
        return cast(os.environ.get(name, default))
    except ValueError:
        return default


class AutoReply:
    DEFAULT_DAILY_LIMIT = 200       # автоответов на один инцидент в сутки (потоковые сутки)
    DEFAULT_MIN_SIM = 0.5           # ниже сходства с кластером ответ не уходит, обращение идёт оператору
    DEFAULT_GLOBAL_DAILY_LIMIT = 600   # автоответов в сутки по всем инцидентам вместе (3 инцидента по 200): страхует от десятка включённых инцидентов

    def __init__(self, audit, daily_limit: int | None = None, min_sim: float | None = None, global_daily_limit: int | None = None):
        self.audit = audit
        self.global_daily_limit = (global_daily_limit if global_daily_limit is not None
                                   else _env_num("RADAR_AUTOREPLY_GLOBAL_DAILY_LIMIT", self.DEFAULT_GLOBAL_DAILY_LIMIT, int))
        self.global_days: dict = {}     # сутки -> сколько автоответов ушло по всем инцидентам
        self._global_audited: set = set()
        self.daily_limit = daily_limit if daily_limit is not None else _env_num("RADAR_AUTOREPLY_DAILY_LIMIT", self.DEFAULT_DAILY_LIMIT, int)
        self.min_sim = min_sim if min_sim is not None else _env_num("RADAR_AUTOREPLY_MIN_SIM", self.DEFAULT_MIN_SIM, float)
        self.incidents: dict = {}
        self.queue: list = []           # мок-очередь автоответов: [{id, incident, ticket_id, text, status, ts}]
        self.mailings: list = []        # мок-очередь рассылок «починили»
        self.operator: list = []        # обращения, отправленные оператору: [{ticket_id, incident, reason, ts}]
        self.assumptions = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ клики человека
    def approve(self, incident: str, product: str, product_name: str, started_at: datetime, now: datetime,
                who: str = "дежурный", seed_ticket_ids: list | None = None) -> dict:
        """Включает автоответ. seed_ticket_ids: обращения, которые уже видны в карточке (они тоже получат «починили»)."""
        replies.render("known", product=product_name, started_at=started_at, incident=incident)   # поля годятся для шаблона, иначе ValueError
        st = self.incidents.get(incident)
        if st and st["state"] == "fixed":
            raise ValueError("инцидент закрыт рассылкой «починили»")
        if st is None:
            st = self.incidents[incident] = {"product": product, "product_name": product_name, "started_at": started_at, "state": "on",
                                             "approved_at": now, "writers": [], "replied": 0, "operator": 0, "days": {}, "proposal": None,
                                             "mailing": None, "excluded": []}
            for t in seed_ticket_ids or []:
                if t not in st["writers"]:
                    st["writers"].append(t)
        st["state"] = "on"
        self.audit.log("autoreply_approved", "human", {"who": who, "incident": incident, "daily_limit": self.daily_limit,
                                                       "min_sim": self.min_sim}, incident, now)
        return self.view(incident)

    def disable(self, incident: str, now: datetime, who: str = "дежурный") -> dict:
        st = self._st(incident)
        if st["state"] == "on":
            st["state"] = "off"
        self.audit.log("autoreply_disabled", "human", {"who": who, "incident": incident}, incident, now)
        return self.view(incident)

    # ------------------------------------------------------------ обращения
    def active(self, incident: str | None = None) -> bool:
        if incident is not None:
            return self.incidents.get(incident, {}).get("state") == "on"
        return any(s["state"] == "on" for s in self.incidents.values())

    def handle_ticket(self, incident: str, ticket_id: str, text: str, now: datetime, sim: float | None = None,
                      reference_texts: list | None = None, extra_texts: list | None = None) -> dict:
        """Решение по одному обращению. sim: сходство с кластером (0..1) от пайплайна; нет его, считаем по reference_texts
        (замаскированные примеры кластера). extra_texts: другие виды текста того же обращения (исходный) только для проверки на инъекцию."""
        from radar_app.reply_safety import unsafe_reason
        from radar_app.semantic import compatible_intent,compatible_object
        previous=next((x for x in self.queue if x['ticket_id']==ticket_id),None)
        if previous is not None:
            return {'action':'duplicate','reason':'обращение уже в очереди','ticket_id':ticket_id,
                    'reply':previous['text'],'queue_id':previous['id']}
        st = self.incidents.get(incident)
        reason = None
        if st is None:
            reason = "инцидент не подтверждён человеком"
        elif st["state"] == "off":
            reason = "автоответ выключен"
        elif st["state"] == "fixed":
            reason = "инцидент закрыт рассылкой «починили»"
        elif injection_in([text] + list(extra_texts or [])):
            reason = "маркер инъекции"
        elif unsafe_reason('\n'.join([text]+list(extra_texts or []))):
            reason = unsafe_reason('\n'.join([text]+list(extra_texts or [])))
        elif st.get('proposal'):
            reason = 'фикс подтверждён; новое обращение требует проверки человеком'
        elif reference_texts and any(not compatible_intent(text,r) or not compatible_object(text,r) for r in reference_texts):
            reason = 'противоречие направления или предмета обращения'
        else:
            if sim is None and reference_texts:
                sim = max((textsim.cosine(text, r) for r in reference_texts), default=0.0)
            if isinstance(sim,bool) or not isinstance(sim,(int,float)) or not math.isfinite(sim) or not 0<=sim<=1 or sim < self.min_sim:
                reason = "низкое сходство с кластером"
            elif st["days"].get(f"{now:%Y-%m-%d}", 0) >= self.daily_limit:
                reason = "дневной лимит"
            elif self.global_days.get(f"{now:%Y-%m-%d}", 0) >= self.global_daily_limit:
                reason = "глобальный дневной лимит"
                if f"{now:%Y-%m-%d}" not in self._global_audited:   # в журнал защиты один раз за сутки, каждый отказ пишет autoreply_escalated
                    self._global_audited.add(f"{now:%Y-%m-%d}")
                    self.audit.log("antispam", "code", {"rule": "autoreply_global_limit", "limit": self.global_daily_limit, "day": f"{now:%Y-%m-%d}"},
                                   incident, now)
        if reason:
            return self._to_operator(incident, st, ticket_id, reason, now)
        day = f"{now:%Y-%m-%d}"
        st["days"][day] = st["days"].get(day, 0) + 1
        self.global_days[day] = self.global_days.get(day, 0) + 1
        st["replied"] += 1
        if ticket_id not in st["writers"]:
            st["writers"].append(ticket_id)
        reply = replies.render("known", product=st["product_name"], started_at=st["started_at"], incident=incident,
                               variant=replies.variant_for(ticket_id))
        item = {"id": f"AR-{len(self.queue) + 1:04d}", "incident": incident, "ticket_id": ticket_id, "text": reply, "status": Q_STATUS,
                "ts": now.isoformat(timespec="minutes")}
        self.queue.append(item)
        self.audit.log("autoreply_sent", "code", {"incident": incident, "ticket_id": ticket_id, "queue_id": item["id"], "kind": "known"},
                       incident, now)
        return {"action": "autoreply", "reason": "", "ticket_id": ticket_id, "reply": reply, "queue_id": item["id"]}

    def _to_operator(self, incident, st, ticket_id, reason, now) -> dict:
        self.operator.append({"ticket_id": ticket_id, "incident": incident, "reason": reason, "ts": now.isoformat(timespec="minutes")})
        if st is not None:
            st["operator"] += 1
            if reason not in {'дневной лимит','глобальный дневной лимит'}:
                if ticket_id not in st['excluded']:st["excluded"].append(ticket_id)
            elif ticket_id not in st["writers"]:
                st["writers"].append(ticket_id)            # написал по теме, ответ не ушёл из-за лимита: «починили» получит
        self.audit.log("autoreply_escalated", "code", {"incident": incident, "ticket_id": ticket_id, "reason": reason}, incident, now)
        return {"action": "operator", "reason": reason, "ticket_id": ticket_id, "reply": None}

    # ------------------------------------------------------------ «починили»
    def propose_fixed(self, incident: str, now: datetime, source: str = "verify") -> dict | None:
        """Предложение рассылки «починили» после подтверждения фикса. Ничего не отправляет: ждёт клика человека."""
        st = self.incidents.get(incident)
        if st is None or st["state"] == "fixed" or st["proposal"]:
            return st["proposal"] if st else None
        st["proposal"] = {"state": "ждёт человека", "source": source, "at": now.isoformat(timespec="minutes")}   # число адресатов живёт в view(): растёт, пока идут обращения
        self.audit.log("autoreply_fixed_proposed", "code", {"incident": incident, "source": source, "recipients": len(st["writers"])}, incident, now)
        return st["proposal"]

    def approve_fixed(self, incident: str, now: datetime, who: str = "дежурный", source: str = "verify") -> dict:
        """Клик человека: формируется рассылка «починили» (id обращений, без контактов). Автоответ по инциденту после этого закрыт."""
        st = self._st(incident)
        if st["state"] == "fixed":
            raise ValueError("рассылка по инциденту уже сформирована")
        if source == "verify" and not st["proposal"]:
            raise ValueError("нет предложения от проверки фикса; для ручной отметки source='человек'")
        ids = [t for t in st["writers"] if t not in st["excluded"]]
        m = {"id": f"ML-{len(self.mailings) + 1:04d}", "incident": incident, "kind": "fixed", "ticket_ids": ids,
             "text": replies.render("fixed", product=st["product_name"], started_at=st["started_at"], incident=incident),
             "status": Q_STATUS, "source": source, "approved_by": who, "ts": now.isoformat(timespec="minutes")}
        self.mailings.append(m)
        st["state"], st["mailing"] = "fixed", m["id"]
        if st["proposal"]:
            st["proposal"]["state"] = "подтверждено человеком"
        self.audit.log("autoreply_fixed_approved", "human", {"who": who, "incident": incident, "mailing": m["id"], "recipients": len(ids),
                                                             "source": source, "ticket_ids": ids}, incident, now)
        return m

    # ------------------------------------------------------------ данные для страницы
    def _st(self, incident: str) -> dict:
        if incident not in self.incidents:
            raise ValueError("автоответ по инциденту не включали")
        return self.incidents[incident]

    def view(self, incident: str, now: datetime | None = None) -> dict:
        st = self.incidents.get(incident)
        if st is None:
            return {"state": "не включён", "replied": 0, "to_operator": 0, "daily_limit": self.daily_limit, "proposal": None, "mailing": None}
        day = f"{now:%Y-%m-%d}" if now else max(st["days"], default=None)
        names = {"on": "включён", "off": "выключен", "fixed": "закрыт рассылкой «починили»"}
        return {"state": names[st["state"]], "on": st["state"] == "on", "approved_at": st["approved_at"].isoformat(timespec="minutes"),
                "replied": st["replied"], "to_operator": st["operator"], "today": st["days"].get(day, 0) if day else 0,
                "daily_limit": self.daily_limit, "recipients": len([t for t in st["writers"] if t not in st["excluded"]]),
                "proposal": st["proposal"], "mailing": st["mailing"],
                "recent_replies": [r for r in self.queue if r["incident"] == incident][-3:]}

    def stats(self) -> dict:
        n = sum(s["replied"] for s in self.incidents.values())
        reasons: dict = {}
        for o in self.operator:
            reasons[o["reason"]] = reasons.get(o["reason"], 0) + 1
        a = self.assumptions
        return {"autoreplied": n, "to_operator": len(self.operator), "by_reason": reasons,
                "incidents_enabled": sum(1 for s in self.incidents.values() if s["state"] == "on"),
                "mailings": len(self.mailings), "mailing_recipients": sum(len(m["ticket_ids"]) for m in self.mailings),
                "daily_limit": self.daily_limit, "global_daily_limit": self.global_daily_limit, "min_sim": self.min_sim,
                "assumed_minutes_saved": {"low": round(n * a["minutes_saved_low"]["value"], 1), "high": round(n * a["minutes_saved_high"]["value"], 1),
                                          "is_assumption": True, "source": a["source"]}}
