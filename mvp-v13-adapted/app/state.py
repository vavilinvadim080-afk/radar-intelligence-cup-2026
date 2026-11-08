"""Состояние демо: проигрыватель записи, карточки, решения человека. Без FastAPI, чтобы тесты вызывали методы напрямую.

Что здесь решает человек, а что агент:
  * агент только пишет карточку (agent.write_card) и кладёт черновики и сообщения в очередь;
  * кнопки «наш баг», «известно», «шум», «подтвердить закрытие», «отправить» это методы decide/confirm_close/approve_outbox,
    их вызывает только человек через HTTP.
"""
from __future__ import annotations

import difflib
import json
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

from radar.config import Config
from radar.models import to_plain

from . import agent as agent_mod
from . import security
from . import textsim
from .audit import Audit
from .autoreply import AutoReply
from .mocks import build_world
from .replay import BIN_MIN, Replay, card_plan, cluster_source_for
from .tools import send_telegram

RECORDED = Path(__file__).resolve().parents[1] / "demo" / "replay.json"
# сходство нового обращения с примерами карточки, с которого демо считает его «отнесённым к кластеру» (в записи прогона нет номера
# кластера по каждому обращению, а radar/pipeline.py не меняем). Строже порога автоответа (AutoReply.min_sim) оно быть не может.
RELATE_SIM = float(os.environ.get("RADAR_AUTOREPLY_RELATE_SIM", "0.3"))
KEY_FILE = RECORDED.parent / ".replay-hmac.key"   # ключ HMAC, если не задан RADAR_HMAC_KEY (создаётся при первом запуске)

# точки входа в сцены демо (потоковое время демо-набора seed 777)
SCENES = {
    1: datetime(2026, 3, 9, 13, 0),    # поток и маскирование
    2: datetime(2026, 3, 9, 14, 30),   # релиз 2.8.0 в 14:45, карточка около 18:00
    3: datetime(2026, 3, 9, 18, 30),   # человек в контуре (затем azs 11.03 и portal 12.03)
    4: datetime(2026, 3, 12, 16, 30),  # петля: фикс, инъекция, проверка фикса
}


class Radar:
    def __init__(self, rep: Replay, mode: str = "replay", llm=None, sync_agent: bool = False, db_path: str = ":memory:",
                 audit_path: str | None = None):
        self.rep = rep
        self.mode = mode                      # replay (записанные ответы) | live (Ollama сейчас) | template (без LLM)
        self.llm = llm
        self.sync_agent = sync_agent
        self.db_path = db_path
        # журнал аудита: файл SQLite, если задан audit_path или RADAR_AUDIT_DB; иначе в памяти (демо, сбрасывается вместе с демо)
        self.audit_path = audit_path or os.environ.get("RADAR_AUDIT_DB") or ":memory:"
        self.audit = None
        self.lock = threading.RLock()
        self.cluster_source = cluster_source_for(rep)
        self.plan = card_plan(rep)
        self.recorded = security.load_recorded(RECORDED, KEY_FILE)   # подпись HMAC проверяется; подмена файла: ReplayTamperedError
        self.base_cfg = Config()
        # бюджет вызовов модели (скользящий час реального времени): на продукт и в целом; исчерпан: шаблонная карточка без модели
        self.llm_budget = security.LlmBudget(int(os.environ.get("RADAR_LLM_PER_PRODUCT_HOUR", "6")), int(os.environ.get("RADAR_LLM_TOTAL_HOUR", "20")))
        self.listeners: list = []             # функции, которым отдаём сообщения (SSE)
        self.reset(SCENES[1])

    # ------------------------------------------------------------ сброс и сообщения
    def reset(self, to: datetime | None = None):
        with self.lock:
            from .mocks import build_world
            self.world = build_world([_rel(r) for r in self.rep.releases], self.rep.ground_truth, self.rep.seed,
                                     self.db_path if self.db_path == ":memory:" else ":memory:", known_topics=["portal_vpn"])
            self.audit_dsn = os.environ.get("RADAR_AUDIT_DSN") or None   # PostgreSQL вместо SQLite (README, «Журнал аудита в PostgreSQL»)
            if self.audit is None or (self.audit_path == ":memory:" and not self.audit_dsn):
                self.audit = Audit(self.audit_path, dsn=self.audit_dsn)   # файловый журнал и PostgreSQL сброс демо не стирает
            self.now = to or SCENES[1]
            self.cards: dict = {}
            self.overrides: dict = {}         # product -> {"z_threshold", "min_users"}, меняет кнопка «шум»
            self.threshold_log: list = []
            self.suppressed: list = []
            self.events_log: list = []        # лента событий для страницы
            self.ev_i = 0
            self.plan_i = 0
            self._closed_at: dict = {}        # burst_id -> когда пайплайн закрыл всплеск (после этого поток темы уже не инцидент)
            self.tk_i = 0                     # указатель в rep.tickets: обращения до него уже разобраны автоответом
            self.autoreply = AutoReply(self.audit)
            if os.environ.get('RADAR_AUTOREPLY_MATCHER')=='semantic':
                if os.environ.get('RADAR_ALLOW_EXPERIMENTAL')!='1':
                    raise ValueError('Local semantic autoreply failed control; experimental override required')
                from radar_app.semantic_runtime import components
                _,matcher=components()
                self.autoreply.min_sim=matcher.threshold
            self._refs: dict = {}             # card_id -> профили 4-грамм замаскированных примеров карточки (для сходства)
            self._semantic_refs: dict = {}
            self._catch_up(self.now)
            self.audit.log("system", "code", {"msg": "сброс демо", "to": self.now.isoformat(), "mode": self.mode}, None, self.now)
            self.emit({"type": "refresh"})

    def emit(self, msg: dict):
        for f in list(self.listeners):
            try:
                f(msg)
            except Exception:
                pass

    def _catch_up(self, to: datetime):
        """События до `to` применяем молча: карточки появляются, но без работы агента (берём записанное или шаблон)."""
        self.ev_i = 0
        self.plan_i = 0
        self.tk_i = 0
        self._process(to, silent=True)

    # ------------------------------------------------------------ ход времени
    def advance(self, to: datetime):
        """Двигает потоковое время вперёд и применяет события, которые случились."""
        with self.lock:
            if to <= self.now:
                return
            self._process(to, silent=False)
            self.now = to

    def jump(self, to: datetime):
        with self.lock:
            if to < self.now:
                self.reset(to)
                return
            self._process(to, silent=True)
            self.now = to
            self.emit({"type": "refresh"})

    def _process(self, to: datetime, silent: bool):
        ev = self.rep.events
        # Restore in time order: a fix must not be applied before its card exists.
        while True:
            e = ev[self.ev_i] if self.ev_i < len(ev) and ev[self.ev_i]["ts"] <= to else None
            p = self.plan[self.plan_i] if self.plan_i < len(self.plan) and self.plan[self.plan_i]["ts"] <= to else None
            if e is None and p is None:
                break
            if e is not None and (p is None or e["ts"] <= p["ts"]):
                self._on_event(e, silent)
                self.ev_i += 1
            else:
                self._open_card(p, silent)
                self.plan_i += 1
        before = (len(self.autoreply.queue), len(self.autoreply.operator))
        self._route_tickets(to)
        if not silent and before != (len(self.autoreply.queue), len(self.autoreply.operator)):
            self.emit({"type": "autoreply_updated"})

    # ------------------------------------------------------------ события конвейера
    def _threshold(self, product: str) -> tuple:
        o = self.overrides.get(product, {})
        return o.get("z_threshold", self.base_cfg.z_threshold), o.get("min_users", self.base_cfg.min_users)

    def _on_event(self, e: dict, silent: bool):
        b = e["burst"]
        self.events_log.append({"ts": e["ts"].isoformat(timespec="minutes"), "kind": e["kind"], "burst_id": b.id,
                                "product": b.product, "alert": b.alert, "z": round(b.z, 1), "reason": b.alert_reason})
        self.events_log = self.events_log[-60:]
        if e["kind"] == "closed":
            self._closed_at[b.id] = e["ts"]
        if e["kind"] == "fix_checked" and b.fix_check:
            self._on_fix(e, silent)

    def _open_card(self, p: dict, silent: bool):
        ci = p["ci"]
        # кнопка «шум» уже подняла порог: такая тревога дежурному не уходит
        if p["kind"] == "alert":
            b = next(x["burst"] for x in self.rep.events if x["burst"].id == p["card_id"] and x["kind"] == "opened")
            zt, mu = self._threshold(b.product)
            if b.product in self.overrides and (b.z < zt or b.total_unique_users < mu):
                self.suppressed.append({"burst_id": b.id, "product": b.product, "z": round(b.z, 1), "threshold": zt})
                self.audit.log("system", "code", {"msg": "тревога подавлена порогом продукта", "z": b.z, "threshold": zt}, b.id, p["ts"])
                return
        tier, susp = "", None
        if p["kind"] == "alert":
            tier = b.tier or "page"                       # v3: page будит, quiet это тихая карточка в канал команды
            susp = getattr(b, "suspicion", None)          # защита от накрутки: пометка, почему карточка тихая (radar/antispam.py)
        card = {"id": p["card_id"], "kind": p["kind"], "tier": tier, "product": ci.product, "product_name": ci.product_name,
                "burst_ids": list(p["burst_ids"]), "created_sim": p["ts"].isoformat(timespec="minutes"),
                "decision": None, "task_key": None, "close_proposal": None, "fix": None, "agent": None, "pending": True,
                "suspicion": susp}
        quiet_susp = bool(susp) and tier == "quiet"
        card["no_model"] = "подозрение на накрутку: карточка собрана шаблоном, модель не вызывается" if quiet_susp else None
        self.cards[card["id"]] = card
        self.emit({"type": "card_opened", "card_id": card["id"], "product": card["product"]})
        if silent and (not self._recorded(card["id"]) or card.get("no_model")):
            self._finish_card(card, self._write_card(card, ci, force_template=True))
            return
        if self.mode == "live" and not silent and not self.sync_agent:
            threading.Thread(target=self._agent_thread, args=(card, ci), daemon=True).start()
        else:
            self._agent_now(card, ci)

    def _recorded(self, card_id: str):
        return self.recorded.get("results", {}).get(card_id)

    def _write_card(self, card: dict, ci, force_template: bool, llm=None) -> dict:
        """Единственное место, где состояние зовёт агента. Модель вызывается, только если нет причин её не звать:
        карточка не помечена накруткой и бюджет вызовов в час не исчерпан (иначе шаблон, причина в fallback_reason и в журнале аудита)."""
        why = None
        if card.get("no_model"):
            force_template, why = True, card["no_model"]
        elif not force_template:
            ok, reason = self.llm_budget.try_acquire(ci.product)
            if not ok:
                force_template, why = True, "бюджет вызовов модели исчерпан: " + reason
                self.audit.log("antispam", "code", {"rule": "llm_budget", "product": ci.product, "reason": reason}, card["id"], ci.detected_at)
        res = agent_mod.write_card(ci, self.world, self.audit, self.cluster_source, llm=llm, force_template=force_template)
        if why:
            res["fallback_reason"] = why
        return res

    def _agent_now(self, card: dict, ci):
        rec = self._recorded(card["id"]) if self.mode == "replay" else None
        if rec and not card.get("no_model"):
            for row in rec.get("audit", []):   # журнал записанного вызова: видно, что делал агент
                self.audit.log(row["kind"], row["actor"], row["payload"], card["id"], ci.detected_at)
            res = {k: v for k, v in rec.items() if k != "audit"}
            res["recorded"] = True
            self._finish_card(card, res)
        else:
            res = self._write_card(card, ci, force_template=(self.mode in ("template", "replay")), llm=self.llm)   # в режиме записи без записи: шаблон, LLM не зовём
            self._finish_card(card, res)

    def _agent_thread(self, card, ci):
        res = self._write_card(card, ci, force_template=False, llm=self.llm)
        with self.lock:
            self._finish_card(card, res)

    def _finish_card(self, card: dict, res: dict):
        card["agent"] = res
        card["pending"] = False
        card["facts"] = res["facts"]
        self.emit({"type": "refresh"})

    def _card_of_burst(self, bid: str):
        return next((c for c in self.cards.values() if bid in c["burst_ids"]), None)

    # ------------------------------------------------------------ проверка фикса
    def _on_fix(self, e: dict, silent: bool):
        b, fc = e["burst"], e["burst"].fix_check
        card = self._card_of_burst(b.id)
        if not card:
            return
        if card["fix"] and card["fix"]["verdict"] == "confirmed":
            return
        res = {"fix_version": fc.fix_version, "fix_ts": fc.fix_ts.isoformat(timespec="minutes"), "verdict": fc.verdict,
               "residual_share": fc.residual_share, "residual_ci_high": fc.residual_ci_high, "n_before": fc.n_before,
               "n_after": fc.n_after, "checked_at": e["ts"].isoformat(timespec="minutes")}
        card["fix"] = res
        if fc.verdict == "confirmed":
            self.autoreply.propose_fixed(card["id"], e["ts"])      # только предложение: рассылку «починили» запускает клик человека
        self._sync_close_proposal(card)
        self.audit.log("system", "code", {"msg": "проверка фикса", **res, "proposal": bool(card["close_proposal"])}, card["id"], e["ts"])
        self.emit({"type": "fix_checked", "card_id": card["id"], "verdict": fc.verdict})
        self.emit({"type": "refresh"})

    def _sync_close_proposal(self, card: dict):
        fc = card.get("fix")
        if fc and fc["verdict"] == "confirmed" and card["task_key"] and not card["close_proposal"]:
            r, hi = fc["residual_share"], fc["residual_ci_high"]
            card["close_proposal"] = {
                "state": "ждёт человека", "task_key": card["task_key"],
                "text": (f"Подтверждено по данным, проверьте: после фикса {fc['fix_version']} тема вернулась к уровню до релиза "
                         f"(остаток всплеска {0 if r is None else round(r * 100)}%, верхняя граница {0 if hi is None else round(hi * 100)}%). "
                         f"Предлагаю закрыть задачу {card['task_key']}. Сам ничего не закрываю.")}

    # ------------------------------------------------------------ решения человека
    def decide(self, card_id: str, action: str, task_key: str | None = None, who: str = "дежурный") -> dict:
        with self.lock:
            card = self.cards[card_id]
            if card["decision"]:
                raise ValueError("решение уже принято")
            out = {"action": action}
            if action == "our_bug":
                draft = self._draft_for(card)
                task = self.world.create_task_from_draft(draft["id"], self.now)
                card["task_key"] = task["key"]
                out["task"] = task
            elif action == "known":
                key = task_key or (card["agent"]["card"].get("existing_ticket") if card["agent"] else None)
                if not key:   # агент не нашёл: берём лучшее совпадение живой задачи по заголовку карточки
                    hits = [t for t in self.world.search_tracker(card["agent"]["card"]["title"], card["product"], self.now)
                            if t["status"] in ("open", "in_progress")]
                    key = hits[0]["key"] if hits else None
                if not key or not self.world.task(key):
                    raise ValueError("нет задачи, к которой привязывать")
                self.world.link_task(key, card_id)
                card["task_key"] = key
                out["task"] = self.world.task(key)
            elif action == "noise":
                zt, mu = self._threshold(card["product"])
                z = (card.get("facts") or {}).get("z", zt)
                new = {"z_threshold": round(max(zt, z) + 0.5, 1), "min_users": mu + 2}
                self.overrides[card["product"]] = new
                entry = {"product": card["product"], "from": {"z_threshold": zt, "min_users": mu}, "to": new,
                         "sim_ts": self.now.isoformat(timespec="minutes"), "card_id": card_id}
                self.threshold_log.append(entry)
                out["threshold"] = entry
            else:
                raise ValueError("неизвестное решение")
            card["decision"] = {"action": action, "at": self.now.isoformat(timespec="minutes"), "who": who, "task_key": card["task_key"]}
            self._sync_close_proposal(card)
            self.audit.log("human", "human", {"decision": action, "who": who, **{k: v for k, v in out.items() if k != "action"}},
                           card_id, self.now)
            self.emit({"type": "refresh"})
            return out

    def _draft_for(self, card: dict) -> dict:
        """Черновик задачи: тот, что агент уже сделал, иначе собираем из карточки."""
        drafts = [d for d in self.world.drafts() if d["burst_id"] == card["id"] and d["status"] == "ждёт подтверждения"]
        if drafts:
            return drafts[-1]
        c = card["agent"]["card"]
        text = "\n".join(f"- {x['text']} ({', '.join(x['ids'])})" for x in c["claims"])
        return self.world.new_draft(card["id"], card["product"], c["title"], text, c["severity"], self.now)

    def confirm_close(self, card_id: str, who: str = "дежурный") -> dict:
        with self.lock:
            card = self.cards[card_id]
            cp = card["close_proposal"]
            if not cp or cp["state"] != "ждёт человека":
                raise ValueError("предложения о закрытии нет")
            task = self.world.set_task_status(cp["task_key"], "done", self.now)
            cp["state"] = "закрыто человеком"
            self.audit.log("human", "human", {"decision": "close_task", "task": cp["task_key"], "who": who}, card_id, self.now)
            self.emit({"type": "refresh"})
            return task

    def approve_outbox(self, oid: str, who: str = "дежурный") -> dict:
        with self.lock:
            o = next(x for x in self.world.outbox() if x["id"] == oid)
            via = send_telegram(f"[{o['product']}] {o['message']}")
            self.world.set_outbox_status(oid, "отправлено" if via == "telegram" else "подтверждено (отправка выключена)", via)
            self.audit.log("human", "human", {"decision": "approve_outbox", "id": oid, "via": via, "who": who}, o["burst_id"], self.now)
            self.emit({"type": "refresh"})
            return {"id": oid, "via": via}

    # ------------------------------------------------------------ автоответ на повторные обращения
    def _plan_of(self, card_id: str):
        return next((p for p in self.plan if p["card_id"] == card_id), None)

    def approve_autoreply(self, card_id: str, who: str = "дежурный") -> dict:
        """Клик человека: включить автоответ по инциденту. Не для «шума» и не для карточек-наблюдений (инъекция)."""
        with self.lock:
            card = self.cards[card_id]
            if card["kind"] != "alert":
                raise ValueError("автоответ включается только по тревогам")
            if card["decision"] and card["decision"]["action"] == "noise":
                raise ValueError("карточка помечена как шум: автоответ не нужен")
            if card.get("suspicion") and card["tier"] == "quiet":
                self.audit.log("antispam", "code", {"rule": "autoreply_blocked_suspicion", "who": who, "reason": card["suspicion"].get("reason", "")},
                               card_id, self.now)
                raise ValueError("подозрение на накрутку: автоответ по такой карточке не включается")
            ci = self._plan_of(card_id)["ci"]
            seed = [e["id"] for e in ci.evidence if not agent_mod.injection_in([e["text"]])]
            self._refs[card_id] = [textsim.profile(e["text"]) for e in ci.evidence if not agent_mod.injection_in([e["text"]])]
            self._semantic_refs[card_id] = [dict(id=e['id'],product=card['product'],kind='bug',
                 text=e['text'],source_verified=True) for e in ci.evidence if not agent_mod.injection_in([e['text']])][:3]
            out = self.autoreply.approve(card_id, card["product"], card["product_name"], datetime.fromisoformat(card["facts"]["started_at"]),
                                         self.now, who, seed_ticket_ids=seed)
            if card["fix"] and card["fix"]["verdict"] == "confirmed":
                self.autoreply.propose_fixed(card_id, self.now)
                out = self.autoreply.view(card_id, self.now)
            self.emit({"type": "refresh"})
            return out

    def disable_autoreply(self, card_id: str, who: str = "дежурный") -> dict:
        with self.lock:
            out = self.autoreply.disable(card_id, self.now, who)
            self.emit({"type": "refresh"})
            return out

    def approve_fixed_mailing(self, card_id: str, who: str = "дежурный") -> dict:
        """Клик человека: рассылка «починили». Есть предложение от проверки фикса: источник verify; нет: человек нажал «починили»."""
        with self.lock:
            st = self.autoreply.incidents.get(card_id)
            source = "verify" if st and st["proposal"] else "человек"
            m = self.autoreply.approve_fixed(card_id, self.now, who, source)
            self.emit({"type": "refresh"})
            return m

    def _route_tickets(self, to: datetime):
        """Обращения до `to`: те, что относятся к подтверждённому инциденту того же продукта, получают решение автоответа.
        Отнесение к инциденту здесь по сходству с примерами карточки (RELATE_SIM); пайплайн своё отнесение в записи не хранит."""
        rows, ar = self.rep.tickets, self.autoreply
        while self.tk_i < len(rows) and rows[self.tk_i][1] <= to:
            row = rows[self.tk_i]
            self.tk_i += 1
            if not ar.active():
                continue
            if os.environ.get('RADAR_AUTOREPLY_MATCHER')=='semantic':
                self._route_semantic_ticket(row)
                continue
            best, best_sim = None, 0.0
            prof = None
            for cid, st in ar.incidents.items():
                if st["state"] != "on" or st["product"] != row[2] or not self._incident_open(cid, row[1]):
                    continue
                prof = prof or textsim.profile(row[7])
                sim = max((textsim.cosine_profiles(prof, r) for r in self._refs.get(cid, [])), default=0.0)
                if sim > best_sim:
                    best, best_sim = cid, sim
            if best is not None and best_sim >= RELATE_SIM:
                ar.handle_ticket(best, row[0], row[7], row[1], sim=best_sim, extra_texts=[row[6]],
                                 reference_texts=[r['text'] for r in self._semantic_refs.get(best,[])])

    def _route_semantic_ticket(self,row):
        from radar_app.reply_policy import ReplyPolicy
        from radar_app.semantic_runtime import components
        _,matcher=components();policy=ReplyPolicy(matcher)
        candidates=[];refusals=[]
        for cid,st in self.autoreply.incidents.items():
            if st['state']!='on' or st['product']!=row[2] or not self._incident_open(cid,row[1]):continue
            payload=dict(event='incoming_ticket',product=st['product'],kind='bug',
                 references=self._semantic_refs.get(cid,[]),
                 query=dict(id=row[0],product=row[2],text=row[7],consent=True),
                 approved=True,state='open',fix_confirmed=bool(st.get('proposal')),
                 human_closed=False,fixed_notice_approved=False,
                 seen_message_ids=[x['ticket_id'] for x in self.autoreply.queue])
            # This replay contains synthetic contacts only. Production ingestion
            # must supply the source's actual permission instead of this demo flag.
            decision=policy.decide(payload)
            score=matcher.last_trace.get('pair_score',0.)
            if decision['action']=='queue_standard':candidates.append((score,cid))
            elif decision['action']!='ignore_duplicate':refusals.append((cid,decision['reason']))
        candidates.sort(reverse=True)
        if candidates:
            score,cid=candidates[0]
            if len(candidates)>1 and score-candidates[1][0]<.1:
                self.autoreply._to_operator(cid,self.autoreply.incidents[cid],row[0],'неоднозначный инцидент',row[1])
            else:self.autoreply.handle_ticket(cid,row[0],row[7],row[1],sim=score,extra_texts=[row[6]])
        elif refusals:
            # Keep only related refusals visible; do not attach every product
            # ticket to an incident or its future fixed-notice recipient list.
            for cid,reason in refusals:
                score=max((textsim.cosine_profiles(textsim.profile(row[7]),p) for p in self._refs.get(cid,[])),default=0.)
                if score>=RELATE_SIM:
                    self.autoreply._to_operator(cid,self.autoreply.incidents[cid],row[0],reason,row[1]);break

    def _incident_open(self, card_id: str, ts: datetime) -> bool:
        """Инцидент открыт на момент ts, если хотя бы один его всплеск ещё не закрыт пайплайном."""
        return any(self._closed_at.get(b) is None or self._closed_at[b] > ts for b in self.cards[card_id]["burst_ids"])

    def autoreply_view(self) -> dict:
        with self.lock:
            ar = self.autoreply
            return {"stats": ar.stats(), "incidents": {cid: ar.view(cid, self.now) for cid in self.cards},
                    "queue": ar.queue[-20:], "mailings": ar.mailings[-10:], "operator": ar.operator[-20:],
                    "settings": {"daily_limit": ar.daily_limit, "min_sim": ar.min_sim, "relate_sim": RELATE_SIM}}

    # ------------------------------------------------------------ данные для страницы
    def series(self, hours: int = 72) -> dict:
        with self.lock:
            rep, now = self.rep, self.now
            i_now = int((now - rep.start).total_seconds() // (BIN_MIN * 60))
            i0 = max(0, i_now - hours * 60 // BIN_MIN + 1)
            t0 = rep.start + timedelta(minutes=i0 * BIN_MIN)
            sl = lambda arr: arr[i0:i_now + 1]
            bursts = {}
            alerts = []
            for e in rep.events:
                if e["ts"] > now:
                    break
                b = e["burst"]
                if e["kind"] == "opened":
                    bursts[b.id] = {"product": b.product, "z": round(b.z, 1), "alert": b.alert, "signal": b.signal,
                                    "count": b.count_in_window, "expected": round(b.baseline_in_window, 1),
                                    "window_min": b.window_min, "opened": e["ts"].isoformat(timespec="minutes"),
                                    "bins": sl(rep.burst_bins[b.id])}
                    alerts.append({"ts": e["ts"].isoformat(timespec="minutes"), "burst_id": b.id, "product": b.product,
                                   "alert": b.alert, "z": round(b.z, 1)})
            return {"now": now.isoformat(timespec="minutes"), "t0": t0.isoformat(timespec="minutes"), "bin_min": BIN_MIN,
                    "products": {p: sl(v) for p, v in rep.bins_total.items()},
                    "bursts": {k: v for k, v in bursts.items() if v["product"] and any(v["bins"])},
                    "releases": [{"product": r["product"], "version": r["version"], "ts": r["ts"].isoformat(timespec="minutes"), "kind": r["kind"]}
                                 for r in rep.releases if t0 <= r["ts"] <= now],
                    "alerts": [a for a in alerts if a["ts"] >= t0.isoformat(timespec="minutes")],
                    "thresholds": self.overrides}

    def tickets_feed(self, limit: int = 8) -> list:
        with self.lock:
            rows, out = self.rep.tickets, []
            # бинарный поиск по времени: последние обращения до now
            lo, hi = 0, len(rows)
            while lo < hi:
                mid = (lo + hi) // 2
                if rows[mid][1] <= self.now:
                    lo = mid + 1
                else:
                    hi = mid
            for r in reversed(rows[max(0, lo - 400):lo]):
                out.append({"id": r[0], "ts": r[1].isoformat(timespec="minutes"), "product": r[2], "platform": r[3], "version": r[4],
                            "channel": r[5], "original": r[6], "masked": r[7], "has_pii": r[8]})
                if len(out) >= limit * 6:
                    break
            with_pii = [x for x in out if x["has_pii"]][:limit]
            for x in with_pii:   # куда в исходном тексте попали токены: подсветка «до»
                sm = difflib.SequenceMatcher(None, x["original"], x["masked"], autojunk=False)
                spans = [[i1, i2] for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("replace", "delete")]
                merged = []
                for sp in spans:   # куски одного номера (пробелы внутри) склеиваем
                    if merged and sp[0] - merged[-1][1] <= 3:
                        merged[-1][1] = sp[1]
                    else:
                        merged.append(sp)
                x["spans"] = merged
            return {"latest": out[:limit], "with_pii": with_pii}

    def fix_status(self) -> list:
        with self.lock:
            res = []
            for c in self.cards.values():
                if c["kind"] != "alert":
                    continue
                fix = next((r for r in self.rep.releases if r["product"] == c["product"] and r["kind"] == "fix"
                            and r["ts"] > datetime.fromisoformat(c["created_sim"]) - timedelta(hours=6)), None)
                if c["fix"]:
                    state = {"confirmed": "подтверждено по данным", "not_confirmed": "не подтверждено", "insufficient_data": "мало данных"}[c["fix"]["verdict"]]
                elif fix and fix["ts"] <= self.now:
                    state = "фикс выкатан, идёт наблюдение"
                else:
                    state = "фикса ещё нет"
                next_event = None
                if fix and fix["ts"] > self.now:
                    next_event = {"kind": "fix_release", "to": fix["ts"].isoformat(timespec="minutes"), "label": "К выпуску фикса"}
                elif not c["fix"] or c["fix"]["verdict"] != "confirmed":
                    checks = [e["ts"] for e in self.rep.events if e["kind"] == "fix_checked"
                              and e["burst"].id in c["burst_ids"] and e["ts"] > self.now]
                    if checks:
                        next_event = {"kind": "verification", "to": min(checks).isoformat(timespec="minutes"), "label": "К результату проверки"}
                res.append({"card_id": c["id"], "product": c["product"], "state": state, "fix_version": fix["version"] if fix and fix["ts"] <= self.now else None,
                            "fix_ts": fix["ts"].isoformat(timespec="minutes") if fix else None, "fix": c["fix"], "proposal": c["close_proposal"],
                            "demo_next": next_event})
            return res


def _rel(r: dict):
    from radar.models import Release
    return Release(r["product"], r["version"], r["ts"], r["kind"], r["changelog"])
