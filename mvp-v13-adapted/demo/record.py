"""Пересчёт записанного прогона: реальные ответы агента (Ollama) на карточки демо-набора -> demo/replay.json.

Запуск: make demo-record   (медленно: каждая карточка это до 6 вызовов модели).
Опции: --only B0003,B0005  записать только эти карточки (остальные в файле остаются).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import agent as agent_mod
from app.audit import Audit
from app.mocks import build_world
from app.replay import card_plan, cluster_source_for, load_or_build
from app.state import _rel

OUT = Path(__file__).parent / "replay.json"
DEFAULT_CARDS = ["B0003", "B0005", "B0006", "W-T012906"]   # карточки, которые видны в четырёх сценах


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(DEFAULT_CARDS))
    args = ap.parse_args()
    want = set(args.only.split(","))
    rep = load_or_build()
    world = build_world([_rel(r) for r in rep.releases], rep.ground_truth, rep.seed, known_topics=["portal_vpn"])
    src = cluster_source_for(rep)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"results": {}}
    llm = agent_mod.OllamaClient()
    if not llm.available():
        sys.exit("Ollama недоступна: запись не сделана")
    for item in card_plan(rep):
        if item["card_id"] not in want:
            continue
        audit = Audit()
        t = time.time()
        res = agent_mod.write_card(item["ci"], world, audit, src, llm=llm)
        res["audit"] = [{"kind": r["kind"], "actor": r["actor"], "payload": r["payload"]} for r in audit.rows(limit=500)]
        data["results"][item["card_id"]] = res
        print(item["card_id"], res["mode"], "попыток", res["attempts"], "шагов", res["steps"], "инструменты", res["tools_used"], f"{time.time() - t:.0f} с", flush=True)
        data.update({"seed": rep.seed, "model": llm.model, "recorded_at": datetime.now().isoformat(timespec="seconds")})
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")   # после каждой карточки
        from app import security   # подпись HMAC: без неё сервер не примет записанный прогон
        security.sign_file(OUT, security.get_key(Path(__file__).parent / ".replay-hmac.key")[0])


if __name__ == "__main__":
    main()
