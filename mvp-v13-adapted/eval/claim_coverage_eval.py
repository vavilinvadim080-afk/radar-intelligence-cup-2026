"""Замер проверки покрытия утверждений (validate_card, claim_coverage) на сохранённых СЫРЫХ утверждениях LLM. Модель не запускается.

Вход: eval/cache/claims-raw-v2-schema.json (31 карточка слепых seeds 301-305, вариант v2 + схема; сырой final модели до валидатора,
тексты обращений из входа и из get_cluster). Файл собран 2026-10-29 коротким прогоном gemma4:e2b (кэш eval/cache/agent_eval-v2-schema хранит
только итоги карточек, без текстов утверждений, поэтому по нему замер невозможен).
Что считает: сколько утверждений отбросится на каждом пороге (ложные отбрасывания) и какую долю подмен ссылок ловит проверка
(тест из research/prod-practices-v1.md G1: ссылки утверждения заменяются на обращения другого кластера).
Запуск: PYTHONPATH=. .venv/bin/python eval/claim_coverage_eval.py [--thresholds 0.25 0.3 0.34 0.4 0.5]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app import agent as A

RAW = Path(__file__).parent / "cache" / "claims-raw-v2-schema.json"


def load(path: Path = RAW) -> list:
    """[(метка карточки, {id: текст}, [утверждения последнего final модели без дублей])]."""
    out = []
    for r in json.loads(path.read_text(encoding="utf-8")):
        texts = {e["id"]: e["text"] for e in r["evidence"]}
        texts.update(r["seen"])
        seen, claims = set(), []
        for c in (r["raw_claims"][-1] if r["raw_claims"] else []):
            k = (c["text"], tuple(c["ids"]))
            if k not in seen:
                seen.add(k)
                claims.append(c)
        out.append((f'{r["seed"]}-{r["burst"]}', texts, claims, [e["id"] for e in r["evidence"][:4]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.3, 0.34, 0.4, 0.5])
    a = ap.parse_args()
    cards = load()
    kinds, checkable = Counter(), []
    for label, texts, claims, _ in cards:
        for c in claims:
            if not c["ids"]:
                kinds["без ids"] += 1
            elif any(i.startswith(A.NON_TICKET_PREFIXES) for i in c["ids"]):
                kinds["id релиза/задачи/инцидента"] += 1
            else:
                ok, _s = A.claim_coverage(c["text"], c["ids"], texts, 0.0)
                kinds["про обращения, проверяется" if ok is not None else "про обращения, не проверяется (масштаб, время, мало слов)"] += 1
                if ok is not None:
                    checkable.append((label, c))
    print(f"карточек {len(cards)}, утверждений {sum(kinds.values())}:", dict(kinds))
    by = {label: (texts, ids4) for label, texts, _, ids4 in cards}
    for th in a.thresholds:
        drop = sum(1 for lb, c in checkable if not A.claim_coverage(c["text"], c["ids"], by[lb][0], th)[0])
        swaps = caught = 0
        for lb, c in checkable:
            for lb2, (t2, ids2) in by.items():
                if lb2 != lb:
                    swaps += 1
                    caught += not A.claim_coverage(c["text"], ids2, t2, th)[0]
        print(f"порог {th}: ложных отбрасываний {drop}/{len(checkable)}; поймано подмен ссылок {caught}/{swaps} ({caught / swaps:.0%})")


if __name__ == "__main__":
    main()
