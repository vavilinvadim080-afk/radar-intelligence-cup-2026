"""Цена темы за неделю. Детерминированная формула, LLM её не считает.

    цена_руб = число_уникальных_обращений * t/60 * C1

t  - минут первой линии на одно обращение, C1 - рублей за час первой линии (shared/numbers.json, сценарий base).
Пример: 40 уникальных обращений, t=10, C1=800: 40 * 10/60 * 800 = 5333.33 руб.
"""
from __future__ import annotations

import json
from pathlib import Path

NUMBERS_PATH = Path(__file__).resolve().parents[1] / "shared" / "numbers.json"


def load_params(scenario: str = "base", path: Path | str | None = None) -> dict:
    """Берёт t и C1 из общего файла цифр. Возвращает {"t": ..., "C1": ...}."""
    data = json.loads(Path(path or NUMBERS_PATH).read_text(encoding="utf-8"))
    inp = data["inputs"]
    return {"t": float(inp["t"][scenario]), "C1": float(inp["C1"][scenario])}


def weekly_cost_rub(n_unique: int, t: float | None = None, C1: float | None = None, scenario: str = "base") -> float:
    if t is None or C1 is None:
        p = load_params(scenario)
        t, C1 = p["t"] if t is None else t, p["C1"] if C1 is None else C1
    return round(n_unique * t / 60.0 * C1, 2)


def unique_users(tickets) -> int:
    """Один человек = одно обращение, сколько бы раз он ни написал."""
    return len({t.user_id for t in tickets})
