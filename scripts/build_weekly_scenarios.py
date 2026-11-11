"""Build separate weekly savings scenarios using only the Python standard library."""

import argparse
import json
from pathlib import Path
import sys


WEEKLY_SAVED_HOURS = (1.5, 3, 6)
COMMON_INPUTS = (
    "staff", "weeks", "hourly_cost", "cash_share", "adoption", "discount", "wage_growth",
)


def calculate_scenario(weekly_saved_hours: float, assumptions: dict) -> dict:
    """Hours and capacity assume full adoption; only cash flows apply adoption."""
    if assumptions["wage_growth"] != 0:
        raise ValueError("Weekly scenarios require wage_growth=0.")
    if assumptions["horizon_years"] != len(assumptions["adoption"]):
        raise ValueError("Each horizon year needs an adoption value.")
    annual_hours = assumptions["staff"] * assumptions["weeks"] * weekly_saved_hours
    annual_capacity = annual_hours * assumptions["hourly_cost"]
    cash_flows = [
        annual_capacity * assumptions["cash_share"] * adoption
        for adoption in assumptions["adoption"]
    ]
    present_value = sum(
        cash_flow / (1 + assumptions["discount"]) ** year
        for year, cash_flow in enumerate(cash_flows, start=1)
    )
    return {
        "weekly_saved_hours": weekly_saved_hours,
        "annual_hours": annual_hours,
        "annual_capacity_rub": annual_capacity,
        "cash_flows": cash_flows,
        "budget_discounted": round(present_value, 2),
    }


def build_payload(source: Path) -> dict:
    """Read common base assumptions without altering the complete legacy scenarios."""
    base = json.loads(source.read_text(encoding="utf-8"))["base"]["inputs"]
    assumptions = {key: base[key] for key in COMMON_INPUTS}
    assumptions["horizon_years"] = len(assumptions["adoption"])
    return {
        "version": 1,
        "provenance": {
            "case": {
                "source": "Официальное условие кейса «Газпром нефть».",
                "facts": ["40 сотрудников", "Бесплатная инфраструктура по условию кейса"],
            },
            "common_inputs": "data/economics.json#/base/inputs; кроме staff, это сценарные допущения, не замеры.",
            "weekly_saved_hours": "Анализ чувствительности: 1,5 / 3 / 6 часов в неделю на одного сотрудника; сценарии, не измеренный минимум.",
            "legacy_scenarios": "Полные conservative/base/optimistic сохранены отдельно в data/economics.json.",
        },
        "assumptions": assumptions,
        "scenarios": [
            calculate_scenario(weekly, assumptions) for weekly in WEEKLY_SAVED_HOURS
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="compare the existing weekly JSON with the calculation without writing",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "economics-weekly.json"
    try:
        expected = build_payload(root / "data" / "economics.json")
        if args.check:
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing != expected:
                print(f"FAIL: {output} differs from the weekly calculation.", file=sys.stderr)
                return 1
            print(f"OK: {output} matches the weekly calculation; no files written.")
        else:
            output.write_text(
                json.dumps(expected, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
