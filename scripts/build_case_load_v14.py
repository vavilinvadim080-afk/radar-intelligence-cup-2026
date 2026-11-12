"""Reproduce the reported case volume; calendar and sizing remain assumptions."""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/case-load-20261112.json"


def nonnegative(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name}: expected a finite {'positive' if positive else 'nonnegative'} number")


def text_slots(daily_total, peak_factor, seconds_per_full_text_path, working_seconds):
    """Concurrent full text paths, not model calls or physical GPU count."""
    for name, value in [("daily_total", daily_total), ("peak_factor", peak_factor),
                        ("seconds_per_full_text_path", seconds_per_full_text_path),
                        ("working_seconds", working_seconds)]:
        nonnegative(value, name, positive=name != "daily_total")
    return math.ceil(daily_total * peak_factor * seconds_per_full_text_path / working_seconds)


def autoreply_hours(daily_total, working_days, repeat_share, eligible_share, minutes_saved):
    """Annual time only; eligible_share is conditional on repeated requests."""
    for name, value in [("daily_total", daily_total), ("working_days", working_days),
                        ("repeat_share", repeat_share), ("eligible_share", eligible_share),
                        ("minutes_saved", minutes_saved)]:
        nonnegative(value, name)
    if repeat_share > 1 or eligible_share > 1:
        raise ValueError("Shares must be between 0 and 1")
    return daily_total * working_days * repeat_share * eligible_share * minutes_saved / 60


def build_payload(*, weeks_per_year=48, working_days_per_week=5, hours_per_day=8,
                  peak_factor=5, seconds_per_full_text_path=30):
    assumptions = dict(weeks_per_year=weeks_per_year,
                       working_days_per_week=working_days_per_week,
                       hours_per_day=hours_per_day, peak_factor=peak_factor,
                       seconds_per_full_text_path=seconds_per_full_text_path)
    for name, value in assumptions.items():
        nonnegative(value, name, positive=True)
    daily_total, products = 480, 40
    working_days = weeks_per_year * working_days_per_week
    working_seconds = hours_per_day * 3600
    illustration = dict(repeat_share=0.2, eligible_share=0.5, minutes_saved_per_eligible=3.5)
    illustration["annual_hours"] = autoreply_hours(
        daily_total, working_days, illustration["repeat_share"],
        illustration["eligible_share"], illustration["minutes_saved_per_eligible"],
    )
    return {
        "version": 14,
        "provenance": "user_report_of_organizer_case_update",
        "source": {
            "reported_on": "2026-11-12",
            "reporter": "user",
            "reported_statement": "организаторы еще дописали что в день им по40продуктам приходит480обращений в день в кейсе",
            "interpretation": "480 обращений в день суммарно по 40 продуктам, не по 480 на каждый.",
            "direct_document_available": False,
            "external_url": None,
            "is_measured_traffic": False,
            "prior_case_context": {
                "path": "reports/case-text-for-v5.txt",
                "support_managers_approximately": 40,
                "separate_from_product_count": True,
            },
        },
        "case_load": {
            "daily_total": daily_total,
            "products": products,
            "daily_per_product_mean": daily_total / products,
            "is_per_product_peak": False,
            "distribution": "unknown_uneven",
        },
        "assumptions": assumptions,
        "derived": {
            "working_days_per_year": working_days,
            "annual_total": daily_total * working_days,
            "working_seconds_per_day": working_seconds,
            "hourly_total_mean": daily_total / hours_per_day,
            "hourly_per_product_mean": daily_total / hours_per_day / products,
        },
        "sizing": {
            "status": "sizing_assumption",
            "formula": "ceil(daily_total * peak_factor * seconds_per_full_text_path / working_seconds_per_day)",
            "text_slots": text_slots(daily_total, peak_factor, seconds_per_full_text_path, working_seconds),
            "seconds_sensitivity": [
                {"seconds_per_full_text_path": seconds,
                 "text_slots": text_slots(daily_total, peak_factor, seconds, working_seconds)}
                for seconds in [15, 30, 60]
            ],
            "scope": "Один полный текстовый путь на новое обращение, включая его вызовы моделей и программные этапы.",
            "not_model_or_gpu_benchmark": True,
            "limitations": [
                "Пик 5x и длительность пути предположены; распределение по продуктам неравномерно и пока неизвестно.",
                "12 в день и 1,5 в час на продукт являются средними, не пиковыми значениями.",
                "Слоты являются параллельными путями, не числом GPU; ASR/VL и повторы планируются отдельно.",
                "Старые замеры на 800/1500/3000 не подтверждают качество или скорость при 480 в день.",
            ],
        },
        "autoreply": {
            "status": "illustrative_assumptions_only",
            "formula": "daily_total * working_days_per_year * repeat_share * eligible_share * minutes_saved_per_eligible / 60",
            "unknown_inputs": ["repeat_share", "eligible_share", "minutes_saved_per_eligible"],
            "eligible_share_definition": "Доля допустимых для автоответа среди повторных обращений; определяется отдельным контролем на данных компании.",
            "illustration": illustration,
            "included_in_base_pv": False,
            "quality_metrics_are_not_eligible_share": True,
            "delivery_connected": False,
            "limitations": "672 ч/год при 20% повторов, 50% допуска среди повторов и 3,5 мин экономии являются иллюстрацией допущений, не измерением. Синтетические Precision/Recall не задают долю допуска. До сложения с другими эффектами нужно исключить двойной счёт.",
        },
        "economics": {
            "source": "data/economics-weekly.json",
            "depends_on_case_volume": False,
            "policy": "Три сценария 1,5/3/6 ч/нед и PV 0,973/1,947/3,893 млн руб. неизменны. Потенциал автоответа к PV не прибавлен.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare without writing")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="JSON path (default: data/case-load-20261112.json)")
    args = parser.parse_args()
    payload = build_payload()
    if args.check:
        try:
            current = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            parser.exit(1, f"Case-load check failed: {error}\n")
        if current != payload:
            parser.exit(1, "Case-load check failed: JSON differs from the calculation\n")
        print("Case-load v14: current; no writes")
    else:
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Case-load v14: {args.output.name}; 480/day, 40 products, 12/product/day; 115200/year; 3 text slots; illustrative 672 h/year")


if __name__ == "__main__":
    main()
