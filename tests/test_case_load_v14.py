"""Case volume, explicit assumptions and a read-only reproducibility check."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_case_load_v14.py"
OUTPUT = ROOT / "data/case-load-20261112.json"


class CaseLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = runpy.run_path(str(SCRIPT))
        cls.payload = cls.model["build_payload"]()

    def test_provenance_preserves_the_report_and_separates_staff_from_products(self):
        p = self.payload
        self.assertEqual(p["provenance"], "user_report_of_organizer_case_update")
        self.assertEqual(p["source"]["reported_on"], "2026-11-12")
        self.assertEqual(p["source"]["reported_statement"],
                         "организаторы еще дописали что в день им по40продуктам приходит480обращений в день в кейсе")
        self.assertFalse(p["source"]["direct_document_available"])
        self.assertIsNone(p["source"]["external_url"])
        self.assertFalse(p["source"]["is_measured_traffic"])
        self.assertEqual(p["source"]["prior_case_context"]["support_managers_approximately"], 40)
        self.assertTrue(p["source"]["prior_case_context"]["separate_from_product_count"])

    def test_480_is_the_total_and_12_is_a_mean_not_a_peak(self):
        load = self.payload["case_load"]
        self.assertEqual(load["daily_total"], 480)
        self.assertEqual(load["products"], 40)
        self.assertEqual(load["daily_per_product_mean"], 12)
        self.assertFalse(load["is_per_product_peak"])
        self.assertEqual(load["distribution"], "unknown_uneven")

    def test_annual_and_hourly_units_use_explicit_calendar_assumptions(self):
        self.assertEqual(self.payload["assumptions"], {
            "weeks_per_year": 48, "working_days_per_week": 5,
            "hours_per_day": 8, "peak_factor": 5,
            "seconds_per_full_text_path": 30,
        })
        self.assertEqual(self.payload["derived"], {
            "working_days_per_year": 240, "annual_total": 115200,
            "working_seconds_per_day": 28800,
            "hourly_total_mean": 60, "hourly_per_product_mean": 1.5,
        })
        changed = self.model["build_payload"](weeks_per_year=40, hours_per_day=4)
        self.assertEqual(changed["derived"]["annual_total"], 96000)
        self.assertEqual(changed["derived"]["hourly_total_mean"], 120)

    def test_sizing_uses_ceiling_and_full_path_seconds(self):
        sizing = self.payload["sizing"]
        self.assertEqual(sizing["status"], "sizing_assumption")
        self.assertTrue(sizing["not_model_or_gpu_benchmark"])
        self.assertEqual(sizing["text_slots"], 3)
        self.assertEqual(sizing["seconds_sensitivity"], [
            {"seconds_per_full_text_path": 15, "text_slots": 2},
            {"seconds_per_full_text_path": 30, "text_slots": 3},
            {"seconds_per_full_text_path": 60, "text_slots": 5},
        ])
        self.assertEqual(self.model["text_slots"](480, 5, 30, 28800), 3)
        self.assertEqual(self.model["text_slots"](480, 1, 60, 28800), 1)
        self.assertEqual(self.model["text_slots"](481, 1, 60, 28800), 2)

    def test_autoreply_is_an_independent_illustration_without_quality_proxy(self):
        auto = self.payload["autoreply"]
        self.assertEqual(auto["status"], "illustrative_assumptions_only")
        self.assertEqual(auto["illustration"], {
            "repeat_share": 0.2, "eligible_share": 0.5,
            "minutes_saved_per_eligible": 3.5, "annual_hours": 672,
        })
        self.assertEqual(len(auto["unknown_inputs"]), 3)
        self.assertFalse(auto["included_in_base_pv"])
        self.assertTrue(auto["quality_metrics_are_not_eligible_share"])
        self.assertFalse(auto["delivery_connected"])
        calculate = self.model["autoreply_hours"]
        self.assertEqual(calculate(480, 240, 0.2, 0.5, 3.5), 672)
        self.assertEqual(calculate(480, 240, 0.2, 0, 3.5), 0)
        self.assertEqual(calculate(480, 240, 0, 0.5, 3.5), 0)
        self.assertFalse(self.payload["economics"]["depends_on_case_volume"])

    def test_calculations_reject_invalid_units_and_shares(self):
        for name, arguments in [
            ("text_slots", (480, 5, 30, 0)),
            ("text_slots", (-480, 5, 30, 28800)),
            ("text_slots", (480, float("nan"), 30, 28800)),
            ("autoreply_hours", (480, 240, 1.1, 0.5, 3.5)),
            ("autoreply_hours", (480, 240, 0.2, -0.5, 3.5)),
            ("autoreply_hours", (480, 240, 0.2, 0.5, -3.5)),
        ]:
            with self.subTest(name=name, arguments=arguments):
                with self.assertRaises(ValueError):
                    self.model[name](*arguments)
        for arguments in [{"hours_per_day": 0}, {"weeks_per_year": -1}]:
            with self.assertRaises(ValueError):
                self.model["build_payload"](**arguments)

    def test_json_is_current_and_check_is_read_only(self):
        self.assertEqual(json.loads(OUTPUT.read_text()), self.payload)
        before = OUTPUT.read_bytes(), OUTPUT.stat().st_mtime_ns
        checked = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--check"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual((OUTPUT.read_bytes(), OUTPUT.stat().st_mtime_ns), before)

    def test_check_rejects_missing_malformed_and_drift_without_writing(self):
        with tempfile.TemporaryDirectory(prefix="radar-load-check-") as directory:
            for label, content in [
                ("missing", None), ("malformed", "{"),
                ("drift", json.dumps(dict(self.payload, case_load={}))),
            ]:
                with self.subTest(case=label):
                    fixture = Path(directory) / f"{label}.json"
                    self.assertFalse(fixture.exists())
                    if content is not None:
                        fixture.write_text(content)
                    before = (fixture.read_bytes(), fixture.stat().st_mtime_ns) if fixture.exists() else None
                    checked = subprocess.run(
                        [sys.executable, "-B", str(SCRIPT), "--check", "--output", str(fixture)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(checked.returncode, 1, checked.stderr)
                    after = (fixture.read_bytes(), fixture.stat().st_mtime_ns) if fixture.exists() else None
                    self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
