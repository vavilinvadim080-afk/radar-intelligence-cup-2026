"""Weekly time scenarios: units, adopted cash, present value and read-only checks."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_weekly_scenarios.py"
SOURCE = ROOT / "data" / "economics.json"


class WeeklyEconomicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = runpy.run_path(str(SCRIPT))
        cls.payload = cls.model["build_payload"](SOURCE)

    def test_weekly_per_employee_units_become_annual_team_capacity(self):
        self.assertEqual(
            self.payload["assumptions"],
            {
                "staff": 40, "weeks": 48, "hourly_cost": 800,
                "cash_share": 0.25, "adoption": [0.5, 1, 1],
                "discount": 0.2, "horizon_years": 3, "wage_growth": 0,
            },
        )
        self.assertEqual(
            [
                (s["weekly_saved_hours"], s["annual_hours"], s["annual_capacity_rub"])
                for s in self.payload["scenarios"]
            ],
            [(1.5, 2880, 2304000), (3, 5760, 4608000), (6, 11520, 9216000)],
        )

    def test_adoption_and_cash_share_are_applied_once_before_discounting(self):
        self.assertEqual(
            [s["cash_flows"] for s in self.payload["scenarios"]],
            [[288000, 576000, 576000],
             [576000, 1152000, 1152000],
             [1152000, 2304000, 2304000]],
        )
        self.assertEqual(
            [s["budget_discounted"] for s in self.payload["scenarios"]],
            [973333.33, 1946666.67, 3893333.33],
        )
        # Independent small example: 100 rubles/year, 50% cash, 50/100% adoption.
        assumptions = dict(
            self.payload["assumptions"], staff=1, weeks=1, hourly_cost=100,
            cash_share=0.5, adoption=[0.5, 1], discount=1, horizon_years=2,
        )
        scenario = self.model["calculate_scenario"](1, assumptions)
        self.assertEqual(scenario["cash_flows"], [25, 50])
        self.assertEqual(scenario["budget_discounted"], 25)

    def test_no_cash_realization_keeps_hours_but_sets_budget_to_zero(self):
        assumptions = dict(self.payload["assumptions"], cash_share=0)
        for weekly, hours, capacity in [
            (1.5, 2880, 2304000), (3, 5760, 4608000), (6, 11520, 9216000),
        ]:
            with self.subTest(weekly_saved_hours=weekly):
                scenario = self.model["calculate_scenario"](weekly, assumptions)
                self.assertEqual(scenario["annual_hours"], hours)
                self.assertEqual(scenario["annual_capacity_rub"], capacity)
                self.assertEqual(scenario["cash_flows"], [0, 0, 0])
                self.assertEqual(scenario["budget_discounted"], 0)

    def test_source_base_is_consumed_without_rewriting_legacy_scenarios(self):
        original = SOURCE.read_bytes()
        source = json.loads(original)
        source["base"]["inputs"]["hourly_cost"] = 400
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "economics.json"
            fixture.write_text(json.dumps(source), encoding="utf-8")
            result = self.model["build_payload"](fixture)
        self.assertEqual(result["scenarios"][0]["annual_capacity_rub"], 1152000)
        self.assertEqual(result["scenarios"][0]["cash_flows"], [144000, 288000, 288000])
        self.assertEqual(result["scenarios"][0]["budget_discounted"], 486666.67)
        self.assertEqual(SOURCE.read_bytes(), original)

    def test_check_accepts_current_data_and_rejects_drift_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory)
            script = sandbox / "scripts" / SCRIPT.name
            source = sandbox / "data" / SOURCE.name
            script.parent.mkdir()
            source.parent.mkdir()
            script.write_bytes(SCRIPT.read_bytes())
            source.write_bytes(SOURCE.read_bytes())
            output = source.with_name("economics-weekly.json")

            def invoke(*arguments):
                return subprocess.run(
                    [sys.executable, "-B", str(script), *arguments],
                    cwd=directory, capture_output=True, text=True, check=False,
                )

            missing = invoke("--check")
            self.assertNotEqual(missing.returncode, 0, missing.stdout)
            self.assertFalse(output.exists())
            generated = invoke()
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertEqual(json.loads(output.read_text()), self.payload)
            for label, content, expected_status in [
                ("current", output.read_bytes(), 0),
                ("drift", json.dumps(dict(self.payload, scenarios=[])).encode(), 1),
                ("malformed", b"{", 1),
            ]:
                with self.subTest(state=label):
                    output.write_bytes(content)
                    before = (output.read_bytes(), output.stat().st_mtime_ns)
                    checked = invoke("--check")
                    self.assertEqual(checked.returncode, expected_status, checked.stderr)
                    self.assertEqual(
                        (output.read_bytes(), output.stat().st_mtime_ns), before,
                    )
            self.assertEqual(source.read_bytes(), SOURCE.read_bytes())


if __name__ == "__main__":
    unittest.main()
