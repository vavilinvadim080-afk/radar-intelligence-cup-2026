"""scale_check: v2 пишет в свои файлы, старые не трогает, прежние объёмы копирует без пересчёта."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import scale_check as S  # noqa: E402

OLD_JSON = ROOT / "shared" / "metrics-scale.json"
OLD_MD = ROOT / "eval" / "REPORT-scale.md"


def _h(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_defaults_point_to_old_files():
    a = S.build_parser().parse_args([])
    assert Path(a.out) == OLD_JSON and Path(a.report) == OLD_MD


def test_out_and_report_override():
    a = S.build_parser().parse_args(["--out", "x.json", "--report", "x.md"])
    assert a.out == "x.json" and a.report == "x.md"


def test_import_legacy_copies_without_recompute():
    old = json.loads(OLD_JSON.read_text(encoding="utf-8"))
    out = {"by_volume": {}, "timing": {}}
    S.import_legacy(out, OLD_JSON, [800, 1500, 3000])
    for v in ("800", "1500", "3000"):
        assert out["by_volume"][v] == old["by_volume"][v]
        assert out["timing"][v] == old["timing"][v]
    assert "metrics-scale.json" in out["legacy_source"]["file"]


def test_main_does_not_touch_old_files(tmp_path, monkeypatch):
    before = (_h(OLD_JSON), _h(OLD_MD))
    fake = lambda jobs, w: [dict(seed=s, noise=n, volume=v, runs=[dict(res=_res(), seconds=1.0)] * 2)
                            for (s, n, v, c) in jobs]
    monkeypatch.setattr(S.R, "run_jobs", fake)
    monkeypatch.setattr(S.R, "by_noise", lambda res, i: {k: {kk: 0.0 for kk in S.KEYS} for k in ("all", "low", "mid", "high")})
    o, r = tmp_path / "m.json", tmp_path / "r.md"
    S.main(["--volumes", "6000", "--out", str(o), "--report", str(r), "--copy-legacy", "800,1500,3000"])
    d = json.loads(o.read_text(encoding="utf-8"))
    assert set(d["by_volume"]) == {"800", "1500", "3000", "6000"}
    assert r.exists()
    assert (_h(OLD_JSON), _h(OLD_MD)) == before


def _res():
    return dict(ttc=[10.0])


def test_no_cache_env_writes_nothing(tmp_path, monkeypatch):
    import datacache
    monkeypatch.setattr(datacache, "CACHE", tmp_path / "ds")
    monkeypatch.setenv("RADAR_NO_CACHE", "1")
    ds, prepared = datacache.get(301, "low", 200)
    assert len(ds.tickets) > 0 and not (tmp_path / "ds").exists()


def test_threshold_experiment_uses_unused_seeds_and_keeps_detector_params():
    import threshold_experiment as E
    assert not set(E.SEEDS) & (set(S.R.TUNE_SEEDS) | set(S.R.FINAL_SEEDS) | set(S.R.BURNED) | {301, 302, 303, 304, 305})
    cfgs = E.build_cfgs([6.0, 8.0])
    tuned = json.loads(S.R.TUNED.read_text())["best"]
    assert [c["z_threshold"] for c in cfgs["poisson"]] == [6.0, 8.0]
    assert all(c["z_version"] == tuned["poisson"]["z_version"] and c["windows_min"] == tuned["poisson"]["windows_min"]
               for c in cfgs["poisson"])


def test_curve_for_deck_built_from_metrics():
    import make_curve_for_deck as C
    m = {"volumes": [800, 1500], "by_volume": {v: {"detector": {k: {"precision": 0.5, "recall": 0.6, "ttc_median_min": 90.0,
         "false_alarms_per_week": 0.4, "extra": 1} for k in ("poisson", "ewma")}} for v in ("800", "1500")}}
    c = C.build(m)
    assert c["volumes"] == [800, 1500]
    assert c["series"]["ewma"]["false_alarms_per_week"] == [0.4, 0.4]
    assert set(c["series"]["poisson"]) == {"false_alarms_per_week", "precision", "recall", "ttc_median_min"}


def _c(x, p=2):
    return "-" if x is None else f"{x:.{p}f}".replace(".", ",")


def test_report_v2_numbers_match_json_and_deck():
    import make_report_v2 as M
    import make_curve_for_deck as C
    m = json.loads((ROOT / "shared" / "metrics-scale-v2.json").read_text(encoding="utf-8"))
    text = M.render(m, None)
    for v, x in m["by_volume"].items():
        for meth in ("poisson", "ewma"):
            d = x["detector"][meth]
            row = next(r for r in text.splitlines() if r.startswith(f"| {v} | {meth} |"))
            for val in (_c(d["precision"]), _c(d["recall"]), _c(d["false_alarms_per_week"]),
                        _c(d["ttc_median_min"], 0), _c(d["ttc_p90_min"], 0), _c(d["caught_within_60_share"] * 100, 0)):
                assert f" {val} " in row, (v, meth, val, row)
    deck = C.build(m)
    for meth in ("poisson", "ewma"):
        for i, v in enumerate(deck["volumes"]):
            assert deck["series"][meth]["false_alarms_per_week"][i] == m["by_volume"][str(v)]["detector"][meth]["false_alarms_per_week"]
    assert "20000" in text and "прерван" in text
