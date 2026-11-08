"""Быстрая проверка вариантов на tune-seeds (только 1-5). Печатает сводку; в отчёт не идёт.
Пример: .venv/bin/python eval/dev_tune.py '{"method":"poisson"}' '{"z_threshold":5.5}'"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import run_eval as RE
if __name__ == "__main__":
    variants = [json.loads(a) for a in sys.argv[1:]] or [{}]
    cfgs = [RE.cfg_dict(**v) for v in variants]
    t0 = time.time()
    res = RE.run_jobs([(s, n, 800, cfgs) for s in RE.TUNE_SEEDS for n in RE.NOISES], 7)
    for i, v in enumerate(variants):
        a = RE.aggregate([r["runs"][i]["res"] for r in res], len(res) * RE.EVAL_DAYS)
        f = a["fix"]
        print(json.dumps(v), f"P {RE.fmt(a['precision'])} R {RE.fmt(a['recall'])} F1 {RE.f1(a):.3f} ttc {RE.fmt(a['ttc_median_min'],0)} "
              f"in60 {RE.fmt(a['recall_within_60'])} FA/wk {a['false_alarms_per_week']:.2f} {a['fa_kinds']} watch/wk {a['watch_per_week']:.1f} "
              f"mob {a['mobile']['caught']}/{a['mobile']['n_events']} web {a['web']['caught']}/{a['web']['n_events']} sig {a['first_hit_signal']} | "
              f"fix eff {f['effective_confirmed']}/{f['n_effective']} (nc {f['effective_not_checked']}) ineff {f['ineffective_false_confirmed']}/{f['n_ineffective']} "
              f"nonfix {f['nonfix_false_confirmed']}/{f['n_nonfix_checks']} nofixreg {f['nofix_regression_false_confirmed']}/{f['n_nofix_regression_checks']} "
              f"res_eff {RE.fmt(f['residual_median_effective'])} res_ineff {RE.fmt(f['residual_median_ineffective'])}", flush=True)
    print(f"{time.time()-t0:.0f} s")
