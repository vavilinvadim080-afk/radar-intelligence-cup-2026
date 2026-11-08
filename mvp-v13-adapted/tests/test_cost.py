from radar.cost import load_params, weekly_cost_rub


def test_cost_formula_and_numbers_file():
    p = load_params("base")
    assert p == {"t": 10.0, "C1": 800.0}               # shared/numbers.json, сценарий base
    assert weekly_cost_rub(40) == 5333.33              # 40 * 10/60 * 800
    assert weekly_cost_rub(0) == 0
    assert weekly_cost_rub(30, t=8, C1=600) == 2400.0   # 30 * 8/60 * 600
