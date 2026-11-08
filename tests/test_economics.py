import pytest
from radar_app.economics import calculate

def params(**kw):
    return dict(staff=40, weeks=48, before_hours=2, after_hours=0.5,
                hourly_cost=800, adoption=[0.5,1,1], cash_share=0.25,
                discount=0.2, wage_growth=0, **kw)

def test_case_hours_cash_and_budget_have_different_meanings():
    r = calculate(params())
    assert r["annual_hours"] == 2880
    assert r["annual_capacity_rub"] == 2304000
    assert r["cash_flows"] == [288000,576000,576000]
    assert r["budget_simple"] == 1440000
    assert r["budget_discounted"] == pytest.approx(973333.3333333333)

def test_no_cash_realization_no_claim_of_money_payback():
    p = params(); p["cash_share"] = 0
    r = calculate(p)
    assert r["annual_hours"] == 2880
    assert r["budget_simple"] == 0

def test_review_time_can_make_effect_negative():
    p = params(); p["after_hours"] = 3
    r = calculate(p)
    assert r["annual_hours"] == -1920
    assert r["budget_simple"] == 0
    assert r["viable"] is False

@pytest.mark.parametrize("key,value", [("staff",41),("cash_share",1.2),
                                      ("adoption",[1,1]),("discount",-1),("weeks",60),
                                      ("before_hours",float("nan"))])
def test_rejects_invalid_inputs(key, value):
    p=params();p[key]=value
    with pytest.raises(ValueError):
        calculate(p)
