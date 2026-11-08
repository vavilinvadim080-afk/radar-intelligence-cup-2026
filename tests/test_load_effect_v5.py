import importlib
import importlib.util

import pytest


def module():
    assert importlib.util.find_spec('radar_app.load_effect'), 'Workload/effect contract is not implemented'
    return importlib.import_module('radar_app.load_effect')


@pytest.mark.parametrize('minutes,want', [(12, 1200), (8, 1800), (4, 3600)])
def test_support_capacity_uses_case_headcount_and_explicit_contact_time(minutes, want):
    r = module().workload_from_support_team(handling_minutes=minutes)
    assert r['daily_capacity'] == want
    assert r['annual_days'] == 240
    assert r['annual_capacity'] == want * 240
    assert r['actual_daily_tickets'] is None
    assert r['is_observed_traffic'] is False
    assert r['weekly_hours_used_per_person'] == 32


def test_workload_rejects_overlapping_impossible_weekly_time_budget():
    with pytest.raises(ValueError, match='weekly'):
        module().workload_from_support_team(contact_hours_per_day=8, analysis_hours_per_week=2)


@pytest.mark.parametrize('kwargs', [{'staff': 41}, {'handling_minutes': 0}, {'staff': True},
                                   {'contact_hours_per_day': float('nan')}, {'days_per_week': 8}])
def test_workload_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        module().workload_from_support_team(**kwargs)


def reply_inputs(**changes):
    p = dict(annual_tickets=1000, repeat_share=.5, eligible_share=.5, approved_share=.5,
             selection_recall=.8, selection_precision=1., before_minutes=8, after_minutes=2,
             error_review_minutes=6, hourly_cost=800, cash_share=.25,
             adoption=[.5, 1, 1], discount=0, delivery_connected=False)
    p.update(changes)
    return p


def test_autoreply_potential_does_not_become_actual_saving_without_delivery():
    r = module().autoreply_capacity(**reply_inputs())
    assert r['eligible_tickets'] == 125
    assert r['useful_replies'] == 100
    assert r['potential_annual_hours'] == 10
    assert r['potential_capacity_rub'] == 8000
    assert r['potential_cash_flows'] == [1000, 2000, 2000]
    assert r['potential_budget_discounted'] == 5000
    assert r['observed_automatic_delivery_effect'] == 0
    assert r['include_in_core_roi'] is False


def test_recall_already_counts_true_positive_replies_precision_accounts_for_errors_separately():
    r = module().autoreply_capacity(**reply_inputs(selection_precision=.8))
    assert r['useful_replies'] == 100
    assert r['expected_false_replies'] == pytest.approx(25)
    assert r['potential_annual_hours'] == pytest.approx(7.5)


def test_connected_transport_is_not_evidence_of_measured_money_effect():
    r = module().autoreply_capacity(**reply_inputs(delivery_connected=True))
    assert r['observed_automatic_delivery_effect'] is None
    assert r['include_in_core_roi'] is False
    assert r['financial_status'] == 'scenario_only'


def test_extra_review_can_make_autoreply_potential_negative():
    r = module().autoreply_capacity(**reply_inputs(selection_precision=.5, error_review_minutes=12))
    assert r['potential_annual_hours'] == -10
    assert r['potential_budget_discounted'] == 0


@pytest.mark.parametrize('kwargs', [{'selection_precision': 0}, {'repeat_share': 1.1},
                                   {'selection_recall': True}, {'adoption': [1, 1]},
                                   {'hourly_cost': float('inf')}])
def test_autoreply_effect_rejects_invalid_assumptions(kwargs):
    with pytest.raises(ValueError):
        module().autoreply_capacity(**reply_inputs(**kwargs))
