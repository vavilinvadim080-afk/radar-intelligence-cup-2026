"""Case-linked capacity estimates, kept separate from observed traffic and savings."""
import math


def _number(name, value, low=0, high=None, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < low
            or (high is not None and value > high) or (positive and value <= 0)):
        raise ValueError(f'Invalid {name}')
    return value


def workload_from_support_team(*, staff=40, contact_hours_per_day=6, handling_minutes=8,
                               weeks=48, days_per_week=5, weekly_work_hours=40,
                               analysis_hours_per_week=2):
    """Estimate tickets per support workday; the case supplies headcount, not traffic.

    Contact time excludes the separate weekly analysis of themes. Their sum must
    fit the working week, so the same hours are not used twice in this estimate.
    """
    _number('staff', staff, high=40)
    _number('contact_hours_per_day', contact_hours_per_day, high=24)
    _number('handling_minutes', handling_minutes, high=1440, positive=True)
    _number('weeks', weeks, high=52)
    _number('days_per_week', days_per_week, high=7, positive=True)
    _number('weekly_work_hours', weekly_work_hours, high=168)
    _number('analysis_hours_per_week', analysis_hours_per_week, high=168)
    weekly = contact_hours_per_day * days_per_week + analysis_hours_per_week
    if weekly > weekly_work_hours:
        raise ValueError('Contact and analysis time exceed the weekly working budget')
    daily = staff * contact_hours_per_day * 60 / handling_minutes
    annual_days = weeks * days_per_week
    return {
        'daily_capacity': daily,
        'annual_days': annual_days,
        'annual_capacity': daily * annual_days,
        'weekly_hours_used_per_person': weekly,
        'actual_daily_tickets': None,
        'is_observed_traffic': False,
        'day_kind': 'support_workday',
        'inputs': dict(staff=staff, contact_hours_per_day=contact_hours_per_day,
                       handling_minutes=handling_minutes, weeks=weeks,
                       days_per_week=days_per_week, weekly_work_hours=weekly_work_hours,
                       analysis_hours_per_week=analysis_hours_per_week),
        'provenance': {
            'staff': 'Official case: approximately 40 support managers in the covered contour',
            'other_inputs': 'Scenario assumptions, not measured company operations',
            'source': 'reports/case-text-for-v5.txt',
        },
    }


def autoreply_capacity(*, annual_tickets, repeat_share, eligible_share, approved_share,
                       selection_recall, selection_precision, before_minutes,
                       after_minutes, error_review_minutes, hourly_cost, cash_share,
                       adoption, discount, delivery_connected=False):
    """Potential effect of correctly prepared replies, including erroneous selection.

    Recall already equals true positives / eligible records. Precision therefore
    estimates extra false positives and their review cost; multiplying recall by
    precision again would incorrectly discount the true-positive count twice.
    These inputs describe a scenario, never an observation of money saved.
    """
    for name, value in dict(repeat_share=repeat_share, eligible_share=eligible_share,
                            approved_share=approved_share, selection_recall=selection_recall,
                            selection_precision=selection_precision, cash_share=cash_share).items():
        _number(name, value, high=1)
    _number('annual_tickets', annual_tickets)
    _number('before_minutes', before_minutes, high=1440)
    _number('after_minutes', after_minutes, high=1440)
    _number('error_review_minutes', error_review_minutes, high=1440)
    _number('hourly_cost', hourly_cost, high=100000)
    _number('discount', discount, high=1)
    if selection_precision <= 0:
        raise ValueError('Positive measured selection_precision is required for this estimate')
    if not isinstance(adoption, (list, tuple)) or len(adoption) != 3:
        raise ValueError('Three adoption shares required')
    for a in adoption:
        _number('adoption', a, high=1)
    if not isinstance(delivery_connected, bool):
        raise ValueError('delivery_connected must be a boolean')

    eligible = annual_tickets * repeat_share * eligible_share * approved_share
    useful = eligible * selection_recall
    erroneous = useful * (1 / selection_precision - 1)
    hours = (useful * (before_minutes - after_minutes) - erroneous * error_review_minutes) / 60
    capacity = hours * hourly_cost
    cash_flows = [capacity * cash_share * a for a in adoption]
    ceiling = max(0, sum(v / (1 + discount) ** (i + 1) for i, v in enumerate(cash_flows)))
    return {
        'eligible_tickets': eligible,
        'useful_replies': useful,
        'expected_false_replies': erroneous,
        'potential_annual_hours': hours,
        'potential_capacity_rub': capacity,
        'potential_cash_flows': cash_flows,
        'potential_budget_discounted': ceiling,
        'observed_automatic_delivery_effect': None if delivery_connected else 0,
        'financial_status': 'scenario_only',
        'include_in_core_roi': False,
        'combination_condition': 'Disjoint work categories and validated delivery/time measurements',
        'inputs': dict(annual_tickets=annual_tickets, repeat_share=repeat_share,
                       eligible_share=eligible_share, approved_share=approved_share,
                       selection_recall=selection_recall, selection_precision=selection_precision,
                       before_minutes=before_minutes, after_minutes=after_minutes,
                       error_review_minutes=error_review_minutes, hourly_cost=hourly_cost,
                       cash_share=cash_share, adoption=list(adoption), discount=discount,
                       delivery_connected=delivery_connected),
    }
