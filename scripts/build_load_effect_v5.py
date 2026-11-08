"""Recompute case-derived workload scenarios and keep provenance alongside values."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from radar_app.economics import SCENARIOS, calculate
from radar_app.load_effect import workload_from_support_team, autoreply_capacity


def main():
    workloads=[]
    for name, minutes in [('long',12),('middle',8),('short',4)]:
        w=workload_from_support_team(handling_minutes=minutes)
        w['id']=name
        w['text_slots_assumption']=math.ceil(w['daily_capacity']*5*30/28800)
        w['slot_inputs']=dict(peak_factor=5,seconds_per_text=30,working_seconds=28800,
                             measured_internal_model=False)
        workloads.append(w)
    core={name:dict(inputs=p,results=calculate(p)) for name,p in SCENARIOS.items()}
    result=dict(schema_version=1,created='2026-11-04',
                title='Сценарии ёмкости контура из условия кейса',
                official_facts=dict(support_managers_approximately=40,actual_daily_tickets=None,
                                    infrastructure_cost_ignored_by_case=True,
                                    source='reports/case-text-for-v5.txt'),
                workloads=workloads,core_economics=core,
                explanation='40 × 6 контактных часов × 60 / 12, 8 или 4 минуты. Это сценарная ёмкость рабочего дня поддержки, не фактическая нагрузка ГПН.',
                assumptions=dict(contact_hours_per_day=6,working_days_per_week=5,working_weeks=48,
                                 weekly_work_hours=40,analysis_hours_per_week=2,
                                 other_weekly_hours=8),
                core_roi_independent_of_ticket_volume=True,
                autoreply=dict(status='awaiting_control_measurement',include_in_core_roi=False,
                               observed_delivery_effect=0,external_delivery_connected=False),
                not_transferable_legacy_numbers=['17.42 million annual net','1060 daily break-even','2500–6000 observed GPN traffic'])
    quality=ROOT/'data/quality-v5.json'
    if quality.exists():
        q=json.loads(quality.read_text())
        measured=q.get('autoreply',{}).get('semantic',{})
        gate=q.get('release_gate',{}).get('autoreply',{}).get('passed',False)
        if not gate:
            result['autoreply'].update(status='candidate_rejected',
                explanation='Новый локальный matcher не прошёл контроль; его метрики не переносятся в денежный эффект. Сохраняются прежний путь и независимые исправления защиты.')
        if gate and measured.get('precision') is not None and measured.get('recall') is not None and measured['precision']>0:
            potentials=[]
            for w in workloads:
                for share in [.1,.2,.3]:
                    item=autoreply_capacity(annual_tickets=w['annual_capacity'],repeat_share=share,
                         eligible_share=.5,approved_share=.8,selection_recall=measured['recall'],
                         selection_precision=measured['precision'],before_minutes=w['inputs']['handling_minutes'],
                         after_minutes=2,error_review_minutes=15,hourly_cost=800,cash_share=.25,
                         adoption=[.5,1,1],discount=.2,delivery_connected=False)
                    item['workload_id']=w['id'];potentials.append(item)
            result['autoreply'].update(status='measured_selection_scenario_effect',scenarios=potentials,
                         metric_source='quality-v5.json; independent synthetic control, not GPN')
    (ROOT/'data/load-effect-v5.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(workloads=[w['daily_capacity'] for w in workloads],
                        investment_ceilings={k:v['results']['budget_discounted'] for k,v in core.items()})))


if __name__=='__main__':main()
