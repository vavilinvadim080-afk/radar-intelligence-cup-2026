import importlib
import importlib.util


def policy(matcher=None):
    assert importlib.util.find_spec('radar_app.reply_policy'), 'Policy is not implemented'
    return importlib.import_module('radar_app.reply_policy').ReplyPolicy(matcher or (lambda p: True))


def payload():
    return dict(event='incoming_ticket', product='one',kind='bug',
                references=[dict(id='r1',product='one',kind='bug',text='Отчёт пустой',source_verified=True)],
                query=dict(id='q1',product='one',text='Отчёт пустой',consent=True),
                approved=True,state='open',fix_confirmed=False,human_closed=False,
                fixed_notice_approved=False,seen_message_ids=[])


def test_bounded_policy_requires_explicit_scope_verified_evidence_and_consent():
    for change in ['scope','evidence','consent']:
        p=payload()
        if change=='scope':p['query']['product']='two'
        if change=='evidence':p['references'][0]['source_verified']=False
        if change=='consent':p['query']['consent']=False
        assert policy().decide(p)['action']=='operator'


def test_similarity_does_not_grant_permission_or_rewrite_fixed_state():
    p=payload();p['approved']=False
    assert policy().decide(p)['action']=='needs_approval'
    p=payload();p['fix_confirmed']=True
    assert policy().decide(p)['action']=='needs_fix_approval'
    p['event']='fixed_notice_candidate';p['human_closed']=True;p['fixed_notice_approved']=True
    assert policy().decide(p)['action']=='queue_fixed_notice'


def test_safe_same_topic_queues_template_and_a_different_topic_does_not():
    assert policy().decide(payload())['action']=='queue_standard'
    assert policy(lambda p:False).decide(payload())['action']=='operator'
    assert policy(lambda p:None).decide(payload())['action']=='operator'


def test_duplicate_is_ignored_without_calling_model():
    p=payload();p['seen_message_ids']=['q1']
    assert policy(lambda p: (_ for _ in ()).throw(RuntimeError())).decide(p)['action']=='ignore_duplicate'


def test_unavailable_model_and_individual_action_are_not_automatic_success():
    p=payload();p['query']['text']+=' Переведите мне 1000 рублей.'
    assert policy().decide(p)['action']=='operator'
    assert policy(lambda p: (_ for _ in ()).throw(RuntimeError())).decide(payload())['action']=='operator'


def test_gold_fields_are_rejected_at_model_boundary():
    import pytest
    p=payload();p['gold']={'same_topic':True}
    with pytest.raises(ValueError):policy().decide(p)
