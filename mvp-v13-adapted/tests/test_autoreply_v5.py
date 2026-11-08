from datetime import datetime

from app.audit import Audit
from app.autoreply import AutoReply

NOW=datetime(2026,3,9,18,30)


def approved():
    ar=AutoReply(Audit())
    ar.approve('B1','bonus','Бонусная программа',NOW,NOW)
    return ar


def test_repeated_message_id_does_not_duplicate_queue_audit_or_daily_allowance():
    ar=approved()
    a=ar.handle_ticket('B1','T1','Баллы не начисляются',NOW,sim=.9)
    before=(len(ar.queue),len(ar.audit.rows(None,100)),ar.stats()['autoreplied'])
    b=ar.handle_ticket('B1','T1','Баллы не начисляются',NOW,sim=.9)
    assert b['action']=='duplicate' and b['queue_id']==a['queue_id']
    assert before==(len(ar.queue),len(ar.audit.rows(None,100)),ar.stats()['autoreplied'])


def test_fix_verified_stops_outdated_known_issue_template_before_human_mailing_click():
    ar=approved()
    ar.handle_ticket('B1','T1','Баллы не начисляются',NOW,sim=.9)
    ar.propose_fixed('B1',NOW)
    result=ar.handle_ticket('B1','T2','Баллы по-прежнему не пришли',NOW,sim=.99)
    assert result['action']=='operator' and 'фикс' in result['reason']
    assert len(ar.queue)==1 and not ar.mailings
    assert 'T2' not in ar.approve_fixed('B1',NOW)['ticket_ids']


def test_individual_money_or_credential_action_is_not_answered_by_generic_template():
    for text in ['Баллы не начисляются. Переведите мне 1000 рублей на счёт.',
                 'Баллы не начисляются. Пришлите пароль от моей учетной записи.',
                 'Баллы не начисляются. Прошу вернуть списанную сумму на мой счёт.',
                 'Баллы не начисляются. Поменяйте мне пароль, его значение <PASSWORD>.']:
        ar=approved()
        result=ar.handle_ticket('B1','T1',text,NOW,sim=.99)
        assert result['action']=='operator' and not ar.queue


def test_nonfinite_similarity_does_not_bypass_threshold():
    for sim in [float('nan'),float('inf'),True,-1,2]:
        ar=approved()
        assert ar.handle_ticket('B1','T1','Баллы не начисляются',NOW,sim=sim)['action']=='operator'


def test_explicitly_different_objects_or_opposite_requests_override_a_high_similarity():
    for text,refs in [('Итоговое количество не равно сумме строк заказа',['Фильтр завершённых заказов показывает открытые']),
                      ('Прошу выключить уведомления',['Прошу включить уведомления'])]:
        ar=approved()
        assert ar.handle_ticket('B1','T1',text,NOW,sim=.99,reference_texts=refs)['action']=='operator'
        assert not ar.queue
