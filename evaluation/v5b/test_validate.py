import copy
import unittest
from validate import validate_bundle

SPEC = dict(needs_rows=2, families=1, family_size=1, unknown=0, duplicate=1,
            reply_rows=1, standard_positive=1, reply_families=1)

def fixture():
    row = dict(id='a', product='portal', author_id='author', source_id='source',
               created_at='2026-11-01T10:00:00+03:00', received_at='2026-11-01T10:00:01+03:00',
               channel='web_form', text='Синтетическая строка теста валидатора.')
    duplicate = {**row, 'received_at': '2026-11-01T10:00:02+03:00'}
    gold = dict(product='portal', kind='bug', topic_id='portal:bug:fixture', direction='restore',
                route='candidate', is_duplicate_delivery=False, synthetic=True, reason='test fixture')
    ng = [dict(id='a', gold=gold), dict(id='a', gold={**gold, 'is_duplicate_delivery': True})]
    ref = dict(id='ref', product='portal', kind='bug', author_id='other', text='Фикстура.', source_verified=True)
    reply = dict(id='reply', product='portal', kind='bug', event='incoming_ticket', references=[ref, {**ref, 'id': 'ref2', 'author_id': 'other2'}],
                 query=dict(id='query', product='portal', author_id='third', created_at=row['created_at'], channel='web_form', consent=True, text='Фикстура.'),
                 approved=True, state='open', fix_confirmed=False, human_closed=False, fixed_notice_approved=False, seen_message_ids=[])
    rg = dict(id='reply', gold=dict(same_topic=True, eligibility=True, safe_action='queue_standard',
              reference_topic_id='portal:bug:fixture', query_kind='bug', query_direction='restore', reason=['same_issue_safe_open_approved'], synthetic=True))
    return [row, duplicate], ng, [reply], [rg]

class ValidatorTests(unittest.TestCase):
    def test_valid_fixture(self):
        self.assertEqual(validate_bundle(*fixture(), expected=SPEC), [])

    def test_input_label_leak_rejected(self):
        data=fixture();data[0][0]['topic_id']='leak'
        self.assertIn('input_label_leak', validate_bundle(*data, expected=SPEC))

    def test_duplicate_content_drift_rejected(self):
        data=fixture();data[0][1]['text']='Изменённая доставка того же id.'
        self.assertIn('duplicate_content_drift', validate_bundle(*data, expected=SPEC))

    def test_gold_alignment_rejected(self):
        data=fixture();data[1][1]['id']='wrong'
        self.assertIn('gold_alignment', validate_bundle(*data, expected=SPEC))

    def test_unsafe_positive_rejected(self):
        data=fixture();data[2][0]['query']['consent']=False
        self.assertIn('unsafe_standard_positive', validate_bundle(*data, expected=SPEC))

    def test_independent_semantics_and_eligibility(self):
        data=fixture();data[2][0]['query']['consent']=False;data[3][0]['gold'].update(eligibility=False,safe_action='operator',reason=['no_consent'])
        self.assertEqual(validate_bundle(*data, expected={**SPEC,'standard_positive':0}), [])

    def test_wrong_topic_cannot_be_standard_positive(self):
        data=fixture();data[3][0]['gold']['same_topic']=False
        self.assertIn('unsafe_standard_positive', validate_bundle(*data, expected=SPEC))

    def test_counts_rejected(self):
        data=fixture()
        self.assertIn('needs_counts',validate_bundle(*data,expected={**SPEC,'needs_rows':3}))

if __name__=='__main__':unittest.main()
