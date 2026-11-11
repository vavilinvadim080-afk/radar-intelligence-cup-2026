"""Behavior checks for the candidate, independent of control answers or neural models."""
import pytest


def test_opposite_notification_requests_never_merge():
    from radar_app.selection_v5c import compatible_direction
    assert not compatible_direction('Отключите письма после покупок', 'Хочу получать письма после каждой покупки')
    assert not compatible_direction('Не присылайте уведомления', 'Нужны уведомления об оплате')
    assert compatible_direction('Не отключайте письма', 'Хочу получать письма')
    assert compatible_direction('После покупки баллы не начислены', 'Бонусы за оплату не поступили')


def test_failed_existing_function_and_requested_feature_remain_distinct():
    from radar_app.selection_v5c import compatible_direction
    assert not compatible_direction('Экспорт раньше работал, теперь упал с ошибкой', 'Добавьте экспорт: такой функции нет')
    assert compatible_direction('Нужна выгрузка в Excel', 'Добавьте скачивание отчёта в xlsx')


@pytest.mark.parametrize('text', [
    'Кнопка не сломана, просто такой настройки нет',
    'Ничего не сломалось, но такой возможности пока нет',
    'Экспорт не сломан, нужна такая функция, которой сейчас нет',
])
def test_negated_failure_does_not_veto_a_missing_feature(text):
    from radar_app.selection_v5c import compatible_direction, functionality_state
    assert functionality_state(text) != 'broken'
    assert compatible_direction(text, 'Добавьте возможность: такой настройки нет')


def test_negation_does_not_erase_a_separate_actual_failure():
    from radar_app.selection_v5c import functionality_state, compatible_direction
    text='Вход не сломан, но экспорт перестал работать'
    assert functionality_state(text)=='broken'
    assert not compatible_direction(text, 'Добавьте экспорт: такой функции нет')


def test_missing_pair_evidence_is_abstention_not_a_fabricated_zero():
    from radar_app.selection_v5c import pair_inputs
    class Features:
        data = {'reranker': {}}
    assert pair_inputs(Features(), 'a', 'b') is None


def test_nonfinite_pair_evidence_is_rejected():
    from radar_app.selection_v5c import pair_inputs
    from scripts.prepare_features_v5b import pair_key
    class Features:
        data = {'reranker': {pair_key('a','b'): float('nan'), pair_key('b','a'): .9}}
    with pytest.raises(ValueError):
        pair_inputs(Features(), 'a', 'b')


class ContextFeatures:
    def kind_scores(self, text):
        return dict(wish=.46, usability=.42, bug=.06, service=.04, unknown=.02)

    def cosine(self, backend, a, b):
        return {'good': .9, 'missing': .8}.get(b, .7)

    def pair_score(self, a, b):
        return {'good': .95, 'missing': None}.get(b, .1)


def test_one_verified_paraphrase_can_support_matching_without_inventing_missing_scores():
    from radar_app.selection_v5c import CandidateSelector
    selector = CandidateSelector(ContextFeatures(), backend='qwen',
                                 config=dict(pair_min=.85, reference_policy='best'))
    assert selector.score_refs('query', ['good', 'missing']) == (.95, .9)
    assert selector.score_refs('query', ['missing']) is None
    assert selector.score_refs('query', ['bad']) is None


def test_strong_topic_context_resolves_close_kind_predictions_only_after_pair_verification():
    from radar_app.selection_v5c import CandidateSelector
    selector = CandidateSelector(ContextFeatures(), backend='qwen', config=dict(
        pair_min=.85, kind_min=0., kind_margin=0., context_kind_margin=.1, reference_policy='best'))
    row = dict(id='q', product='p', text='query')
    topic = dict(topic_key='known', label='known label', kind='usability', quote='good')
    result = selector.analyze(row, [topic])
    assert result['topic_key'] == 'known' and result['kind'] == 'usability'
    assert selector.trace['initial_kind'] == 'wish'
    topic['quote'] = 'bad'
    assert selector.analyze(row, [topic])['kind'] == 'wish'
    topic.update(kind='bug', quote='good')
    assert selector.analyze(row, [topic])['kind'] == 'wish'


def test_unknown_cannot_be_forced_into_a_topic_by_context():
    from radar_app.selection_v5c import CandidateSelector
    class Unknown(ContextFeatures):
        def kind_scores(self, text):
            return dict(wish=.1, usability=.1, bug=.1, service=.1, unknown=.6)
    selector = CandidateSelector(Unknown(), backend='qwen', config=dict(
        pair_min=.85, context_kind_margin=1., reference_policy='best'))
    result = selector.analyze(dict(id='q', product='p', text='query'), [dict(
        topic_key='known', label='known', kind='usability', quote='good')])
    assert result['kind'] == 'unknown' and result['topic_key'] == 'unknown'


def test_uncertain_unknown_needs_requires_stronger_existing_topic_evidence():
    from radar_app.selection_v5c import CandidateSelector
    class Uncertain(ContextFeatures):
        def kind_scores(self, text):
            return dict(wish=.35, usability=.18, bug=.04, service=.03, unknown=.4)
    selector = CandidateSelector(Uncertain(), backend='qwen', config=dict(
        pair_min=.6, unknown_pair_min=.9, context_unknown_max=.5,
        context_kind_margin=1., reference_policy='best'))
    row = dict(id='q', product='p', text='query')
    topic = dict(topic_key='known', label='known', kind='wish', quote='good')
    assert selector.analyze(row, [])['kind'] == 'unknown'
    assert selector.analyze(row, [topic])['topic_key'] == 'known'
    selector.config['unknown_pair_min'] = .99
    assert selector.analyze(row, [topic])['kind'] == 'unknown'


def test_polynomial_pair_features_are_symmetric_and_deterministic():
    from radar_app.selection_v5c import transform_pair_inputs
    assert transform_pair_inputs([2.,3.], 'poly2') == [2.,3.,4.,6.,9.]
    assert transform_pair_inputs([2.,3.], 'identity') == [2.,3.]
    with pytest.raises(ValueError):
        transform_pair_inputs([2.,3.], 'unverified')
