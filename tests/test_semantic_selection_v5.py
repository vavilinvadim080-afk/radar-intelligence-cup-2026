"""Behavioral checks independent of tuning/control gold files."""
import importlib
import importlib.util
import math

import pytest


def api():
    assert importlib.util.find_spec('radar_app.semantic'), 'Semantic selection is not implemented'
    return importlib.import_module('radar_app.semantic')


def test_opposite_requested_directions_are_incompatible():
    s = api()
    assert s.intent_direction('Хочу отключить сообщения о заявке') == 'disable'
    assert s.intent_direction('Прошу включить оповещения о заявке') == 'enable'
    assert s.intent_direction('Не отключайте уведомления') == 'enable'
    assert s.intent_direction('Не включайте уведомления') == 'disable'
    assert not s.compatible_intent('Перестаньте присылать уведомления', 'Начните присылать уведомления')
    assert s.compatible_intent('Сломался экспорт', 'Файл экспорта пустой')


def test_generic_kind_cues_distinguish_new_function_regression_and_usability():
    s = api()
    assert s.kind_cue('Добавьте поддержку нового формата, такой функции нет') == 'wish'
    assert s.kind_cue('После обновления существующая выгрузка перестала работать') == 'bug'
    assert s.kind_cue('Кнопка работает, но её трудно найти') == 'usability'
    assert s.kind_cue('Предлагаю отдельный новый сервис обмена инструментами') == 'service'


class FixtureModels:
    def embed(self, texts):
        # Semantic fixture deliberately has no common words between the first two.
        v = {'Хочу сохранять покупки файлом': [1., 0.], 'Дайте выгрузку истории': [1., 0.],
             'Не приходит код': [0., 1.], 'Неясно': [.70710678, .70710678]}
        return [v[t] for t in texts]

    def rerank(self, pairs):
        return [2. if 'код' not in b else -5. for a, b in pairs]


def selector():
    return api().SemanticAnalyzer(FixtureModels(), kind_predictor=lambda text, vector: ('wish', .9),
                                  config={'join_cosine': .55, 'join_margin': .06,
                                          'reranker_min': -4., 'use_reranker': True,'content_weight':0})


def test_paraphrase_joins_dynamic_catalog_without_a_predefined_topic_dictionary():
    m = selector()
    a = m('Хочу сохранять покупки файлом', [])
    assert a['kind'] == 'wish' and a['quote'] == 'Хочу сохранять покупки файлом'
    b = m('Дайте выгрузку истории', [dict(a, examples=['Хочу сохранять покупки файлом'])])
    assert a['topic_key'] == b['topic_key']


def test_two_indistinguishable_neighbors_are_sent_for_review_instead_of_arbitrary_merge():
    m = selector()
    c = [dict(topic_key=k, kind='wish', label=k, examples=['Хочу сохранять покупки файлом']) for k in ['one','two']]
    assert m('Дайте выгрузку истории', c)['kind'] == 'unknown'


def test_different_kind_does_not_join_even_if_the_text_embedding_is_identical():
    m = selector()
    a = m('Дайте выгрузку истории', [dict(topic_key='bug', kind='bug', label='bug', examples=['Хочу сохранять покупки файлом'])])
    assert a['kind'] == 'wish' and a['topic_key'] != 'bug'


def test_model_exception_is_not_replaced_with_fake_semantic_success():
    m = selector()
    m.models.embed = lambda _: (_ for _ in ()).throw(RuntimeError('offline model unavailable'))
    with pytest.raises(RuntimeError): m('Дайте выгрузку истории', [])


def test_pair_features_are_symmetric_and_keep_the_two_representations_separate():
    s=api()
    a,b=[1.,0.],[0.,1.]
    c,d=[.6,.8],[.6,.8]
    assert s.pair_features(a,b,c,d)==s.pair_features(b,a,d,c)
    assert s.pair_features(a,b,c,d)[-2:]==[0.,1.]


def test_explicit_operational_objects_cannot_be_joined_by_shared_order_vocabulary():
    s=api()
    assert not s.compatible_object('Итоговое количество не равно сумме строк заказа',
                                   'Фильтр завершённых заказов показывает открытые')
    assert s.compatible_object('Экспорт заказов пустой','Выгрузка покупок без строк')
    assert not s.compatible_object('Не могу загрузить документ', 'Не могу скачать документ')
    assert s.compatible_object('Новая тема про инструмент', 'Синоним пока неизвестен')
