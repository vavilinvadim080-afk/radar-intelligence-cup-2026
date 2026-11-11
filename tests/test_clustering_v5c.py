"""History clustering uses only observed messages and refuses contradictory bridges."""
from radar_app.selection_v5c import CandidateSelector


class Evidence:
    def __init__(self, scores):
        self.scores = scores

    def kind_scores(self, text):
        if text == 'early':
            return dict(wish=.34, bug=.04, service=.02, usability=.1, unknown=.5)
        return dict(wish=.8, bug=.05, service=.05, usability=.05, unknown=.05)

    def cosine(self, backend, a, b):
        return .9

    def pair_score(self, a, b):
        return self.scores.get(frozenset((a,b)))


def rows(*texts):
    return [dict(id=str(i), source_id=str(i), product='p', text=text, created_at=str(i))
            for i,text in enumerate(texts)]


def selector(scores):
    return CandidateSelector(Evidence(scores),backend='qwen',config=dict(
        cluster_policy='history', pair_min=.8, pair_veto=.2, kind_min=0.,
        kind_margin=0., context_kind_margin=1., context_kind_min=.03,
        context_unknown_max=.6, unknown_pair_min=.9))


def test_later_observed_paraphrase_can_resolve_an_early_uncertain_message():
    predicted = selector({frozenset(('early','clear')):.97}).predict(rows('early','clear'))
    assert predicted[0]['prediction'] == predicted[1]['prediction']
    assert predicted[0]['kind'] == 'wish'


def test_weak_endpoint_blocks_transitive_bridge():
    predicted = selector({frozenset(('a','b')):.98, frozenset(('b','c')):.95,
                          frozenset(('a','c')):.1}).predict(rows('a','b','c'))
    assert predicted[0]['prediction'] == predicted[1]['prediction']
    assert predicted[2]['prediction'] != predicted[0]['prediction']


def test_duplicate_delivery_never_adds_a_second_vote():
    data = rows('a','b'); data.append(dict(data[0], received_at='later'))
    predicted = selector({frozenset(('a','b')):.98}).predict(data)
    assert len(predicted) == 2


def test_opposite_requests_and_different_products_stay_apart():
    a,b='Отключите письма','Хочу получать письма'
    pred = selector({frozenset((a,b)):.99}).predict(rows(a,b))
    assert pred[0]['prediction'] != pred[1]['prediction']
    data=rows('a','b');data[1]['product']='another'
    pred = selector({frozenset(('a','b')):.99}).predict(data)
    assert pred[0]['prediction'] != pred[1]['prediction']
