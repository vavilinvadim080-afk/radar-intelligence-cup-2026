import numpy as np
import pytest


def test_runtime_closes_each_model_before_opening_the_next():
    from radar_app.local_selection_runtime_v5c import LiveStore
    active=[];events=[]
    class Client:
        def __init__(self,name):self.name=name
        def __enter__(self):
            assert not active
            active.append(self.name);events.append(('open',self.name));return self
        def __exit__(self,*args):active.pop();events.append(('close',self.name))
        def embed(self,texts):return [[1.]+[0.]*1023 for _ in texts]
        def nli(self,pairs):return [dict(entailment=.8,contradiction=.1,neutral=.1) for _ in pairs]
        def rerank(self,pairs):return [.9]*len(pairs)
    store=LiveStore(lambda n:Client(n),lambda:Client('reranker'))
    store.prepare('query',['a','b'])
    assert [kind for event,kind in events if event=='open']==['qwen','nli','reranker']
    assert len(store.kind_scores('query'))==5
    assert len(store.data['reranker'])==4
    assert not active
    store.prepare('query',['query'])
    assert len(store.data['reranker'])==1
    assert store.relation('query','query')['entailment']==.8
    store.prepare('kind',['reference'])
    assert len(store.kind_scores('kind'))==5
    assert store.relation('kind','reference')['entailment']==.8


def test_auto_reply_is_refused_before_loading_any_model_when_release_failed():
    from radar_app.local_selection_runtime_v5c import LiveAnalyzer
    analyzer=object.__new__(LiveAnalyzer)
    analyzer.quality=dict(models={'candidate':dict(gates=dict(target_80_met=False,zero_false_positive_replies=True))})
    result=analyzer.decide_reply(dict(query=dict(text='Нужен отчёт'),references=[]),config={})
    assert result['action']=='operator' and result['eligible'] is False


def test_unprepared_reference_is_abstention_not_a_silent_rule_fallback():
    from radar_app.local_selection_runtime_v5c import LiveStore
    store=LiveStore(None,None)
    with pytest.raises(KeyError):store.relation('unprepared','reference')
