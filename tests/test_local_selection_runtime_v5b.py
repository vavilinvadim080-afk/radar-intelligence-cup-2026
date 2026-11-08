import importlib
import json


def test_live_adapter_closes_each_model_before_opening_the_other_and_keeps_evidence(tmp_path):
    from radar_app.core import validate_analysis
    api=importlib.import_module('radar_app.local_selection_runtime_v5b')
    heads={'qwen':{'kind':{'classes':['wish','unknown'],'coefficients':[[2.,0.],[-2.,0.]],'intercept':[0.,0.]},
                   'pair':{'coefficients':[0.]*7,'intercept':2.}}}
    path=tmp_path/'heads.json';path.write_text(json.dumps(heads))
    class Fake:
        live=set();events=[]
        def __init__(self,backend):self.backend=backend
        def __enter__(self):
            assert not self.live;self.live.add(self.backend);self.events.append(('open',self.backend));return self
        def __exit__(self,*args):self.live.remove(self.backend);self.events.append(('close',self.backend))
        def embed(self,texts):return [[1.,0.] for _ in texts]
        def nli(self,pairs):return [dict(entailment=.8,neutral=.18,contradiction=.02) for _ in pairs]
    analyzer=api.LiveAnalyzer(backend='qwen',heads_path=path,config={'pair_min':.7,'risk_min':.99},client_factory=Fake)
    text='Нужна история заказов'
    result=analyzer(text,[dict(topic_key='existing',kind='wish',label='История покупок',examples=['Хочу список прошлых покупок'])])
    assert result['topic_key']=='existing' and result['quote']==text
    validate_analysis(result,text)
    assert Fake.events==[('open','qwen'),('close','qwen'),('open','nli'),('close','nli')]
    assert not Fake.live


def test_fourth_reference_still_participates_in_permission_and_product_checks(tmp_path):
    api=importlib.import_module('radar_app.local_selection_runtime_v5b')
    heads={'qwen':{'kind':{'classes':['wish','unknown'],'coefficients':[[2.,0.],[-2.,0.]],'intercept':[0.,0.]},
                   'pair':{'coefficients':[0.]*7,'intercept':2.}}}
    path=tmp_path/'heads.json';path.write_text(json.dumps(heads))
    class Fake:
        def __init__(self,backend):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def embed(self,texts):return [[1.,0.] for _ in texts]
        def nli(self,pairs):return [dict(entailment=.8,neutral=.18,contradiction=.02) for _ in pairs]
    analyzer=api.LiveAnalyzer(backend='qwen',heads_path=path,config={'pair_min':.7},client_factory=Fake)
    ref=dict(text='Нужна история заказов',product='p',kind='wish',source_verified=True)
    for fourth in [dict(ref,source_verified=False),dict(ref,product='other')]:
        payload=dict(id='r',event='incoming_ticket',product='p',kind='wish',approved=True,state='open',
            query=dict(id='q',text='Нужна история заказов',product='p',consent=True),references=[dict(ref) for _ in range(3)]+[fourth])
        assert analyzer.decide_reply(payload,config={'pair_min':.7,'risk_min':.99})['action']=='operator'
