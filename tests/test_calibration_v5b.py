import numpy as np
from radar_app.calibrated_v5b import probabilities,pair_vector


def test_pair_features_are_symmetric_without_losing_contradiction():
    a=np.array([1.,0.]);b=np.array([.6,.8])
    f=dict(entailment=.8,contradiction=.1);r=dict(entailment=.3,contradiction=.4)
    x=pair_vector(a,b,f,r)
    np.testing.assert_allclose(x,pair_vector(b,a,r,f))
    np.testing.assert_allclose(x[-3:],[.6,.3,.4])


def test_large_logits_use_stable_softmax_and_preserve_the_class_order():
    head=dict(coefficients=[[1000.,0.],[0.,1000.]],intercept=[0.,0.],classes=['wish','bug'])
    p=probabilities(head,np.array([1.,0.]))
    assert p['wish']==1 and p['bug']==0 and sum(p.values())==1


def test_calibrated_matching_uses_the_learned_pair_score_not_a_hard_nli_veto():
    from radar_app.calibrated_v5b import CalibratedSelector
    class Features:
        def cosine(self,backend,a,b):return .8
        def pair_score(self,backend,a,b):return .9
        def relation(self,a,b):raise AssertionError('The calibrated feature provider owns NLI features')
    selector=CalibratedSelector(Features(),backend='qwen',config={'pair_min':.8})
    assert selector.score_refs('Нужна история заказов',['Хочу список прошлых покупок'])==(.9,.8)
    selector.config['pair_min']=.95
    assert selector.score_refs('Нужна история заказов',['Хочу список прошлых покупок']) is None
