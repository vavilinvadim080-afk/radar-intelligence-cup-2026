import copy
import pytest
from scripts.prepare_features_v5b import key,validate_feature_provenance,KIND_HYPOTHESES,RISK_HYPOTHESES


def fixture():
    needs=[dict(id='a',source_id='a',product='p',text='История заказов')]
    reply=[dict(id='r',query={'text':'История заказов'},references=[{'text':'История заказов'}])]
    metadata={k:{'dimension':2,'revision':'pinned'} for k in ['tfidf','e5','qwen','nli']}
    data=dict(source_sha256={'input':'abc'},needs_inputs=needs,reply_inputs=reply,
        preparation_contract_sha256='contract',metadata=metadata,
        vectors={k:{key('История заказов'):[1.,0.]} for k in ['tfidf','e5','qwen']},
        kinds={key('История заказов'):{k:.1 for k in KIND_HYPOTHESES}},
        risks={key('История заказов'):{k:.1 for k in RISK_HYPOTHESES}},relations={})
    return data,dict(sources={'input':'abc'},needs=needs,reply=reply,contract='contract',metadata=metadata)


def test_complete_feature_inventory_is_bound_to_inputs_models_and_contract():
    data,args=fixture();validate_feature_provenance(data,**args)
    for field,value in [('source_sha256',{'input':'wrong'}),('needs_inputs',[]),
                        ('preparation_contract_sha256','different')]:
        broken=copy.deepcopy(data);broken[field]=value
        with pytest.raises(RuntimeError):validate_feature_provenance(broken,**args)
    broken=copy.deepcopy(data);broken['metadata']['qwen']['revision']='other'
    with pytest.raises(RuntimeError):validate_feature_provenance(broken,**args)


def test_missing_risk_or_invalid_vector_is_not_a_silent_partial_evaluation():
    data,args=fixture();data['risks'][key('История заказов')].pop('money')
    with pytest.raises(RuntimeError):validate_feature_provenance(data,**args)
    data,args=fixture();data['vectors']['e5'][key('История заказов')]=[float('nan'),0.]
    with pytest.raises(RuntimeError):validate_feature_provenance(data,**args)
