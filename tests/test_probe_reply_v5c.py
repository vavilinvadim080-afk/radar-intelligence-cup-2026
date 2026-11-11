import pytest
from scripts.probe_reply_v5c import reply_examples, prepare_folds


def test_pair_supervision_uses_semantics_never_permission_labels(monkeypatch):
    import scripts.probe_reply_v5c as p
    monkeypatch.setattr(p.t,'pair_inputs',lambda *_:[.8,.9,.9,.7,.1])
    row=dict(id='r',product='p',query=dict(product='p',text='a'),references=[dict(product='p',text='b')])
    examples=reply_examples(None,[row],{'r':dict(same_topic=True,safe_action='operator',eligibility=False)})
    assert examples[0]['label']==1
    other=dict(row,id='different')
    with pytest.raises(ValueError,match='Contradictory'):
        reply_examples(None,[row,other],{'r':dict(same_topic=True),'different':dict(same_topic=False)})


def test_mixed_unknown_references_and_repeated_queries_stay_in_one_fold(monkeypatch):
    import scripts.probe_reply_v5c as p
    monkeypatch.setattr(p.t,'training_pairs',lambda *_:[])
    needs=[];gold={};replies=[]
    for i in range(12):
        ident=str(i);text='text'+ident
        needs.append(dict(id=ident,text=text,product='p'))
        gold[ident]=dict(kind='unknown' if i<2 else 'wish',topic_id=None if i<2 else 'family'+ident)
        replies.append(dict(id='r'+ident,query=dict(text='query'+ident),references=[dict(text=text)]))
    replies[0]['references'].append(dict(text='text1'))
    replies[3]['query']['text']=replies[2]['query']['text']
    replies[-1]['references']=[]
    folds=prepare_folds(None,needs,gold,replies)
    assignments={row['id']:fold['fold'] for fold in folds for row in fold['validation']}
    assert assignments['r0']==assignments['r1']
    assert assignments['r2']==assignments['r3']
    assert len(assignments)==12
    for fold in folds:
        assert not {r['id'] for r in fold['train_replies']} & {r['id'] for r in fold['validation']}
