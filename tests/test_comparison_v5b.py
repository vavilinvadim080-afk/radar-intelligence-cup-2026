from scripts.compare_semantic_v5b import needs_rank,score_needs,score_reply


def test_metrics_count_pairs_and_unknowns_instead_of_inventing_card_quality():
    gold=[dict(id=k,gold=dict(kind='wish',topic_id=topic)) for k,topic in [('a','x'),('b','x'),('c','y')]]
    pred=[dict(id=k,kind='wish',prediction=['p','wish',topic]) for k,topic in [('a','one'),('b','one'),('c','one')]]
    m=score_needs(pred,gold)
    assert m['n']==3 and m['pair_metrics']['tp']==1 and m['pair_metrics']['fp']==2
    assert m['pair_metrics']['precision']==1/3 and m['pair_metrics']['recall']==1


def test_standard_and_fixed_notice_are_distinct_denominators():
    gold=[dict(id=k,gold={'safe_action':action}) for k,action in [('a','queue_standard'),('b','operator'),('c','queue_fixed_notice')]]
    pred=[dict(id=k,action=action) for k,action in [('a','queue_standard'),('b','queue_standard'),('c','queue_fixed_notice')]]
    m=score_reply(pred,gold)
    assert m['tp']==1 and m['fp']==1 and m['fixed_notice']['tp']==1
    assert m['expected_standard']==1 and m['queued_standard']==2


def test_model_selection_obeys_declared_precision_constraint_before_f1():
    high_precision={'metrics':{'pair_metrics':{'precision':.9,'recall':.5}}}
    high_recall={'metrics':{'pair_metrics':{'precision':.8,'recall':.95}}}
    assert needs_rank(high_precision)>needs_rank(high_recall)
