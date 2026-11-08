import pytest
from radar_app.evaluation import pair_metrics

def test_unknown_is_abstention_not_one_giant_cluster():
    r=pair_metrics(['a','a','b','b'],['x','x',None,None])
    assert r['tp']==1 and r['fp']==0 and r['fn']==1
    assert r['precision']==1 and r['recall']==.5
    assert r['coverage']==.5

def test_overmerge_and_split_are_counted_separately():
    r=pair_metrics(['a','a','b','b'],['x','x','x','y'])
    assert (r['tp'],r['fp'],r['fn'])==(1,2,1)
    assert r['precision']==pytest.approx(1/3)
    assert r['recall']==.5

def test_no_pairs_is_not_perfect_score():
    r=pair_metrics(['a','b'],[None,None])
    assert r['precision'] is None and r['recall'] is None
