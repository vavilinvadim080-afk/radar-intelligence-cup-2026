"""Pure contract tests; no inference-library imports."""
import importlib
import pytest


def module():
    return importlib.import_module('scripts.reranker_worker_mlx_v5c')


def test_right_padding_preserves_valid_token_positions():
    rows=[[1,2,3],[4,5]]
    padded,positions,count=module().right_pad(rows)
    assert count==2 and positions==[2,1,2,2]
    assert len(padded)==4 and len(padded[0])==32
    assert padded[0][:3]==rows[0] and padded[1][:2]==rows[1]
    assert all(v==0 for v in padded[1][2:])


@pytest.mark.parametrize('rows',[[],[[]],[[True]],[[1]]*5,[[1]*513]])
def test_invalid_token_batches_fail_before_inference(rows):
    with pytest.raises(ValueError):module().right_pad(rows)


def test_shared_prefix_must_match_every_input_exactly():
    assert module().strip_prefix([[1,2,3],[1,2,4]],[1,2])==[[3],[4]]
    with pytest.raises(ValueError):module().strip_prefix([[1,3,4]],[1,2])
    with pytest.raises(ValueError):module().strip_prefix([[1,2]],[1,2])
