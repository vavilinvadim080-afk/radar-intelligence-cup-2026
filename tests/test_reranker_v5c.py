"""No model loading: contract guards for the separate pair verifier."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def worker():
    spec = importlib.util.spec_from_file_location('reranker_worker', ROOT/'scripts/reranker_worker_v5c.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_instruction_is_for_equivalence_not_generic_relevance():
    mod = worker()
    prompt = mod.format_pair('Не присылайте письма', 'Нужны письма после покупки')
    assert '<Query>: Не присылайте письма' in prompt
    assert '<Document>: Нужны письма после покупки' in prompt
    assert 'same user intent' in prompt and 'opposite' in prompt
    assert 'answer can only be' in mod.PREFIX


@pytest.mark.parametrize('value', [None, [], [['a']], [['a', '']], [['a', 'b', 'c']], [['a'*4001, 'b']], [['a', 'b']]*5])
def test_rejects_unbounded_or_invalid_pairs(value):
    with pytest.raises(ValueError):
        worker().validate_pairs(value)


def test_valid_pair_and_gold_are_separate():
    mod = worker()
    mod.validate_pairs([['исходный текст', 'проверяемый текст']])
    with pytest.raises(ValueError):
        mod.validate_pairs([{'query': 'a', 'document': 'b', 'gold': True}])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1, True])
def test_invalid_probability_never_becomes_a_match(value):
    with pytest.raises(RuntimeError):
        worker().check_score(value)
