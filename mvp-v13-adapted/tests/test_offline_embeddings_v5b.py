import importlib
import importlib.util
import numpy as np
import pytest


def kind():
    assert importlib.util.find_spec('radar_app.legacy_semantic'),'Offline semantic adapter is not implemented'
    return importlib.import_module('radar_app.legacy_semantic').OfflineSemanticEmbedder


class FakeClient:
    calls=[]
    def __init__(self,backend):self.backend=backend;self.closed=False
    def __enter__(self):return self
    def __exit__(self,*_):self.closed=True;self.calls.append('closed')
    def describe(self):return dict(encoder='fixture',revision='test-revision',dimension=2)
    def embed(self,texts):self.calls.append(texts);return [[1.,0.] if 'истори' in t else [0.,1.] for t in texts]


def test_adapter_has_no_torch_dependency_and_reuses_duplicate_masked_texts():
    FakeClient.calls=[]
    m=kind()('e5',client_factory=FakeClient)
    assert m.fit(['Прогрев']) is m
    v=m.transform(['Нужна история','Нужна история','Код входа'])
    np.testing.assert_array_equal(v,[[1.,0.],[1.,0.],[0.,1.]])
    assert FakeClient.calls==[['Нужна история','Код входа'],'closed']


def test_semantic_adapter_failure_does_not_become_a_frequency_result():
    class Broken(FakeClient):
        def embed(self,texts):raise RuntimeError('Model unavailable')
    with pytest.raises(RuntimeError):kind()('e5',client_factory=Broken).transform(['Новая история'])


def test_factory_keeps_baseline_and_exposes_explicit_semantic_options():
    from radar.embed import make_embedder,TfidfEmbedder
    assert isinstance(make_embedder('tfidf'),TfidfEmbedder)
    assert isinstance(make_embedder('local-e5'),kind())
    assert isinstance(make_embedder('local-qwen'),kind())


def test_explicit_missing_ollama_and_unknown_kind_fail_instead_of_switching_algorithm(monkeypatch):
    from radar.embed import make_embedder,OllamaEmbedder
    monkeypatch.setattr(OllamaEmbedder,'available',staticmethod(lambda:False))
    with pytest.raises(RuntimeError):make_embedder('nomic')
    with pytest.raises(ValueError):make_embedder('typo-semantic')


def test_legacy_replay_semantic_cache_never_reuses_the_default_frequency_file(tmp_path,monkeypatch):
    from app import replay
    from datetime import datetime
    import radar.embed as embed
    monkeypatch.setattr(replay,'CACHE',tmp_path)
    monkeypatch.setattr(embed,'embedding_identity',lambda kind: {'kind':kind,'revision':'fixture'})
    builds=[]
    def build(seed=777,noise='mid',volume=800,*,cfg=None):
        chosen=cfg.embedder if cfg else 'tfidf';builds.append(chosen)
        return replay.Replay(seed,datetime(2026,1,1),datetime(2026,1,2),[],{}, {},[],[],{},embedder=chosen+'@fixture')
    monkeypatch.setattr(replay,'build',build)
    a=replay.load_or_build();b=replay.load_or_build(embedder='local-e5',sim_threshold=.8)
    c=replay.load_or_build(embedder='local-e5',sim_threshold=.8)
    assert a.embedder.startswith('tfidf') and b.embedder==c.embedder
    assert builds==['tfidf','local-e5']
    assert len(list(tmp_path.glob('*.json')))==2


def test_a_new_encoder_requires_its_own_explicit_clustering_threshold(monkeypatch):
    from app import replay
    monkeypatch.delenv('RADAR_SIM_THRESHOLD',raising=False)
    def not_build(*args,**kwargs):raise AssertionError('Reject missing threshold before building a replay')
    monkeypatch.setattr(replay,'build',not_build)
    with pytest.raises(ValueError,match='threshold'):
        replay.load_or_build(embedder='local-e5')
