import importlib
import importlib.util
import os

import pytest


def api():
    assert importlib.util.find_spec('radar_app.semantic_v5b'),'Bounded backend comparison is not implemented'
    return importlib.import_module('radar_app.semantic_v5b')


def test_only_pinned_named_backends_can_be_selected():
    with pytest.raises(ValueError):api().BackendClient('https://arbitrary-host/model')


def test_nli_label_mapping_and_probability_validation():
    check=api().check_nli
    assert check([.8,.15,.05])=={'entailment':.8,'neutral':.15,'contradiction':.05}
    for v in [[.8,.15],[1.,1.,0.],[float('nan'),0.,1.],[-.1,.1,1.],[True,0.,0.]]:
        with pytest.raises(RuntimeError):check(v)


def test_symmetric_equivalence_requires_both_entailment_directions_and_blocks_contradiction():
    f=api().equivalence
    assert f({'entailment':.9,'contradiction':.02},{'entailment':.2,'contradiction':.01})==.2
    assert f({'entailment':.8,'contradiction':.2},{'entailment':.9,'contradiction':.01})==0


def test_model_integrity_rejects_a_truncated_file_even_if_local_manifest_matches(tmp_path):
    import hashlib,json,struct
    from scripts.semantic_worker_v5b import verify_weights
    header=json.dumps({'a':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
    path=tmp_path/'model.safetensors'
    path.write_bytes(struct.pack('<Q',len(header))+header+b'1234')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='size'):
        verify_weights(path,digest)


@pytest.mark.skipif(os.environ.get('RADAR_TEST_V5B_MODELS')!='1',reason='Explicit sequential offline model smoke')
@pytest.mark.parametrize('backend',['e5','qwen','nli'])
def test_real_pinned_backend_smoke(backend,tmp_path):
    import json,time
    from pathlib import Path
    start=time.perf_counter()
    with api().BackendClient(backend,cache_dir=tmp_path/backend) as c:
        d=c.describe();assert d['offline'] is True and d['revision']
        if backend=='nli':
            values=c.nli([('Выгрузка заказов перестала работать.','Выгрузка заказов не работает.'),
                          ('Выгрузка заказов не работает.','Погода сегодня солнечная.')])
            assert values[0]['entailment']>values[1]['entailment']
            evidence={'relations':values}
        else:
            vectors=c.embed(['Нужна выгрузка прошлых заказов','Хочу скачать историю покупок файлом','Не приходит код входа'])
            assert len(vectors[0])==d['dimension']
            dot=lambda a,b:sum(x*y for x,y in zip(a,b))
            assert dot(vectors[0],vectors[1])>dot(vectors[0],vectors[2])
            evidence={'cosine':[[dot(a,b) for b in vectors] for a in vectors]}
        report=dict(metadata=d,evidence=evidence,elapsed_seconds=time.perf_counter()-start,
                    resources=c._request('stats'),meaning='Protocol and simple ranking smoke; not task quality')
        out=Path('reports/quality-v5b');out.mkdir(parents=True,exist_ok=True)
        (out/('smoke-'+backend+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
