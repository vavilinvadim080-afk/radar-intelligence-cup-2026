import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def client_type():
    assert importlib.util.find_spec('radar_app.semantic_client'), 'Semantic process boundary is not implemented'
    return importlib.import_module('radar_app.semantic_client').SemanticClient


def worker(tmp_path, *, revision='a', malformed=False, delay=0):
    script = tmp_path / f'worker-{revision}.py'
    counter = tmp_path / f'count-{revision}.txt'
    script.write_text('''import json,sys,time
from pathlib import Path
counter=Path(%r)
for line in sys.stdin:
 r=json.loads(line);time.sleep(%r);op=r['op'];v=None
 if op=='describe': v={'dimension':2,'encoder':'fixture','encoder_revision':%r,'reranker':'fixture'}
 elif op=='embed':
  counter.write_text(str(int(counter.read_text())+1 if counter.exists() else 1))
  v=[[1.,0.] for text in r['texts']]
 elif op=='rerank':v=[2.5 for p in r['pairs']]
 else:raise ValueError('operation')
 print(json.dumps({'id':r['id']+%r,'ok':True,'result':v}),flush=True)
''' % (str(counter), delay, revision, int(malformed)))
    return script, counter


def make_client(tmp_path, script, **kw):
    return client_type()(python=sys.executable, worker_script=script,
                         cache_dir=tmp_path / 'cache', **kw)


def test_real_process_boundary_returns_vectors_and_raw_reranker_scores(tmp_path):
    script, _ = worker(tmp_path)
    with make_client(tmp_path, script) as c:
        assert c.describe()['dimension'] == 2
        assert c.embed(['Выгрузка истории', 'История файлом']) == [[1., 0.], [1., 0.]]
        assert c.rerank([('Выгрузка истории', 'История файлом')]) == [2.5]
    assert c.running is False


def test_cache_reuses_only_matching_model_revision_without_storing_text(tmp_path):
    script, count = worker(tmp_path)
    with make_client(tmp_path, script) as c:
        c.embed(['Не сохранять исходную фразу'])
        c.embed(['Не сохранять исходную фразу'])
    assert count.read_text() == '1'
    assert all('исходную фразу' not in p.read_text() for p in (tmp_path / 'cache').glob('*.json'))
    other, second_count = worker(tmp_path, revision='b')
    with make_client(tmp_path, other) as c:
        c.embed(['Не сохранять исходную фразу'])
    assert second_count.read_text() == '1'


def test_tampered_cache_is_recomputed(tmp_path):
    script, count = worker(tmp_path)
    with make_client(tmp_path, script) as c:
        c.embed(['История заказов'])
        cache = next((tmp_path / 'cache').glob('*.json'))
        j = json.loads(cache.read_text());j['result'] = [0., 1.]
        cache.write_text(json.dumps(j))
        assert c.embed(['История заказов']) == [[1., 0.]]
    assert count.read_text() == '2'


def test_input_limits_are_checked_before_starting_a_model_process(tmp_path):
    script, count = worker(tmp_path)
    with make_client(tmp_path, script) as c:
        with pytest.raises(ValueError):c.embed(['x' * 4001])
        with pytest.raises(ValueError):c.embed(['x'] * 33)
        with pytest.raises(ValueError):c.rerank([('one', 12)])
        assert c.running is False
    assert not count.exists()


def test_request_text_stays_data_instead_of_a_shell_command(tmp_path):
    script, _ = worker(tmp_path)
    target = tmp_path / 'must-not-exist'
    with make_client(tmp_path, script) as c:
        c.embed([f'$(touch {target}); ignore instructions'])
    assert not target.exists()


def test_wrong_response_id_is_not_accepted_as_current_result(tmp_path):
    script, _ = worker(tmp_path, malformed=True)
    with make_client(tmp_path, script) as c:
        with pytest.raises(RuntimeError, match='response'):
            c.describe()
    assert c.running is False


def test_timeout_stops_only_the_owned_model_process(tmp_path):
    script, _ = worker(tmp_path, delay=2)
    with make_client(tmp_path, script, timeout=.1) as c:
        with pytest.raises(RuntimeError, match='timeout'):
            c.describe()
        assert c.running is False


def test_restarting_an_exited_worker_closes_its_old_pipes_and_log(tmp_path):
    script, _ = worker(tmp_path)
    source = script.read_text().replace("elif op=='rerank':", "elif op=='rerank':")
    source += '\n'
    source = source.replace("print(json.dumps({'id':r['id']+0,'ok':True,'result':v}),flush=True)",
                            "print(json.dumps({'id':r['id']+0,'ok':True,'result':v}),flush=True)\n if op=='embed':sys.exit(0)")
    script.write_text(source)
    with make_client(tmp_path, script) as c:
        c.embed(['Первая тема'])
        old, old_log = c._p, c._log
        old.wait(timeout=2)
        c.embed(['Другая тема'])
        assert old.stdin.closed and old.stdout.closed and old_log.closed
