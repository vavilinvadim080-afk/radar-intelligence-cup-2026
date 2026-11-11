"""Local access links must use the server's token and never start another server."""
import importlib.util
from pathlib import Path
import sys
import json
from urllib.parse import parse_qs,urlsplit
import pytest


@pytest.fixture
def launcher(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'path',list(sys.path))
    path=Path(__file__).resolve().parents[1]/'run.py'
    spec=importlib.util.spec_from_file_location('submission_launcher',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    directory=tmp_path/'mvp-v13-adapted/runtime';directory.mkdir(parents=True)
    (directory/'.secrets.json').write_text(json.dumps({'token':'test-generated-local-access'}))
    monkeypatch.setattr(module.runpy,'run_path',lambda *a,**k:pytest.fail('Access-link helper must not start a server'))
    return module


@pytest.mark.parametrize('token',[None,'test-explicit-server-token','test +/#&=server-token'])
def test_access_link_reuses_correct_token_without_server_or_dependencies(launcher,monkeypatch,capsys,token):
    monkeypatch.setenv('RADAR_API_TOKEN',token or '')
    monkeypatch.setattr(sys,'argv',['run.py','--access-url','--port','9891'])
    monkeypatch.setattr(launcher.importlib.util,'find_spec',lambda *_:pytest.fail('Link helper needs no server dependencies'))
    launcher.main()
    output=capsys.readouterr().out
    url=next(word for word in output.split() if word.startswith('http://'))
    assert url.startswith('http://127.0.0.1:9891/#')
    assert parse_qs(urlsplit(url).fragment)['token']==[token or 'test-generated-local-access']


def test_link_helper_does_not_generate_a_secret_when_server_has_never_started(launcher,monkeypatch):
    path=launcher.ROOT/'mvp-v13-adapted/runtime/.secrets.json';path.unlink()
    monkeypatch.setenv('RADAR_API_TOKEN','')
    monkeypatch.setattr(sys,'argv',['run.py','--access-url'])
    with pytest.raises(SystemExit) as error:launcher.main()
    assert error.value.code==2 and not path.exists()
