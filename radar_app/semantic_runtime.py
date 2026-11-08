"""One shared optional worker per application process, shut down at exit."""
import atexit
import json
from pathlib import Path
import threading

from .semantic import SemanticAnalyzer,SemanticReplyMatcher
from .semantic_client import SemanticClient

_lock=threading.RLock()
_client=None
_analyzer=None
_matcher=None


def components():
    global _client,_analyzer,_matcher
    with _lock:
        if _client is None:
            config=json.loads((Path(__file__).resolve().parents[1]/'data/semantic-config-v5.json').read_text())
            _client=SemanticClient();atexit.register(_client.close)
            _analyzer=SemanticAnalyzer(_client,config=config['needs'])
            _matcher=SemanticReplyMatcher(_client,threshold=config['reply']['threshold'])
        return _analyzer,_matcher
