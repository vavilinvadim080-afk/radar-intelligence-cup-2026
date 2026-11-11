"""Separate bounded verifier client; frozen semantic clients remain unchanged."""
import copy
import json
import math
import os
import selectors
import time
from .semantic_client import SemanticClient, ROOT


class RerankerClient(SemanticClient):
    def __init__(self, **kwargs):
        kwargs.setdefault('worker_script', ROOT/'scripts/reranker_worker_v5c.py')
        kwargs.setdefault('cache_dir', ROOT/'runtime/reranker-v5c')
        kwargs.setdefault('timeout', 180)
        super().__init__(**kwargs)
        self._generation = None
        self._retired = False

    def _start(self):
        if self._closed:
            raise RuntimeError('Semantic client is closed')
        if self._retired:
            raise RuntimeError('Verifier worker stopped; create a new client')
        if self._generation is not None:
            if self._p is not self._generation or not self.running:
                self._stop()
                raise RuntimeError('Verifier worker exited; create a new client')
            return
        try:
            super()._start()
        except (OSError, RuntimeError):
            self._stop()
            raise
        self._generation = self._p

    def _stop(self):
        if self._generation is not None:
            self._retired = True
        self._desc = None
        super()._stop()

    def describe(self):
        with self._lock:
            self._start()
            return copy.deepcopy(super().describe())

    @staticmethod
    def _valid_score(value):
        # Range first also rejects arbitrarily large JSON integers without overflow.
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and 0 <= value <= 1 and math.isfinite(value))

    def _read(self, key):
        value = super()._read(key)
        return value if self._valid_score(value) else None

    def _write(self, key, result):
        if not self._valid_score(result):
            self._stop()
            raise RuntimeError('Invalid verifier score; cache write refused')
        self._start()
        super()._write(key, result)

    def _request(self, op, **args):
        # Keep this stricter envelope boundary local to v5c; v5b stays frozen.
        with self._lock:
            self._start()
            self._serial += 1
            try:
                body = json.dumps(dict(id=self._serial, op=op, **args),
                                  ensure_ascii=False, allow_nan=False).encode() + b'\n'
                if len(body) > 1024**2:
                    raise RuntimeError('Model request exceeds limit')
                self._p.stdin.write(body)
                self._p.stdin.flush()
                deadline = time.monotonic() + self.timeout
                buf = bytearray()
                with selectors.DefaultSelector() as selector:
                    selector.register(self._p.stdout, selectors.EVENT_READ)
                    while b'\n' not in buf:
                        left = deadline - time.monotonic()
                        if left <= 0 or not selector.select(left):
                            raise RuntimeError('Local semantic model timeout')
                        part = os.read(self._p.stdout.fileno(), 65536)
                        if not part:
                            raise RuntimeError('Local semantic model closed its response')
                        buf.extend(part)
                        if len(buf) > 1024**2:
                            raise RuntimeError('Local semantic response exceeds limit')
                response = json.loads(bytes(buf).strip())
                if (not isinstance(response, dict)
                        or type(response.get('id')) is not int
                        or response['id'] != self._serial or response.get('ok') is not True
                        or 'result' not in response):
                    raise RuntimeError('Invalid local semantic response')
                result = response['result']
                if op == 'rerank' and (not isinstance(result, list)
                        or len(result) != len(args['pairs'])
                        or any(not self._valid_score(score) for score in result)):
                    raise RuntimeError('Invalid verifier scores or response size')
                return result
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                self._stop()
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError('Invalid local semantic response') from exc

    def rerank(self, pairs):
        if not isinstance(pairs, (list, tuple)) or len(pairs) > 4:
            raise ValueError('At most four pairs per verifier call')
        with self._lock:
            result = super().rerank(pairs)
            if pairs:
                self._start()  # Cache hits must also belong to a live generation.
            return result
