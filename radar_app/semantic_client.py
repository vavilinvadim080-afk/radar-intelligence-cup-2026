"""Bounded local process for optional semantic models; no network inference."""
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


class SemanticClient:
    def __init__(self, *, python=None, worker_script=None, cache_dir=None, timeout=120):
        self.python = str(python or os.environ.get('RADAR_SEMANTIC_PYTHON') or
                          ROOT.parent / 'Intelligence-Cup-2026/promo/.venv-tts/bin/python')
        self.worker_script = Path(worker_script or ROOT / 'scripts/semantic_worker.py').resolve()
        self.cache_dir = Path(cache_dir or ROOT / 'runtime/semantic-cache').resolve()
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 300:
            raise ValueError('Invalid model timeout')
        self.timeout = timeout
        self._p = None
        self._log = None
        self._desc = None
        self._serial = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def running(self):
        return self._p is not None and self._p.poll() is None

    def _start(self):
        if self.running:
            return
        if self._p is not None:
            self._stop()
        if not Path(self.python).is_file() or not self.worker_script.is_file():
            raise RuntimeError('Local semantic executable or worker is unavailable')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.cache_dir.parent / 'semantic-worker.log'
        self._log = open(log_path, 'ab') if not log_path.exists() or log_path.stat().st_size < 1024**2 else open(os.devnull, 'ab')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        self._p = subprocess.Popen([self.python, '-u', '-B', str(self.worker_script)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=self._log, env=env, bufsize=0)

    def _stop(self):
        p, self._p = self._p, None
        if p is not None:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(timeout=2)
            for stream in (p.stdin, p.stdout):
                if stream is not None:
                    stream.close()
        if self._log is not None:
            self._log.close(); self._log = None

    def _request(self, op, **args):
        if self._closed:
            raise RuntimeError('Semantic client is closed')
        self._start()
        self._serial += 1
        body = json.dumps(dict(id=self._serial, op=op, **args), ensure_ascii=False).encode() + b'\n'
        if len(body) > 1024**2:
            raise ValueError('Model request exceeds limit')
        try:
            self._p.stdin.write(body); self._p.stdin.flush()
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
            if response.get('id') != self._serial or response.get('ok') is not True:
                raise RuntimeError('Invalid local semantic response')
            return response['result']
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self._stop()
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError('Invalid local semantic response') from exc

    def describe(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('Semantic client is closed')
            if self._desc is None:
                d = self._request('describe')
                if (not isinstance(d, dict) or isinstance(d.get('dimension'), bool)
                        or not isinstance(d.get('dimension'), int) or not 1 <= d['dimension'] <= 8192):
                    self._stop(); raise RuntimeError('Invalid model description response')
                self._desc = d
            return dict(self._desc)

    def _key(self, op, value):
        raw = json.dumps([self.describe(), op, value], ensure_ascii=False, sort_keys=True,
                         separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()

    def _read(self, key):
        try:
            j = json.loads((self.cache_dir / (key + '.json')).read_text())
            raw = json.dumps(j['result'], separators=(',', ':'), allow_nan=False).encode()
            if hashlib.sha256(raw).hexdigest() != j['sha256']:
                return None
            return j['result']
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write(self, key, result):
        # Bounded cache: stop adding files at capacity; never evict user files.
        if sum(1 for _ in self.cache_dir.glob('*.json')) >= 4096:
            return
        raw = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
        blob = json.dumps({'result': result, 'sha256': hashlib.sha256(raw).hexdigest()})
        tmp = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=self.cache_dir, prefix='semantic-', suffix='.tmp', delete=False) as f:
                tmp = Path(f.name); f.write(blob)
            os.replace(tmp, self.cache_dir / (key + '.json'))
        except OSError:
            pass  # A full/read-only cache must not lose a successfully computed result.
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()

    @staticmethod
    def _texts(texts):
        if (not isinstance(texts, (list, tuple)) or len(texts) > 32
                or any(not isinstance(t, str) or not t.strip() or len(t) > 4000 for t in texts)):
            raise ValueError('At most 32 nonempty strings of up to 4000 characters required')

    def embed(self, texts):
        self._texts(texts)
        if not texts:
            return []
        with self._lock:
            dimension = self.describe()['dimension']
            keys = [self._key('embed', t) for t in texts]
            values = {key: self._read(key) for key in keys}
            missing = list(dict((key, text) for key, text in zip(keys, texts) if values[key] is None).items())
            if missing:
                answer = self._request('embed', texts=[t for _, t in missing])
                if not isinstance(answer, list) or len(answer) != len(missing):
                    self._stop(); raise RuntimeError('Invalid embedding response size')
                for (key, _), vector in zip(missing, answer):
                    self._check_vector(vector, dimension)
                    values[key] = vector; self._write(key, vector)
            for value in values.values():
                self._check_vector(value, dimension)
            return [values[k] for k in keys]

    @staticmethod
    def _check_vector(vector, dimension):
        if (not isinstance(vector, list) or len(vector) != dimension
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector)
                or abs(sum(x*x for x in vector) - 1) > .02):
            raise RuntimeError('Invalid normalized embedding response')

    def rerank(self, pairs):
        if not isinstance(pairs, (list, tuple)) or len(pairs) > 32:
            raise ValueError('At most 32 text pairs required')
        for p in pairs:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError('Text pair required')
            self._texts(p)
        if not pairs:
            return []
        with self._lock:
            keys = [self._key('rerank', list(p)) for p in pairs]
            scores = {k: self._read(k) for k in keys}
            missing = list(dict((k, list(p)) for k, p in zip(keys, pairs) if scores[k] is None).items())
            if missing:
                answer = self._request('rerank', pairs=[p for _, p in missing])
                if not isinstance(answer, list) or len(answer) != len(missing):
                    self._stop(); raise RuntimeError('Invalid reranker response size')
                for (key, _), score in zip(missing, answer):
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        self._stop(); raise RuntimeError('Invalid reranker response score')
                    scores[key] = score; self._write(key, score)
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores.values()):
                raise RuntimeError('Invalid cached reranker response')
            return [scores[k] for k in keys]

    def close(self):
        with self._lock:
            self._closed = True; self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
