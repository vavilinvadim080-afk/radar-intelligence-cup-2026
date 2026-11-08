"""Эмбеддинг текста в вектор. Два варианта:

 tfidf: символьные n-граммы 3-5 + TruncatedSVD. Работает офлайн, быстро, терпит опечатки
        (опечатка портит несколько n-грамм из сотни).
 nomic: nomic-embed-text через локальный Ollama (http://localhost:11434). Нужен запущенный Ollama.

Оба возвращают строки длины 1 (L2-нормировка), поэтому скалярное произведение = косинус.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .mask import strip_tokens

OLLAMA_URL = "http://localhost:11434"


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, svd_dim: int = 128, seed: int = 0):
        # char_wb: n-граммы внутри слов с пробелами по краям; sublinear_tf гасит влияние повторов; max_df=0.2 выкидывает n-граммы, которые есть в каждом пятом тексте (приветствия, "не работает")
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.2, sublinear_tf=True, lowercase=True)
        self.svd_dim = svd_dim
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=seed)

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        """Обучаем словарь и SVD на прогревочных данных (первые дни). Дальше только transform."""
        clean = [strip_tokens(t) for t in texts]
        try:
            X = self.vec.fit_transform(clean)
        except ValueError:   # на крошечной выборке max_df выкидывает всё: снимаем ограничение
            self.vec.set_params(max_df=1.0)
            X = self.vec.fit_transform(clean)
        self.svd.n_components = min(self.svd_dim, X.shape[1] - 1)
        self.svd.fit(X)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        X = self.vec.transform([strip_tokens(t) for t in texts])
        return normalize(self.svd.transform(X))


class OllamaEmbedder:
    name = "nomic"

    def __init__(self, model: str = "nomic-embed-text", cache_dir: str | None = None, batch: int = 64):
        self.model, self.batch = model, batch
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def available() -> bool:
        try:
            with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2) as r:
                names = [m["name"] for m in json.load(r)["models"]]
            return any(n.startswith("nomic-embed-text") for n in names)
        except Exception:
            return False

    def fit(self, texts):   # обучения нет, оставлено для единого интерфейса
        return self

    def _call(self, texts: list[str]) -> list:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(OLLAMA_URL + "/api/embed", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)["embeddings"]

    def transform(self, texts: list[str]) -> np.ndarray:
        clean = [strip_tokens(t).strip() or "пусто" for t in texts]
        out = [None] * len(clean)
        todo = []
        for i, t in enumerate(clean):
            f = self.cache / (hashlib.md5(t.encode()).hexdigest() + ".npy") if self.cache else None
            if f is not None and f.exists():
                out[i] = np.load(f)
            else:
                todo.append(i)
        # одинаковые тексты считаем один раз
        uniq = {}
        for i in todo:
            uniq.setdefault(clean[i], []).append(i)
        keys = list(uniq)
        for s in range(0, len(keys), self.batch):
            chunk = keys[s:s + self.batch]
            for k, v in zip(chunk, self._call(chunk)):
                v = np.asarray(v, dtype=np.float32)
                if self.cache:
                    np.save(self.cache / (hashlib.md5(k.encode()).hexdigest() + ".npy"), v)
                for i in uniq[k]:
                    out[i] = v
        return normalize(np.vstack(out))


def make_embedder(kind: str, svd_dim: int = 128, cache_dir: str | None = None):
    """Explicit tfidf, nomic, local-e5 or local-qwen; never substitute an algorithm."""
    if kind in {'local-e5','local-qwen'}:
        from radar_app.legacy_semantic import OfflineSemanticEmbedder
        return OfflineSemanticEmbedder(kind.removeprefix('local-'))
    if kind == "nomic":
        if not OllamaEmbedder.available():raise RuntimeError('Requested nomic embedding is unavailable')
        return OllamaEmbedder(cache_dir=cache_dir)
    if kind == 'tfidf':return TfidfEmbedder(svd_dim)
    raise ValueError('Unknown embedding backend')


def embedding_identity(kind):
    if kind in {'local-e5','local-qwen'}:
        from radar_app.legacy_semantic import embedding_identity as identity
        return identity(kind)
    return {'kind':kind}
