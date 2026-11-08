"""Кэш наборов для оценки: синтетика + замаскированные тексты + эмбеддинги TF-IDF.

Генерация, маскирование и эмбеддинг детерминированы по (seed, noise, volume), поэтому их
незачем считать заново на каждый конфиг. Кэш лежит в eval/cache/ds/, удалять можно в любой момент.
pickle здесь безопасен: файлы пишет и читает только этот скрипт на этой машине, чужие файлы сюда не кладутся.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "eval" / "cache" / "ds"


def get(seed: int, noise: str, volume: int = 800):
    """(ds, prepared): ds из radar.gen.generate, prepared = результат radar.pipeline._prepare (tfidf)."""
    from radar.config import Config
    from radar.gen import generate
    from radar.pipeline import _prepare
    if os.environ.get("RADAR_NO_CACHE"):                  # большие объёмы: кэш на диск не пишем и не читаем
        ds = generate(seed, noise, volume)
        tickets, texts, reverse, emb, X = _prepare(ds.tickets, Config())
        return ds, (tickets, texts, reverse, emb, X.astype("float32"))
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{seed}_{noise}_{volume}.pkl"
    if f.exists():
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            f.unlink(missing_ok=True)
    ds = generate(seed, noise, volume)
    tickets, texts, reverse, emb, X = _prepare(ds.tickets, Config())
    prepared = (tickets, texts, reverse, emb, X.astype("float32"))
    with open(f, "wb") as fh:
        pickle.dump((ds, prepared), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return ds, prepared
