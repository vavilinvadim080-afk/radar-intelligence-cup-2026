"""Кластеризация: онлайн (на лету) + суточная пересборка HDBSCAN для склейки расползшихся тем.

Онлайн: берём ближайший по косинусу центр кластера; если сходство выше порога, обращение
идёт в него (центр сдвигается), иначе рождается новый кластер. Работает по одному обращению.
HDBSCAN раз в сутки смотрит на последние дни целиком и подсказывает, какие онлайн-кластеры
на самом деле одна тема. Мы тогда их склеиваем.

Дедупликация «один user_id = один голос» живёт в pipeline.py, там, где считаются голоса.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN


class OnlineClusterer:
    """Центры кластеров одного продукта. Номера кластеров = индексы строк."""

    def __init__(self, dim: int, threshold: float):
        self.threshold = threshold
        self.sums = np.zeros((64, dim))      # сумма векторов членов кластера
        self.unit = np.zeros((64, dim))      # та же сумма, но нормированная: по ней считаем косинус
        self.counts = np.zeros(64, dtype=int)
        self.n = 0                           # сколько кластеров создано
        self.alias: dict[int, int] = {}      # влитый кластер -> тот, в который влили

    def resolve(self, k: int) -> int:
        while k in self.alias:
            k = self.alias[k]
        return k

    def assign(self, v: np.ndarray) -> int:
        """Возвращает номер кластера для вектора v (длина 1) и обновляет его центр."""
        if not np.any(v):   # пустой текст (одни токены или знаки): отправляем в общий «мусорный» кластер
            v = np.zeros_like(v)
            v[0] = 1.0
        if self.n:
            sims = self.unit[:self.n] @ v   # у влитых кластеров строка нулевая, они не выиграют
            k = int(sims.argmax())
            if sims[k] >= self.threshold:
                self._add(k, v)
                return k
        return self._new(v)

    def _new(self, v) -> int:
        if self.n == len(self.sums):         # место кончилось: удваиваем
            self.sums = np.vstack([self.sums, np.zeros_like(self.sums)])
            self.unit = np.vstack([self.unit, np.zeros_like(self.unit)])
            self.counts = np.concatenate([self.counts, np.zeros_like(self.counts)])
        k = self.n
        self.n += 1
        self._add(k, v)
        return k

    def _add(self, k, v):
        self.sums[k] += v
        self.counts[k] += 1
        self.unit[k] = self.sums[k] / np.linalg.norm(self.sums[k])

    def merge(self, keep: int, drop: int):
        """Вливаем drop в keep. Дальше drop в assign не участвует (его строка обнуляется)."""
        self.sums[keep] += self.sums[drop]
        self.counts[keep] += self.counts[drop]
        self.unit[keep] = self.sums[keep] / np.linalg.norm(self.sums[keep])
        self.sums[drop] = 0
        self.unit[drop] = 0
        self.alias[drop] = keep


def hdbscan_groups(X: np.ndarray, online_ids: np.ndarray, min_cluster_size: int = 25, merge_share: float = 0.6,
                   max_points: int = 4000, seed: int = 0) -> list[list[int]]:
    """Какие онлайн-кластеры склеить. X: векторы последних дней (строки длины 1), online_ids: онлайн-кластер каждой строки.

    Возвращает список групп [[3, 17, 40], [5, 9]]: внутри группы кластеры, которые HDBSCAN считает одной темой.
    Кластер попадает в группу, только если не менее merge_share его членов лежит в одной группе HDBSCAN
    (иначе он «пограничный», трогать не будем).
    """
    if len(X) > max_points:   # HDBSCAN дорог по числу точек, берём выборку
        pick = np.random.default_rng(seed).choice(len(X), max_points, replace=False)
        X, online_ids = X[pick], online_ids[pick]
    if len(X) < 2 * min_cluster_size:
        return []
    # для строк длины 1 евклидово расстояние монотонно косинусу: ||a-b||^2 = 2 - 2cos
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, metric="euclidean", copy=True).fit_predict(X)
    by_label: dict[int, list[int]] = {}
    for oid in np.unique(online_ids):
        lab = labels[online_ids == oid]
        vals, cnt = np.unique(lab[lab >= 0], return_counts=True)
        if len(vals) and cnt.max() / len(lab) >= merge_share:
            by_label.setdefault(int(vals[cnt.argmax()]), []).append(int(oid))
    return [g for g in by_label.values() if len(g) > 1]
