"""Признак координированной накрутки: много разных людей пишут почти один и тот же текст за короткое время.

Какие поля в данных есть (Ticket): время, продукт, канал, платформа, версия приложения, user_id, текст. Возраста аккаунта, устройства
и истории аккаунта в потоке нет, поэтому признак строится по тексту и времени:
  * доля людей окна, чей текст почти совпадает с самой большой группой (dup_share): токены слов длиной от 3 букв, сходство Жаккара
    не ниже brigade_dup_sim. Один пользователь считается один раз (последний текст), поэтому флуд одного аккаунта сюда не попадает;
  * равномерность по времени внутри группы (gap_cv, отношение стандартного отклонения пауз к средней паузе): у людей около 1
    (поток Пуассона), у скрипта с таймером около 0.
Подозрение: группа не меньше brigade_min_group людей и доля не ниже brigade_dup_share, ИЛИ доля не ниже brigade_timer_share при
gap_cv ниже brigade_timer_cv. Короткие тексты (меньше brigade_min_tokens слов) доказательством не считаются: «не работает» пишут многие
и честно. fresh_share (доля людей группы, которых раньше не видели в потоке) только считается и пишется в карточку, в решении не участвует:
в синтетике и у настоящих всплесков почти все авторы «новые», признак не различает (см. STATUS-antispam.md).

Без pipeline и без app: чистые функции, чтобы тесты вызывали их с руками собранными строками."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Callable, Optional

import numpy as np

_TOK = re.compile(r"[a-zа-яё]+")
MAX_ROWS = 400          # на анализ идут не больше стольких самых свежих людей: порядок цены O(n * групп)


def tokens(text: str) -> frozenset:
    return frozenset(w for w in _TOK.findall((text or "").lower().replace("ё", "е")) if len(w) >= 3)


def _jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def users_without_duplicates(n_users: int, dup_users: int) -> int:
    """Сколько людей остаётся, если группу из dup_users одинаковых считать одним голосом."""
    return n_users - max(dup_users, 1) + 1


def assess(rows: list, cfg, seen_before: Optional[Callable[[str], bool]] = None) -> dict:
    """rows: [(user_id, ts, masked_text)] обращений окна. Возвращает показатели и вердикт suspect (см. модуль)."""
    last: dict = {}
    for user, ts, text in sorted(rows, key=lambda r: r[1]):
        last[user] = (ts, text)                       # один человек = один голос, берём его последний текст
    people = list(last.items())[-MAX_ROWS:]
    out = dict(suspect=False, reason="", n_users=len(last), n_assessed=0, dup_users=0, dup_share=0.0, gap_cv=None,
               fresh_share=None, group_users=[])
    items = [(u, ts, tokens(tx)) for u, (ts, tx) in people]
    items = [it for it in items if len(it[2]) >= cfg.brigade_min_tokens]
    out["n_assessed"] = len(items)
    if len(items) < cfg.brigade_min_users:
        return out
    reps, groups = [], []                             # жадные группы по первому представителю
    for it in items:
        for k, r in enumerate(reps):
            if _jaccard(it[2], r) >= cfg.brigade_dup_sim:
                groups[k].append(it)
                break
        else:
            reps.append(it[2])
            groups.append([it])
    top = max(groups, key=len)
    out["dup_users"], out["dup_share"] = len(top), len(top) / len(items)
    out["group_users"] = [u for u, _, _ in top]
    times = sorted(ts.timestamp() for _, ts, _ in top)
    gaps = np.diff(times)
    if len(gaps) >= 2 and gaps.mean() > 0:
        out["gap_cv"] = float(gaps.std() / gaps.mean())
    if seen_before is not None:
        out["fresh_share"] = sum(1 for u, _, _ in top if not seen_before(u)) / len(top)
    by_share = len(top) >= cfg.brigade_min_group and out["dup_share"] >= cfg.brigade_dup_share
    by_timer = (len(top) >= cfg.brigade_min_group and out["dup_share"] >= cfg.brigade_timer_share
                and out["gap_cv"] is not None and out["gap_cv"] < cfg.brigade_timer_cv)
    if by_share or by_timer:
        out["suspect"] = True
        out["reason"] = (f"подозрение на накрутку: {len(top)} из {len(items)} людей пишут почти одинаково (доля {out['dup_share']:.2f})"
                         + (f", паузы равномерные (gap_cv {out['gap_cv']:.2f})" if by_timer and not by_share else ""))
    return out
