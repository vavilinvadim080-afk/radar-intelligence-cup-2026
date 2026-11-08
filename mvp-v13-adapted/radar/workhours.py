"""Рабочие минуты: сколько минут между двумя моментами приходится на будни с start_h до end_h.

Нужно оценке веб-продуктов (b2b, portal, partners): регрессия, начавшаяся в 18:00 в пятницу, до утра понедельника
жалоб почти не собирает, и «6 часов» по календарю для неё не то же самое, что для приложения, которым пользуются ночью."""
from __future__ import annotations

from datetime import datetime, timedelta


def work_minutes(a: datetime, b: datetime, start_h: int = 9, end_h: int = 19) -> float:
    """Рабочие минуты на отрезке [a, b] (пн-пт, start_h-end_h, локальное время). Если b <= a: 0."""
    if b <= a:
        return 0.0
    total = 0.0
    day = a.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= b:
        if day.weekday() < 5:
            lo = max(a, day.replace(hour=start_h))
            hi = min(b, day.replace(hour=end_h)) if end_h < 24 else min(b, day + timedelta(days=1))
            if hi > lo:
                total += (hi - lo).total_seconds() / 60
        day += timedelta(days=1)
    return total
