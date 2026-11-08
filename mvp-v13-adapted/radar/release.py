"""Привязка всплеска к релизу. Результат называется «подозреваемый релиз»: это корреляция
по времени и версии приложения, а не доказательство причины.

Правила (простые, проверяются руками):
 1. кандидат: последний релиз продукта до начала всплеска, не старше release_max_hours;
 2. born_after_release: первое обращение кластера пришло уже после релиза (тема новая);
 3. new_version_share: какая доля обращений всплеска (с известной версией) пришла с версии релиза,
    рядом доля той же версии у всего продукта за то же время (ожидаемая «случайная» доля);
 4. confidence: high, если доля версии не меньше порога И минимум в 1.5 раза выше доли по продукту
    И кластер родился после релиза; medium, если выполнено что-то одно; иначе low.
    Если версий нет (веб-продукты), high не бывает: максимум medium, когда кластер родился после релиза.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from .models import Release, SuspectedRelease


def _share(versions: list, target: str) -> Optional[float]:
    known = [v for v in versions if v]
    if len(known) < 5:    # меньше 5 обращений с версией: доля ничего не говорит
        return None
    return sum(1 for v in known if v == target) / len(known)


def find_suspect(releases: list[Release], product: str, started_at: datetime, first_seen: datetime,
                 burst_versions: list[str], product_versions: list[str],
                 max_hours: float = 48.0, share_min: float = 0.6) -> Optional[SuspectedRelease]:
    cands = [r for r in releases if r.product == product and r.ts <= started_at]
    if not cands:
        return None
    rel = max(cands, key=lambda r: r.ts)
    hours = (started_at - rel.ts).total_seconds() / 3600
    if hours > max_hours:
        return None
    born = first_seen >= rel.ts
    nvs = _share(burst_versions, rel.version)
    pvs = _share(product_versions, rel.version)
    if nvs is None:
        conf = "medium" if born else "low"
    else:
        strong = nvs >= share_min and (pvs is None or nvs >= 1.5 * pvs)
        if strong and born:
            conf = "high"
        elif strong or born:
            conf = "medium"
        else:
            conf = "low"
    return SuspectedRelease(rel.version, rel.ts, round(hours, 2), born, None if nvs is None else round(nvs, 3),
                            None if pvs is None else round(pvs, 3), conf)
