"""Типы данных. Всё простые dataclass: поле = колонка, без логики внутри."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Ticket:
    """Одно обращение пользователя (то, что приходит из канала)."""
    id: str            # "T000123", уникален в наборе
    ts: datetime       # когда пришло (локальное время, без часового пояса)
    product: str       # ключ продукта: azs / bonus / b2b / portal / partners
    channel: str       # app_chat / email / hotline / store_review / web_form
    platform: str      # android / ios / web
    app_version: str   # "4.12.0" у мобильных, "" если версии нет
    user_id: str       # "u12345": по нему считаем «один человек = один голос»
    text: str          # исходный текст, МОЖЕТ содержать ПДн (до маскирования)


@dataclass
class Release:
    """Релиз продукта (из календаря релизов)."""
    product: str
    version: str       # "4.12.0", у веб-продуктов "r7"
    ts: datetime       # момент выкатки
    kind: str = "regular"   # regular | fix (патч x.y.1)
    changelog: str = ""     # короткая запись для карточки


@dataclass
class Cluster:
    """Кластер похожих обращений внутри одного продукта."""
    id: str
    product: str
    size: int                    # сколько обращений всего (с повторами)
    first_seen: datetime
    last_seen: datetime


@dataclass
class SuspectedRelease:
    """Подозреваемый релиз: корреляция по времени и версии, а не доказательство."""
    version: str
    ts: datetime
    hours_since_release: float   # через сколько часов после релиза начался всплеск
    born_after_release: bool     # первый тикет кластера пришёл после релиза
    new_version_share: Optional[float]   # доля обращений всплеска на новой версии (None если версий нет)
    product_new_version_share: Optional[float]  # доля новой версии у всего продукта в то же время
    confidence: str              # high / medium / low


@dataclass
class FixCheck:
    """Результат проверки фикса (verify.py)."""
    fix_version: str
    fix_ts: datetime
    verdict: str                 # confirmed / not_confirmed / insufficient_data
    drop_percent: Optional[float]
    ratio: Optional[float]       # интенсивность после / до (с поправкой)
    ratio_ci_low: Optional[float]
    ratio_ci_high: Optional[float]
    p_value: Optional[float]
    n_before: int
    n_after: int
    used_control: bool
    checked_at: datetime
    mode: str = "ratio"                         # ratio (v1) | baseline (v2)
    residual_share: Optional[float] = None      # baseline: какая доля всплеска осталась после фикса (0 = вернулись к норме)
    residual_ci_high: Optional[float] = None    # верхняя граница 95% для residual_share
    n_baseline: int = 0                         # обращений темы в норме (дни до релиза)
    clusters: list = field(default_factory=list)  # по каким кластерам считали


@dataclass
class Burst:
    """Всплеск обращений по кластеру: главный выход детектора."""
    id: str                      # "B0001"
    product: str
    cluster_id: str
    method: str                  # ewma / poisson
    window_min: int              # окно, в котором сработало (30 или 60)
    started_at: datetime         # оценка начала: первое обращение в окне срабатывания
    detected_at: datetime        # когда детектор сработал (потоковое время)
    last_seen: datetime          # последнее обращение всплеска
    status: str                  # open / closed
    z: float                     # значение z на момент срабатывания
    count_in_window: int         # уникальных людей в окне срабатывания
    baseline_in_window: float    # сколько ожидалось в этом окне
    total_unique_users: int      # уникальных людей за всё время всплеска
    cluster_size: int
    platforms: dict = field(default_factory=dict)    # {"android": 31, ...}
    versions: dict = field(default_factory=dict)     # {"4.12.0": 28, ...}
    example_ids: list = field(default_factory=list)  # id ближайших к центру обращений
    examples: list = field(default_factory=list)     # [{"id":..., "text":...}] ТОЛЬКО замаскированный текст
    suspected_release: Optional[SuspectedRelease] = None
    weekly_cost_rub: float = 0.0
    fix_check: Optional[FixCheck] = None
    # --- v2 ---
    cluster_ids: list = field(default_factory=list)   # все кластеры-осколки всплеска (скан подмножеств), cluster_id первый
    signal: str = "topic"                              # topic: число людей темы; version: люди темы на новой версии
    alert: bool = True                                 # True: тревога дежурному; False: только наблюдение/продолжение
    alert_reason: str = ""                             # почему тревога или почему нет
    z_version: Optional[float] = None                  # z сигнала версии в момент срабатывания (мобильные)
    version_count: int = 0                             # людей на новой версии в окне
    version_baseline: float = 0.0                      # сколько ждали на новой версии
    # --- v3: два уровня тревоги ---
    tier: str = ""                                     # "page": будим дежурного; "quiet": тихая карточка в канал команды; "" у alert=False
    judge: Optional[dict] = None                       # ответ проверки {problem, items, share, ok}; None: не проверялась
    # --- защита от накрутки (radar/antispam.py) ---
    suspicion: Optional[dict] = None                   # признак накрутки {reason, dup_share, dup_users, n_users, gap_cv, fresh_share, kept_page, ...}; None: не замечено


@dataclass
class SuspectedCardInput:
    """Вход для агента следующей волны: всё, что нужно, чтобы написать карточку.
    Числа посчитаны кодом, модель их только переносит. Текст только замаскированный."""
    burst_id: str
    product: str
    product_name: str
    detected_at: datetime
    started_at: datetime
    unique_users: int
    window_min: int
    z: float
    platforms: dict
    versions: dict
    suspected_release: Optional[SuspectedRelease]
    weekly_cost_rub: float
    evidence: list              # [{"id":"T000123","text":"..."}]
    allowed_ids: list           # id, на которые карточка вправе ссылаться
    min_users_for_status: int = 5   # меньше этого: status = "наблюдаем"


def to_plain(obj):
    """dataclass (в том числе вложенный) -> dict/list, datetime -> ISO-строка. Для JSON."""
    if is_dataclass(obj):
        return {k: to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="minutes")
    return obj
