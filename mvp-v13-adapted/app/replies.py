"""Шаблоны автоответов. Текст собирает код из готовых фраз, поля подставляются только из кода.

Что сюда НЕ попадает: текст обращения пользователя и текст от модели. Это защита от инъекций (обращение не может
написать ответ другим людям) и от утечки ПДн (в ответ не уходит ничего, что человек написал в обращении).
Подставляются три поля: название продукта (из справочника кода), время начала (datetime из детектора), номер инцидента (id карточки).
Каждое поле проверяется по белому списку символов: сюда нельзя «протащить» чужой текст, даже если ошибся вызывающий.
"""
from __future__ import annotations

import re
import zlib
from datetime import datetime

TEMPLATES = {
    "known": [
        "Здравствуйте! Мы уже знаем о проблеме в сервисе «{product}», она началась {started}. Команда её исправляет. "
        "Номер инцидента: {incident}. Как только исправим, сообщим вам.",
        "Добрый день! Ваше обращение связано с известной проблемой в сервисе «{product}» (с {started}). "
        "Мы работаем над исправлением, инцидент {incident}. Сообщим, когда всё заработает.",
        "Спасибо, что написали. Проблема в сервисе «{product}» известна с {started}, команда уже чинит. "
        "Ваше обращение учтено в инциденте {incident}, о решении сообщим.",
    ],
    "fixed": [
        "Здравствуйте! Проблему в сервисе «{product}», о которой вы писали (инцидент {incident}), мы исправили. "
        "Если сбой повторится, напишите нам ещё раз.",
        "Добрый день! Сообщаем: неполадка в сервисе «{product}» устранена (инцидент {incident}). "
        "Спасибо, что дали знать. Если что-то осталось, ответьте на это сообщение.",
        "Спасибо за терпение. Инцидент {incident} в сервисе «{product}» закрыт, всё должно работать. "
        "Если вы всё ещё видите проблему, напишите нам.",
    ],
}

_PRODUCT_RE = re.compile(r"^[\w «»\-.,()]{1,60}$")
_INCIDENT_RE = re.compile(r"^[A-Za-z0-9\-]{1,32}$")


def render(kind: str, *, product: str, started_at: datetime, incident: str, variant: int = 0) -> str:
    """Готовый текст ответа. Любое поле не по формату: ValueError (ничего не отправляется)."""
    if kind not in TEMPLATES:
        raise ValueError(f"нет шаблона {kind!r}")
    if not isinstance(started_at, datetime):
        raise ValueError("время начала должно быть datetime из кода")
    if not isinstance(product, str) or not _PRODUCT_RE.match(product):
        raise ValueError("название продукта не по формату")
    if not isinstance(incident, str) or not _INCIDENT_RE.match(incident):
        raise ValueError("номер инцидента не по формату")
    variants = TEMPLATES[kind]
    return variants[variant % len(variants)].format(product=product, started=f"{started_at:%d.%m.%Y %H:%M}", incident=incident)


def variant_for(ticket_id: str) -> int:
    """Номер варианта по id обращения (стабильно, без случайности: тот же id даёт тот же текст). Текст обращения не читается."""
    return zlib.crc32(str(ticket_id).encode("utf-8"))
