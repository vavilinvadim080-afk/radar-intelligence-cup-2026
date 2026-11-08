"""Необязательный слой маскирования: имена и адреса через Natasha (NER, работает офлайн, модели лежат в пакете).

Включается флагом Config.mask_ner (по умолчанию выключен: не меняет метрики и не требует пакета natasha).
Установка: uv pip install --python .venv/bin/python -r requirements-ner.txt
Берём тип PER (человек) и LOC только когда за ним идёт улица с номером дома («Екатеринбург, Малышева 51»), голый
город сам по себе не ПДн.
"""
from __future__ import annotations

import re

_STATE = {}


def available() -> bool:
    try:
        import natasha  # noqa: F401
        return True
    except Exception:   # noqa: BLE001
        return False


def _load():
    if not _STATE:
        from natasha import NewsEmbedding, NewsNERTagger, Segmenter
        _STATE["seg"], _STATE["ner"] = Segmenter(), NewsNERTagger(NewsEmbedding())
    return _STATE["seg"], _STATE["ner"]


_STREET_AFTER = re.compile(r"^,?\s*(?:(?:ул\.?|улица|проспект|пр-т|пер\.|переулок|бульвар)\s*)?[А-ЯЁ][\w-]+(?:\s[А-ЯЁ][\w-]+)?,?\s*(?:д\.?\s*|дом\s*)?\d")
_TOKEN = re.compile(r"<[A-Z_]+_\d+>")


def mask_ner(text: str, state: dict) -> str:
    from natasha import Doc
    seg, ner = _load()
    doc = Doc(text)
    doc.segment(seg)
    doc.tag_ner(ner)
    spans = []
    for sp in doc.spans:
        if _TOKEN.search(text[sp.start:sp.stop]):
            continue
        if sp.type == "PER":
            spans.append((sp.start, sp.stop, "PERSON"))
        elif sp.type == "LOC" and _STREET_AFTER.match(text[sp.stop:]):
            spans.append((sp.start, sp.stop, "ADDRESS"))
    for s, e, kind in sorted(spans, reverse=True):
        value = text[s:e]
        key = (kind, value.lower())
        if key not in state["seen"]:
            state["count"][kind] = state["count"].get(kind, 0) + 1
            state["seen"][key] = f"<{kind}_{state['count'][kind]}>"
            state["map"][state["seen"][key]] = value
        text = text[:s] + state["seen"][key] + text[e:]
    return text
