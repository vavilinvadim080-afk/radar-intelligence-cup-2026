"""Маскирование ПДн ДО любой модели.

Идея: найти в тексте телефон, e-mail, карту, госномер, адрес, ФИО и заменить на токены
<PHONE_1>, <EMAIL_1>, <CARD_1>, <CAR_PLATE_1>, <ADDRESS_1>, <PERSON_1>.
Настоящие значения кладём в отдельный словарь (token -> оригинал). Словарь НЕ идёт в модель:
он нужен только человеку, который хочет увидеть оригинал.

Порядок правил важен: сначала то, что однозначно (e-mail), потом телефоны, потом длинные
цифровые карты, иначе кусок карты съест правило телефона.
"""
from __future__ import annotations

import re

# --- регулярки ---
# разрешаем и записи «ivan(at)mail.ru», «ivan собака mail.ru», «ivan @ mail.ru»
EMAIL = re.compile(r"[\w.+-]+\s?(?:@|\(at\)|\sсобака\s)\s?[\w-]+(?:\.[\w-]+)+")
# +7 (912) 345-67-89, 8 912 345 67 89, 89123456789, 8(912)345-67-89
PHONE = re.compile(r"(?<![\d])(?:\+7|8|7)[\s\-.()]*\d{3}[\s\-.()]*\d{3}[\s\-.()]*\d{2}[\s\-.()]*\d{2}(?![\d])")
# 13-19 цифр, можно через пробел или дефис группами (платёжная карта или карта лояльности)
CARD = re.compile(r"(?<![\d])\d(?:[\s\-]?\d){12,18}(?![\d])")
# госномер РФ: буква, 3 цифры, 2 буквы, регион 2-3 цифры. Буквы только те, что есть на номерах
_L = "АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx"   # плюс латинские двойники
PLATE = re.compile(rf"(?<![\w])[{_L}]\s?\d{{3}}\s?[{_L}]{{2}}\s?\d{{2,3}}(?![\w])")
# адрес: «г. Омск, ул. Мира, д. 12, кв. 5» или «ул. Мира, 15»
ADDRESS = re.compile(
    r"(?:г\.\s*[А-ЯЁ][\w-]+,?\s*)?(?:ул\.?|улица|пр-т|проспект|пер\.)\s*[А-ЯЁ][\w-]+(?:\s[А-ЯЁ][\w-]+)?,?\s*"
    r"(?:д\.?\s*)?\d+[а-я]?(?:\s*(?:к\.|корп\.)\s*\d+)?(?:,?\s*кв\.?\s*\d+)?")
# «Мира 15 кв 3» без слова «улица»
ADDRESS2 = re.compile(r"\b[А-ЯЁ][а-яё]+\s\d{1,3}\s?кв\.?\s?\d+")

# --- словари для имён ---
FIRST_NAMES = set("""Александр Сергей Дмитрий Андрей Алексей Максим Иван Михаил Николай Артём Артем Ильдар Рустем Тимур Олег
Виктор Евгений Владимир Павел Денис Роман Константин Игорь Антон Анатолий Василий Юрий Станислав Руслан Ренат Марат Айрат
Анна Мария Елена Ольга Наталья Татьяна Ирина Светлана Екатерина Юлия Гульнара Алсу Людмила Галина Надежда Виктория
Анастасия Дарья Марина Вера Любовь Оксана Алина Диана Регина Лилия Эльвира""".split())
PATRONYMIC = re.compile(r"^[А-ЯЁ][а-яё]+(?:ович|евич|ьич|овна|евна|ична|инична)(?:а|у|ем|е|ой|ы)?$")
SURNAME = re.compile(r"^[А-ЯЁ][а-яё]+(?:ов|ев|ёв|ин|ын|ова|ева|ёва|ина|ына|ский|ская|цкий|цкая|ко|ук|юк|ян|ову|еву|ину|овой|евой|иной)$")
# склонения имён («Сергею», «Анне»): сравниваем по основе без последней буквы
NAME_STEMS = {n[:-1] for n in FIRST_NAMES if len(n) >= 4}
WORD = re.compile(r"[А-ЯЁA-Z][а-яёa-z]+|[А-ЯЁ]\.[А-ЯЁ]\.")


def _replace_all(text: str, pattern, kind: str, state: dict) -> str:
    """Заменяет каждое совпадение на токен. Одинаковое значение получает один и тот же токен."""
    def repl(m):
        value = m.group(0)
        key = (kind, re.sub(r"\W", "", value).lower())
        if key not in state["seen"]:
            state["count"][kind] = state["count"].get(kind, 0) + 1
            token = f"<{kind}_{state['count'][kind]}>"
            state["seen"][key] = token
            state["map"][token] = value
        return state["seen"][key]
    return pattern.sub(repl, text)


def _mask_names(text: str, state: dict) -> str:
    """Имя из словаря, отчество по суффиксу; соседняя фамилия (по суффиксу) входит в тот же токен."""
    words = list(WORD.finditer(text))
    spans = []   # (start, end) кусков, которые считаем ФИО
    for i, m in enumerate(words):
        w = m.group(0)
        is_core = w in FIRST_NAMES or bool(PATRONYMIC.match(w)) or (len(w) >= 4 and w[:-1] in NAME_STEMS and len(w) <= len(w[:-1]) + 2) \
            or (len(w) >= 5 and w[:-2] in NAME_STEMS)
        is_initials = bool(re.match(r"^[А-ЯЁ]\.[А-ЯЁ]\.$", w))
        if not (is_core or is_initials):
            continue
        start, end = m.start(), m.end()
        # слово перед: фамилия, а перед отчеством любое слово с заглавной буквы (скорее всего имя, даже редкое)
        if i > 0 and text[words[i - 1].end():m.start()].strip() == "":
            prev = words[i - 1].group(0)
            if SURNAME.match(prev) or (PATRONYMIC.match(w) and len(prev) >= 3 and prev.isalpha()):
                start = words[i - 1].start()
        # слово после: отчество или фамилия, если рядом через пробел
        j = i + 1
        while j < len(words) and text[end:words[j].start()].strip() == "" and (
                PATRONYMIC.match(words[j].group(0)) or SURNAME.match(words[j].group(0)) or re.match(r"^[А-ЯЁ]\.[А-ЯЁ]\.$", words[j].group(0))):
            end = words[j].end()
            j += 1
        if is_initials and start == m.start():
            continue   # «И.О.» без фамилии рядом ничего не значит
        spans.append((start, end))
    # склеиваем пересекающиеся куски и подставляем справа налево (чтобы индексы не плыли)
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    tokens = []
    for s, e in merged:   # номера токенов раздаём слева направо, чтобы <PERSON_1> был первым в тексте
        value = text[s:e]
        key = ("PERSON", value.lower())
        if key not in state["seen"]:
            state["count"]["PERSON"] = state["count"].get("PERSON", 0) + 1
            token = f"<PERSON_{state['count']['PERSON']}>"
            state["seen"][key] = token
            state["map"][token] = value
        tokens.append(state["seen"][key])
    for (s, e), token in reversed(list(zip(merged, tokens))):
        text = text[:s] + token + text[e:]
    return text


def mask_text(text: str) -> tuple[str, dict]:
    """Возвращает (замаскированный текст, {токен: оригинал}). Словарь модели не отдавать."""
    state = {"seen": {}, "count": {}, "map": {}}
    out = _replace_all(text, EMAIL, "EMAIL", state)
    out = _replace_all(out, PHONE, "PHONE", state)
    out = _replace_all(out, CARD, "CARD", state)
    out = _replace_all(out, PLATE, "CAR_PLATE", state)
    out = _replace_all(out, ADDRESS, "ADDRESS", state)
    out = _replace_all(out, ADDRESS2, "ADDRESS", state)
    out = _mask_names(out, state)
    return out, state["map"]


def mask_tickets(tickets: list) -> tuple[list, dict]:
    """Для списка обращений: ([замаскированные тексты в том же порядке], {ticket_id: словарь_обратной_подстановки})."""
    texts, reverse = [], {}
    for t in tickets:
        masked, mapping = mask_text(t.text)
        texts.append(masked)
        if mapping:
            reverse[t.id] = mapping
    return texts, reverse


TOKEN_RE = re.compile(r"<[A-Z_]+_\d+>")


def strip_tokens(text: str) -> str:
    """Токены убираем перед эмбеддингом: иначе обращения с ПДн кучкуются из-за слов <PHONE_1>."""
    return TOKEN_RE.sub(" ", text)
