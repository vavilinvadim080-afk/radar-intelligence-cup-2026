"""Лексическое сходство без LLM и без внешних библиотек: символьные 4-граммы слов.

Зачем: (1) проверка покрытия утверждений в validate_card (ссылка на обращение должна подтверждать утверждение);
(2) сходство нового обращения с кластером для автоответа (app/autoreply.py).
Почему не лемматизатор: pymorphy в зависимостях ядра нет, а 4-граммы переживают окончания и приставки
(«блокировке» ~ «заблокирована»: общие «блок», «локи», «окир», «киро», «иров»). Цена: грубее лемм, поэтому порог настраивается.
"""
from __future__ import annotations

import math
import re
from collections import Counter

WORD_RE = re.compile(r"[a-zа-яё]+")
N = 4                       # длина символьной граммы
MIN_WORD = 4                # слова короче не несут содержания («чек», «вход» уже 4)
WORD_GRAM_SHARE = 0.6       # слово «найдено» в тексте, если в нём есть хотя бы такая доля его 4-грамм

# служебные слова
STOP = frozenset("""и в не на по с к о из за что как при для или но это то же бы от до а ли уже еще ещё все всех всё его ее её их они он она
был была были было будет есть нет после перед через также только когда если чтобы там тут так такой очень можно нужно
могут может могла могли могу можем хочу хотел хотела свою свой своя мой моя мои наш наша наши ваш ваша ваши меня мне нас вас вам нам себя тоже""".split())

# «мета-слова» утверждений: слова про сам кластер, а не про симптом; в обращениях их искать бессмысленно
META_PREFIXES = ("достиг", "пика", "пике", "начавш", "сработав", "завершивш", "закончивш", "жалу", "затро", "затра", "среди", "неизвест", "unknown", "общее", "общий", "число", "составля", "существу", "сталкив", "устройств",
                 "верси", "этом","пользовател", "обращен", "человек", "людей", "людям", "наблюда", "всплеск", "всплес", "касают", "касает", "сообща",
                 "жалоб", "платформ", "уверенн", "релиз", "подозрева", "начал", "начин", "сработ", "связан", "совпал",
                 "уникальн", "кластер", "инцидент", "карточк", "утвержд", "проблем", "ситуаци", "завершил", "доля", "процент", "около",
                 "android", "ios", "web", "низк", "высок", "средн", "массов", "сообщен", "написал", "пишут", "указыва", "отмечаю",
                 "связь", "причин", "возможн", "вероятн", "похож", "однотипн", "разнородн", "разных", "разные", "разный", "другие", "других")


def words(text: str) -> list[str]:
    """Слова текста в нижнем регистре, ё как е. Числа и id (T000123, REL-...) в слова не попадают: в них нет букв подряд длиннее приставки."""
    t = re.sub(r"\b(?:T\d+|REL-[\w.\-]*|INC-[\w.\-]*|RAD-[\w.\-]*|DR-[\w.\-]*|KB-[\w.\-]*)", " ", text or "")
    return WORD_RE.findall(t.lower().replace("ё", "е"))


def content_words(text: str) -> list[str]:
    """Содержательные слова утверждения: без служебных, без мета-слов, без коротких. Порядок и повторы сохраняем без дублей."""
    out, seen = [], set()
    for w in words(text):
        if len(w) < MIN_WORD or w in STOP or w.startswith(META_PREFIXES) or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def grams(word: str) -> set:
    return {word[i:i + N] for i in range(len(word) - N + 1)} if len(word) >= N else {word}


def gram_set(text: str) -> set:
    """Множество 4-грамм всех слов текста (длиной от MIN_WORD)."""
    g: set = set()
    for w in words(text):
        if len(w) >= MIN_WORD:
            g |= grams(w)
    return g


def word_in(word: str, text_grams: set) -> bool:
    """Слово «есть» в тексте, если доля его 4-грамм среди 4-грамм текста не меньше WORD_GRAM_SHARE (ловит приставки: «блокировке» ~ «заблокирована»)."""
    g = grams(word)
    return bool(g) and len(g & text_grams) / len(g) >= WORD_GRAM_SHARE


def same_root(a: str, b: str) -> bool:
    """Общее начало слов (ловит окончания: «входом» ~ «вход», «кода» ~ «код», «участвовать» ~ «участвую»):
    общая приставка от 4 букв, либо от 3, если слово короче на одну букву и отличается только окончанием."""
    cp = 0
    for x, y in zip(a, b):
        if x != y:
            break
        cp += 1
    return cp >= 4 or (cp >= 3 and cp >= min(len(a), len(b)) - 1)


def word_matches(word: str, ticket_words: set, text_grams: set) -> bool:
    return word_in(word, text_grams) or any(same_root(word, t) for t in ticket_words)


SCALE_RE = re.compile(r"платформ|версии|версий|уникальн|\bunknown\b|\bандроид|\bios\b|\bandroid\b|\(\d+\)", re.I)


def is_scale_claim(text: str) -> bool:
    """Утверждение про масштаб (сколько людей, какие платформы и версии): симптома в нём нет, сверять с текстом обращений нечего.
    Числа в нём проверяет отдельное правило валидатора (числа только из входа)."""
    return bool(SCALE_RE.search(text or ""))


def overlap(claim_text: str, ticket_text: str) -> tuple:
    """(сколько содержательных слов утверждения нашлось в тексте, сколько всего содержательных слов).
    ticket_text может быть склейкой нескольких обращений: слова считаются по объединению."""
    cw = content_words(claim_text)
    tw = {w for w in words(ticket_text) if len(w) >= 3}
    tg = gram_set(ticket_text)
    return sum(1 for w in cw if word_matches(w, tw, tg)), len(cw)


def profile(text: str) -> tuple:
    """(частоты 4-грамм, длина вектора) текста. Считается один раз на пример кластера, потом cosine_profiles на каждое обращение."""
    c = Counter(g for w in words(text) if len(w) >= MIN_WORD for g in _all_grams(w))
    return c, math.sqrt(sum(v * v for v in c.values()))


def cosine_profiles(a: tuple, b: tuple) -> float:
    (ca, na), (cb, nb) = a, b
    if not na or not nb:
        return 0.0
    if len(ca) > len(cb):
        ca, cb = cb, ca
    return sum(v * cb.get(k, 0) for k, v in ca.items()) / (na * nb)


def cosine(a: str, b: str) -> float:
    """Косинус частот 4-грамм двух текстов (0..1). Для сходства нового обращения с примерами кластера."""
    return cosine_profiles(profile(a), profile(b))


def _all_grams(word: str) -> list:
    return [word[i:i + N] for i in range(len(word) - N + 1)] or [word]
