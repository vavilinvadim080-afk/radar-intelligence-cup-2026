"""Генератор синтетики. Всё берётся из random/numpy с одним seed, поэтому набор воспроизводится.

Что делает, по шагам (см. функцию generate):
 1. календарь релизов по продуктам;
 2. случайно выбирает события (регрессии, фиксы, медленный рост, шумовые всплески, инъекции);
 3. фон: для каждого продукта, темы и часа число обращений ~ Пуассон с суточным и недельным ритмом;
 4. добавки событий: ещё обращения по тем же темам в нужные часы;
 5. превращает «тема + время» в готовый текст (шаблон, синонимы, опечатки, ПДн).
Выход: tickets, releases, ground_truth, labels. labels нужны только для оценки.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from radar.models import Release, Ticket
from . import topics as T

START = datetime(2026, 3, 2, 0, 0)   # понедельник 00:00, день 0
NOISE = {
    # typo: доля слов с опечаткой; cross: вторая фраза из соседней темы; generic: доля пустых «не работает»;
    # repeat: доля людей, написавших второй раз; hour_k: чем меньше, тем «рваннее» поток по часам;
    # micro: вероятность случайного микровсплеска на тему в сутки; pii_hard: доля ПДн в «трудной» записи
    # (склонённые имена, латинские буквы в номере, слитные адреса), которую простые правила пропускают
    "low": dict(typo=0.012, cross=0.05, generic=0.03, repeat=0.08, hour_k=14.0, micro=0.02, pii_hard=0.05),
    "mid": dict(typo=0.035, cross=0.14, generic=0.09, repeat=0.13, hour_k=6.0, micro=0.05, pii_hard=0.15),
    "high": dict(typo=0.075, cross=0.28, generic=0.19, repeat=0.20, hour_k=3.0, micro=0.10, pii_hard=0.30),
}
MOBILE_PROFILE = [0.2, 0.12, 0.08, 0.06, 0.08, 0.2, 0.5, 0.9, 1.2, 1.1, 1.0, 1.1, 1.3, 1.2, 1.1, 1.2, 1.4, 1.7, 1.9, 1.8, 1.5, 1.0, 0.6, 0.35]
WEB_PROFILE = [0.05, 0.03, 0.02, 0.02, 0.05, 0.2, 0.6, 1.3, 2.2, 2.5, 2.3, 1.9, 1.4, 1.9, 2.3, 2.2, 1.8, 1.2, 0.6, 0.3, 0.2, 0.12, 0.08, 0.06]


def _norm(profile):
    m = sum(profile) / len(profile)
    return [p / m for p in profile]


MOBILE_PROFILE, WEB_PROFILE = _norm(MOBILE_PROFILE), _norm(WEB_PROFILE)
TOPIC_BY_ID = {t[0]: t for t in T.TOPICS}


@dataclass
class Dataset:
    tickets: list
    releases: list
    ground_truth: dict
    labels: dict      # ticket_id -> {"topic", "event", "pii": [[type, value]], "injection", "kind"}


# ---------------------------------------------------------------- вспомогательное
def day_factor(product: str, day: int) -> float:
    """Недельный ритм: мобильные в выходные чуть активнее, веб-продукты почти пустые."""
    weekend = (day % 7) >= 5
    if T.PRODUCTS[product][2]:
        return 1.15 if weekend else 1.0
    return 0.25 if weekend else 1.0


def profile_for(product: str):
    return MOBILE_PROFILE if T.PRODUCTS[product][2] else WEB_PROFILE


def snap_workday(dt: datetime, rng: random.Random) -> datetime:
    """Релизы выкатывают в будни с 11 до 17."""
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.replace(hour=rng.randint(11, 16), minute=rng.choice([0, 0, 15, 30, 45]), second=0)


def poisson(nprng, lam: float) -> int:
    return int(nprng.poisson(max(lam, 0.0)))


# ---------------------------------------------------------------- релизы
def make_releases(rng: random.Random, days: int) -> list:
    releases = []
    for product, (_, _, mobile, _, ver, cadence) in T.PRODUCTS.items():
        counter = 0
        t = START - timedelta(days=rng.randint(5, 9))   # стартовая версия, вышла до начала данных
        while t < START + timedelta(days=days):
            version = f"{ver[0]}.{ver[1] + counter}.0" if mobile else f"r{12 + counter}"
            releases.append(Release(product, version, t, "regular", f"Плановый релиз {version}"))
            counter += 1
            t = snap_workday(t + timedelta(days=cadence + rng.randint(-2, 2)), rng)
    releases.sort(key=lambda r: r.ts)
    return releases


def versions_at(releases: list, product: str, ts: datetime) -> list:
    """Релизы продукта, вышедшие к моменту ts, от новых к старым."""
    vs = [r for r in releases if r.product == product and r.ts <= ts]
    vs.sort(key=lambda r: r.ts, reverse=True)
    return vs


def pick_version(rng: random.Random, releases: list, product: str, ts: datetime) -> str:
    """Доля новой версии растёт по мере выкатки: 0 до ~85% за несколько дней."""
    vs = versions_at(releases, product, ts)
    if not vs:
        return ""
    tau = (ts - vs[0].ts).total_seconds() / 86400
    p_new = 0.85 * (1 - math.exp(-tau / 2.0))
    r = rng.random()
    if r < p_new or len(vs) == 1:
        return vs[0].version
    if r < p_new + (1 - p_new) * 0.88 or len(vs) == 2:
        return vs[1].version
    return vs[2].version


# ---------------------------------------------------------------- тексты
def fill_slots(tpl: str, rng: random.Random) -> str:
    def repl(m):
        options = T.SLOTS.get(m.group(1))
        return fill_slots(rng.choice(options), rng) if options else m.group(0)
    return re.sub(r"\{(\w+)\}", repl, tpl)


KEY_NEIGHBORS = {"а": "ыпв", "о": "лрщ", "е": "нкуг", "и": "мтс", "н": "егш", "т": "ьиро", "с": "чым", "р": "оптк",
                 "в": "аыуч", "л": "одд", "к": "уеа", "м": "чсит", "д": "лжщ", "п": "аро", "у": "цкй", "я": "чфы",
                 "ы": "фвц", "ь": "тб", "б": "ьюл", "з": "хщ", "ч": "яс", "г": "нш", "ш": "гщ", "й": "ц", "ж": "дэ"}


def add_typos(text: str, rate: float, rng: random.Random) -> str:
    out = []
    for w in text.split(" "):
        if len(w) >= 4 and rng.random() < rate:
            i = rng.randint(1, len(w) - 2)
            op = rng.choice(["swap", "drop", "double", "neighbor"])
            if op == "swap":
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif op == "drop":
                w = w[:i] + w[i + 1:]
            elif op == "double":
                w = w[:i] + w[i] + w[i:]
            else:
                c = w[i].lower()
                w = w[:i] + rng.choice(KEY_NEIGHBORS.get(c, c)) + w[i + 1:]
        out.append(w)
    return " ".join(out)


def make_phone(rng, hard=False):
    a, b, c, d = rng.randint(900, 999), rng.randint(100, 999), rng.randint(10, 99), rng.randint(10, 99)
    if hard:
        return rng.choice([f"8-{a}-{b}-{c}-{d}", f"+7 {a} {b}{c}{d}", f"7{a}{b}{c}{d}", f"8.{a}.{b}.{c}.{d}", f"+7({a}){b}{c}{d}"])
    return rng.choice([f"+7 ({a}) {b}-{c}-{d}", f"8 {a} {b} {c} {d}", f"8{a}{b}{c}{d}", f"+7{a}{b}{c}{d}", f"8({a}){b}-{c}-{d}"])


def make_email(rng, hard=False):
    user = rng.choice(["ivan", "sergey", "a.petrova", "driver", "mihail", "olga", "info", "alex", "tim"]) + rng.choice(["", ".", "_"]) + str(rng.randint(1, 9999))
    dom = rng.choice(T.EMAIL_DOMAINS)
    if hard:
        return rng.choice([f"{user} собака {dom}", f"{user}(at){dom}", f"{user} @ {dom}"])
    return f"{user}@{dom}"


def make_plate(rng, hard=False):
    L = T.PLATE_LETTERS
    p = f"{rng.choice(L)}{rng.randint(100, 999)}{rng.choice(L)}{rng.choice(L)}{rng.choice(T.PLATE_REGIONS)}"
    if hard:   # латинские двойники кириллицы
        lat = dict(zip("АВЕКМНОРСТУХ", "ABEKMHOPCTYX"))
        return "".join(lat.get(ch, ch) for ch in p)
    if rng.random() < 0.3:
        p = f"{p[0]} {p[1:4]} {p[4:6]} {p[6:]}"
    return p.lower() if rng.random() < 0.25 else p


def make_card(rng, hard=False):
    if rng.random() < 0.6:  # платёжная карта, 16 цифр
        d = "".join(str(rng.randint(0, 9)) for _ in range(16))
        if hard:
            return f"{d[:4]} {d[4:12]} {d[12:]}"
        sep = rng.choice([" ", "", "-"])
        return sep.join(d[i:i + 4] for i in range(0, 16, 4))
    return "".join(str(rng.randint(0, 9)) for _ in range(13))  # карта лояльности


def make_addr(rng, hard=False):
    street, house = rng.choice(T.STREETS), rng.randint(1, 120)
    if hard:
        return rng.choice([f"{street} {house} кв {rng.randint(1, 99)}", f"ул.{street} д.{house}", f"проспект {street} {house}"])
    if rng.random() < 0.5:
        return f"г. {rng.choice(T.CITIES)}, ул. {street}, д. {house}, кв. {rng.randint(1, 200)}"
    return f"ул. {street}, {house}"


RARE_NAMES = ["Рамиль", "Феликс", "Эдуард", "Лейла", "Заур"]


def make_fio(rng, hard=False):
    female = rng.random() < 0.35
    sur = rng.choice(T.FEM_SURNAMES if female else T.SURNAMES)
    name = rng.choice(T.NAMES_F if female else T.NAMES_M)
    pat = rng.choice(T.PATRONYMICS_F if female else T.PATRONYMICS_M)
    if hard:
        dec = sur[:-1] + "ой" if female else sur + "у"
        return rng.choice([f"{dec} {name[:-1]}ю {pat}", f"{rng.choice(RARE_NAMES)} {pat}", f"{sur.lower()} {name.lower()}", f"{sur[:-2]}{sur[-1]} {name}"])
    return rng.choice([f"{sur} {name} {pat}", f"{name} {pat}", f"{sur} {name[0]}.{pat[0]}.", name, f"{sur} {name}"])


PII_MAKERS = {"phone": make_phone, "email": make_email, "plate": make_plate, "card": make_card, "addr": make_addr, "fio": make_fio}
PII_PHRASES = {
    "phone": ["Мой номер {v}.", "Перезвоните на {v}.", "тел. {v}", "связаться со мной можно по {v}"],
    "email": ["Почта {v}", "ответ пришлите на {v}", "мой email {v}"],
    "plate": ["Авто {v}.", "госномер {v}", "машина {v}", "номер авто {v}"],
    "card": ["Карта {v}.", "номер карты {v}", "карта лояльности {v}", "карта {v}"],
    "addr": ["АЗС {v}.", "Адрес: {v}", "была на {v}"],
    "fio": ["Меня зовут {v}.", "{v}", "С уважением, {v}", "Это {v}."],
}
PII_PROB_BY_CHANNEL = {"email": 0.6, "hotline": 0.4, "app_chat": 0.25, "web_form": 0.4, "store_review": 0.08}


def make_text(topic_id: str, product: str, channel: str, noise: dict, rng: random.Random, after_update=False, repeat=False):
    """Возвращает (текст, список вставленных ПДн [(тип, значение)])."""
    templates = TOPIC_BY_ID[topic_id][5]
    body = fill_slots(rng.choice(templates), rng)
    if after_update:
        body = rng.choice(T.AFTER_UPDATE) + " " + body
    if rng.random() < noise["cross"]:
        body += ", " + rng.choice(T.CROSS_FRAGMENTS[product])
    if repeat:
        body = rng.choice(["Ещё раз пишу.", "Повторное обращение.", "Никто не ответил.", "Пишу снова."]) + " " + body
    if rng.random() < 0.3:
        body = body + rng.choice(["", ".", "!", "!!", "..."])
    parts = []
    if rng.random() < 0.3:
        parts.append(rng.choice(T.GREET))
    parts.append(body)
    if rng.random() < 0.25:
        parts.append(rng.choice(T.CLOSE))
    text = " ".join(p for p in parts if p).strip()
    text = add_typos(text, noise["typo"], rng)
    r = rng.random()   # регистр: часть людей пишет всё строчными, редко капсом
    if r < 0.3:
        text = text.lower()
    elif r < 0.34:
        text = text.upper()
    else:
        text = text[:1].upper() + text[1:]
    if channel == "hotline":   # расшифровка звонка: без знаков препинания
        text = re.sub(r"[.,!?]", "", text).lower()
    if channel == "store_review":
        text = rng.choice(["Ужас. ", "Не рекомендую. ", "Одна звезда. ", "", ""]) + text
    pii = []
    if rng.random() < PII_PROB_BY_CHANNEL.get(channel, 0.2):
        kinds = ["phone", "email", "fio", "plate", "card", "addr"]
        weights = [3, 2, 3, 2 if product == "azs" else 1, 2 if product in ("azs", "bonus", "b2b") else 0.5, 1 if product == "azs" else 0.3]
        for kind in dict.fromkeys(rng.choices(kinds, weights, k=rng.choice([1, 1, 2, 3]))):   # без повторов, порядок фиксирован (set зависит от PYTHONHASHSEED)
            value = PII_MAKERS[kind](rng, hard=rng.random() < noise["pii_hard"])
            pii.append((kind, value))
            phrase = rng.choice(PII_PHRASES[kind]).format(v=value)
            text = phrase + " " + text if rng.random() < 0.25 else text + " " + phrase
    return text, pii


# ---------------------------------------------------------------- основная функция
def generate(seed: int, noise: str = "mid", daily_volume: int = 800, days: int = 28) -> Dataset:
    assert noise in NOISE, f"noise: low/mid/high, получено {noise}"
    nz = NOISE[noise]
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    scale = daily_volume / 800.0
    releases = make_releases(rng, days)
    end = START + timedelta(days=days)
    plans = []           # события: по ним генерируем добавки и пишем ground_truth

    # ---- 2. события
    regular = [r for r in releases if 7 <= (r.ts - START).total_seconds() / 86400 <= 18]
    rng.shuffle(regular)
    n_reg = rng.randint(2, 3)
    used_products, used_topics = set(), set()
    for rel in regular:
        if len(used_products) >= n_reg:
            break
        if rel.product in used_products:
            continue
        cands = [t for t in T.TOPICS if t[1] == rel.product and t[4]]
        topic = rng.choice(cands)
        onset = rel.ts + timedelta(minutes=rng.randint(20, 180))
        fix_kind = rng.choices(["effective", "ineffective", "none"], [0.55, 0.2, 0.25])[0]
        fix = None
        if fix_kind != "none":
            ft = snap_workday(onset + timedelta(hours=rng.randint(26, 70)), rng)
            if ft < START + timedelta(days=24):
                mobile_p = T.PRODUCTS[rel.product][2]
                v = rel.version[:-1] + "1" if mobile_p else rel.version + ".1"
                fix = Release(rel.product, v, ft, "fix", f"Исправление: {topic[2].lower()}")
                releases.append(fix)
        mobile = T.PRODUCTS[rel.product][2]
        bias = rng.choice(["android", "ios"]) if mobile and rng.random() < 0.7 else None
        plans.append(dict(type="regression", id=f"E{len(plans) + 1}", product=rel.product, topic=topic[0], onset=onset,
                          release=rel, peak=math.exp(rng.uniform(math.log(3.0), math.log(18.0))) * scale * math.sqrt(T.PRODUCTS[rel.product][1] / 0.25), fix=fix, fix_effective=(fix_kind == "effective"),
                          platform_bias=bias))
        used_products.add(rel.product)
        used_topics.add(topic[0])
    releases.sort(key=lambda r: r.ts)

    # медленный рост без всплеска
    topic = rng.choice([t for t in T.TOPICS if t[0] not in used_topics and t[3] >= 0.07])
    used_topics.add(topic[0])
    plans.append(dict(type="slow_growth", id=f"E{len(plans) + 1}", product=topic[1], topic=topic[0],
                      onset=START + timedelta(days=rng.randint(4, 9)), factor_end=rng.uniform(2.8, 4.0)))
    # шумовые всплески без причины (новость, акция)
    for _ in range(rng.randint(1, 2)):
        for _try in range(20):
            topic = rng.choice([t for t in T.TOPICS if t[3] >= 0.07 and t[0] not in used_topics])
            onset = START + timedelta(days=rng.randint(8, 25), hours=rng.randint(9, 19), minutes=rng.randint(0, 59))
            near = [r for r in releases if r.product == topic[1] and abs((r.ts - onset).total_seconds()) < 3 * 3600]
            if not near:
                break
        used_topics.add(topic[0])
        plans.append(dict(type="noise_spike", id=f"E{len(plans) + 1}", product=topic[1], topic=topic[0], onset=onset,
                          dur_min=rng.randint(100, 200), n=rng.uniform(30, 75) * scale))
    # prompt injection
    for _ in range(rng.randint(2, 4)):
        topic = rng.choice(T.TOPICS)
        plans.append(dict(type="prompt_injection", id=f"E{len(plans) + 1}", product=topic[1], topic=topic[0],
                          onset=START + timedelta(days=rng.uniform(3, days - 1)), template=rng.randrange(len(T.INJECTIONS))))

    raw = []   # (ts, product, topic, event_id, kind, extra)

    # ---- 3. фон
    pool = {p: max(300, int(daily_volume * v[1] * 25)) for p, v in T.PRODUCTS.items()}
    slow = next(p for p in plans if p["type"] == "slow_growth")
    slow_day0 = (slow["onset"] - START).days
    for product, (_, share, _, _, _, _) in T.PRODUCTS.items():
        prof = profile_for(product)
        tp = [t for t in T.TOPICS if t[1] == product]
        wsum = sum(t[3] for t in tp)
        for day in range(days):
            day_noise = math.exp(nprng.normal(0, 0.05 + 0.04 * (nz["hour_k"] < 7)))
            for t in tp:
                grow = 1.0
                if slow["topic"] == t[0] and day >= slow_day0:
                    grow = 1 + (slow["factor_end"] - 1) * (day - slow_day0) / (days - slow_day0)
                base_day = daily_volume * share * day_factor(product, day) * day_noise * t[3] / wsum * grow
                for h in range(24):
                    g = nprng.gamma(nz["hour_k"], 1 / nz["hour_k"])   # рваность потока по часам
                    for _ in range(poisson(nprng, base_day * prof[h] / 24 * g)):
                        ts = START + timedelta(days=day, hours=h, minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
                        raw.append((ts, product, t[0], slow["id"] if (slow["topic"] == t[0] and grow > 1.2) else None, "base", {}))
                if rng.random() < nz["micro"]:   # случайный микровсплеск: часть фона, в события не пишется
                    start_m = START + timedelta(days=day, hours=rng.randint(8, 21), minutes=rng.randint(0, 59))
                    for _ in range(max(2, int(rng.uniform(6, 14) * scale))):
                        raw.append((start_m + timedelta(minutes=rng.randint(0, 40)), product, t[0], None, "micro", {}))

    # ---- 4. добавки событий
    for p in plans:
        if p["type"] == "regression":
            prof = profile_for(p["product"])
            t = p["onset"]
            while t < end:
                hrs = (t - p["onset"]).total_seconds() / 3600
                rate = p["peak"] * min(1.0, hrs / 0.75 + 0.15) * 0.5 ** (hrs / 24 / 7)   # разгон 38 мин (до пика), потом медленный спад
                if p["fix"] and t > p["fix"].ts + timedelta(hours=3):
                    k = min(1.0, (t - p["fix"].ts).total_seconds() / 3600 / 9 - 0.33)   # после фикса спад за ~9 часов
                    floor = 0.06 if p["fix_effective"] else 0.7
                    rate *= 1 - (1 - floor) * max(k, 0)
                rate *= prof[t.hour] * day_factor(p["product"], (t - START).days)
                for _ in range(poisson(nprng, rate * 10 / 60)):
                    raw.append((t + timedelta(seconds=rng.randint(0, 599)), p["product"], p["topic"], p["id"], "regression", {}))
                t += timedelta(minutes=10)
        elif p["type"] == "noise_spike":
            for _ in range(poisson(nprng, p["n"])):
                off = rng.triangular(0, p["dur_min"], p["dur_min"] * 0.3)
                raw.append((p["onset"] + timedelta(minutes=off), p["product"], p["topic"], p["id"], "noise_spike", {}))
        elif p["type"] == "prompt_injection":
            raw.append((p["onset"], p["product"], p["topic"], p["id"], "injection", {"tpl": p["template"]}))

    # ---- 5. превращаем в обращения
    raw.sort(key=lambda x: x[0])
    tickets, labels = [], {}
    plan_by_id = {p["id"]: p for p in plans}
    products = list(T.PRODUCTS)

    def build(ts, product, topic_id, event_id, kind, extra, user=None, repeat=False):
        mobile = T.PRODUCTS[product][2]
        r = rng.random()
        if mobile:
            channel = "app_chat" if r < 0.45 else "store_review" if r < 0.65 else "email" if r < 0.8 else "hotline"
        else:
            channel = "web_form" if r < 0.45 else "email" if r < 0.8 else "hotline"
        platform = rng.choices(["android", "ios"], [0.62, 0.38])[0] if mobile else "web"
        version = pick_version(rng, releases, product, ts) if mobile else ""
        after_update = False
        if kind == "regression":
            plan = plan_by_id[event_id]
            if plan["platform_bias"] and rng.random() < 0.9:
                platform = plan["platform_bias"]
            if mobile:
                vs = versions_at(releases, product, ts)
                prev = vs[1].version if len(vs) > 1 else version
                version = plan["release"].version if rng.random() < 0.88 else prev
            after_update = rng.random() < 0.3
        if channel in ("email", "hotline") and rng.random() < 0.6:
            version = ""   # у части обращений версии нет
        if channel == "hotline" and rng.random() < 0.5:
            platform = "unknown"
        if user is None:
            user = f"u{products.index(product)}{rng.randrange(pool[product]):05d}"
        generic = kind in ("base", "micro") and rng.random() < nz["generic"]
        if kind == "injection":
            text, pii = make_text(topic_id, product, channel, nz, rng)
            text = T.INJECTIONS[extra["tpl"]].format(topic=text)
        elif generic:
            text, pii = rng.choice(T.GENERIC), []
        else:
            text, pii = make_text(topic_id, product, channel, nz, rng, after_update=after_update, repeat=repeat)
        tk = Ticket("", ts, product, channel, platform, version, user, text)
        tickets.append(tk)
        labels[id(tk)] = {"topic": "generic" if generic else topic_id, "event": event_id,
                          "pii": [list(x) for x in pii], "injection": kind == "injection",
                          "kind": "repeat" if repeat else kind}
        return tk

    for ts, product, topic_id, event_id, kind, extra in raw:
        if ts >= end:
            continue
        tk = build(ts, product, topic_id, event_id, kind, extra)
        if kind != "injection" and rng.random() < nz["repeat"]:   # тот же человек пишет ещё раз
            for _ in range(rng.choice([1, 1, 2])):
                t2 = ts + timedelta(minutes=rng.randint(8, 240))
                if t2 < end:
                    build(t2, product, topic_id, event_id, kind, extra, user=tk.user_id, repeat=True)

    tickets.sort(key=lambda t: t.ts)
    final_labels = {}
    for i, tk in enumerate(tickets, 1):
        old = labels[id(tk)]
        tk.id = f"T{i:06d}"
        final_labels[tk.id] = old

    # ---- ground truth
    by_event = {}
    for tid, lab in final_labels.items():
        if lab["event"]:
            by_event.setdefault(lab["event"], []).append(tid)
    events = []
    for p in plans:
        e = dict(id=p["id"], type=p["type"], product=p["product"], topic=p["topic"],
                 topic_name=TOPIC_BY_ID[p["topic"]][2], onset_ts=p["onset"].isoformat(timespec="minutes"),
                 n_tickets=len(by_event.get(p["id"], [])))
        if p["type"] == "regression":
            e.update(release_version=p["release"].version, release_ts=p["release"].ts.isoformat(timespec="minutes"),
                     platform_bias=p["platform_bias"], peak_per_hour=round(p["peak"], 2),
                     fix_version=p["fix"].version if p["fix"] else None,
                     fix_ts=p["fix"].ts.isoformat(timespec="minutes") if p["fix"] else None,
                     fix_effective=p["fix_effective"] if p["fix"] else None)
        if p["type"] == "noise_spike":
            e.update(duration_min=p["dur_min"], note="всплеск без причины в релизах (новость или акция)")
        if p["type"] == "slow_growth":
            e.update(factor_end=round(p["factor_end"], 2), note="рост без всплеска, для дайджеста, тревоги быть не должно")
        if p["type"] == "prompt_injection":
            e["ticket_ids"] = by_event.get(p["id"], [])
        events.append(e)
        if p["type"] == "regression" and p["fix"]:
            events.append(dict(id=p["id"] + "-fix", type="fix", product=p["product"], topic=p["topic"], fixes_event=p["id"],
                               onset_ts=p["fix"].ts.isoformat(timespec="minutes"), fix_version=p["fix"].version,
                               effective=p["fix_effective"]))
    gt = dict(seed=seed, noise=noise, daily_volume=daily_volume, days=days, start=START.isoformat(), events=events)
    return Dataset(tickets, releases, gt, final_labels)


def save(ds: Dataset, out_dir) -> None:
    """Пишет tickets.jsonl, releases.json, ground_truth.json (+ labels.json для оценки)."""
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tickets.jsonl", "w", encoding="utf-8") as f:
        for t in ds.tickets:
            f.write(json.dumps({**t.__dict__, "ts": t.ts.isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
    rel = [{**r.__dict__, "ts": r.ts.isoformat(timespec="minutes")} for r in ds.releases]
    (out / "releases.json").write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "ground_truth.json").write_text(json.dumps(ds.ground_truth, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "labels.json").write_text(json.dumps(ds.labels, ensure_ascii=False), encoding="utf-8")


def load(dir_):
    """Читает то, что записал save: (tickets, releases, ground_truth, labels)."""
    import json
    from pathlib import Path
    d = Path(dir_)
    tickets = []
    for line in open(d / "tickets.jsonl", encoding="utf-8"):
        o = json.loads(line)
        o["ts"] = datetime.fromisoformat(o["ts"])
        tickets.append(Ticket(**o))
    releases = [Release(**{**r, "ts": datetime.fromisoformat(r["ts"])}) for r in json.loads((d / "releases.json").read_text(encoding="utf-8"))]
    gt = json.loads((d / "ground_truth.json").read_text(encoding="utf-8"))
    labels = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    return tickets, releases, gt, labels
