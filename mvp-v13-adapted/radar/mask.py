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
import threading
import warnings

# --- регулярки ---
# разрешаем и записи «ivan(at)mail.ru», «ivan собака mail.ru», «ivan @ mail.ru»
# длины ограничены (локальная часть до 64 знаков, как в RFC 5321): без границ регулярка квадратична на длинных цифровых и буквенных строках
EMAIL = re.compile(r"[\w.+-]{1,64}\s?(?:@|\(at\)|\sсобака\s)\s?[\w-]{1,63}(?:\.[\w-]{1,63})+")
MAX_TEXT_CHARS = 4000   # дальше текст обрезается ДО маскирования: хвост в модель не уходит, а время на обращение ограничено
# +7 (912) 345-67-89, 8 912 345 67 89, 89123456789, 8(912)345-67-89
PHONE = re.compile(r"(?<![\d])(?:\+7|8|7)[\s\-.()]*\d{3}[\s\-.()]*\d{3}[\s\-.()]*\d{2}[\s\-.()]*\d{2}(?![\d])")
# v2: телефон без кода страны, 10 цифр на 9 (мобильные РФ): 9123456789, 912 345-67-89
PHONE10 = re.compile(r"(?<![\d])9\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?![\d])")
# v2: международный номер с плюсом, кроме +7: +375 29 123 45 67
PHONE_INTL = re.compile(r"(?<![\d\w])\+(?!7)\d{1,3}(?:[\s\-()]*\d){8,11}(?![\d])")
# v2: расчётный счёт, 20 цифр, можно группами через пробел (40817 810 0 9991 0004312). Идёт ДО карты
ACCOUNT = re.compile(r"(?<![\d])\d(?:\s?\d){19}(?![\d])")
# v2: паспорт РФ: рядом слово «паспорт»/«серия» или формат «4510 123456» (серия 4 цифры, номер 6)
PASSPORT = re.compile(r"(?:[Пп]аспорт\w*|[Сс]ерия)[^\d\n]{0,20}\d{2}\s?\d{2}(?:[^\d\n]{0,10})?\d{6}(?![\d])"
                      r"|(?<![\d])\d{2}\s?\d{2}\s\d{6}(?![\d])")
# v2: СНИЛС: 123-456-789 01 или слово «снилс» и 11 цифр
SNILS = re.compile(r"(?<![\d])\d{3}[\-\s]\d{3}[\-\s]\d{3}[\s\-]\d{2}(?![\d])|(?i:снилс)\D{0,10}\d{11}(?![\d])")
# v3: слово «снилс» и номер в любом формате (с дефисами или слитно): маскируем без проверки суммы, опечатка остаётся ПДн
SNILS_KW = re.compile(r"(?i)снилс\D{0,10}?(?<![\d])(\d{3}[\-\s]?\d{3}[\-\s]?\d{3}[\s\-]?\d{2})(?![\d])")
# 13-19 цифр, можно через пробел или дефис группами (платёжная карта или карта лояльности)
CARD = re.compile(r"(?<![\d])\d(?:[\s\-]?\d){12,18}(?![\d])")
# госномер РФ: буква, 3 цифры, 2 буквы, регион 2-3 цифры. Буквы только те, что есть на номерах
_L = "АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx"   # плюс латинские двойники
PLATE = re.compile(rf"(?<![\w])[{_L}]\s?\d{{3}}\s?[{_L}]{{2}}\s?\d{{2,3}}(?![\w])")
# v2: прицеп и мотоцикл: две буквы, 4 цифры, регион (АВ 1234 77)
PLATE2 = re.compile(rf"(?<![\w])[{_L}]{{2}}\s?\d{{4}}\s?\d{{2,3}}(?![\w])")
# адрес: «г. Омск, ул. Мира, д. 12, кв. 5» или «ул. Мира, 15»
ADDRESS = re.compile(
    r"(?:г\.\s*[А-ЯЁ][\w-]+,?\s*)?(?:ул\.?|улица|пр-т|проспект|пер\.)\s*[А-ЯЁ][\w-]+(?:\s[А-ЯЁ][\w-]+)?,?\s*"
    r"(?:д\.?\s*)?\d+[а-я]?(?:\s*(?:к\.|корп\.)\s*\d+)?(?:,?\s*кв\.?\s*\d+)?")
# «Мира 15 кв 3» без слова «улица»
ADDRESS2 = re.compile(r"\b[А-ЯЁ][а-яё]+\s\d{1,3}\s?кв\.?\s?\d+")
# v2: адреса, которые v1 пропускал
_CITY_LIST = ["Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск", "Омск", "Тюмень", "Уфа", "Самара", "Пермь",
              "Краснодар", "Ростов-на-Дону", "Челябинск", "Красноярск", "Воронеж", "Волгоград", "Саратов", "Тольятти", "Ижевск",
              "Барнаул", "Иркутск", "Хабаровск", "Владивосток", "Ярославль", "Томск", "Оренбург", "Кемерово", "Сургут",
              "Нижневартовск", "Ханты-Мансийск", "Тула", "Курск", "Липецк", "Набережные Челны", "Нижний Новгород"]
CITIES = "|".join(re.escape(c) for c in _CITY_LIST)
STREET_KW = r"(?:ул\.?|улица|улице|улицу|пр-т|проспект|проспекте|пер\.|переулок|бульвар|б-р|шоссе)"
# «Казань, улица Баумана, дом 7», «Новосибирск, Красный проспект, 25», «Екатеринбург, Малышева 51»
ADDRESS3 = re.compile(
    rf"(?:г\.\s*)?\b(?:{CITIES}),\s*(?:{STREET_KW}\s*)?[А-ЯЁ][\w-]+(?:\s(?:{STREET_KW}|[А-ЯЁ][\w-]+))?,?\s*(?:д\.?\s*|дом\s*)?\d+[а-я]?(?:\s*(?:к\.|корп\.)\s*\d+)?(?:,?\s*кв\.?\s*\d+)?")
# «улице Советской 25», «Красный проспект 25»: название и номер дома без города
ADDRESS4 = re.compile(
    rf"(?:\b{STREET_KW}\s*[А-ЯЁ][\w-]+(?:\s[А-ЯЁ][\w-]+)?|\b[А-ЯЁ][а-яё]+(?:ый|ий|ой|ая)\s{STREET_KW})"
    r",?\s*(?:д\.?\s*|дом\s*)?\d+[а-я]?(?:\s*(?:к\.|корп\.)\s*\d+)?(?:,?\s*кв\.?\s*\d+)?")
# «пункт на Ленина 5», «по Гагарина 12»: предлог, название в родительном падеже, номер дома
ADDRESS5 = re.compile(r"\b(?:на|по|ул\.?)\s+[А-ЯЁ][а-яё]+(?:ина|ова|ева|ёва|ого|ы|ой)\s\d{1,3}[а-я]?\b")
# слова перед номером заказа/накладной: 13 цифр там не карта
NOT_CARD_CONTEXT = re.compile(r"(?i)(?:заказ\w*|накладн\w*|трек\w*|отслежив\w*|посылк\w*)\W{0,12}$")

# --- v3: контрольные суммы и новые типы ПДн ---
def luhn_ok(digits: str) -> bool:
    """Проверка Луна (платёжные карты). Пробелы и дефисы игнорируются."""
    d = [int(c) for c in re.sub(r"\D", "", digits)][::-1]
    return len(d) >= 13 and (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0


def inn_ok(digits: str) -> bool:
    """Контрольная сумма ИНН: 10 цифр (юрлицо) или 12 (физлицо). Алгоритм ФНС."""
    d = [int(c) for c in re.sub(r"\D", "", digits)]
    def ctrl(w, n):
        return sum(a * b for a, b in zip(w, d[:n])) % 11 % 10
    if len(d) == 10:
        return ctrl([2, 4, 10, 3, 5, 9, 4, 6, 8], 9) == d[9]
    if len(d) == 12:
        return ctrl([7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 10) == d[10] and ctrl([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8], 11) == d[11]
    return False


def snils_ok(digits: str) -> bool:
    """Контрольное число СНИЛС: сумма цифр с весами 9..1, остаток от 101 (100 и 101 дают 00); последние две цифры = контрольное число."""
    d = [int(c) for c in re.sub(r"\D", "", digits)]
    if len(d) != 11:
        return False
    s = sum(a * b for a, b in zip(range(9, 0, -1), d[:9]))
    c = s if s < 100 else (0 if s in (100, 101) else s % 101 % 100)
    return c == d[9] * 10 + d[10]


# рядом слово «карта»: тогда 13-19 цифр маскируем как карту и без Луна (опечатка в номере остаётся ПДн)
CARD_WORD = re.compile(r"(?i)(?:карт\w*|card|visa|mastercard|maestro|мир|pan)[^\d\n]{0,24}$")
# слова перед номером обращения, заказа, заявки: десять цифр там не ИНН, даже если контрольная сумма случайно сошлась
NOT_ID_CONTEXT = re.compile(r"(?i)(?:заказ\w*|накладн\w*|трек\w*|отслежив\w*|посылк\w*|обращени\w*|заявк\w*|тикет\w*|№|номер\w*)\W{0,12}$")
# ИНН: слово «ИНН» и 10 или 12 цифр (контрольная сумма не нужна: опечатка в ИНН остаётся ПДн)
INN_KW = re.compile(r"(?i)\bинн\b[^\d\n]{0,20}?(?<![\d])(\d{12}|\d{10})(?![\d])")
# ИНН без слова: 10 или 12 цифр подряд, берём только при сошедшейся контрольной сумме
INN_BARE = re.compile(r"(?<![\d.,\-])(?:\d{12}|\d{10})(?![\d]|[.,]\d)")
# полис ОМС: 16 цифр (можно группами по 4) рядом со словом «полис» или «ОМС»; без слова это неотличимо от карты
OMS = re.compile(r"(?i)(?:полис\w*|\bомс\b)[^\d\n]{0,20}?(?<![\d])(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})(?![\d])")
# водительское: слово («водительское удостоверение», «права», «ву», «в/у») и серия+номер: 99 12 345678 или 77 АВ 123456
DRIVER_LICENSE = re.compile(r"(?i)(?:водительск\w*(?:\s+удостоверени\w+)?|удостоверени\w+|\bправа\b|\bву\b|\bв/у\b)[^\d\n]{0,25}?"
                            r"(?<![\d])(\d{2}\s?(?:\d{2}|[А-ЯA-Z]{2})\s?\d{6})(?![\d])")
_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4 = re.compile(rf"(?<![\w.]){_OCT}(?:\.{_OCT}){{3}}(?![\w]|\.\d)")
# перед четырьмя числами через точку слово «версия/build»: это номер сборки
NOT_IP_CONTEXT = re.compile(r"(?i)(?:верси\w*|version|build|ver|сборк\w*|релиз\w*|release)\.?\s*$")
_MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
# дата рождения только по контексту: просто дата в тексте обращения не ПДн
BIRTH_DATE = re.compile(
    r"(?i)(?:\bд\.\s?р\.?|\bд/р\b|\bдр\b|дат\w*\s+рождения|год\w*\s+рождения|родил(?:ся|ась))\W{0,8}(?:в\s+)?"
    rf"(\d{{1,2}}[./\-]\d{{1,2}}[./\-](?:\d{{4}}|\d{{2}})(?!\d)|\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}(?:\s?г(?:ода|\.)?)?|(?:19|20)\d{{2}}(?:\s?г(?:оду|\.)?)?(?!\d))")
# Telegram: @ник, 5-32 знака; перед @ не буква (иначе это e-mail), после ника не «.домен»
TELEGRAM = re.compile(r"(?<![\w@.])@[A-Za-z][A-Za-z0-9_]{4,31}(?![\w@])(?!\.[A-Za-z])")

# --- словари для имён ---
FIRST_NAMES = set("""Александр Сергей Дмитрий Андрей Алексей Максим Иван Михаил Николай Артём Артем Ильдар Рустем Тимур Олег
Виктор Евгений Владимир Павел Денис Роман Константин Игорь Антон Анатолий Василий Юрий Станислав Руслан Ренат Марат Айрат
Анна Мария Елена Ольга Наталья Татьяна Ирина Светлана Екатерина Юлия Гульнара Алсу Людмила Галина Надежда Виктория
Анастасия Дарья Марина Вера Любовь Оксана Алина Диана Регина Лилия Эльвира""".split())
EXTRA_NAMES = set("""Рамиль Радик Тимофей Кирилл Владислав Нурия Аделина Ринат Пётр Петр Ильнур Артур Эдуард Глеб Ярослав Вадим Данил Даниил Фёдор Федор
Лев Георгий Степан Леонид Валерий Геннадий Борис Никита Егор Илья Матвей Ильяс Азат Булат Динар Камиль Салават Эльдар Зуфар Фарит
Ирек Фанис Рафаэль Рафик Лейсан Зульфия Резеда Айгуль Альфия Венера Гузель Ляйсан Милена Полина Софья София Ксения Валентина
Тамара Нина Зоя Лариса Лидия Алёна Алена Карина Кристина Яна Влада Маргарита Евгения Тихон""".split())
# слова, которые бывают и именем, и обычным словом: маскируем только рядом с фамилией или отчеством
AMBIGUOUS = {"Вера", "Надежда", "Любовь", "Лев", "Яна"}
# «меня зовут Ильдар», «зовут меня иван петров», «моё имя Рамиль»
CONTEXT_NAME = re.compile(r"(?i:меня\s+зовут|зовут\s+меня|мо[её]\s+имя|имя\s+мо[её]|мо[её]\s+фио)[:\s]+([А-ЯЁа-яё]+)(?:\s+([А-ЯЁа-яё]+))?")
PATRONYMIC = re.compile(r"^[А-ЯЁ][а-яё]+(?:ович|евич|ьич|овна|евна|ична|инична)(?:а|у|ем|е|ой|ы)?$")
PATRONYMIC_EXT = re.compile(r"^[А-ЯЁ][а-яё]+(?:(?:ович|евич|ьич|овна|евна|ична|инична)(?:а|у|ем|е|ой|ы)?|овн[ыеуо]й?|евн[ыеуо]й?)$")
SURNAME = re.compile(r"^[А-ЯЁ][а-яё]+(?:ов|ев|ёв|ин|ын|ова|ева|ёва|ина|ына|ский|ская|цкий|цкая|ко|ук|юк|ян|ову|еву|ину|овой|евой|иной)$")
# склонения имён («Сергею», «Анне»): сравниваем по основе без последней буквы
NAME_STEMS = {n[:-1] for n in FIRST_NAMES if len(n) >= 4}
NAME_STEMS_EXT = {n[:-1] for n in FIRST_NAMES | EXTRA_NAMES if len(n) >= 4}
WORD = re.compile(r"[А-ЯЁA-Z][а-яёa-z]+|[А-ЯЁ]\.[А-ЯЁ]\.")


# --- каскад: нужен ли NER ---
# Слова, которые в обращениях встречаются с заглавной буквы и не являются именами: приветствия, служебные слова, продукты, платформы.
# Нужны, чтобы правило «заглавное слово не в начале предложения» не срабатывало на каждом втором тексте.
_KNOWN_TERMS = set("""здравствуйте добрый день вечер утро доброе привет приветствую коллеги уважаемые спасибо благодарю пожалуйста помогите прошу
срочно ужас повторное повторно уже еще ещё что как где почему когда зачем сколько вчера сегодня завтра опять снова
подскажите подскажи скажите объясните исправьте сообщаю обращаюсь пишу жду нет да всё все при после перед
приложение приложении апп карта карты картой карту бонус бонусы бонусная баллы балл акция акции возврат оплата чек
вход авторизация доступ загрузка выгрузка выплата отчёт отчет заявка заявки запросы документы акт блокировка
android ios iphone ipad web windows macos linux chrome safari firefox edge huawei samsung xiaomi
excel word pdf csv telegram whatsapp viber google play store appstore apple wifi sms push qr pin face touch id email
портал сервис партнёров партнеров топливных топливная азс сотрудника сотрудников программа авто адрес""".split())
for _name in ("Приложение АЗС", "Бонусная программа", "B2B-портал топливных карт", "Портал сотрудника", "Сервис для партнёров"):
    _KNOWN_TERMS.update(w.lower() for w in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", _name))

_TOKEN_STRIP = re.compile(r"<[A-Z_]+_\d+>")
_CAP_WORD = re.compile(r"(?<![\w-])[А-ЯЁA-Z][а-яёa-z]{2,}(?![\w])")
_SENT_END = ".!?…\n"
_DIGIT_RUN = re.compile(r"\d+")
_LATIN_WORD = re.compile(r"(?<![\w])[A-Za-z0-9]{3,}(?![\w-])")
_cascade_lock = threading.Lock()
_CASCADE = {"texts": 0, "ner_calls": 0}
_CASCADE_WARNED = False
_TRIG_CACHE: dict = {}


def _compiled(patterns, weak: bool = False):
    key = (tuple(patterns), weak)
    if key not in _TRIG_CACHE:
        if weak:   # слабый триггер: «это Мансур», «я Тагир» (следом слово с заглавной, даже в начале предложения)
            _TRIG_CACHE[key] = re.compile("(?i:" + "|".join(patterns) + r")\s+[А-ЯЁA-Z][а-яёa-z]+") if patterns else None
        else:
            _TRIG_CACHE[key] = re.compile("|".join(patterns), re.I) if patterns else None
    return _TRIG_CACHE[key]


def needs_ner(text: str, cfg=None) -> bool:
    """Решающее правило каскада: нужен ли NER для текста, УЖЕ прошедшего регулярки. Работает только по строке, без моделей, быстро.
    Истина, если в остатке (токены <PHONE_1> и т.п. вырезаны) есть хотя бы одно из:
      1. слова с заглавной буквы не в начале предложения и не из словаря терминов (>= cfg.cascade_min_caps);
      2. триггер: «меня зовут», «с уважением», «ул.», «проспект», «г.», «д. 5» (cfg.cascade_triggers) или «это/я» + слово с заглавной;
      3. цифры подряд >= cfg.cascade_min_digits, латинско-цифровая смесь или латинское слово не из словаря длиной >= cfg.cascade_min_alnum.
    Все пороги в radar.config.Config (cascade_*)."""
    if cfg is None:
        from .config import Config
        cfg = Config()
    t = _TOKEN_STRIP.sub(" ", text)
    if not t.strip():
        return False
    strong = _compiled(cfg.cascade_triggers)
    if strong and strong.search(t):
        return True
    weak = _compiled(cfg.cascade_weak_triggers, weak=True)
    if weak and weak.search(t):
        return True
    if any(len(m.group(0)) >= cfg.cascade_min_digits for m in _DIGIT_RUN.finditer(t)):
        return True
    for m in _LATIN_WORD.finditer(t):
        w = m.group(0)
        if len(w) < cfg.cascade_min_alnum:
            continue
        has_d, has_l = any(c.isdigit() for c in w), any(c.isalpha() for c in w)
        if has_d and has_l:
            return True   # смесь букв и цифр длиной >= порога
        if has_l and w.lower() not in _KNOWN_TERMS:
            return True   # латинское слово не из словаря
    caps = 0
    for m in _CAP_WORD.finditer(t):
        before = t[:m.start()].rstrip()
        if not before or before[-1] in _SENT_END:
            continue   # начало предложения: заглавная обязательна, ничего не говорит
        if m.group(0).lower() in _KNOWN_TERMS:
            continue
        caps += 1
        if caps >= cfg.cascade_min_caps:
            return True
    return False


def cascade_stats(reset: bool = False) -> dict:
    """Сколько текстов прошло через режим ner="auto" и в скольких из них вызван NER (для замеров eval)."""
    with _cascade_lock:
        out = dict(_CASCADE)
        if reset:
            _CASCADE["texts"] = _CASCADE["ner_calls"] = 0
    return out


def _cascade_ner_ok() -> bool:
    """Natasha на месте? Если нет, каскад работает как «только регулярки» и один раз предупреждает."""
    global _CASCADE_WARNED
    from . import mask_ner
    if mask_ner.available():
        return True
    with _cascade_lock:
        if not _CASCADE_WARNED:
            _CASCADE_WARNED = True
            warnings.warn("mask_ner='auto': пакет natasha не установлен, маскирование работает только на регулярках "
                          "(uv pip install -r requirements-ner.txt)", RuntimeWarning, stacklevel=3)
    return False


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


def _mask_names(text: str, state: dict, ext: bool = False) -> str:
    """Имя из словаря, отчество по суффиксу; соседняя фамилия (по суффиксу) входит в тот же токен.
    ext=True: расширенный словарь имён, отчества в косвенных падежах, «Вера/Надежда» только рядом с фамилией, две фамилии подряд."""
    names = FIRST_NAMES | EXTRA_NAMES if ext else FIRST_NAMES
    stems = NAME_STEMS_EXT if ext else NAME_STEMS
    PAT = PATRONYMIC_EXT if ext else PATRONYMIC
    words = list(WORD.finditer(text))
    spans = []   # (start, end) кусков, которые считаем ФИО
    for i, m in enumerate(words):
        w = m.group(0)
        is_core = w in names or bool(PAT.match(w)) or (len(w) >= 4 and w[:-1] in stems and len(w) <= len(w[:-1]) + 2) \
            or (len(w) >= 5 and w[:-2] in stems)
        is_initials = bool(re.match(r"^[А-ЯЁ]\.[А-ЯЁ]\.$", w))
        if not (is_core or is_initials):
            continue
        lone_ok = not (ext and w in AMBIGUOUS)
        start, end = m.start(), m.end()
        # слово перед: фамилия, а перед отчеством любое слово с заглавной буквы (скорее всего имя, даже редкое)
        if i > 0 and text[words[i - 1].end():m.start()].strip() == "":
            prev = words[i - 1].group(0)
            if SURNAME.match(prev) or (PAT.match(w) and len(prev) >= 3 and prev.isalpha()):
                start = words[i - 1].start()
                # ext: «Тарасов Игнат Львович» (ядро отчество, имя вне словаря, перед ним фамилия)
                if ext and PAT.match(w) and i > 1 and text[words[i - 2].end():words[i - 1].start()].strip() == "" \
                        and SURNAME.match(words[i - 2].group(0)):
                    start = words[i - 2].start()
        # слово после: отчество или фамилия, если рядом через пробел
        j = i + 1
        while j < len(words) and text[end:words[j].start()].strip() == "" and (
                PAT.match(words[j].group(0)) or SURNAME.match(words[j].group(0)) or re.match(r"^[А-ЯЁ]\.[А-ЯЁ]\.$", words[j].group(0))):
            end = words[j].end()
            j += 1
        if is_initials and start == m.start():
            continue   # «И.О.» без фамилии рядом ничего не значит
        if not lone_ok and (start, end) == (m.start(), m.end()):
            continue   # «Вера», «Надежда» без фамилии рядом: скорее обычное слово
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


def _replace_checked(text: str, pattern, kind: str, state: dict, group: int = 0, skip=None, valid=None) -> str:
    """Маскирует группу group каждого совпадения (остальное, например слово «ИНН», остаётся в тексте).
    skip(текст_до_совпадения, значение) истинно: пропустить; valid(значение) ложно: пропустить."""
    def repl(m):
        a, b = (m.start(group), m.end(group))
        value = m.string[a:b]
        if (skip and skip(m.string[max(0, a - 30):a], value)) or (valid and not valid(value)):
            return m.group(0)
        key = (kind, re.sub(r"\W", "", value).lower())
        if key not in state["seen"]:
            state["count"][kind] = state["count"].get(kind, 0) + 1
            state["seen"][key] = f"<{kind}_{state['count'][kind]}>"
            state["map"][state["seen"][key]] = value
        return m.string[m.start():a] + state["seen"][key] + m.string[b:m.end()]
    return pattern.sub(repl, text)


def _replace_all_guarded(text: str, pattern, kind: str, state: dict, skip) -> str:
    """Как _replace_all, но совпадение пропускается, если skip(text до совпадения) истинно."""
    def repl(m):
        if skip(m.string[max(0, m.start() - 25):m.start()]):
            return m.group(0)
        return _replace_all(m.group(0), re.compile(re.escape(m.group(0))), kind, state)
    return pattern.sub(repl, text)


def _mask_context_names(text: str, state: dict) -> str:
    """«меня зовут X [Y]»: слово после такой фразы это имя, даже если его нет в словаре."""
    def repl(m):
        start, end = m.start(1), m.end(1)
        w2 = m.group(2)
        if w2 and (w2[0].isupper() or SURNAME.match(w2.capitalize()) or w2.capitalize()[:-1] in NAME_STEMS):
            end = m.end(2)
        value = m.string[start:end]
        key = ("PERSON", value.lower())
        if key not in state["seen"]:
            state["count"]["PERSON"] = state["count"].get("PERSON", 0) + 1
            state["seen"][key] = f"<PERSON_{state['count']['PERSON']}>"
            state["map"][state["seen"][key]] = value
        return m.string[m.start():start] + state["seen"][key] + m.string[end:m.end()]
    return CONTEXT_NAME.sub(repl, text)


def mask_text(text: str, ner=False, ctx_names: bool = False, cascade_cfg=None) -> tuple[str, dict]:
    """Возвращает (замаскированный текст, {токен: оригинал}). Словарь модели не отдавать.
    ner=True: поверх регулярок ещё имена и адреса через Natasha (radar/mask_ner.py, нужен пакет natasha).
    ner="auto": каскад. NER запускается только если needs_ner(остаток после регулярок) истинно; без natasha деградирует к регуляркам (одно предупреждение).
    cascade_cfg: Config с порогами cascade_* (по умолчанию значения Config()).
    ctx_names=True: расширенные правила имён (словарь, падежи отчеств, правило «меня зовут X Y»). Выключено по умолчанию: они маскируют имена с опечатками и в нижнем регистре,
    которые правила v1 пропускали, и это чуть меняет текст для эмбеддинга (на seed 202 сдвигает состав кластеров всплеска)."""
    state = {"seen": {}, "count": {}, "map": {}}
    text = text[:MAX_TEXT_CHARS]
    out = _replace_all(text, EMAIL, "EMAIL", state)
    out = _replace_checked(out, DRIVER_LICENSE, "DRIVER_LICENSE", state, group=1)   # до паспорта: «права 99 12 345678» иначе съест правило паспорта
    out = _replace_all(out, PASSPORT, "PASSPORT", state)
    out = _replace_checked(out, SNILS_KW, "SNILS", state, group=1)
    out = _replace_checked(out, SNILS, "SNILS", state, valid=lambda v: bool(re.search(r"[^\W\d_]", v)) or snils_ok(v))   # со словом «снилс» без суммы, голый формат только с верной суммой
    out = _replace_checked(out, OMS, "OMS", state, group=1)   # до карты: 16 цифр рядом со словом «полис»
    out = _replace_checked(out, INN_KW, "INN", state, group=1)
    out = _replace_all(out, PHONE, "PHONE", state)
    out = _replace_all(out, PHONE_INTL, "PHONE", state)
    out = _replace_all(out, ACCOUNT, "ACCOUNT", state)
    out = _replace_checked(out, CARD, "CARD", state,   # Луна верна, либо рядом слово «карта»; номер заказа и штрихкод не карта
                           skip=lambda before, v: bool(NOT_CARD_CONTEXT.search(before)) or not (luhn_ok(v) or CARD_WORD.search(before)))
    out = _replace_all(out, PHONE10, "PHONE", state)
    out = _replace_checked(out, INN_BARE, "INN", state, skip=lambda before, v: bool(NOT_ID_CONTEXT.search(before)), valid=inn_ok)
    out = _replace_checked(out, IPV4, "IP_ADDRESS", state, skip=lambda before, v: bool(NOT_IP_CONTEXT.search(before)))
    out = _replace_checked(out, BIRTH_DATE, "BIRTH_DATE", state, group=1)
    out = _replace_all(out, TELEGRAM, "TELEGRAM", state)
    out = _replace_all(out, PLATE, "CAR_PLATE", state)
    out = _replace_all(out, PLATE2, "CAR_PLATE", state)
    out = _replace_all(out, ADDRESS3, "ADDRESS", state)
    out = _replace_all(out, ADDRESS, "ADDRESS", state)
    out = _replace_all(out, ADDRESS4, "ADDRESS", state)
    out = _replace_all(out, ADDRESS2, "ADDRESS", state)
    out = _replace_all(out, ADDRESS5, "ADDRESS", state)
    out = _mask_names(out, state, ext=ctx_names)
    if ctx_names:
        out = _mask_context_names(out, state)   # после словарей: «Меня зовут Смирнова Ольга Николаевна» целиком уже токен
    if ner == "auto":
        with _cascade_lock:
            _CASCADE["texts"] += 1
        if needs_ner(out, cascade_cfg) and _cascade_ner_ok():
            with _cascade_lock:
                _CASCADE["ner_calls"] += 1
            from . import mask_ner as _mn
            out = _mn.mask_ner(out, state)
    elif ner:
        from . import mask_ner as _mn
        out = _mn.mask_ner(out, state)
    return out, state["map"]


def mask_tickets(tickets: list, ner=False, ctx_names: bool = False) -> tuple[list, dict]:
    """Для списка обращений: ([замаскированные тексты в том же порядке], {ticket_id: словарь_обратной_подстановки})."""
    texts, reverse = [], {}
    for t in tickets:
        masked, mapping = mask_text(t.text, ner=ner, ctx_names=ctx_names)
        texts.append(masked)
        if mapping:
            reverse[t.id] = mapping
    return texts, reverse


TOKEN_RE = re.compile(r"<[A-Z_]+_\d+>")


def strip_tokens(text: str) -> str:
    """Токены убираем перед эмбеддингом: иначе обращения с ПДн кучкуются из-за слов <PHONE_1>."""
    return TOKEN_RE.sub(" ", text)
