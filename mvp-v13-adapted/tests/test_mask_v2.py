"""Маскирование v2: структурные ПДн, размеченный набор, неизменность текста для эмбеддинга, необязательные слои."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar.config import Config
from radar.gen import generate
from radar.mask import mask_text, mask_tickets

DATA = ROOT / "tests" / "data"


def gold(name):
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def v1():
    spec = importlib.util.spec_from_file_location("mask_v1", ROOT / "eval" / "baseline" / "mask_v1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.mask_text


@pytest.mark.parametrize("text,kind", [
    ("Расчётный счёт 40817810099910004312 указан неверно", "ACCOUNT"),
    ("перечислите на счёт 40702 810 4 0000 1234567", "ACCOUNT"),
    ("Паспорт 4510 123456, снимите блокировку", "PASSPORT"),
    ("серия 45 10 номер 123456", "PASSPORT"),
    ("СНИЛС 123-456-789 01 не принимается", "SNILS"),
    ("номер телефона 9123456789 не принимает", "PHONE"),
    ("в профиле указан +375 29 123 45 67", "PHONE"),
    ("Прицеп АВ 1234 77 не вносится", "CAR_PLATE"),
    ("авто K123MX 56 не добавляется", "CAR_PLATE"),
    ("номер A123BC77 введён латиницей", "CAR_PLATE"),
    ("номер А123BC 777 смесь букв", "CAR_PLATE"),
    ("Казань, улица Баумана, дом 7", "ADDRESS"),
    ("г. Омск, ул. Мира, д. 12", "ADDRESS"),
    ("АЗС на улице Советской 25 закрыта", "ADDRESS"),
    ("Новосибирск, Красный проспект, 25", "ADDRESS"),
])
def test_structural_pii_is_masked(text, kind):
    out, mp = mask_text(text)
    assert f"<{kind}_1>" in out and kind in "".join(mp), (out, mp)


def test_city_prefix_g_dot_is_not_left_behind():
    out, _ = mask_text("была на г. Новосибирск, ул. Центральная, д. 18, кв. 108")
    assert out == "была на <ADDRESS_1>"


@pytest.mark.parametrize("text", [
    "Заказ 1234567890123 не отображается в истории",
    "по накладной 1234567890123456 посылка не дошла",
])
def test_order_numbers_are_not_cards(text):
    assert "<CARD" not in mask_text(text)[0]


def test_real_card_still_masked_next_to_word_order():
    assert "<CARD_1>" in mask_text("заказ оплачен, карта 4276 1234 5678 9012")[0]


def test_dev_set_structural_types_fully_caught():
    for r in gold("mask_gold.jsonl"):
        out, _ = mask_text(r["text"])
        for p in r["pii"]:
            if p["type"] in ("PHONE", "EMAIL", "CARD", "ACCOUNT", "CAR_PLATE", "PASSPORT", "SNILS", "ADDRESS"):
                left = {w.lower() for w in re.findall(r"\w+", out)}
                assert not any(w.lower() in left for w in re.findall(r"\w{2,}", p["value"])), (r["text"], out)


def test_no_pii_lines_do_not_get_structural_tokens():
    for r in gold("mask_gold.jsonl") + gold("mask_gold_holdout.jsonl"):
        if not r["pii"]:
            out, mp = mask_text(r["text"])
            bad = {t: v for t, v in mp.items() if not re.match(r"<(PERSON|PHONE)_", t)}   # «Вера», телефон горячей линии: известные границы
            assert not bad, (r["text"], bad)


def test_default_text_for_embedding_is_unchanged_on_generated_set():
    """Главная гарантия: v2 по умолчанию даёт тот же текст, что v1, на синтетике eval (иначе сдвинутся кластеры и метрики)."""
    old = v1()
    ds = generate(5, "mid", 300)
    diff = [t.text for t in ds.tickets if old(t.text)[0] != mask_text(t.text)[0]]
    assert diff == []


@pytest.mark.parametrize("text", [
    "С уважением, Фёдорв Рустем",                     # имя с опечаткой в фамилии: v1 маскирует только «Рустем»
    "Меня зовут Васильв Иван.",
    "Меня зовут кузнецов виктор.",                   # нижний регистр: v1 пропускает, расширенные правила маскируют
    "Меня зовут Смирнова Ольга Николаевна.",
])
def test_default_masking_equals_v1_on_known_divergent_cases(text):
    assert mask_text(text) == v1()(text)


def test_ctx_names_off_by_default_and_on_catches_lowercase_name():
    s = "Меня зовут кузнецов виктор. Не работает вход"
    assert "<PERSON" not in mask_text(s)[0]
    assert "<PERSON_1>" in mask_text(s, ctx_names=True)[0]


def test_ctx_names_does_not_mask_ambiguous_word_alone_but_masks_with_surname():
    assert "<PERSON" not in mask_text("Вера в то, что исправите, потеряна", ctx_names=True)[0]
    assert "<PERSON_1>" in mask_text("Прошу ответить Вера Лаптева", ctx_names=True)[0]


def test_extended_patronymic_case_and_two_surnames():
    out, _ = mask_text("Прошу ответить Петровой Анне Викторовне и Тарасов Игнат Львович", ctx_names=True)
    assert "Викторовне" not in out and "Тарасов" not in out and "Львович" not in out


def test_mask_tickets_passes_flags_from_config(monkeypatch):
    from radar import pipeline
    seen = {}
    monkeypatch.setattr(pipeline, "mask_tickets", lambda tickets, ner=False, ctx_names=False: seen.update(ner=ner, ctx=ctx_names) or ([t.text for t in tickets], {}))
    ds = generate(5, "low", 50)
    monkeypatch.setattr(pipeline, "make_embedder", lambda *a, **k: (_ for _ in ()).throw(StopIteration))
    with pytest.raises(StopIteration):
        pipeline._prepare(ds.tickets, Config(mask_ner=True, mask_ctx_names=True))
    assert seen == {"ner": True, "ctx": True}


def test_config_defaults_keep_optional_layers_off():
    c = Config()
    assert c.mask_ner is False and c.mask_ctx_names is False


natasha = pytest.importorskip("natasha", reason="NER необязателен: uv pip install -r requirements-ner.txt")


def test_ner_catches_name_outside_dictionary_and_not_plain_text():
    out, _ = mask_text("Здравствуйте, это Мансур Каримов, пишу по заявке 778", ner=True)
    assert "Каримов" not in out and "<PERSON_1>" in out
    assert mask_text("Приложение вылетает на Android 14", ner=True)[0] == "Приложение вылетает на Android 14"


def test_ner_does_not_mask_bare_city_names():
    text = "Страница тарифов Пермь и Уфа не открывается"
    assert mask_text(text, ner=True)[0] == text


@pytest.mark.parametrize("text", ["9" * 30000, "А" * 30000, "а" * 30000, "a" * 30000 + "@", "1 " * 15000])
def test_long_hostile_text_is_masked_in_bounded_time(text):
    """Раньше регулярка e-mail была квадратичной: 20 000 цифр подряд держали процесс около 20 секунд."""
    import time
    t = time.perf_counter()
    mask_text(text)
    assert time.perf_counter() - t < 2.0


def test_text_over_limit_is_truncated_not_passed_through():
    from radar.mask import MAX_TEXT_CHARS
    tail_phone = "x" * (MAX_TEXT_CHARS + 10) + " +7 912 345 67 89"
    out, _ = mask_text(tail_phone)
    assert len(out) <= MAX_TEXT_CHARS and "912" not in out       # хвост с ПДн в модель не уходит вовсе


def test_normal_email_forms_unchanged_by_length_bounds():
    for e in ("ivan.petrov@mail.ru", "a_b+c@sub.domain-x.co.uk", "i.ivanov собака mail.ru", "sergey1985(at)gmail.com"):
        assert "<EMAIL_1>" in mask_text("пишите " + e + " сегодня")[0], e
