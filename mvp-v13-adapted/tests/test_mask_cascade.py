"""Каскад маскирования: регулярки всегда, NER только когда в остатке есть подозрительные места (Config.mask_ner = "auto")."""
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar import mask as M
from radar import mask_ner
from radar.config import Config
from radar.mask import mask_text, needs_ner


@pytest.mark.parametrize("text", [
    "Здравствуйте, это Мансур Каримов, пишу по заявке",     # заглавное слово не в начале предложения + триггер «это»
    "Меня зовут Тагир, не приходит код",                     # триггер «меня зовут»
    "С уважением, Рамзия",                                   # триггер подписи
    "не могу войти, код подтверждения 482913 не работает",   # цифровая последовательность >= 5
    "пароль от кабинета Qx7Zk2m9 не принимается",            # латинско-цифровая смесь >= 5
    "пишите ivan[at]mail.ru",                                # e-mail, который регулярка не поймала
    "проживаю в посёлке Заречный, дом 4",                    # заглавное слово + адресный триггер
])
def test_needs_ner_fires_on_suspicious_residue(text):
    assert needs_ner(text) is True


@pytest.mark.parametrize("text", [
    "Приложение вылетает на Android 14",                      # продукт/термин из словаря
    "не открывается страница оплаты",
    "Заказ не отображается. Помогите, пожалуйста",            # заглавное только в начале предложения
    "<PHONE_1> не отвечает, перезвоните",                     # токены не считаются остатком
    "после обновления 2.8.0 бонусы не начисляются",           # версия: цифры короткими группами
    "",
])
def test_needs_ner_quiet_on_clean_text(text):
    assert needs_ner(text) is False


def test_needs_ner_thresholds_live_in_config():
    c = Config()
    assert c.cascade_min_digits == 5 and c.cascade_min_alnum == 5 and c.cascade_min_caps == 1
    assert needs_ner("код 123456", c) is True
    assert needs_ner("код 123456", Config(cascade_min_digits=7)) is False
    assert needs_ner("это Мансур", Config(cascade_min_caps=2, cascade_triggers=[], cascade_weak_triggers=[])) is False


def test_auto_calls_ner_only_when_rule_fires(monkeypatch):
    calls = []
    monkeypatch.setattr(mask_ner, "available", lambda: True)
    monkeypatch.setattr(mask_ner, "mask_ner", lambda text, state: calls.append(text) or text)
    mask_text("не открывается страница оплаты", ner="auto")
    assert calls == []
    mask_text("Здравствуйте, это Мансур Каримов, пишу по заявке", ner="auto")
    assert len(calls) == 1


def test_auto_equals_regex_when_rule_quiet():
    t = "Приложение вылетает на Android 14, звоните 89123456789"
    assert mask_text(t, ner="auto") == mask_text(t, ner=False)


def test_old_boolean_values_work_as_before(monkeypatch):
    calls = []
    monkeypatch.setattr(mask_ner, "mask_ner", lambda text, state: calls.append(1) or text)
    mask_text("не открывается страница оплаты", ner=True)       # True: NER всегда
    assert calls == [1]
    mask_text("Здравствуйте, это Мансур Каримов", ner=False)    # False: никогда
    assert calls == [1]


def test_auto_degrades_to_regex_with_single_warning(monkeypatch):
    monkeypatch.setattr(mask_ner, "available", lambda: False)
    monkeypatch.setattr(M, "_CASCADE_WARNED", False)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("NER не должен вызываться"))
    monkeypatch.setattr(mask_ner, "mask_ner", boom)
    s = "Здравствуйте, это Мансур Каримов, пишу по заявке"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = mask_text(s, ner="auto")
        b = mask_text(s, ner="auto")
    assert a == b == mask_text(s, ner=False)
    assert len([x for x in w if "natasha" in str(x.message).lower()]) == 1


def test_cascade_stats_count_texts_and_ner_calls(monkeypatch):
    monkeypatch.setattr(mask_ner, "available", lambda: True)
    monkeypatch.setattr(mask_ner, "mask_ner", lambda text, state: text)
    M.cascade_stats(reset=True)
    mask_text("не открывается страница оплаты", ner="auto")
    mask_text("Здравствуйте, это Мансур Каримов, пишу по заявке", ner="auto")
    assert M.cascade_stats() == {"texts": 2, "ner_calls": 1}


def test_config_accepts_auto_and_booleans():
    assert Config(mask_ner="auto").mask_ner == "auto"
    assert Config(mask_ner=True).mask_ner is True
    with pytest.raises(Exception):
        Config(mask_ner="always")


def test_real_natasha_cascade_catches_name_outside_dictionary():
    pytest.importorskip("natasha")
    out, _ = mask_text("Здравствуйте, это Мансур Каримов, пишу по заявке 778", ner="auto")
    assert "Каримов" not in out
