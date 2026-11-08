"""Маскирование v3: ИНН, СНИЛС и карта с контрольными суммами, ОМС, водительское, IP, дата рождения, Telegram-ник.
Позитивы и негативы: числа заказов, версии 4.12.0, суммы не должны маскироваться ложно."""
import pytest

from radar.mask import luhn_ok, inn_ok, snils_ok, mask_text


def kinds(text):
    return sorted({t.strip("<>").rsplit("_", 1)[0] for t in mask_text(text)[1]})


# --- контрольные суммы ---
def test_luhn():
    assert luhn_ok("4111111111111111") and luhn_ok("4111 1111 1111 1111")
    assert not luhn_ok("4276123456789012")


def test_inn_checksums():
    assert inn_ok("7707083893") and inn_ok("500100732259")      # ПАО Сбербанк; публичный пример физлица
    assert not inn_ok("7707083894") and not inn_ok("500100732250")
    assert not inn_ok("12345")


def test_snils_checksum():
    assert snils_ok("11223344595")
    assert not snils_ok("11223344596")


# --- ИНН ---
@pytest.mark.parametrize("text", [
    "ИНН 7707083893 не принимается", "инн организации: 7707083893", "физлицо 500100732259 в заявке",
    "ИНН 7707083894 с опечаткой",   # слово ИНН рядом: маскируем и без контрольной суммы (безопасная ошибка)
])
def test_inn_masked(text):
    out, mp = mask_text(text)
    assert "<INN_1>" in out and not any(c.isdigit() for c in out.replace("<INN_1>", ""))


@pytest.mark.parametrize("text", [
    "заказ 7707083893 не пришёл",                 # контрольная сумма сошлась, но это номер заказа
    "версия приложения 4.12.0 вылетает",
    "списали 1234567890 рублей?",                 # без ИНН-контекста и контрольная сумма не сходится
    "номер обращения 1234567890",
    "сумма 12 500,00 руб.",
])
def test_inn_not_false_positive(text):
    assert "<INN" not in mask_text(text)[0]


# --- СНИЛС ---
def test_snils_with_valid_checksum_bare():
    assert "<SNILS_1>" in mask_text("мой номер 112-233-445 95 прошу проверить")[0]


def test_snils_bad_checksum_bare_not_masked_but_keyword_masks():
    assert "<SNILS" not in mask_text("код 123-456-789 01 в чеке")[0]
    assert "<SNILS_1>" in mask_text("СНИЛС 123-456-789 01 не принимается")[0]


# --- карта: Луна ---
def test_card_luhn_valid_masked_without_context():
    assert "<CARD_1>" in mask_text("списали с 4111 1111 1111 1111 дважды")[0]


def test_card_luhn_invalid_without_context_not_card():
    assert "<CARD" not in mask_text("штрихкод 4276123456789012 на упаковке")[0]


def test_card_luhn_invalid_with_card_word_still_masked():
    assert "<CARD_1>" in mask_text("карта 4276 1234 5678 9012 заблокирована")[0]


# --- ОМС ---
def test_oms_masked_with_context():
    out = mask_text("полис ОМС 1234567890123456 не найден")[0]
    assert "<OMS_1>" in out and "<CARD" not in out
    assert "<OMS_1>" in mask_text("полис 1234 5678 9012 3456")[0]


def test_oms_not_without_context():
    assert "<OMS" not in mask_text("заказ 1234567890123456 в пути")[0]


# --- водительское ---
@pytest.mark.parametrize("text", [
    "водительское удостоверение 99 12 345678 просрочено", "ву 9912 345678", "права 77 АВ 123456 утеряны", "в/у 99 12 345678",
])
def test_driver_license_masked(text):
    out = mask_text(text)[0]
    assert "<DRIVER_LICENSE_1>" in out and "345678" not in out and "123456" not in out


def test_driver_license_requires_context():
    assert "<DRIVER" not in mask_text("номер 99 12 345678 в списке")[0]
    assert "<DRIVER" not in mask_text("ручки на ву 3 штуки")[0]


# --- IP ---
@pytest.mark.parametrize("text,ip", [("заходил с 192.168.1.15 вчера", "192.168.1.15"), ("ip 8.8.8.8.", "8.8.8.8"), ("сервер 10.0.0.255:8080", "10.0.0.255")])
def test_ip_masked(text, ip):
    out = mask_text(text)[0]
    assert "<IP_ADDRESS_1>" in out and ip not in out


@pytest.mark.parametrize("text", [
    "версия 4.12.0 не открывается", "build 1.2.3.4 вылет", "версия приложения v2.3.4.5", "999.999.999.999", "курс 1.234.567.890 руб", "сумма 12.50", "раздел 1.2.3.4.5",
])
def test_ip_not_false_positive(text):
    assert "<IP" not in mask_text(text)[0]


# --- дата рождения ---
@pytest.mark.parametrize("text", [
    "д.р. 05.03.1990 проверьте", "дата рождения: 5 марта 1990 г.", "родился 05/03/1990 в Омске", "Дата рождения 05-03-90", "ДР 01.02.1985",
])
def test_birth_date_masked(text):
    out = mask_text(text)[0]
    assert "<BIRTH_DATE_1>" in out and "1990" not in out and "1985" not in out and "-90" not in out


def test_plain_dates_not_masked():
    for t in ("списали 05.03.2026 в 12:30", "обновление от 10.10.2026", "заказ от 5 марта 2026 г."):
        assert "<BIRTH" not in mask_text(t)[0]


def test_birth_keeps_keyword():
    assert mask_text("дата рождения 05.03.1990")[0].startswith("дата рождения")


# --- Telegram ---
def test_telegram_masked():
    out = mask_text("пишите в телеграм @ivan_petrov_77 или звоните")[0]
    assert "<TELEGRAM_1>" in out and "ivan_petrov" not in out


@pytest.mark.parametrize("text", ["пишите на ivan@mail.ru", "почта @mail.ru", "оператор @ рядом", "команда @abc"])
def test_telegram_not_email_or_short(text):
    assert "<TELEGRAM" not in mask_text(text)[0]


def test_email_still_email():
    assert kinds("ivan@mail.ru и @ivan_petrov") == ["EMAIL", "TELEGRAM"]


# --- боевой режим NER через окружение (умолчание Config остаётся False: протокол оценки заморожен) ---
def test_mask_ner_default_false_and_env_switches_to_auto(monkeypatch):
    from radar.config import Config
    monkeypatch.delenv("RADAR_MASK_NER", raising=False)
    assert Config().mask_ner is False
    monkeypatch.setenv("RADAR_MASK_NER", "auto")
    assert Config().mask_ner == "auto"
    monkeypatch.setenv("RADAR_MASK_NER", "мусор")
    assert Config().mask_ner is False          # непонятное значение не роняет запуск
    assert Config(mask_ner=True).mask_ner is True   # явный аргумент сильнее окружения
