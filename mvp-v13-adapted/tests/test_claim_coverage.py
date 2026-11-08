"""Проверка покрытия утверждений (аудит prod-practices-v1, раздел 4, G1): ссылка должна лексически подтверждать утверждение."""
from datetime import datetime

import pytest

from app import agent as A
from app import textsim
from app.audit import Audit
from app.mocks.world import MockWorld
from app.tools import Toolbox
from radar.models import Release, SuspectedCardInput, SuspectedRelease

NOW = datetime(2026, 3, 9, 18, 0)
TICKETS = [{"id": "T1", "text": "после обновления баллы за заправку не начислились на карту"},
           {"id": "T2", "text": "приложение не открывается, белый экран при входе"},
           {"id": "T3", "text": "карта заблокирована, превышен лимит по топливу"}]


@pytest.fixture
def env():
    w = MockWorld()
    w.add_release(Release("bonus", "2.8.0", datetime(2026, 3, 9, 14, 45), "regular", "Плановый релиз"))
    w.add_task("RAD-101", "bonus", "Баллы за заправку не начислились", "in_progress", datetime(2026, 3, 9, 17, 0), "")
    w.commit()
    ci = SuspectedCardInput(
        burst_id="B1", product="bonus", product_name="Бонусная программа", detected_at=NOW, started_at=datetime(2026, 3, 9, 17, 30),
        unique_users=40, window_min=60, z=6.1, platforms={"android": 30}, versions={"2.8.0": 35},
        suspected_release=SuspectedRelease("2.8.0", datetime(2026, 3, 9, 14, 45), 2.8, True, 0.9, 0.2, "high"),
        weekly_cost_rub=12000.0, evidence=TICKETS, allowed_ids=["T1", "T2", "T3"], min_users_for_status=5)
    return w, ci, Toolbox(w, Audit(), NOW, "B1", "bonus", ci.allowed_ids)


def card(*claims):
    return {"title": "Баллы не начисляются", "status": "новая", "severity": "medium", "injection_suspected": False,
            "claims": [{"text": t, "ids": ids} for t, ids in claims]}


def validate(env, *claims, **kw):
    w, ci, tb = env
    return A.validate_card(card(*claims), ci, tb, "Людей 40, версия 2.8.0, через 2.8 ч", **kw)


# ---------------------------------------------------------------- textsim
def test_content_words_drop_meta_numbers_and_stopwords():
    ws = textsim.content_words("Пользователи сообщают: 40 человек на android не могут оплатить заправку 2026-03-09")
    assert ws == ["оплатить", "заправку"]


def test_word_match_survives_prefixes_and_endings():
    grams = textsim.gram_set("карта заблокирована после превышения лимита")
    assert textsim.word_in("блокировке", grams)          # заблокирована ~ блокировке
    assert textsim.word_in("лимит", grams)
    assert not textsim.word_in("оплата", grams)


def test_overlap_counts_matched_content_words():
    m, n = textsim.overlap("баллы за заправку не начислились", "баллы за заправку не пришли на карту")
    assert (m, n) == (2, 3)                      # баллы, заправку совпали; начислились нет


# ---------------------------------------------------------------- validate_card
def test_supported_symptom_kept(env):
    out, rep = validate(env, ("Баллы за заправку не начислились на карту", ["T1"]))
    assert len(out["claims"]) == 1 and not rep["dropped"]


def test_unsupported_symptom_dropped_with_reason(env):
    out, rep = validate(env, ("Баллы за заправку не начислились на карту", ["T2"]), ("Баллы за заправку не начислились", ["T1"]))
    assert [c["ids"] for c in out["claims"]] == [["T1"]]
    assert rep["dropped"][0]["why"] == "ссылка не подтверждает"
    assert rep["coverage"]["dropped"] == 1


def test_one_supporting_ticket_among_cited_is_enough(env):
    out, _ = validate(env, ("Карта заблокирована из-за превышения лимита", ["T2", "T3"]))
    assert len(out["claims"]) == 1


def test_release_task_incident_ids_checked_by_existence_only(env):
    out, rep = validate(env, ("Подозреваемый релиз 2.8.0 вышел до всплеска жалоб", ["REL-bonus-2.8.0"]),
                        ("Похожая задача уже есть в трекере разработки", ["RAD-101"]))
    assert len(out["claims"]) == 2 and not rep["dropped"]


def test_claim_mixing_release_and_ticket_not_lexically_checked(env):
    out, _ = validate(env, ("Подозреваемый релиз 2.8.0: всплеск жалоб начался через 2.8 ч после выхода", ["REL-bonus-2.8.0", "T2"]))
    assert len(out["claims"]) == 1


def test_scale_claim_without_symptom_words_not_checked(env):
    out, _ = validate(env, ("Людей 40, платформа android, версия 2.8.0", ["T2"]))
    assert len(out["claims"]) == 1


def test_threshold_is_configurable(env):
    # 1 слово из 3 содержательных совпало: при мягком пороге проходит, при жёстком нет
    text = ("Баллы за заправку исчезли после перезапуска терминала", ["T1"])
    soft, _ = validate(env, text, min_overlap=0.2)
    hard, rep = validate(env, text, ("Баллы за заправку не начислились", ["T1"]), min_overlap=0.9)
    assert len(soft["claims"]) == 1
    assert [c["text"] for c in hard["claims"]] == ["Баллы за заправку не начислились"] and rep["coverage"]["dropped"] == 1


def test_threshold_from_env(env, monkeypatch):
    monkeypatch.setenv("RADAR_CLAIM_MIN_OVERLAP", "0.95")
    assert A.claim_min_overlap() == 0.95
    monkeypatch.setenv("RADAR_CLAIM_MIN_OVERLAP", "мусор")
    assert A.claim_min_overlap() == A.CLAIM_MIN_OVERLAP_DEFAULT


def test_text_of_tool_seen_ticket_is_used(env):
    w, ci, tb = env
    tb.seen_ids.add("T9")
    tb.seen_texts["T9"] = "в приложении не открывается экран входа"
    out, _ = A.validate_card(card(("Не открывается экран входа в приложении", ["T9"])), ci, tb, "")
    assert len(out["claims"]) == 1


def test_ticket_without_known_text_is_not_a_proof_nor_a_refutation(env):
    w, ci, tb = env
    tb.seen_ids.add("T9")         # id есть, текст неизвестен: проверить нельзя, утверждение остаётся (проверка id как раньше)
    out, rep = A.validate_card(card(("Не открывается экран входа в приложении", ["T9"])), ci, tb, "")
    assert len(out["claims"]) == 1 and rep["coverage"]["unchecked"] == 1


# ---------------------------------------------------------------- найдено на сохранённых карточках (замер 2026-10-29)
def test_prefix_match_handles_case_endings_and_short_roots():
    """«входом» ~ «ВХОД», «кода» ~ «КОД», «участвовать» ~ «участвую», «цены» ~ «цену»: окончания не мешают."""
    m, n = textsim.overlap("проблемы с входом и получением кода из SMS", "ЖДУ КОД ИЗ СМС, НИЧЕГО НЕТ, ВХОД НЕ РАБОТАЕТ")
    assert (m, n) == (2, 3)
    m, n = textsim.overlap("не могут участвовать в акциях по начислению бонусных баллов", "участвую в акции, когда начислят баллы лояльности")
    assert m == 4 and n == 5


def test_scale_claim_is_not_lexically_checked():
    """«Проблема наблюдается на платформах iOS (4), Android (22) и неизвестных устройствах»: про масштаб, симптома нет."""
    assert textsim.is_scale_claim("Проблема наблюдается на платформах iOS (4), Android (22) и неизвестных устройствах (1).")
    assert textsim.is_scale_claim("Уникальных пользователей: 28 человек на платформе web и unknown.")
    assert not textsim.is_scale_claim("Пользователи не могут скачать закрывающие документы.")


def test_words_of_all_cited_tickets_are_pooled(env):
    """Утверждение собрано из двух обращений (регистрация из одного, выплата из другого): хватает объединения их слов."""
    w, ci, tb = env
    tb.seen_texts.update({"T8": "не проходит регистрация", "T9": "выплата не пришла"})
    tb.seen_ids.update({"T8", "T9"})
    out, _ = A.validate_card(card(("Проблемы с регистрацией и выплатой партнёров", ["T8", "T9"])), ci, tb, "")
    assert len(out["claims"]) == 1


def test_timing_words_are_meta():
    """«всплеск, начавшийся 17:32 и сработавший 18:02» говорит про время, а не про симптом: эти слова в обращениях искать не надо."""
    assert textsim.content_words("Наблюдается всплеск обращений по бонусной программе, начавшийся 2026-03-09 17:32 и сработавший 2026-03-09 18:02.") \
        == ["бонусной", "программе"]


def test_peak_claim_is_timing_not_symptom(env):
    """«Всплеск жалоб начался ... и достиг пика ...»: время, симптома нет, лексически проверять нечего (не «ссылка не подтверждает»)."""
    assert textsim.content_words("Всплеск жалоб начался 2026-03-07 18:09 и достиг пика в 2026-03-07 20:06.") == []
