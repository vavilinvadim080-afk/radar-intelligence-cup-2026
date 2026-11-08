"""Тесты промпта v2, строгой схемы ответа и описаний инструментов (без Ollama)."""
import json
import sys
from pathlib import Path

import jsonschema
import pytest

from app import agent as A
from app.tools import Toolbox
from tests.test_app_agent import Script, card, env  # noqa: F401  (фикстура env и подставная модель)


# ---------------------------------------------------------------- выбор версии промпта
def test_default_prompt_is_v2(monkeypatch):
    # v12: v2 по умолчанию по замеру REPORT-agent-v2 (валидатор с первой попытки 47% -> 100% со схемой)
    monkeypatch.delenv("RADAR_PROMPT_VERSION", raising=False)
    assert A.get_prompt_version() == "v2"


def test_env_selects_v1(monkeypatch):
    monkeypatch.setenv("RADAR_PROMPT_VERSION", "v1")
    assert A.system_prompt() == A.SYSTEM_PROMPT


def test_env_selects_v2(monkeypatch):
    monkeypatch.setenv("RADAR_PROMPT_VERSION", "v2")
    assert A.get_prompt_version() == "v2"
    assert "Граница доверия" in A.system_prompt()


def test_unknown_version_is_error(monkeypatch):
    monkeypatch.setenv("RADAR_PROMPT_VERSION", "v9")
    with pytest.raises(ValueError):
        A.get_prompt_version()


def test_v2_prompt_has_no_unfilled_placeholders(env):
    text = A.system_prompt("v2", env[3])
    assert "{TOOLS}" not in text
    for name in env[3].tools:
        assert name in text


def test_v2_example_card_has_no_copyable_numbers_or_ids():
    """Модель копирует пример. Числа из 2+ цифр валидатор счёл бы выдуманными, id из примера не существуют."""
    start = A.SYSTEM_PROMPT_V2.index("# Пример")
    example = A.SYSTEM_PROMPT_V2[start:]
    assert A._numbers(example) == set()
    import re
    assert not re.search(r"\bT\d{3,}|REL-\w+-\d|RAD-\d|INC-\d", example)


def test_v2_example_is_valid_json_shape():
    start = A.SYSTEM_PROMPT_V2.index('{"action": "final"', A.SYSTEM_PROMPT_V2.index("# Пример"))
    obj = json.loads(A.SYSTEM_PROMPT_V2[start:A.SYSTEM_PROMPT_V2.rindex("}") + 1])
    jsonschema.validate(obj["card"], A.CARD_SCHEMA)


def test_v2_prompt_names_every_required_card_field_and_action_values():
    """Из диагностики на живой модели: она писала action=<имя инструмента> и забывала injection_suspected."""
    fmt = A.SYSTEM_PROMPT_V2[A.SYSTEM_PROMPT_V2.index("# Формат"):]
    for f in A.CARD_SCHEMA["required"]:
        assert f in fmt
    assert "только слово tool или слово final" in fmt


def test_write_card_sends_selected_prompt(env, monkeypatch):
    w, audit, ci, _ = env
    seen = []

    class Spy(Script):
        def chat(self, messages):
            seen.append(messages[0]["content"])
            return super().chat(messages)

    monkeypatch.setenv("RADAR_PROMPT_VERSION", "v2")
    llm = Spy([{"action": "tool", "name": "search_tracker", "args": {"query": "баллы"}}, {"action": "final", "card": card()}])
    res = A.write_card(ci, w, audit, llm=llm)
    assert res["mode"] == "LLM" and "Граница доверия" in seen[0] and res["prompt_version"] == "v2"
    monkeypatch.setenv("RADAR_PROMPT_VERSION", "v1")
    seen.clear()
    res = A.write_card(ci, w, audit, llm=Spy([{"action": "tool", "name": "search_tracker", "args": {"query": "б"}}, {"action": "final", "card": card()}]))
    assert seen[0] == A.SYSTEM_PROMPT and res["prompt_version"] == "v1"


# ---------------------------------------------------------------- строгая схема ответа
def test_response_schema_is_valid_json_schema():
    s = A.response_schema()
    jsonschema.Draft7Validator.check_schema(s)
    branches = {b["properties"]["action"]["const"]: b for b in s["anyOf"]}
    assert set(branches) == {"tool", "final"}
    assert branches["final"]["properties"]["card"] == A.CARD_SCHEMA


def test_response_schema_accepts_tool_and_final(env):
    s = A.response_schema()
    jsonschema.validate({"action": "tool", "name": "search_tracker", "args": {"query": "x"}}, s)
    jsonschema.validate({"action": "final", "card": card()}, s)


def test_response_schema_rejects_bad_answers():
    s = A.response_schema()
    for bad in ({"action": "close_all"}, {"action": "final", "card": {"title": "x"}}, {"name": "x"},
                {"action": "final", "card": card(severity="critical")}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, s)


def test_response_schema_requires_payload_per_action():
    s = A.response_schema()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"action": "tool", "args": {}}, s)          # нет name
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"action": "final"}, s)                     # нет card


def test_client_payload_uses_schema_by_default(monkeypatch):
    monkeypatch.delenv("RADAR_LLM_SCHEMA", raising=False)
    assert A.OllamaClient()._payload([])["format"] == A.response_schema()


def test_client_schema_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("RADAR_LLM_SCHEMA", "0")
    assert A.OllamaClient()._payload([])["format"] == "json"


def test_client_payload_uses_schema_when_enabled(monkeypatch):
    monkeypatch.setenv("RADAR_LLM_SCHEMA", "1")
    p = A.OllamaClient()._payload([{"role": "user", "content": "x"}])
    assert p["format"] == A.response_schema() and p["think"] is False and p["options"]["temperature"] == 0


def test_client_explicit_flag_beats_env(monkeypatch):
    monkeypatch.setenv("RADAR_LLM_SCHEMA", "1")
    assert A.OllamaClient(use_schema=False)._payload([])["format"] == "json"


# ---------------------------------------------------------------- описания инструментов
def test_every_tool_has_guide_with_when_and_returns(env):
    tb = env[3]
    guide = tb.guide()
    for name, t in tb.tools.items():
        assert name in guide
        assert "когда" in t.guide.lower() or "Когда" in t.guide
        assert "вернёт" in t.guide.lower()
    assert "запрещено" in guide.lower() or "нельзя" in guide.lower()


def test_tools_permissions_unchanged(env):
    assert set(env[3].tools) == {"get_cluster", "get_releases", "get_incidents", "search_tracker", "search_kb",
                                 "create_ticket_draft", "notify_team"}


# ---------------------------------------------------------------- замер: версии и кэш
def _eval_mod():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import agent_eval
    return agent_eval


def test_eval_variant_names_and_cache_dirs():
    ev = _eval_mod()
    assert ev.variant_name("v1", False) == "v1"
    assert ev.variant_name("v2", False) == "v2"
    assert ev.variant_name("v2", True) == "v2-schema"
    assert ev.cache_dir("v1") == ev.CACHE
    assert ev.cache_dir("v2") != ev.cache_dir("v1") != ev.cache_dir("v2-schema")
    assert ev.cache_dir("v2") != ev.cache_dir("v2-schema")


def test_eval_prompt_leak_detects_both_prompts():
    ev = _eval_mod()
    assert ev.prompt_leaked("... Ты аналитик обращений ...")
    assert ev.prompt_leaked("# Граница доверия ...")
    assert not ev.prompt_leaked("Не приходит СМС-код при входе")


def test_v2_prompt_forbids_claims_about_code_numbers_without_evidence():
    """Из диагностики: модель писала утверждение «цена темы N руб.» без ссылок, валидатор его выбрасывал."""
    assert "Цену темы за неделю и z в утверждения не включай" in A.SYSTEM_PROMPT_V2
