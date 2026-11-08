"""Блок D (2026-10-27): страница объясняет себя сама. Тесты читают исходник dashboard.html (браузерный прогон: eval/screens-v10/)."""
import re
from pathlib import Path

import pytest

HTML = (Path(__file__).resolve().parents[1] / "app" / "static" / "dashboard.html").read_text(encoding="utf-8")


def test_help_panel_exists_with_button_and_title():
    assert 'id="help"' in HTML and 'id="helpbtn"' in HTML and "Как читать экран" in HTML
    assert 'aria-modal="true"' in HTML and "Escape" in HTML       # закрывается клавишей


@pytest.mark.parametrize("phrase", [
    "Сцена 1", "Сцена 2", "Сцена 3", "Сцена 4",            # что делает каждая сцена
    "1 с реального", "Пуск", "Пауза",                       # скорость и пуск
    "Запись ответов", "Живая модель", "Без модели (шаблон)",
    "Наш баг", "Известно", "Шум",                           # кнопки решений и что они меняют
    "Проверка фикса", "Мок-трекер", "Журнал аудита", "Качество на слепых seeds", "Маскирование",
])
def test_help_and_ui_mention_every_control_and_panel(phrase):
    assert phrase in HTML, phrase


def test_every_button_group_has_label_and_tooltip():
    for attr in ("data-scene", "data-speed", "data-mode", "data-mview"):
        tags = re.findall(r"<button[^>]*" + attr + r"=[^>]*>", HTML)
        assert tags and all("title=" in t for t in tags), attr
    for ident in ("play", "theme", "helpbtn"):
        assert re.search(r'<button[^>]*id="' + ident + r'"[^>]*title=', HTML), ident
    assert HTML.count('class="ctl"') >= 4 and HTML.count('class="lbl"') >= 4      # подпись над каждой группой


def test_panels_have_one_line_hint_and_metrics_points_to_source():
    assert HTML.count('class="hint"') >= 7
    assert "metrics-v3.json" in HTML


def test_agent_modes_use_plain_words_but_api_values_unchanged():
    for v in ('data-mode="replay"', 'data-mode="live"', 'data-mode="template"'):
        assert v in HTML


def test_no_external_resources():
    assert not re.search(r'(?:src|href)=["\']https?://', HTML) and "cdn." not in HTML and "@import" not in HTML


def test_header_wraps_instead_of_overlapping():
    assert re.search(r"header\{[^}]*flex-wrap:wrap", HTML)


def test_token_flow_present():
    assert "sessionStorage" in HTML and "Authorization" in HTML and "401" in HTML


def test_esc_still_wraps_data_fields():
    assert "const esc = s =>" in HTML or "const esc=s=>" in HTML
    for raw in ("${t.key}</td>", "${d.id}:", "${o.id} →", 'data-id="${c.id}"'):
        assert raw not in HTML, raw


def test_autoreply_controls_in_dashboard():
    """Автоответ: кнопки включить/выключить/починили, счётчик закрытого автоответом, пометка допущения."""
    for phrase in ("Включить автоответ", "Выключить автоответ", "Разослать «починили»", "подготовлено автоответов", "допущение, не измерение",
                   "/api/autoreply", "autoreply/approve", "fixed/approve"):
        assert phrase in HTML, phrase
