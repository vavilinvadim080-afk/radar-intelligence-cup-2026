"""Защита от спама и накрутки в ядре (radar/antispam.py, RadarStream, LiveRadar). Сначала красные тесты, потом код.
Модель угроз и замеры: STATUS-antispam.md. Здесь: флуд одним аккаунтом, лимит обращений на пользователя, признак накрутки."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from radar import antispam
from radar.config import Config
from radar.live import LiveRadar
from radar.models import Release, Ticket
from radar.pipeline import RadarStream

T0 = datetime(2026, 3, 2)
B0 = T0 + timedelta(days=5, hours=12)
ATTACK = "Бонусы не начислили за заправку срочно верните баллы ужасное приложение"
WORDS = ["бонусы", "баллы", "заправка", "карта", "приложение", "вход", "оплата", "чек", "акция", "списали", "пришли", "пропали",
         "после", "обновления", "ошибка", "экран", "долго", "грузится", "не", "работает", "станция", "колонка", "кэшбэк", "уровень"]


def varied(rng, k=7):
    return " ".join(rng.choice(WORDS, size=k, replace=False))


def tk(i, ts, user, text, product="azs"):
    return Ticket(f"T{i:06d}", ts, product, "app_chat", "android", "4.1.0", user, text)


def base_rows(rng):
    """Фон: в среднем 2 человека в час в одну тему, разные тексты, 7 суток; релиз продукта за 2 часа до B0 (иначе тревоги не будет)."""
    rows = []
    for h in range(7 * 24):
        for _ in range(rng.poisson(2)):
            rows.append((T0 + timedelta(hours=h, minutes=float(rng.uniform(0, 60))), f"b{len(rows)}", varied(rng)))
    return rows


def run_stream(rows, cfg=None, hook=None):
    cfg = cfg or Config(warmup_days=3, min_users=6)
    rel = [Release("azs", "4.2.0", B0 - timedelta(hours=2), "regular")]
    s = RadarStream(rel, cfg, dim=4, t0=T0, start_ts=T0)
    if hook:
        s.on_antispam = hook
    vec = np.array([1.0, 0, 0, 0], dtype=np.float32)
    rows = sorted(rows)
    for i, (ts, user, text) in enumerate(rows):
        s.push(tk(i, ts, user, text), text, vec)
    s.finish(rows[-1][0] + timedelta(hours=1))
    return s


def attack_rows(n, minutes=20, text=ATTACK, start=B0, uniform=False, prefix="atk"):
    if uniform:
        return [(start + timedelta(minutes=minutes * i / n), f"{prefix}{i}", text) for i in range(n)]
    rng = np.random.default_rng(5)
    return [(start + timedelta(minutes=float(rng.uniform(0, minutes))), f"{prefix}{i}", text) for i in range(n)]


# ---------------------------------------------------------------- признак накрутки: чистая функция
def rows_of(texts, step_s=60.0, t=B0):
    return [(f"u{i}", t + timedelta(seconds=step_s * i), tx) for i, tx in enumerate(texts)]


def test_assess_flags_many_users_with_same_text():
    r = antispam.assess(rows_of([ATTACK] * 30), Config())
    assert r["suspect"] and r["dup_users"] == 30 and r["dup_share"] == 1.0 and "накрутк" in r["reason"]


def test_assess_flags_near_duplicates_with_small_mutations():
    texts = [ATTACK + (" !!" if i % 2 else "") + (f" {i}" if i % 3 == 0 else "") for i in range(30)]
    assert antispam.assess(rows_of(texts), Config())["suspect"]


def test_assess_does_not_flag_varied_texts():
    rng = np.random.default_rng(1)
    r = antispam.assess(rows_of([varied(rng) for _ in range(30)]), Config())
    assert not r["suspect"] and r["dup_share"] < 0.2


def test_assess_needs_enough_users():
    assert not antispam.assess(rows_of([ATTACK] * 5), Config())["suspect"]      # мало людей: мало данных, не обвиняем


def test_assess_short_texts_are_not_evidence():
    """«Не работает» пишут многие и честно: короткий текст не доказательство накрутки."""
    assert not antispam.assess(rows_of(["не работает"] * 40), Config())["suspect"]


def test_assess_counts_one_vote_per_user():
    """Один пользователь с 40 одинаковыми обращениями это один голос, а не группа из 40."""
    rows = [("same", B0 + timedelta(seconds=30 * i), ATTACK) for i in range(40)]
    r = antispam.assess(rows, Config())
    assert not r["suspect"] and r["n_users"] == 1


def test_assess_timer_rule_catches_even_spacing_at_lower_share():
    rng = np.random.default_rng(2)
    texts = [ATTACK] * 8 + [varied(rng) for _ in range(14)]          # доля группы 0.36: ниже основного порога 0.5
    even = antispam.assess(rows_of(texts, step_s=60.0), Config())
    assert even["suspect"] and even["gap_cv"] < 0.1
    rows = rows_of(texts)
    rows = [(u, B0 + timedelta(seconds=float(x)), t) for (u, _, t), x in zip(rows, np.cumsum(rng.exponential(60, len(rows))))]
    assert not antispam.assess(rows, Config())["suspect"]            # те же тексты, но времена как у людей


def test_assess_reports_fresh_share_as_diagnostic_only():
    r = antispam.assess(rows_of([ATTACK] * 30), Config(), seen_before=lambda u: u.endswith("0"))
    assert 0.0 < r["fresh_share"] < 1.0 and r["suspect"]


def test_recount_removes_duplicates_but_keeps_one_vote():
    assert antispam.users_without_duplicates(60, 41) == 20


# ---------------------------------------------------------------- 1. флуд одним аккаунтом
def test_single_user_with_500_tickets_does_not_raise_alert():
    rng = np.random.default_rng(3)
    rows = base_rows(rng) + [(B0 + timedelta(minutes=0.5 * i), "spammer", varied(rng)) for i in range(500)]
    s = run_stream(rows, Config(warmup_days=3, min_users=6, user_cap_max=0))      # лимит выключен: держит сам счёт уникальных людей
    assert not [b for b in s.bursts if b.alert]


def test_user_cap_drops_excess_and_counts_them():
    rng = np.random.default_rng(3)
    rows = base_rows(rng) + [(B0 + timedelta(minutes=0.5 * i), "spammer", varied(rng)) for i in range(500)]
    base = len(base_rows(np.random.default_rng(3)))
    s = run_stream(rows, Config(warmup_days=3, min_users=6, user_cap_max=20, user_cap_window_min=1440))
    assert s.n_capped == 480 and s.capped_by_user == {"spammer": 480}
    assert len(s.tickets) == base + 20                       # лишние в детектор не попали
    assert not [b for b in s.bursts if b.alert]


def test_user_cap_window_slides():
    s = RadarStream([], Config(user_cap_max=3, user_cap_window_min=60), dim=4, t0=T0, start_ts=T0)
    vec = np.array([1.0, 0, 0, 0], dtype=np.float32)
    verdicts = [s.push(tk(i, T0 + timedelta(minutes=10 * i), "u1", "текст обращения номер один"), "x", vec) for i in range(12)]
    # 3 за час разрешено: 10-минутный шаг даёт 6 в окне, значит пропускаем 3, потом по одному на каждые освободившиеся места
    assert verdicts[:3] == ["ok"] * 3 and verdicts[3] == "user_capped"
    assert s.n_capped >= 1 and "ok" in verdicts[6:]          # окно уехало: обращения снова принимаются


def test_user_cap_does_not_touch_normal_users():
    s = RadarStream([], Config(), dim=4, t0=T0, start_ts=T0)
    vec = np.array([1.0, 0, 0, 0], dtype=np.float32)
    for i in range(20):
        assert s.push(tk(i, T0 + timedelta(hours=i), f"u{i}", "x"), "x", vec) == "ok"
        assert s.push(tk(100 + i, T0 + timedelta(hours=i, minutes=5), f"u{i}", "x"), "x", vec) == "ok"
    assert s.n_capped == 0


def test_user_cap_audit_hook_is_throttled():
    events = []
    rng = np.random.default_rng(3)
    rows = base_rows(rng) + [(B0 + timedelta(minutes=0.5 * i), "spammer", varied(rng)) for i in range(500)]
    run_stream(rows, Config(warmup_days=3, min_users=6, user_cap_max=20), hook=lambda kind, p: events.append((kind, p)))
    caps = [p for k, p in events if k == "antispam_user_capped"]
    assert 1 <= len(caps) <= 6 and caps[0]["user_id"] == "spammer" and caps[0]["dropped"] == 1      # первый и каждый сотый, не 480 записей


def test_live_radar_logs_user_capped_to_audit_callback():
    log = []
    live = LiveRadar([], Config(user_cap_max=2, user_cap_window_min=60, warmup_days=1), log=lambda k, p: log.append(k))
    rng = np.random.default_rng(4)
    warm = [tk(i, T0 + timedelta(minutes=30 * i), f"w{i}", varied(rng)) for i in range(40)]
    assert all(live.ingest(t) == "ok" for t in warm)
    spam = [tk(100 + i, T0 + timedelta(days=1, minutes=i), "u1", varied(rng)) for i in range(6)]
    verdicts = [live.ingest(t) for t in spam]
    assert verdicts == ["ok", "ok"] + ["user_capped"] * 4 and log.count("antispam_user_capped") == 1
    assert "ticket_user_capped" not in log                  # одна запись о срабатывании, а не по записи на каждое отброшенное


# ---------------------------------------------------------------- 2. накрутка
def test_baseline_attack_without_protection_pages():
    """Эталон угрозы: без защиты 40 свежих аккаунтов с одним текстом будят дежурного."""
    rng = np.random.default_rng(3)
    s = run_stream(base_rows(rng) + attack_rows(40), Config(warmup_days=3, min_users=6, brigade_check=False))
    al = [b for b in s.bursts if b.alert]
    assert al and al[0].tier == "page" and al[0].suspicion is None


def test_brigading_burst_is_quiet_with_reason_and_marker():
    rng = np.random.default_rng(3)
    events = []
    s = run_stream(base_rows(rng) + attack_rows(40), hook=lambda k, p: events.append(k))
    al = [b for b in s.bursts if b.alert]
    assert al and all(b.tier == "quiet" for b in al)
    assert al[0].suspicion and al[0].suspicion["dup_share"] >= 0.5 and "накрутк" in al[0].alert_reason
    assert "antispam_brigading" in events


def test_brigading_does_not_block_later_real_alert_on_same_cluster():
    """Подозрительная тревога не ставит cluster в «уже тревожили»: настоящая регрессия в той же теме через день будит."""
    rng = np.random.default_rng(3)
    day2 = B0 + timedelta(days=1)
    real = [(day2 + timedelta(minutes=float(rng.uniform(0, 25))), f"r{i}", varied(rng)) for i in range(40)]
    rel = Release("azs", "4.3.0", day2 - timedelta(hours=2), "regular")
    cfg = Config(warmup_days=3, min_users=6)
    s = RadarStream([Release("azs", "4.2.0", B0 - timedelta(hours=2), "regular"), rel], cfg, dim=4, t0=T0, start_ts=T0)
    vec = np.array([1.0, 0, 0, 0], dtype=np.float32)
    rows = sorted(base_rows(rng)[:0] + base_rows(np.random.default_rng(3)) + attack_rows(40) + real)
    for i, (ts, user, text) in enumerate(rows):
        s.push(tk(i, ts, user, text), text, vec)
    s.finish(rows[-1][0] + timedelta(hours=1))
    page = [b for b in s.bursts if b.alert and b.tier == "page"]
    assert page and page[-1].detected_at > day2


def test_attack_on_top_of_real_regression_does_not_hide_it():
    """Настоящая регрессия (60 разных людей) и накрутка (40 одинаковых) в одной теме: после вычета копий остаток всё ещё тревога."""
    rng = np.random.default_rng(3)
    real = [(B0 + timedelta(minutes=float(rng.uniform(0, 20))), f"r{i}", varied(rng)) for i in range(60)]
    s = run_stream(base_rows(rng) + real + attack_rows(40))
    al = [b for b in s.bursts if b.alert]
    assert al and al[0].tier == "page", [(b.id, b.tier, b.alert_reason) for b in al]


def test_real_regression_with_varied_texts_stays_page():
    rng = np.random.default_rng(3)
    real = [(B0 + timedelta(minutes=float(rng.uniform(0, 20))), f"r{i}", varied(rng)) for i in range(40)]
    s = run_stream(base_rows(rng) + real)
    al = [b for b in s.bursts if b.alert]
    assert al and al[0].tier == "page" and al[0].suspicion is None


def test_config_defaults_are_conservative():
    c = Config()
    assert c.brigade_check is True and c.user_cap_max >= 10 and c.brigade_dup_share >= 0.5 and c.brigade_min_tokens >= 4
