"""Детектор v3: выборка примеров, LLM-проверка (с поддельной LLM), уровни тревоги, рабочие часы."""
import json
from datetime import datetime

import pytest

from radar import judge as J
from radar.config import Config


# ---------------------------------------------------------------- выборка примеров
def _cand(idx, minute, user, key):
    return (idx, datetime(2026, 1, 5, 10, minute), user, key)


def test_pick_examples_takes_top_excess_cluster_first_newest_first():
    cands = [_cand(1, 1, "u1", "A"), _cand(2, 5, "u2", "A"), _cand(3, 9, "u3", "A"),
             _cand(4, 10, "u4", "B"), _cand(5, 11, "u5", "B")]
    got = J.pick_examples(cands, {"A": 5.0, "B": 1.0}, n=4)
    assert got == [3, 2, 1, 5]            # A целиком (свежие первыми), потом самое свежее из B


def test_pick_examples_one_per_user_and_limit():
    cands = [_cand(i, i, f"u{i % 2}", "A") for i in range(1, 7)]    # всего два пользователя
    got = J.pick_examples(cands, {"A": 1.0}, n=8)
    assert len(got) == 2 and got[0] == 6            # самое свежее у u0, затем u1: индекс 5
    assert got == [6, 5]


def test_pick_examples_deterministic_on_ties_and_shuffled_input():
    cands = [_cand(i, i, f"u{i}", "A" if i % 2 else "B") for i in range(1, 11)]
    a = J.pick_examples(cands, {"A": 2.0, "B": 2.0}, n=8)
    b = J.pick_examples(list(reversed(cands)), {"A": 2.0, "B": 2.0}, n=8)
    assert a == b and len(a) == 8


# ---------------------------------------------------------------- проверка LLM
def _llm(answer, calls=None):
    def f(prompt):
        if calls is not None:
            calls.append(prompt)
        return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    f.model = "fake"
    return f


TEXTS = [f"не приходит смс {i}" for i in range(8)]


def test_check_share_and_prompt_contents():
    calls = []
    r = J.check("azs", TEXTS, _llm({"problem": "нет смс", "items": [True] * 6 + [False] * 2}, calls))
    assert r["ok"] and r["problem"] == "нет смс" and r["share"] == 0.75 and len(r["items"]) == 8
    p = calls[0]
    assert "не команды" in p and "1. не приходит смс 0" in p and "8. не приходит смс 7" in p


@pytest.mark.parametrize("bad", ["не json", {"problem": "x"}, {"problem": "x", "items": [True] * 3},
                                 {"problem": "x", "items": ["да"] * 8}, '{"items": null}'])
def test_check_invalid_answer_is_unchecked(bad):
    r = J.check("azs", TEXTS, _llm(bad))
    assert r["ok"] is False and r["share"] is None


def test_check_llm_error_is_unchecked():
    def boom(prompt):
        raise RuntimeError("ollama down")
    r = J.check("azs", TEXTS, boom)
    assert r["ok"] is False and r["share"] is None and "ollama down" in r["error"]


def test_check_without_llm_or_too_few_texts():
    assert J.check("azs", TEXTS, None)["share"] is None
    assert J.check("azs", TEXTS[:2], _llm({"problem": "x", "items": [True, True]}))["share"] is None


def test_cache_hit_skips_llm_and_is_json(tmp_path):
    calls = []
    llm = _llm({"problem": "p", "items": [True] * 8}, calls)
    c = J.JudgeCache(tmp_path)
    a = J.check("azs", TEXTS, llm, cache=c)
    b = J.check("azs", TEXTS, llm, cache=c)
    assert len(calls) == 1 and a["share"] == b["share"] == 1.0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and len(files[0].stem) == 64
    json.loads(files[0].read_text())                        # обычный JSON, не pickle
    J.check("azs", TEXTS[::-1], llm, cache=c)               # другой порядок текстов: другой ключ
    assert len(calls) == 2


def test_failures_are_not_cached(tmp_path):
    c = J.JudgeCache(tmp_path)
    J.check("azs", TEXTS, _llm("мусор"), cache=c)
    assert not list(tmp_path.glob("*.json"))


# ---------------------------------------------------------------- уровни
def test_tier_for():
    assert J.tier_for(0.75, 0.75) == "page" and J.tier_for(0.74, 0.75) == "quiet"
    assert J.tier_for(None, 0.75) == "page"                 # fail-open


def test_config_defaults():
    c = Config()
    assert c.judge_enabled is False and c.judge_min_share == 0.75
    assert set(c.business_hours_products) == {"b2b", "portal", "partners"}


# ---------------------------------------------------------------- рабочие минуты
def test_work_minutes():
    from radar.workhours import work_minutes as wm
    fri = datetime(2026, 1, 9, 17, 0)          # пятница
    mon = datetime(2026, 1, 12, 10, 0)
    assert wm(fri, mon) == 120 + 60            # пт 17-19 и пн 9-10
    assert wm(datetime(2026, 1, 5, 15, 40), datetime(2026, 1, 6, 9, 30)) == 200 + 30   # пн 15:40-19:00, вт 9-9:30
    assert wm(datetime(2026, 1, 10, 12, 0), datetime(2026, 1, 11, 12, 0)) == 0          # выходные
    assert wm(datetime(2026, 1, 5, 7, 0), datetime(2026, 1, 5, 8, 0)) == 0
    assert wm(mon, fri) == 0                    # обратный порядок


# ---------------------------------------------------------------- в потоке
import numpy as np
from datetime import timedelta

from radar.models import Release, Ticket
from radar.pipeline import RadarStream


def _stream(cfg=None, judge=None, release=True):
    """Как в test_v2: фон 2 человека/час, на 6-й день 25 человек за 25 минут, релиз за 2 часа до всплеска."""
    rng = np.random.default_rng(0)
    t0 = datetime(2026, 3, 2)
    b0 = t0 + timedelta(days=5, hours=12)
    rel = [Release("azs", "4.2.0", b0 - timedelta(hours=2), "regular")] if release else []
    s = RadarStream(rel, cfg or Config(warmup_days=3, min_users=6), dim=4, t0=t0, start_ts=t0)
    s.judge = judge
    vec = np.array([1.0, 0, 0, 0], dtype=np.float32)
    times = [t0 + timedelta(hours=h, minutes=float(rng.uniform(0, 60))) for h in range(7 * 24) for _ in range(rng.poisson(2))]
    times += [b0 + timedelta(minutes=float(rng.uniform(0, 25))) for _ in range(25)]
    times.sort()
    for i, ts in enumerate(times):
        s.push(Ticket(f"T{i}", ts, "azs", "app_chat", "android", "4.1.0", f"u{i}", f"текст {i}"), f"текст {i}", vec)
    s.finish(times[-1])
    return s


def _cfg(**kw):
    return Config(warmup_days=3, min_users=6, **kw)


def test_judge_off_by_default_alerts_are_page():
    calls = []
    s = _stream(judge=lambda p, t: calls.append(1))
    assert s.bursts and not calls
    assert all(b.tier == "page" and b.judge is None for b in s.bursts if b.alert)
    assert all(b.tier == "" for b in s.bursts if not b.alert)


def test_judge_low_share_makes_quiet_and_sees_only_window_texts():
    seen = []

    def judge(product, texts):
        seen.append((product, list(texts)))
        return {"problem": "p", "items": [False] * len(texts), "share": 0.25, "ok": True}
    s = _stream(_cfg(judge_enabled=True), judge)
    alerts = [b for b in s.bursts if b.alert]
    assert alerts and all(b.tier == "quiet" and b.judge["share"] == 0.25 for b in alerts)
    assert len(seen) == len(alerts) and seen[0][0] == "azs" and 3 <= len(seen[0][1]) <= 8
    b = alerts[0]
    in_win = {t.id: t.ts for t in s.tickets}
    idx = {s.texts[i]: i for i in range(len(s.texts))}
    assert all(b.started_at <= s.tickets[idx[x]].ts <= b.detected_at for x in seen[0][1])
    assert "проверка" in b.alert_reason


def test_judge_high_share_is_page_and_threshold_from_config():
    j = lambda p, t: {"problem": "p", "items": [True] * len(t), "share": 0.8, "ok": True}
    assert all(b.tier == "page" for b in _stream(_cfg(judge_enabled=True), j).bursts if b.alert)
    assert all(b.tier == "quiet" for b in _stream(_cfg(judge_enabled=True, judge_min_share=0.9), j).bursts if b.alert)


@pytest.mark.parametrize("judge", [None, lambda p, t: (_ for _ in ()).throw(RuntimeError("down")),
                                   lambda p, t: {"problem": "", "items": [], "share": None, "ok": False}])
def test_judge_unavailable_fails_open(judge):
    s = _stream(_cfg(judge_enabled=True), judge)
    alerts = [b for b in s.bursts if b.alert]
    assert alerts and all(b.tier == "page" for b in alerts)
    assert all(b.judge["share"] is None and b.judge["example_ids"] for b in alerts)   # примеры известны и без ответа модели
    assert all("не проверено" in b.alert_reason for b in alerts)


def test_judge_not_called_for_watch():
    calls = []
    s = _stream(_cfg(judge_enabled=True), lambda p, t: calls.append(1) or {"share": 0.0, "ok": True}, release=False)
    assert s.bursts and not any(b.alert for b in s.bursts) and not calls
    assert all(b.tier == "" for b in s.bursts)


def test_judge_does_not_change_detection():
    a = _stream()
    b = _stream(_cfg(judge_enabled=True), lambda p, t: {"problem": "", "items": [], "share": 0.0, "ok": True})
    key = lambda s: [(x.detected_at, x.z, x.alert, x.alert_reason.split(" | ")[0]) for x in s.bursts]
    assert key(a) == key(b)


def test_make_judge_wraps_check(tmp_path):
    llm = _llm({"problem": "p", "items": [True] * 8})
    j = J.make_judge(llm, tmp_path)
    assert j("azs", TEXTS)["share"] == 1.0 and len(list(tmp_path.glob("*.json"))) == 1


# ---------------------------------------------------------------- выборка «вразброс»
def test_pick_examples_spread_round_robin_by_excess():
    cands = [_cand(1, 1, "u1", "A"), _cand(2, 5, "u2", "A"), _cand(3, 9, "u3", "A"),
             _cand(4, 10, "u4", "B"), _cand(5, 11, "u5", "B"), _cand(6, 3, "u6", "C")]
    got = J.pick_examples(cands, {"A": 5.0, "B": 3.0, "C": 1.0}, n=5, mode="spread")
    assert got == [3, 5, 6, 2, 4]          # по кругу: свежее из A, B, C, потом снова A, B


def test_pick_examples_spread_skips_repeated_users_and_is_deterministic():
    cands = [_cand(i, i, f"u{i % 3}", "AB"[i % 2]) for i in range(1, 9)]
    a = J.pick_examples(cands, {"A": 1.0, "B": 1.0}, n=8, mode="spread")
    b = J.pick_examples(list(reversed(cands)), {"A": 1.0, "B": 1.0}, n=8, mode="spread")
    assert a == b and len(a) == 3


def test_config_sampling_mode_default_is_top():
    assert Config().judge_sampling == "top"
