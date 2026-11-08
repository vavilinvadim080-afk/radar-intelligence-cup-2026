"""Все настраиваемые параметры в одном месте. Pydantic проверяет типы полей; диапазоны значений (границы) намеренно не проверяются: протокол оценки заморожен."""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field


def _env_mask_ner():
    """Умолчание mask_ner из окружения: RADAR_MASK_NER=auto|1|true включает NER в боевом запуске без правки кода.
    Не задано или непонятно: False (так замерены метрики eval; "auto" меняет ~2% текстов синтетики, то есть вход эмбеддера)."""
    import os
    v = os.environ.get("RADAR_MASK_NER", "").strip().lower()
    return "auto" if v == "auto" else True if v in ("1", "true", "on") else False


class Config(BaseModel):
    # --- эмбеддинг ---
    embedder: str = "tfidf"            # tfidf | nomic (Ollama) | local-e5 | local-qwen
    svd_dim: int = 128                 # размерность после TruncatedSVD
    warmup_days: int = 3               # на каких первых днях обучаем TF-IDF и не даём тревог

    # --- маскирование ---
    mask_ner: Union[bool, Literal["auto"]] = Field(default_factory=_env_mask_ner)  # поверх регулярок имена и адреса через Natasha (radar/mask_ner.py).
                                       # False: никогда (метрики eval замерены так, пакет natasha не нужен); True: всегда;
                                       # "auto": КАСКАД, NER только для текстов, где после регулярок остались подозрительные места
                                       # (radar.mask.needs_ner). Без natasha "auto" молча-с-одним-предупреждением работает как False.
                                       # По умолчанию False, потому что протокол оценки заморожен; для боя ставить "auto"
    # пороги правила каскада (radar.mask.needs_ner), все проверяются ДО NER и дёшевы (регулярки по остатку текста)
    cascade_min_caps: int = 1          # слов с заглавной буквы не в начале предложения и не из словаря терминов
    cascade_min_digits: int = 5        # цифр подряд, не замаскированных
    cascade_min_alnum: int = 5         # длина латинско-цифровой смеси (пароль, номер) или латинского слова не из словаря
    cascade_triggers: list[str] = Field(default_factory=lambda: [
        r"меня\s+зовут", r"зовут\s+меня", r"мо[её]\s+(?:имя|фио)", r"с\s+уважением", r"\bфио\b",
        r"\bул\.", r"\bулиц\w*", r"\bпроспект\w*", r"\bпр-т\b", r"\bпер\.", r"\bг\.", r"\bд\.\s*\d", r"\bкв\.",
        r"\[(?:at|dot|собака|точка)\]"])   # сильные триггеры: срабатывают сами
    cascade_weak_triggers: list[str] = Field(default_factory=lambda: [r"\bэто\b", r"\bя\b"])   # слабые: только если следом слово с заглавной

    mask_ctx_names: bool = False       # правило «меня зовут X Y»: ловит имена с опечатками и в нижнем регистре, но меняет текст для
                                       # эмбеддинга на ~0.3% обращений синтетики и сдвигает состав кластеров (seed 202), поэтому выключено

    # --- приём потока (RadarStream.push, radar/live.py) ---
    late_tolerance_min: int = 30       # обращение старше последнего принятого не больше чем на столько минут принимается
                                       # (со временем прихода), дальше отбрасывается с записью в журнал. В замере не срабатывает:
                                       # eval подаёт обращения по порядку времени

    # --- защита от спама и накрутки (radar/antispam.py, STATUS-antispam.md) ---
    user_cap_max: int = 20             # не больше стольких обращений одного user_id принимается за user_cap_window_min минут; лишние не идут
                                       # в детектор и считаются (RadarStream.n_capped). 0: лимит выключен. В eval на dev (seeds 201-210)
                                       # у реального пользователя не больше 8 обращений за сутки, лимит ничего не отбрасывает
    user_cap_window_min: int = 1440
    brigade_check: bool = True         # тревога с признаком накрутки не будит (tier "quiet", пометка suspicion), автоответ не включается
    brigade_min_users: int = 8         # меньше людей с содержательным текстом: мало данных, не обвиняем
    brigade_min_group: int = 5         # и в самой большой группе почти одинаковых текстов не меньше стольких людей
    brigade_min_tokens: int = 4        # короткий текст («не работает») доказательством не считается
    brigade_dup_sim: float = 0.8       # сходство Жаккара по множествам слов, с которого тексты считаются «почти одинаковыми»
    brigade_dup_share: float = 0.5     # доля группы среди людей окна: у настоящих всплесков dev не выше 0.27 (STATUS-antispam.md)
    brigade_timer_share: float = 0.35  # более низкая доля, но вместе с равномерными паузами
    brigade_timer_cv: float = 0.4      # gap_cv ниже этого = «как по таймеру» (у людей около 1)

    # --- кластеризация ---
    sim_threshold: float = 0.5         # косинус до центра, выше которого берём существующий кластер
    use_hdbscan: bool = False          # суточная пересборка HDBSCAN и склейка (v1: True). В v2 выключена: 85% времени
                                       # прогона, а склейка по центрам не отделяет осколки темы от чужих тем
                                       # (см. eval/DIAGNOSIS.md); осколки собирает скан подмножеств в детекторе
    hdbscan_min_cluster_size: int = 25
    hdbscan_merge_share: float = 0.6   # доля кластера в одной группе HDBSCAN, чтобы склеить

    # --- детектор ---
    method: str = "ewma"               # poisson | ewma (v2: ewma, лучший F1 на tune-seeds 1-5, eval/tuned_params.json)
    windows_min: list[int] = Field(default_factory=lambda: [30, 60, 120])   # v2: подобрано на tune-seeds (v1: 30, 60)
    step_min: int = 5                  # как часто проверяем (минуты потокового времени), v1: 10
    z_threshold: float = 6.0           # v2: подобрано на tune-seeds 1-5 (v1: 4.5; порог выше, потому что скан подмножеств
                                       # берёт максимум по многим наборам кластеров)
    min_users: int = 3                 # минимум разных людей в окне
    clear_ratio: float = 0.5           # всплеск закрыт, когда z упал ниже z_threshold*clear_ratio...
    clear_hold_min: int = 90           # ...и так держится столько минут
    baseline_days: int = 7             # глубина истории базовой линии
    guard_hours: int = 3               # последние часы в базовую линию не берём (там может быть начало всплеска)
    lambda_floor: float = 1.0          # нижняя граница ожидания в окне, чтобы z не взрывался на новых кластерах
    ewma_halflife_h: float = 24.0      # у EWMA память короче, чем у «семидневного среднего»
    subset_scan: bool = True           # v2: сумма осколков одной темы (скан подмножеств кластеров продукта, LTSS)
    scan_min_member: int = 2           # кластер идёт в сумму, только если в нём не меньше стольких людей в окне
    scan_max_k: int = 8                # не больше стольких кластеров в одном всплеске
    scan_member_z: float = 1.5         # ...и только если собственный z кластера не ниже этого

    # --- v2: сигнал новой версии (мобильные продукты) ---
    version_signal: bool = True        # считать людей темы на версии последнего релиза против ожидания по доле версии
    z_version: float = 5.5             # порог сигнала версии (подобран на tune-seeds 1-5)
    min_users_version: int = 3         # минимум людей на новой версии в окне
    version_floor: float = 0.1         # нижняя граница ожидания сигнала версии
    adoption_hours: int = 3            # за сколько часов считаем долю новой версии в потоке продукта
    version_q_max: float = 0.3         # сигнал версии работает, пока новая версия у меньшинства (контраст >= 1/q);
                                       # дальше её доля сама по себе высокая и сигнал почти не отличает тему от фона

    # --- v2: кого будить (тревога) и кого только показывать (наблюдение) ---
    alert_policy: str = "release"      # release: тревога только в окне release_max_hours после релиза продукта;
                                       # all: тревога на любой всплеск (поведение v1)
    z_watch: float = 5.5               # вне окна релиза всплеск открывается только с таким z и идёт в «наблюдение»
    realert_days: int = 7              # всплеск на кластере, по которому уже была тревога за эти дни, это продолжение
    release_incident_hours: float = 0.0  # >0: новая тревога того же продукта по тому же релизу в эти часы после первой
                                       # тоже продолжение (одна карточка на релиз, кластеры дописываются)
    dedup_min: int = 60                # один user_id в этом окне в кластере = один голос
    product_overrides: dict = Field(default_factory=dict)  # {"azs": {"z_threshold": 8, "min_users": 8}}

    # --- релиз ---
    release_max_hours: float = 48.0    # дальше этого после релиза связь не ищем
    new_version_share_min: float = 0.6

    # --- проверка фикса ---
    verify_days: int = 3
    verify_lag_hours: int = 12         # после выкатки фикса версия расходится не мгновенно
    verify_alpha: float = 0.01
    verify_max_ratio: float = 0.8      # «подтверждено» только если верхняя граница отношения ниже этого
    verify_min_events: int = 10        # меньше событий до фикса: insufficient_data
    verify_use_control: bool = True
    verify_mode: str = "baseline"      # baseline (v2): вернулась ли тема к уровню до релиза; ratio (v1): упала ли на 20%
    verify_max_residual: float = 0.35  # baseline: «подтверждено», если верхняя граница остатка всплеска ниже этого
    verify_baseline_days: int = 7      # baseline: сколько дней до релиза берём как норму темы

    # --- v3: вторая проверка тревоги и рабочие часы ---
    judge_enabled: bool = False        # LLM читает примеры тревоги (radar/judge.py) и делит её на page/quiet. По умолчанию выключено:
                                       # старые замеры не меняются; боевой режим включает и передаёт judge= в run_full
    judge_min_share: float = 0.75      # доля обращений, упоминающих главную проблему, с которой тревога будит (page)
    judge_examples: int = 8            # сколько обращений показываем модели
    judge_sampling: Literal["top", "spread"] = "top"   # top: кластеры по очереди, самый избыточный целиком; spread: по кругу по одному из каждого
    business_hours_products: list[str] = Field(default_factory=lambda: ["b2b", "portal", "partners"])   # веб-продукты рабочего времени
    business_hours: tuple[int, int] = (9, 19)   # пн-пт, с какого по какой час (для оценки в рабочих минутах)

    # --- карточка ---
    n_examples: int = 8

    def for_product(self, product: str) -> tuple[float, int]:
        """Порог z и минимум людей для продукта (с учётом переопределений)."""
        o = self.product_overrides.get(product, {})
        return o.get("z_threshold", self.z_threshold), o.get("min_users", self.min_users)
