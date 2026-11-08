# Публичный интерфейс ядра Радара

Агент, бот и дашборд следующей волны опираются только на этот файл. Внутренности модулей (`detect.py`, `cluster.py` и так далее) можно менять, пока этот интерфейс не ломается.

Принципы, которые интерфейс гарантирует:

- Тексты, которые отдаёт ядро, только замаскированные (`<PHONE_1>`, `<PERSON_1>`, ...). Оригиналы лежат в отдельном словаре `stream.reverse_maps` и в агента не передаются.
- Числа (всплеск, цена темы, доля версии, проверка фикса) считает код. Агент их переносит и ссылается на `id` обращений из `allowed_ids`; утверждение без ссылки валидатор отбрасывает.
- Ядро ничего не закрывает и ничего никому не отправляет.
- Результат детерминирован: те же входные данные и Config дают те же всплески.

## Импорт

```python
from radar.gen import generate, load           # синтетика: в память или с диска (data/<seed>/)
from radar.config import Config                # все параметры
from radar.pipeline import run, run_full, iter_run, card_input, group_incidents
from radar.models import Ticket, Release, Burst, SuspectedCardInput
```

Запускать из папки `mvp/` (или с `PYTHONPATH=.`), интерпретатор `.venv/bin/python`.

## Вход

`tickets: list[Ticket]` (порядок не важен, ядро сортирует по времени), `releases: list[Release]`.

| Ticket | тип | смысл |
|---|---|---|
| `id` | str | `T000123`, уникален |
| `ts` | datetime | время обращения, без часового пояса |
| `product` | str | `azs`, `bonus`, `b2b`, `portal`, `partners` |
| `channel` | str | `app_chat`, `email`, `hotline`, `store_review`, `web_form` |
| `platform` | str | `android`, `ios`, `web`, `unknown` |
| `app_version` | str | `4.12.0` или пустая строка, если версии нет |
| `user_id` | str | один человек, по нему идёт дедупликация |
| `text` | str | исходный текст, может содержать ПДн; ядро маскирует его само до любой модели |

| Release | тип | смысл |
|---|---|---|
| `product` | str | продукт |
| `version` | str | `4.12.0`, у веб-продуктов `r14` |
| `ts` | datetime | момент выкатки |
| `kind` | str | `regular` или `fix` (патч) |
| `changelog` | str | короткая запись |

Файлы синтетики (`radar.gen.load("data/1")` возвращает `tickets, releases, ground_truth, labels`): `tickets.jsonl` (поля Ticket, `ts` в ISO), `releases.json` (поля Release), `ground_truth.json` (что вставил генератор), `labels.json` (метки для оценки, агенту не показывать).

## Главная функция

Единственная точка входа для следующей волны: `radar.pipeline.run`. Всё остальное (`run_full`, `iter_run`, `card_input`, `group_incidents`) её обёртки и помощники. Контракт v2 (сигнатура не менялась с v1, добавились поля результата): вход `tickets`, `releases`; выход `list[Burst]` в порядке `detected_at`; тревога дежурному = `b.alert is True`; одна проблема может дать несколько всплесков, их склеивает `group_incidents`.

```python
run(tickets, releases, config=None, until=None) -> list[Burst]
```

Прогоняет поток по времени: обращения подаются строго по порядку, раз в `config.step_min` минут (по времени обращений, а не по часам компьютера) детектор делает шаг. Первые `config.warmup_days` дней идут на прогрев, тревог нет. `until` останавливает поток на этом моменте (перемотка).

Пример:

```python
from radar.gen import generate
from radar.pipeline import run

ds = generate(seed=1, noise="mid", daily_volume=800)
bursts = run(ds.tickets, ds.releases)
b = bursts[0]
print(b.product, b.detected_at, b.total_unique_users, b.weekly_cost_rub, b.suspected_release)
```

### Burst

| поле | тип | смысл |
|---|---|---|
| `id` | str | `B0001` |
| `product`, `cluster_id` | str | продукт и кластер (`azs:17`) |
| `method` | str | `poisson` или `ewma` |
| `window_min` | int | окно, в котором сработало (30, 60 или 120) |
| `started_at` | datetime | оценка начала: первое обращение в окне срабатывания |
| `detected_at` | datetime | момент срабатывания (потоковое время). Время до карточки = `detected_at` минус начало проблемы |
| `last_seen` | datetime | последнее обращение всплеска на сейчас |
| `status` | str | `open` или `closed` |
| `z` | float | число сигм: насколько редко видеть столько людей при ожидаемом |
| `count_in_window` | int | разных людей в окне срабатывания |
| `baseline_in_window` | float | сколько людей ждали в этом окне |
| `total_unique_users` | int | разных людей за весь всплеск (обновляется, пока он открыт) |
| `cluster_size` | int | обращений в кластере всего |
| `platforms`, `versions` | dict | `{"android": 31}`, `{"4.12.0": 28}` по обращениям всплеска |
| `example_ids`, `examples` | list | до `n_examples` обращений, ближайших к центру всплеска. `examples = [{"id", "text"}]`, текст замаскирован |
| `suspected_release` | SuspectedRelease или None | подозреваемый релиз (корреляция, не доказательство) |
| `weekly_cost_rub` | float | цена темы за последние 7 дней: уникальные люди кластера * t/60 * C1, числа из `shared/numbers.json`, сценарий base |
| `fix_check` | FixCheck или None | заполняется позже, когда прошло `verify_lag_hours + verify_days` после релиза-фикса |
| `cluster_ids` | list[str] | v2: все кластеры всплеска (осколки одной темы собирает скан подмножеств); `cluster_id` это первый из них |
| `signal` | str | v2: `topic` (число людей темы) или `version` (люди темы на версии последнего релиза, мобильные) |
| `alert` | bool | v2: `True` будить дежурного; `False` только показать на дашборде (наблюдение или продолжение прошлой тревоги) |
| `alert_reason` | str | v2: почему: «через 1.5 ч после релиза 4.12.0, сигнал version», «наблюдение: нет релиза...», «продолжение B0012» |
| `z_version`, `version_count`, `version_baseline` | float/int | v2: z сигнала версии, людей на новой версии в окне и сколько их ждали (`None`/0, если сигнала версии нет) |

**SuspectedRelease**: `version`, `ts`, `hours_since_release`, `born_after_release` (кластер родился после релиза), `new_version_share` (доля обращений всплеска на версии релиза, `None` если версий нет), `product_new_version_share` (та же версия у всего продукта в то же время), `confidence` (`high`/`medium`/`low`).

**FixCheck**: `fix_version`, `fix_ts`, `verdict` (`confirmed`/`not_confirmed`/`insufficient_data`), `drop_percent`, `ratio`, `ratio_ci_low`, `ratio_ci_high`, `p_value`, `n_before`, `n_after`, `used_control`, `checked_at`; v2: `mode` (`baseline`), `residual_share` (доля всплеска, оставшаяся после фикса: 0 = тема вернулась к уровню до релиза), `residual_ci_high`, `n_baseline`, `clusters` (по каким кластерам считали: все тревоги инцидента). В режиме `baseline` поле `p_value` это p-value «всплеск был выше нормы».

Тревоги дежурному: `[b for b in bursts if b.alert]`. Всплески с `alert=False` в ленту тревог не идут. Одна тема всё ещё режется на несколько кластеров, и тревог по одной проблеме может прийти несколько; для карточек используйте `group_incidents([b for b in bursts if b.alert])`: группы одного продукта с паузой не больше 2 часов.

## Потоковый режим для демо

```python
from datetime import datetime
from radar.pipeline import iter_run

for ts, kind, burst in iter_run(ds.tickets, ds.releases, skip_until=datetime(2026, 3, 12, 0, 0)):
    print(ts, kind, burst.id)       # kind: "opened", "closed", "fix_checked"
```

`iter_run` подаёт обращения по времени и отдаёт события в момент, когда они случились. `burst` это копия состояния на этот момент (число людей, примеры, цена такие, какими они были тогда). `skip_until` перематывает: всё раньше этого момента обрабатывается молча (состояние копится), события отдаются после. Паузы под скорость демо делает вызывающий код (`time.sleep` между событиями по разнице `ts`).

## Состояние для дашборда и инструментов агента

```python
stream = run_full(tickets, releases, config)       # RadarStream, те же всплески в stream.bursts
```

| что | зачем |
|---|---|
| `stream.bursts` | все всплески |
| `stream.tickets`, `stream.texts` | обращения и их замаскированные тексты (одинаковый порядок) |
| `stream.reverse_maps` | `{ticket_id: {"<PHONE_1>": "+7 ..."}}`. Только для человека, который нажал «показать оригинал». В агента не передавать |
| `stream.cluster_ticket_indexes(burst)` | индексы обращений кластера всплеска |
| `stream.vote_times[(product, cluster_number)]` | времена засчитанных голосов, ряд для графика |
| `stream.fix_candidate(burst)` | релиз, который проверяется как фикс |
| `stream.verify_cluster(key, fix_ts, now)` | проверка фикса v1 по одному кластеру (отношение «после/до») |
| `stream.incident_keys(burst)`, `stream.verify_keys(keys, fix_ts, now, burst_start, release_ts)` | v2: кластеры инцидента и проверка «вернулась ли тема к норме до релиза» (инструмент `verify_fix` агента) |
| `stream.burst_keys(burst)` | v2: ключи `(product, cluster)` всех кластеров всплеска |
| `stream.cfg.product_overrides` | порог продукта: `{"azs": {"z_threshold": 6.0, "min_users": 8}}`. Кнопка «шум» меняет это поле, действует со следующего шага |

## Вход для агента

```python
from radar.pipeline import card_input
card = card_input(stream, burst)      # SuspectedCardInput
```

`SuspectedCardInput`: `burst_id`, `product`, `product_name`, `detected_at`, `started_at`, `unique_users`, `window_min`, `z`, `platforms`, `versions`, `suspected_release`, `weekly_cost_rub`, `evidence` (`[{"id", "text"}]`, замаскированные), `allowed_ids` (на эти id карточка вправе ссылаться), `min_users_for_status` (меньше этого людей: `status = "наблюдаем"`). В промпт текст обращений идёт только внутри `<tickets>...</tickets>` как данные. В генераторе есть обращения с prompt injection (`ground_truth.json`, тип `prompt_injection`, `ticket_ids`) для сцены демо и проверки устойчивости.

## Ожидаемый формат карточки (решение SPEC от 2026-10-13, для следующей волны)

Ядро карточку не пишет, но задаёт формат, который проверит валидатор. Карточка содержит массив `claims`: каждое утверждение отдельным элементом со списком id обращений (и необязательным полем `release` с версией релиза, если утверждение про него), на которые оно опирается.

```json
{
  "title": "...", "product": "azs", "status": "новая|известная|наблюдаем", "severity": "low|medium|high",
  "injection_suspected": false,
  "claims": [
    {"text": "Всплеск начался через 40 минут после релиза 4.12.0", "ids": ["T012345", "T012351"]},
    {"text": "Обращения идут с Android", "ids": ["T012345"], "release": null}
  ]
}
```

Правила валидатора (его пишет следующая волна, ядро даёт данные): утверждение без `ids` или с id вне `SuspectedCardInput.allowed_ids` отбрасывается; числа (`unique_users`, `weekly_cost_rub`, `z`, доли версий) берутся из `SuspectedCardInput`, а не из текста модели; связь с релизом называется «подозреваемый релиз».

## Config (главное)

`Config()` проверяет типы (pydantic); границы значений не проверяются. Ключевые поля; полный список и комментарии в `radar/config.py`.

| поле | по умолчанию | смысл |
|---|---|---|
| `embedder` | `tfidf` | или `nomic` (нужен Ollama; если не отвечает, молча tfidf, имя в `stream.embedder_name`) |
| `method` | `ewma` | или `poisson`; `ewma` лучший F1 на tune-seeds 1-5 |
| `windows_min` | `[30, 60, 120]` | окна детектора |
| `step_min` | 5 | шаг детектора в минутах потокового времени |
| `z_threshold`, `z_version`, `min_users` | 6.0, 5.5, 3 (подобраны на seeds 1-5, `eval/tuned_params.json`) | порог числа сигм по теме, порог сигнала версии, минимум разных людей |
| `alert_policy` | `release` | тревога только в окне `release_max_hours` (48 ч) после релиза; `all` как в v1 |
| `subset_scan`, `version_signal` | `True`, `True` | скан подмножеств кластеров и сигнал версии (выключатели для абляций) |
| `verify_mode` | `baseline` | проверка фикса «вернулась ли тема к норме до релиза»; `ratio` как в v1 |
| `dedup_min` | 60 | один `user_id` в кластере за это время = один голос |
| `sim_threshold` | 0.5 | порог косинуса онлайн-кластеризации |
| `warmup_days` | 3 | прогрев без тревог |
| `verify_days`, `verify_lag_hours` | 3, 12 | окно проверки фикса и пауза на раскатку версии |

## Скорость и ограничения

- 28 дней по 800 обращений в день (около 27 тысяч): около 10 секунд маскирование и эмбеддинг плюс около 14 секунд детектор на занятой машине M1 (`shared/metrics.json`, поле `timing`); при 3000 в день детектор около 56 секунд. Время до карточки от потока зависит: `eval/REPORT-scale.md`.
- Для демо нужен прогрев: первые 3 дня данных тревог нет.
- Эмбеддинг считается пакетом заранее на всех обращениях (словарь TF-IDF обучен только на прогревочных днях), а детектор потом идёт строго по времени. Для демо это неотличимо от настоящего потока.
- `fix_check.verdict == confirmed` (v2, возврат к норме) проверен на синтетике, цифры в `eval/REPORT.md`. Всё равно подавать как «подтверждено по данным, проверьте», решает человек.
- Тревога (`alert=True`) только в 48 часах после релиза продукта. Регрессия без релиза (бэкенд, партнёр) приходит как «наблюдение» с порогом `z_watch`.
- Привязка к релизу называется «подозреваемый релиз». Писать «виновник» нельзя.
- Метрики качества: `../shared/metrics.json` и `eval/REPORT.md`. Это метрики на синтетике, а не на реальном потоке.
