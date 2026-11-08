"""eval/REPORT-scale-v2.md из shared/metrics-scale-v2.json (и shared/experiment-threshold-v2.json, если он есть).
Все цифры берутся из JSON; руками в отчёте правится только этот шаблон. Запуск:  python eval/make_report_v2.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "shared" / "metrics-scale-v2.json"
EXP = ROOT / "shared" / "experiment-threshold-v2.json"
DST = ROOT / "mvp" / "eval" / "REPORT-scale-v2.md"
METH = ("poisson", "ewma")


def c(x, p=2):
    return "-" if x is None else f"{x:.{p}f}".replace(".", ",")


def render(m: dict, exp) -> str:
    vols = sorted(int(v) for v in m["by_volume"])
    D = lambda v, meth, k: m["by_volume"][str(v)]["detector"][meth][k]
    vmax, legacy = vols[-1], m["legacy_source"]["volumes"]
    L = ["# Масштаб потока v2: от 800 до %d обращений в день, те же пороги" % vmax, "",
         f"Seeds {m['seeds']}, шум low/mid/high, 42 события на точку. Пороги из `eval/tuned_params.json` без перенастройки, "
         "код детектора не менялся. Сырьё: `shared/metrics-scale-v2.json`, таблица собрана `eval/make_report_v2.py`.", "",
         f"Источник цифр: объёмы {', '.join(map(str, legacy))} скопированы из `{m['legacy_source']['file']}` "
         f"(замер {m['legacy_source']['generated_at']}) без пересчёта. Объёмы {', '.join(str(v) for v in vols if v not in legacy)} "
         "посчитаны заново на тех же seeds в этом заходе (seeds 301-305 пороги не подбирались).", "",
         "Колонка «≤60 мин, % пойманных» это доля пойманных событий, у которых карточка вышла за 60 минут от начала всплеска. "
         "Доля от всех событий (включая пропущенные) лежит в JSON как `recall_within_60`.", "",
         "| обращений/день | метод | precision | recall | ложных тревог/нед | медиана до карточки, мин | p90, мин | ≤60 мин, % пойманных |",
         "|---|---|---|---|---|---|---|---|"]
    for v in vols:
        for meth in METH:
            L.append(f"| {v} | {meth} | {c(D(v, meth, 'precision'))} | {c(D(v, meth, 'recall'))} | {c(D(v, meth, 'false_alarms_per_week'))} | "
                     f"{c(D(v, meth, 'ttc_median_min'), 0)} | {c(D(v, meth, 'ttc_p90_min'), 0)} | {c(D(v, meth, 'caught_within_60_share') * 100, 0)} |")
    L += ["", "Время счёта: " + ", ".join(f"{v}: {m['timing'][str(v)]} с" for v in vols if str(v) in m["timing"])
          + " (15 наборов, 4 процесса, оба детектора; объёмы 6000 и 10000 считались при параллельной нагрузке другой сессии, "
          "время несравнимо с прежним).", ""]
    for v, why in m.get("aborted", {}).items():
        L += [f"Объём {v} не посчитан: {why}. 30000 не запускали: по памяти и времени он тяжелее прерванного 20000.", ""]
    fa = lambda meth: " → ".join(c(D(v, meth, "false_alarms_per_week")) for v in vols)
    pe = lambda meth: " → ".join(c(D(v, meth, "precision")) for v in vols)
    first_low = next((v for v in vols if D(v, "ewma", "precision") < 0.5 and D(v, "poisson", "precision") < 0.5), None)
    rmax = max(vols, key=lambda v: D(v, "ewma", "recall"))
    L += ["## Выводы по цифрам", "",
          f"- Ложные тревоги в неделю при {', '.join(map(str, vols))} в день. ewma: {fa('ewma')}. poisson: {fa('poisson')}. "
          f"Рост резкий до 3000 ({c(D(800, 'ewma', 'false_alarms_per_week'))} → {c(D(1500, 'ewma', 'false_alarms_per_week'))} → "
          f"{c(D(3000, 'ewma', 'false_alarms_per_week'))} у ewma, как и в прежнем замере), дальше медленнее: "
          f"{c(D(6000, 'ewma', 'false_alarms_per_week'))} при 6000 и {c(D(vmax, 'ewma', 'false_alarms_per_week'))} при {vmax}.",
          f"- Precision, ewma: {pe('ewma')}; poisson: {pe('poisson')}. Ниже 0,5 у обоих методов она становится с {first_low} в день: "
          f"ложных карточек больше, чем настоящих. При {vmax} в день ложны около {c((1 - D(vmax, 'ewma', 'precision')) * 100, 0)}% карточек (precision {c(D(vmax, 'ewma', 'precision'))} у ewma). "
          f"Это моя граница «неприемлемо» для ручного разбора, "
          "порога приёмки в проекте нет.",
          f"- Время до карточки падает: медиана {c(D(800, 'ewma', 'ttc_median_min'), 0)} мин при 800, {c(D(3000, 'ewma', 'ttc_median_min'), 0)} при 3000, "
          f"{c(D(vmax, 'ewma', 'ttc_median_min'), 0)} при {vmax}; доля пойманных в пределах 60 минут "
          f"{c(D(800, 'ewma', 'caught_within_60_share') * 100, 0)}% → {c(D(3000, 'ewma', 'caught_within_60_share') * 100, 0)}% → "
          f"{c(D(vmax, 'ewma', 'caught_within_60_share') * 100, 0)}%.",
          f"- Recall не растёт монотонно: максимум у ewma при {rmax} ({c(D(rmax, 'ewma', 'recall'))}), при {vmax} уже "
          f"{c(D(vmax, 'ewma', 'recall'))} (poisson {c(D(vmax, 'poisson', 'recall'))}). Причину не разбирали. Возможное объяснение, не проверенное: "
          "с ростом числа тем и окон проверок больше, а порог z один, и часть настоящих событий проигрывает фону в выборе карточки.",
          "- Объяснение роста ложных тревог: порог z выбран на 800 в день и больше не менялся. В генераторе (`radar/gen/generator.py`, "
          "`scale = объём / 800`) пик события и фоновый шум растут пропорционально потоку, и число проверок «тема × окно» с редкими случайными "
          "превышениями растёт вместе с потоком. Измерено только, что ложных тревог больше; что именно число проверок их даёт, не изолировано отдельным опытом.",
          "- Метод poisson и ewma различаются слабо: на разных объёмах впереди оказывается то один, то другой, на 42 событиях это внутри шума. "
          "Разница в 1-2 события (3-5 п.п. recall) значимой не считается.", ""]
    L += ["## Что делать на 100 тыс. в день (гипотезы, НЕ ИЗМЕРЕНО)", "",
          "Ни одно утверждение этого раздела не проверено замером на 100 тыс. в день. Это планы, что пробовать.", "",
          "1. Пороги по продуктам. В `Config.product_overrides` уже есть `z_threshold` и `min_users` на продукт. Крупным продуктам порог выше, малым ниже. "
          "Не проверялось.",
          "2. Поправка на множественные проверки. Если за сутки проверяется N пар «тема × окно», то при одном пороге случайных превышений около N·p. "
          "Порог можно привязать к N (например, по Бонферрони p = α/N) вместо одного числа. Не проверялось.",
          "3. Порог z растёт с числом кластеров. Частный случай пункта 2: пока кластеров мало, порог 6, при росте потока и числа тем он поднимается по правилу, "
          "подобранному на отдельных seeds. Не проверялось в виде правила; ниже только грубый опыт с общим повышением порога.",
          "4. Требовать два последовательных окна выше порога (или два метода) перед карточкой. Цена: время до карточки. Не проверялось.", ""]
    L += ["## Эксперимент, не часть замка протокола", ""]
    if exp is None:
        L += ["Эксперимент не выполнен или ещё не записан.", ""]
    else:
        zs = exp["z_values"]
        L += [f"Что сделано: на свободных seeds {exp['seeds']} (не из подбора, не из финальной оценки, не из замера масштаба) "
              f"при {exp['volume']} обращениях в день менялся только `z_threshold` в конфиге ({', '.join(c(z, 0) for z in zs)}). "
              "Остальные параметры из `tuned_params.json`, код детектора не менялся. Файл: `shared/experiment-threshold-v2.json`, "
              "скрипт `eval/threshold_experiment.py`. Это опыт на малой выборке и не часть замка протокола: порог здесь не выбирается.", "",
              "| метод | z_threshold | precision | recall | ложных тревог/нед | медиана до карточки, мин | p90, мин |", "|---|---|---|---|---|---|---|"]
        for meth in METH:
            for z in zs:
                d = exp["by_method"][meth][str(float(z))]
                L.append(f"| {meth} | {c(z, 0)} | {c(d['precision'])} | {c(d['recall'])} | {c(d['false_alarms_per_week'])} | "
                         f"{c(d['ttc_median_min'], 0)} | {c(d['ttc_p90_min'], 0)} |")
        L += [""]
        for meth in METH:
            b, t = exp["by_method"][meth][str(float(zs[0]))], exp["by_method"][meth][str(float(zs[-1]))]
            L.append(f"- {meth}: z {c(zs[0], 0)} → {c(zs[-1], 0)}: ложных тревог/нед {c(b['false_alarms_per_week'])} → {c(t['false_alarms_per_week'])}, "
                     f"recall {c(b['recall'])} → {c(t['recall'])}, медиана до карточки {c(b['ttc_median_min'], 0)} → {c(t['ttc_median_min'], 0)} мин.")
        L += ["", "Для сверки: при z=6 и том же объёме на seeds 401-405 ложных тревог/нед "
              f"{c(exp['by_method']['poisson'][str(float(zs[0]))]['false_alarms_per_week'])} (poisson) и "
              f"{c(exp['by_method']['ewma'][str(float(zs[0]))]['false_alarms_per_week'])} (ewma), "
              f"на seeds 301-305 было {c(D(3000, 'poisson', 'false_alarms_per_week'))} и {c(D(3000, 'ewma', 'false_alarms_per_week'))}: картина воспроизводится. "
              "Повышение порога снижает ложные тревоги заметно у poisson и слабо у ewma, recall и время до карточки почти не меняются. "
              "Выборка 42 события, поэтому это намёк, а не доказательство; на 6000 и 10000 в день опыт не удалось запустить (машина перегружена).", ""]
    L += ["## Честные границы", "",
          "- Данные синтетические: генератор `radar/gen/generator.py`, реальных обращений нет, на реальных потоках цифры могут быть иными в любую сторону.",
          "- Допущение пропорционального роста: при увеличении потока пик события и число людей в нём растут пропорционально потоку. Если реальный поток вырос без роста аварий, "
          "выигрыша во времени до карточки не будет.",
          "- Фоновый шум тоже масштабируется по формуле генератора, один шумовой профиль (low/mid/high в рамках одного генератора).",
          "- 42 события на точку (по 14 на уровень шума): разница в несколько событий находится внутри шума.",
          f"- Максимум замера {vmax} в день; 100 тыс. это экстраполяция ещё на порядок вверх. Тренд называть можно, цифру нет.",
          "- Пороги подобраны на 800 в день (seeds 1-5) и не менялись; на больших потоках они заведомо не оптимальны.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    m = json.loads(MET.read_text(encoding="utf-8"))
    e = json.loads(EXP.read_text(encoding="utf-8")) if EXP.exists() else None
    DST.write_text(render(m, e), encoding="utf-8")
    print("записан", DST)
