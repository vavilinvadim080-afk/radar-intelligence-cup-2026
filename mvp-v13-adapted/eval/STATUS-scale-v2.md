# STATUS: масштаб v2 (2026-10-20)

Итог: кривая продолжена до 10000 обращений в день (seeds 301-305, пороги из tuned_params.json без изменений).

- Объёмы: 800, 1500, 3000 скопированы из shared/metrics-scale.json без пересчёта (`legacy_source` в JSON). Посчитаны заново: 6000 (1265 с) и 10000 (2109 с), 4 процесса, `RADAR_NO_CACHE=1`.
- 20000 прерван: своп 9,8 из 10 ГБ, воркеры за 84 мин стенного времени набрали по ~13 мин CPU. 30000 не запускали. Машина была перегружена другими сессиями (load average до 18 на 8 ядрах).
- Эксперимент с z_threshold (6, 8, 10), seeds 401-405, 3000 в день: готов, shared/experiment-threshold-v2.json. Попытка на 6000 встала и убита. Не часть замка протокола.
- Файлы: shared/metrics-scale-v2.json, shared/curve-for-deck.json (eval/make_curve_for_deck.py), shared/experiment-threshold-v2.json, mvp/eval/REPORT-scale-v2.md (eval/make_report_v2.py; переименовать в REPORT-scale-v2.md), eval/scale_check.py (--out, --report, --copy-legacy, --legacy), eval/datacache.py (RADAR_NO_CACHE), eval/threshold_experiment.py, tests/test_scale_v2.py (8 тестов, написаны до кода, красные подтверждены).
- Проверка: pytest -q tests: 145 passed (65 прежних плюс тесты других сессий и 8 новых). sha256 до и после совпали: metrics-scale.json 672c0c8c..., REPORT-scale.md 2557d512....
- Не вышло: 20000 и 30000; эксперимент на 6000 и 10000.
- Замечание: `pkill -f spawn_main` при остановке зависшего эксперимента мог задеть чужие процессы multiprocessing на машине.
