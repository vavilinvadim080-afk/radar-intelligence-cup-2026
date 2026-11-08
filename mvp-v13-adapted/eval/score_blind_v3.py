"""Метрики по слепым вердиктам сильной модели на тревогах финального прогона 401-410.

Вход: файл вердиктов {id: {"problem": "...", "items": [true/false, ...]}} для id из eval/cache/blind-401-410/alerts_blind.json
(ключ с метками лежит отдельно: key.json; смотреть его до вердиктов нельзя). Доля = среднее items; id без валидного вердикта
считается «не проверено» и будит. Печатает те же метрики, что eval/run_v3.py, для порогов 0.625/0.75/0.875 и для замороженного,
рядом даёт ответы gemma на тех же тревогах. Запись: shared/metrics-v3-strong.json.
Запуск: PYTHONPATH=. .venv/bin/python eval/score_blind_v3.py verdicts.json [папка слепого файла]
Без второго аргумента берётся основной вариант (eval/cache/blind-401-410), для второго указать его подпапку."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import run_v3 as R3          # noqa: E402
import v3_metrics as M       # noqa: E402


def main(path: str, directory: str = ""):
    d = Path(directory) if directory else R3.BLIND
    key = json.loads((d / "key.json").read_text())
    items = json.loads((d / "alerts_blind.json").read_text())
    verdicts = json.loads(Path(path).read_text())
    strong = M.shares_from_verdicts(verdicts, items)
    # у тревог без текстов (в слепой файл не попали) доли нет: «не проверено», как и у gemma
    ids = {f"{r['seed']}-{r['noise']}-{b['id']}" for r in key["runs"] for b in r["bursts"]}
    strong = {i: strong.get(i) for i in ids}
    runs = key["runs"]
    out = {}
    for name, ov in (("strong", strong), ("gemma", None)):
        tab = {None: M.aggregate(runs, None)}
        for t in R3.THRESHOLDS:
            tab[t] = M.aggregate(runs, t, ov)
        print(f"\n== {name}")
        print(R3.rows_md(tab, judge=name))
        print(R3.loss_md(tab))
        print(R3.quiet_md(tab))
        out[name] = {R3.key(k): v for k, v in tab.items()}
    n_valid = sum(1 for v in strong.values() if v is not None)
    print(f"\nвалидных вердиктов: {n_valid} из {len(strong)}")
    (ROOT / "shared" / ("metrics-v3-strong.json" if d == R3.BLIND else f"metrics-v3-strong-{d.name}.json")).write_text(
        json.dumps(dict(valid=n_valid, total=len(strong), threshold_frozen=key["threshold"], tables=out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
