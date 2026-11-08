"""python -m radar.gen --seeds 1 2 3 --noise mid --volume 800  ->  data/<seed>/"""
import argparse
from pathlib import Path

from . import generate, save

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 101])
ap.add_argument("--noise", default="mid", choices=["low", "mid", "high"])
ap.add_argument("--volume", type=int, default=800, help="обращений в день по всем продуктам")
ap.add_argument("--days", type=int, default=28)
ap.add_argument("--out", default="data")
a = ap.parse_args()
for s in a.seeds:
    name = str(s) if a.noise == "mid" else f"{s}-{a.noise}"
    ds = generate(s, a.noise, a.volume, a.days)
    save(ds, Path(a.out) / name)
    types = {}
    for e in ds.ground_truth["events"]:
        types[e["type"]] = types.get(e["type"], 0) + 1
    print(f"seed={s} noise={a.noise}: {len(ds.tickets)} обращений, события {types} -> {a.out}/{name}")
