"""Сводка eval/cache/diagnose.json -> таблицы для DIAGNOSIS.md (печатает в stdout)."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
D = json.loads((Path(__file__).parent / "cache" / "diagnose.json").read_text())
def agg(rows, name):
    ev = sum(r["runs"][name]["res"]["n_events"] for r in rows); al = sum(r["runs"][name]["res"]["n_alerts"] for r in rows)
    tp = sum(r["runs"][name]["res"]["tp_alerts"] for r in rows); ca = sum(r["runs"][name]["res"]["caught"] for r in rows)
    ttc = [t for r in rows for t in r["runs"][name]["res"]["ttc"]]
    fa = Counter(); [fa.update(r["runs"][name]["fa_kinds"]) for r in rows]
    weeks = len(rows) * 25 / 7
    sec = np.mean([r["runs"][name]["seconds"] for r in rows])
    return f"| {name} | {tp/al:.2f} | {ca/ev:.2f} ({ca}/{ev}) | {np.median(ttc):.0f} | {np.percentile(ttc,90):.0f} | {(al-tp)/weeks:.1f} | {dict(fa)} | {sec:.0f} |"
groups = {"tune 1-5": [r for r in D if r["seed"] < 100], "сожжённые 101-110": [r for r in D if r["seed"] > 100], "все": D}
for g, rows in groups.items():
    print(f"\n### {g}: {len(rows)} прогонов\n| вариант | precision | recall | медиана, мин | p90 | ложных/нед | ложные по причинам | с/прогон |\n|---|---|---|---|---|---|---|---|")
    for name in ("oracle_poisson", "oracle_ewma", "tfidf_poisson"):
        if name in rows[0]["runs"]: print(agg(rows, name))
fr = [r["runs"]["tfidf_poisson"]["frag"] for r in D if r["runs"].get("tfidf_poisson")]
for n in ("low", "mid", "high"):
    f = [r["runs"]["tfidf_poisson"]["frag"] for r in D if r["noise"] == n]
    print(f"frag {n}: purity {np.mean([x['purity'] for x in f]):.3f} clusters/topic(80%) {np.mean([x['frag_mean'] for x in f]):.2f} n_clusters {np.mean([x['n_clusters'] for x in f]):.0f}")
S = [s for r in D for s in r["strength"]]
print("\nсобытий:", len(S), "мобильных:", sum(s["mobile"] for s in S))
for W in (30, 60, 120, 240):
    z = np.array([s[f"z{W}"] for s in S]); ex = np.array([s[f"extra{W}"] for s in S]); lam = np.array([s[f"lam{W}"] for s in S])
    print(f"W={W}: extra median {np.median(ex):.0f} [p10 {np.percentile(ex,10):.0f}, p90 {np.percentile(ex,90):.0f}], baseline median {np.median(lam):.1f}, ideal z median {np.median(z):.1f}, share z>=4.5: {np.mean(z>=4.5):.2f}, z>=3: {np.mean(z>=3):.2f}")
M = [s for s in S if s["mobile"]]
for W in (60, 120):
    zv = np.array([s[f"ver_z{W}"] for s in M]); zt = np.array([s[f"z{W}"] for s in M])
    print(f"mobile W={W}: topic z median {np.median(zt):.1f} share>=4.5 {np.mean(zt>=4.5):.2f} | version z median {np.median(zv):.1f} share>=4.5 {np.mean(zv>=4.5):.2f}; q median {np.median([s[f'ver_q{W}'] for s in M]):.3f}")
mic = [m for r in D for m in r["micro"]]
print("micro bursts per run:", {n: np.mean([len(r["micro"]) for r in D if r["noise"] == n]).round(1) for n in ("low","mid","high")}, "size median", np.median(mic))
# пойманные оракулом: задержка по мобильным/веб
for name in ("oracle_poisson","tfidf_poisson"):
    d = defaultdict(list)
    for r in D:
        mob = {s["id"]: s["mobile"] for s in r["strength"]}
        for e, t in r["runs"][name]["caught"].items(): d[mob[e]].append(t)
    print(name, "ttc mobile median", np.median(d[True]), "n", len(d[True]), "| web median", np.median(d[False]), "n", len(d[False]))
