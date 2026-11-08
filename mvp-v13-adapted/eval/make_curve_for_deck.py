"""shared/curve-for-deck.json из shared/metrics-scale-v2.json: только то, что идёт на график. Руками не правится."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "shared" / "metrics-scale-v2.json"
DST = ROOT / "shared" / "curve-for-deck.json"
FIELDS = ("false_alarms_per_week", "precision", "recall", "ttc_median_min")


def build(m: dict) -> dict:
    vols = sorted(int(v) for v in m["by_volume"])
    return dict(source="shared/metrics-scale-v2.json", volumes=vols,
                series={meth: {f: [m["by_volume"][str(v)]["detector"][meth][f] for v in vols] for f in FIELDS}
                        for meth in ("poisson", "ewma")})


if __name__ == "__main__":
    DST.write_text(json.dumps(build(json.loads(SRC.read_text(encoding="utf-8"))), ensure_ascii=False, indent=1), encoding="utf-8")
    print("записан", DST)
