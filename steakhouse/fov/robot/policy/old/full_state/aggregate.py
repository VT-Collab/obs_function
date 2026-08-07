"""
Summarise the observation comparison: for each (layout, fov) show delivery + time
for old (17-dim) / flat (48) / grid (CNN), and overall means per encoding.

Run: python -m fov.robot.policy.full_state.aggregate [results_dir]
"""
import csv
import glob
import os
import sys
from collections import defaultdict

KINDS = ["old", "flat", "grid"]


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "carc_results_fs"
    files = sorted(glob.glob(os.path.join(rdir, "fullstate_*.csv")))
    if not files:
        print(f"no CSVs in {rdir}", file=sys.stderr)
        sys.exit(1)

    data = defaultdict(dict)                 # (layout,fov) -> kind -> (deliv,t)
    for fn in files:
        with open(fn) as f:
            for r in csv.DictReader(f):
                data[(r["layout"], int(r["fov"]))][r["kind"]] = (
                    float(r["deliv"]), float(r["t"]))

    hdr = f"{'layout':>13} {'fov':>4} | " + " | ".join(
        f"{k+'_dlv':>8} {k+'_t':>7}" for k in KINDS)
    print(hdr); print("-" * len(hdr))
    agg = {k: {"dlv": [], "t": []} for k in KINDS}
    for lay in sorted({k[0] for k in data}):
        for fov in sorted({k[1] for k in data if k[0] == lay}):
            d = data[(lay, fov)]
            cells = []
            for k in KINDS:
                dv, t = d.get(k, (float("nan"), float("nan")))
                cells.append(f"{dv:>8.2f} {t:>7.1f}")
                if k in d:
                    agg[k]["dlv"].append(dv); agg[k]["t"].append(t)
            print(f"{lay:>13} {fov:>4} | " + " | ".join(cells))
        print()

    print("=" * 60)
    print("OVERALL MEANS across all (layout,fov):")
    for k in KINDS:
        if agg[k]["dlv"]:
            print(f"  {k:>5}: delivered={sum(agg[k]['dlv'])/len(agg[k]['dlv']):.3f}  "
                  f"time={sum(agg[k]['t'])/len(agg[k]['t']):.1f}")


if __name__ == "__main__":
    main()
