"""
Aggregate the per-layout CSVs from carc_run into a readable verdict table.

For each (layout, fov): compare the baseline against the BEST override config
(lowest completion time among override_k0/k3/k6 that is NO WORSE on delivery).
Verdict: WIN (no worse delivery AND faster) / tie / WORSE.

Run: python -m fov.robot.policy.override_v2.aggregate [results_dir]
"""
import csv
import glob
import os
import sys
from collections import defaultdict

EPS = 1e-9


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "carc_results"
    files = sorted(glob.glob(os.path.join(rdir, "override_v2_*.csv")))
    if not files:
        print(f"no CSVs in {rdir}", file=sys.stderr)
        sys.exit(1)

    # data[(layout,fov)][config] = dict(deliv,t,override,cone)
    data = defaultdict(dict)
    for fn in files:
        with open(fn) as f:
            for row in csv.DictReader(f):
                key = (row["layout"], int(row["fov"]))
                data[key][row["config"]] = dict(
                    deliv=float(row["deliv"]), t=float(row["t"]),
                    override=float(row["override"]), cone=float(row["cone"]))

    wins = ties = worse = 0
    tot_delta = 0.0
    n_delta = 0
    layouts = sorted({k[0] for k in data})
    hdr = (f"{'layout':>13} {'fov':>4} | {'base_t':>6} {'b_dlv':>5} | "
           f"{'best_cfg':>11} {'ovr_t':>6} {'o_dlv':>5} {'ovr%':>5} {'cone%':>5} | verdict")
    print(hdr)
    print("-" * len(hdr))
    for lay in layouts:
        for fov in sorted({k[1] for k in data if k[0] == lay}):
            d = data[(lay, fov)]
            base = d.get("baseline")
            if base is None:
                continue
            ovr_cfgs = {c: v for c, v in d.items() if c.startswith("override")}
            # best override that is no-worse on delivery, by lowest time
            no_worse = {c: v for c, v in ovr_cfgs.items()
                        if v["deliv"] >= base["deliv"] - EPS}
            pool = no_worse or ovr_cfgs
            best_c = min(pool, key=lambda c: pool[c]["t"])
            best = pool[best_c]
            nw = best["deliv"] >= base["deliv"] - EPS
            faster = best["t"] < base["t"] - EPS
            verdict = "WIN" if (nw and faster) else ("tie" if nw else "WORSE")
            if verdict == "WIN":
                wins += 1
            elif verdict == "WORSE":
                worse += 1
            else:
                ties += 1
            if nw:
                tot_delta += (base["t"] - best["t"]); n_delta += 1
            print(f"{lay:>13} {fov:>4} | {base['t']:>6.1f} {base['deliv']:>5.2f} | "
                  f"{best_c:>11} {best['t']:>6.1f} {best['deliv']:>5.2f} "
                  f"{best['override']:>4.0%} {best['cone']:>4.0%} | {verdict}")
        print()

    n = wins + ties + worse
    print("=" * 60)
    print(f"cells: {n}   WIN={wins} ({wins/max(1,n):.0%})   "
          f"tie={ties}   WORSE={worse}")
    if n_delta:
        print(f"mean time saved on no-worse-delivery cells: "
              f"{tot_delta/n_delta:+.1f} steps  (positive = faster)")


if __name__ == "__main__":
    main()
