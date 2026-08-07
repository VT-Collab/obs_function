"""Aggregate the big mode sweep (mods_*.csv). Per decision mode: a per-FOV
breakdown (averaged over layouts) + an overall line, so we can see which rule
(assist vs subtask) actually beats the baseline and where.

Run: python -m fov.robot.policy.override_v2.aggregate_mods [results_dir]
"""
import csv
import glob
import os
import sys
from collections import defaultdict

FOVS = [30, 60, 90, 120, 180, 360]


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else "carc_results_mods"
    files = sorted(glob.glob(os.path.join(rdir, "mods_*.csv")))
    if not files:
        print(f"no CSVs in {rdir}", file=sys.stderr); sys.exit(1)

    rows = []
    for fn in files:
        with open(fn) as f:
            rows += list(csv.DictReader(f))
    modes = sorted({r["mode"] for r in rows})
    layouts = sorted({r["layout"] for r in rows})
    print(f"{len(layouts)} layouts, modes={modes}, "
          f"n_seeds={rows[0]['n'] if rows else '?'}\n")

    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        print(f"===== mode = {mode} =====")
        print(f"{'fov':>4} | {'base_dlv':>8} {'mod_dlv':>7} {'dDlv':>6} | "
              f"{'base_t3':>7} {'mod_t3':>6} {'dT3':>6} | {'winrate':>7} {'rescues':>7}")
        oa = defaultdict(list)
        for fov in FOVS:
            fr = [r for r in mr if int(r["fov"]) == fov]
            if not fr:
                continue
            bd = sum(float(r["base_dlv"]) for r in fr) / len(fr)
            md = sum(float(r["mod_dlv"]) for r in fr) / len(fr)
            bt = sum(float(r["base_t3"]) for r in fr) / len(fr)
            mt = sum(float(r["mod_t3"]) for r in fr) / len(fr)
            wr = sum(float(r["winrate"]) for r in fr) / len(fr)
            rc = sum(int(r["rescues"]) for r in fr)
            oa["bd"].append(bd); oa["md"].append(md); oa["bt"].append(bt)
            oa["mt"].append(mt); oa["wr"].append(wr); oa["rc"].append(rc)
            print(f"{fov:>4} | {bd:>8.2f} {md:>7.2f} {md-bd:>+6.2f} | "
                  f"{bt:>7.1f} {mt:>6.1f} {mt-bt:>+6.1f} | {wr:>6.0%} {rc:>7}")
        if oa["bd"]:
            bd = sum(oa["bd"]) / len(oa["bd"]); md = sum(oa["md"]) / len(oa["md"])
            bt = sum(oa["bt"]) / len(oa["bt"]); mt = sum(oa["mt"]) / len(oa["mt"])
            wr = sum(oa["wr"]) / len(oa["wr"]); rc = sum(oa["rc"])
            print(f"{'ALL':>4} | {bd:>8.2f} {md:>7.2f} {md-bd:>+6.2f} | "
                  f"{bt:>7.1f} {mt:>6.1f} {mt-bt:>+6.1f} | {wr:>6.0%} {rc:>7}")
        print()


if __name__ == "__main__":
    main()
