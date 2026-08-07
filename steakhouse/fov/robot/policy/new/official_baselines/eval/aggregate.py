"""
Turn the JSONL rows from fov_human_eval.py into the tables you report.

    python aggregate.py /scratch1/$USER/steakhouse_zsc/eval/*.jsonl
    python aggregate.py .../eval/*.jsonl --metric reward
    python aggregate.py .../eval/*.jsonl --per_layout

Nothing is ever recomputed from a printed table -- every number here is derived
from the rows on disk, so a table can be regenerated, re-sliced or corrected
without replaying a single episode.

===========================================================================
WHAT THE +/- IS
===========================================================================
Standard deviation ACROSS POLICY SEEDS, not across episodes. Each seed's
episodes are averaged first, then the spread of those per-seed means is
reported.

That is the number the 11-seed re-run exists to produce, and it answers a
different question from episode variance:

    episode std   "how noisy is one policy's play?"
    seed std      "how much does the ANSWER depend on which run I happened to
                  train?"

The second is what makes a baseline comparison honest. The old seed-1 table
could not report it at all. Untrained rows (noop, random) have one "seed" by
construction, so their +/- is 0.00 and means nothing -- read their spread off
n_episodes instead.

===========================================================================
COMPLETION TIME, AND WHY DNFs ARE NOT DROPPED
===========================================================================
A team that never finished is scored at horizon + DNF_PENALTY (500 with the
defaults), NOT excluded. Dropping DNFs would rank a policy that finishes 30% of
the time and is fast when it does above one that always finishes -- the
classic survivorship bug. The finish rate is printed beside every cell so a
mean dragged up by non-finishes is visible rather than mysterious.
"""

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

_ORDER = {"noop": 0, "random": 1, "sp_old": 2, "sp": 3, "sp_eps": 4, "e3t": 5}


def load(paths):
    rows = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return rows


def cell(sub, metric):
    """One (algo, fov) cell -> mean, std-across-seeds, finish rate, n."""
    per_seed = collections.defaultdict(list)
    for r in sub:
        per_seed[r["policy_seed"]].append(r[metric])
    seed_means = [float(np.mean(v)) for v in per_seed.values()]
    return {
        "mean": float(np.mean([r[metric] for r in sub])),
        "std": float(np.std(seed_means)) if len(seed_means) > 1 else 0.0,
        "finish": float(np.mean([r["finished"] for r in sub])),
        "n": len(sub),
        "n_seeds": len(per_seed),
    }


def table(rows, metric, title, lower_better):
    fovs = sorted({r["fov"] for r in rows})
    algos = sorted({r["algo"] for r in rows}, key=lambda a: (_ORDER.get(a, 9), a))

    print("\n" + "=" * 92)
    print(f"{title}   ({'LOWER' if lower_better else 'HIGHER'} IS BETTER)")
    print("mean +/- std ACROSS POLICY SEEDS.  (f=..) finish rate.")
    print("=" * 92)
    print(f"{'algo':<9}{'seeds':>6}{'eps':>7}  "
          + "".join(f"{'fov' + str(f):>19}" for f in fovs))

    best = {}
    for f in fovs:
        vals = []
        for a in algos:
            sub = [r for r in rows if r["algo"] == a and r["fov"] == f]
            if sub:
                vals.append((cell(sub, metric)["mean"], a))
        if vals:
            best[f] = (min if lower_better else max)(vals)[1]

    for a in algos:
        cells, n_seeds, n_eps = [], 0, 0
        for f in fovs:
            sub = [r for r in rows if r["algo"] == a and r["fov"] == f]
            if not sub:
                cells.append(f"{'-':>19}")
                continue
            c = cell(sub, metric)
            n_seeds, n_eps = c["n_seeds"], c["n"]
            mark = "*" if best.get(f) == a else " "
            cells.append(f"{mark}{c['mean']:7.1f}±{c['std']:<5.1f}(f{c['finish']:.2f})")
        print(f"{a:<9}{n_seeds:>6}{n_eps:>7}  " + "".join(cells))
    print("=" * 92)
    print("* = best in column")


def deltas(rows, metric="completion_time", floor="random"):
    """Trained arms against an untrained floor, per FOV.

    This is the sentence the table has to support: does a policy trained with
    zero FOVHuman exposure actually HELP a limited-FOV partner, or does it get
    in the way? Negative delta = the robot earned its place on the board.

    THE FLOOR IS `random` BY DEFAULT, NOT min(noop, random). Measured reason:
    `noop` parks in its start cell for the entire episode, and on a 5x5 like
    steak_gc00 that is a large fraction of the walkable floor. It scored a
    finish rate of 0.00 at five of six FOVs there -- it does not measure "the
    robot did not help", it measures "the robot is furniture". Its one
    non-degenerate cell (fov180, 140.9) then looked like the best number in the
    column and dragged every delta with it.

    `noop` is still worth REPORTING -- a wedged kitchen is a real fact about a
    layout -- but it is a pathological reference point. `--floor noop` or
    `--floor best` if you want it anyway.
    """
    fovs = sorted({r["fov"] for r in rows})
    untrained = [a for a in ("noop", "random") if any(r["algo"] == a for r in rows)]
    if floor == "best":
        floors = untrained
    else:
        floors = [a for a in untrained if a == floor]
    trained = sorted({r["algo"] for r in rows} - set(untrained),
                     key=lambda a: (_ORDER.get(a, 9), a))
    if not floors or not trained:
        return

    print("\n" + "=" * 92)
    print(f"HELP OR HINDER: trained arm minus the '{floor}' floor, on {metric}")
    print("negative = the trained robot helped.  positive = it got in the way.")
    print("=" * 92)
    print(f"{'algo':<9}  " + "".join(f"{'fov' + str(f):>13}" for f in fovs))
    for a in trained:
        cells = []
        for f in fovs:
            sub = [r for r in rows if r["algo"] == a and r["fov"] == f]
            base = [cell([r for r in rows if r["algo"] == fl and r["fov"] == f],
                         metric)["mean"]
                    for fl in floors
                    if any(r["algo"] == fl and r["fov"] == f for r in rows)]
            if not sub or not base:
                cells.append(f"{'-':>13}")
                continue
            d = cell(sub, metric)["mean"] - min(base)
            cells.append(f"{d:+12.1f}")
        print(f"{a:<9}  " + "".join(cells))
    print("=" * 92)
    print("floor = %s per fov" % (" min of " + ", ".join(floors)
                                  if len(floors) > 1 else floors[0]))


def main(argv=None):
    p = argparse.ArgumentParser("aggregate FOV-human eval rows")
    p.add_argument("paths", nargs="+", help="jsonl files or globs")
    p.add_argument("--metric", default="completion_time",
                   choices=["completion_time", "reward", "deliveries",
                            "h_explore", "h_wasted", "steps"])
    p.add_argument("--per_layout", action="store_true")
    p.add_argument("--floor", default="random", choices=["random", "noop", "best"],
                   help="reference for the help-or-hinder table. See deltas().")
    args = p.parse_args(argv)

    rows = load(args.paths)
    if not rows:
        print("no rows found")
        return 1

    lower_better = args.metric in ("completion_time", "h_explore", "h_wasted",
                                   "steps")
    layouts = sorted({r["layout"] for r in rows})
    print(f"loaded {len(rows)} rows | layouts {layouts} | "
          f"algos {sorted({r['algo'] for r in rows})}")

    if args.per_layout:
        for lay in layouts:
            sub = [r for r in rows if r["layout"] == lay]
            table(sub, args.metric, f"{lay} -- {args.metric}", lower_better)
            deltas(sub, args.metric, args.floor)
    else:
        table(rows, args.metric, f"ALL LAYOUTS POOLED -- {args.metric}",
              lower_better)
        deltas(rows, args.metric, args.floor)
        print("\n(pooled across layouts. use --per_layout -- kitchens differ "
              "enormously in how much room a second cook has.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
