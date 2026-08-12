"""Emit the RESULTS.md tables straight from a grid directory. Do not hand-type them.

    python -m robot.filter.analysis.results_tables /path/to/grid > tables.md

Every number in RESULTS.md comes from here. The one table in this package that
was ever transcribed by hand went stale when the layouts changed underneath it,
and a whole conclusion had to be withdrawn -- so the rule now is that a number in
a document is either generated or it is suspect.

Prints, in order: the headline win/tie/loss per pairing, the full dishes/ticks
table, the two controls, the cost line, and the layout fingerprints the grid was
run against.

ONE THING THIS CANNOT DETECT FOR YOU. `handoff` names a different object before
and after the baselines were made to draw their sub-task: grids run earlier hold
a DETERMINISTIC handoff under that same name, and nothing in the row says so. A
grid with no `layout_sha` predates that change and should be read as historical,
not compared against a current one. Fingerprints are printed at the end for
exactly this reason -- check them before quoting a number next to another grid's.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(ROOT)))
sys.path.insert(0, ROOT)

from robot.filter.analysis.report import NAME_FIX, PAIRS          # noqa: E402

LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table", "divide", "pantry"]
FOVS = [30, 60, 90, 180, 360]
# baseline -> filter, in the order the table columns appear.
COLS = [("greedy", "qmdp-greedy"), ("solo", "qmdp-solo"),
        ("handoff", "qmdp"), ("bayes", "qmdp-bayes")]



def load(d):
    by, sha = collections.defaultdict(list), {}
    files = glob.glob(os.path.join(d, "*.jsonl"))
    if not files:
        raise SystemExit("no .jsonl under %s" % d)
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["method"] = NAME_FIX.get(r["method"], r["method"])
            by[(r["layout"], r["fov"], r["method"])].append(r)
            sha.setdefault(r["layout"], r.get("layout_sha", "unrecorded"))
    A = {}
    for k, v in by.items():
        n = len(v)
        A[k] = {"d": sum(x["delivered"] for x in v) / n,
                "t": sum(x["ticks"] for x in v) / n,
                "n": n,
                "dev": sum(x.get("deviated_frac", 0) for x in v) / n,
                "ms": sum(x["ms_per_tick"] for x in v) / n}
    return A, sha


def verdict(b, m):
    """Pass rule, stated once: more dishes wins; at equal dishes, fewer ticks."""
    if m["d"] > b["d"] + 1e-9:
        return "win"
    if abs(m["d"] - b["d"]) < 1e-9:
        if m["t"] < b["t"] - 1e-9:
            return "win"
        if abs(m["t"] - b["t"]) < 1e-9:
            return "tie"
    return "loss"


def cell(x):
    return "%.1f/%d" % (x["d"], round(x["t"]))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("grid")
    a = p.parse_args(argv)
    A, sha = load(a.grid)
    present = {k[2] for k in A}
    cols = [(b, q) for b, q in COLS if b in present and q in present]
    eps = sum(v["n"] for v in A.values())

    print("episodes: %d   methods: %d   cells: %d"
          % (eps, len(present), len({(k[0], k[1]) for k in A})))
    print("\n## headline\n")
    print("| pairing | win | tie | loss |")
    print("|---|---|---|---|")
    for b, q in cols:
        c = collections.Counter()
        for lay in LAYOUTS:
            for f in FOVS:
                bb, qq = A.get((lay, f, b)), A.get((lay, f, q))
                if bb and qq:
                    c[verdict(bb, qq)] += 1
        print("| `%s` vs %s | %d | %d | %d |" % (q, b, c["win"], c["tie"], c["loss"]))

    print("\n## whole table\n")
    print("```")
    print("%-13s|%4s | %s" % ("layout", "fov",
                              " | ".join("%-8s +qmdp   " % b for b, _ in cols)))
    for lay in LAYOUTS:
        for f in FOVS:
            row = "%-13s|%4d " % (lay, f)
            parts = []
            for b, q in cols:
                bb, qq = A.get((lay, f, b)), A.get((lay, f, q))
                if not (bb and qq):
                    parts.append("%-19s" % "?")
                    continue
                v = verdict(bb, qq)
                parts.append("%-8s %-8s %s"
                             % (cell(bb), cell(qq),
                                "*" if v == "win" else ("X" if v == "loss" else " ")))
            print(row + "| " + " | ".join(parts))
    print("```")
    print("\n`*` the layer beat its baseline, `X` lost, blank tie. dishes/ticks,"
          " mean over seeds.")

    # THE SUMMARY GOES UNDER THE TABLE TOO, and it carries BOTH views because they
    # disagree and the disagreement is the finding. Cell counts treat a one-tick win
    # and a whole-dish win alike; total dishes weights them by what a dish is worth.
    # A pairing can be level on cells and down two dishes across the grid, which is
    # exactly what happened to the ladder pairings before the candidate budgets came
    # off -- so quoting either number alone overstates the case.
    print("\n### summary\n")
    print("| pairing | win | tie | loss | dishes: baseline -> layer | change |")
    print("|---|---|---|---|---|---|")
    for b, q in cols:
        c = collections.Counter()
        db = dq = 0.0
        for lay in LAYOUTS:
            for f in FOVS:
                bb, qq = A.get((lay, f, b)), A.get((lay, f, q))
                if bb and qq:
                    c[verdict(bb, qq)] += 1
                    db += bb["d"]
                    dq += qq["d"]
        print("| `%s` vs %s | %d | %d | %d | %.1f -> %.1f | %+.1f |"
              % (q, b, c["win"], c["tie"], c["loss"], db, dq, dq - db))
    print("\nwin/tie/loss counts CELLS (more dishes wins; at equal dishes, fewer"
          " ticks).\n`dishes` sums the per-cell mean over all cells both sides"
          " completed, so it is\ncomparable across pairings only where the same"
          " cells are present.")

    print("\n## controls\n")
    for name, base, what in (("qmdp-base", "handoff", "must be IDENTICAL"),
                             ("qmdp-fixed", "qmdp", "qmdp is the inferring one")):
        if name not in present or base not in present:
            continue
        c, det = collections.Counter(), []
        for lay in LAYOUTS:
            for f in FOVS:
                bb, qq = A.get((lay, f, base)), A.get((lay, f, name))
                if not (bb and qq):
                    continue
                v = verdict(bb, qq)
                c[v] += 1
                if v != "tie":
                    det.append("    %-13s fov%-4d %-11s %-9s  %-8s %s"
                               % (lay, f, name, cell(qq), base, cell(bb)))
        print("%-11s vs %-8s  %s -- better %d, identical %d, worse %d"
              % (name, base, what, c["win"], c["tie"], c["loss"]))
        for d in det:
            print(d)
        print()

    print("## cost\n")
    for m in sorted(present):
        if not m.startswith("qmdp"):
            continue
        v = [A[k] for k in A if k[2] == m]
        print("    %-13s deviates %4.1f%% of ticks   %4.0f ms/tick"
              % (m, 100 * sum(x["dev"] for x in v) / len(v),
                 sum(x["ms"] for x in v) / len(v)))

    print("\n## layouts this grid ran against\n")
    for lay in LAYOUTS:
        if lay in sha:
            print("    %-13s %s" % (lay, sha[lay]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
