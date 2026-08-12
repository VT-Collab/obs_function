"""Diff two grids cell by cell. Did a change actually move anything?

    python -m robot.filter.analysis.compare_grids OLD_DIR NEW_DIR

Paired win/tie/loss per pairing for both runs side by side, then every cell whose
result CHANGED. A change that moves nothing is worth knowing about -- most of the
edits in this package's history moved nothing, and the ones that did were not
always the ones predicted to.

TWO GUARDS AGAINST READING A DIFF THAT IS NOT ABOUT THE CODE, both added after
being caught out by exactly this:

  1. LAYOUT DRIFT. Four .layout files were edited between two grids. Diffing
     them attributed a layout change to a code change -- and the layouts in
     question were the ones carrying the conclusion. Rows now carry `layout_sha`
     and a layout whose fingerprint differs is EXCLUDED from the tallies and
     listed separately. Old rows without the field are reported as unknown
     rather than assumed to match.

  2. BASELINE DRIFT. The nominal baselines share no code with the filter, so a
     baseline cell that moves between two grids is proof that something other
     than the filter changed -- a layout, a seed convention, a shared geometry
     helper. That is checked and printed BEFORE the pairing tallies, because it
     invalidates them.
"""
import argparse
import collections
import glob
import json
import os
import sys

# Each filter against the baseline it wraps -- see report.PAIRS for why.
PAIRS = [("solo", "qmdp-solo"), ("greedy", "qmdp-greedy"),
         ("handoff", "qmdp"), ("bayes", "qmdp-bayes")]
# Grids written under earlier vocabularies are normalised on load.
NAME_FIX = {"greedy-stoch": "greedy", "solo-stoch": "solo",
            "handoff-stoch": "handoff", "bayes-post": "bayes",
            "bayes-prior": "bayes-noip"}
LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table", "divide", "pantry"]
FOVS = [30, 60, 90, 180, 360]


# The baseline rows. A cell of one of these that MOVES between two grids is
# proof the diff is not about the filter -- they share no code with it.
BASE = ["solo", "greedy", "handoff", "bayes"]


def load(d):
    by = collections.defaultdict(list)
    sha = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                r["method"] = NAME_FIX.get(r["method"], r["method"])
                by[(r["layout"], r["fov"], r["method"])].append(r)
                sha.setdefault(r["layout"], r.get("layout_sha", "unknown"))
    A = {k: (sum(x["delivered"] for x in v) / len(v),
             sum(x["ticks"] for x in v) / len(v), len(v)) for k, v in by.items()}
    return A, sha


def comparable(so, sn):
    """Split the layouts three ways by what the fingerprints can prove.

    `unverified` exists because grids written before `layout_sha` have no
    fingerprint at all, and refusing to diff them would make this tool useless
    on exactly the history it was written to explain. For those, BASELINE DRIFT
    is the fallback proof -- a nominal baseline that moved cannot have been moved
    by the filter -- and main() demotes an unverified layout to drifted when its
    baselines disagree.
    """
    same, drift, unverified = [], [], []
    for lay in LAYOUTS:
        a, b = so.get(lay), sn.get(lay)
        if a is None or b is None:
            continue
        if "unknown" in (a, b):
            unverified.append(lay)
        elif a == b:
            same.append(lay)
        else:
            drift.append(lay)
    return same, drift, unverified


def baseline_drift(O, N, layouts):
    """Baseline cells that moved. Each one is proof the diff is not about the filter."""
    out = []
    for lay in layouts:
        for fov in FOVS:
            for b in BASE:
                o, n = O.get((lay, fov, b)), N.get((lay, fov, b))
                if o and n and (abs(o[0] - n[0]) > 1e-9 or abs(o[1] - n[1]) > 1e-9):
                    out.append((lay, fov, b, o, n))
    return out


def verdict(base, mine):
    """'win' / 'tie' / 'loss'. Ties are exact equality, not near-equality."""
    if mine[0] > base[0] + 1e-9 or (abs(mine[0] - base[0]) < 1e-9 and mine[1] < base[1] - 1e-9):
        return "win"
    if abs(mine[0] - base[0]) < 1e-9 and abs(mine[1] - base[1]) < 1e-9:
        return "tie"
    return "loss"


def tally(A, layouts):
    out = {}
    for b, q in PAIRS:
        c = collections.Counter()
        for lay in layouts:
            for fov in FOVS:
                bb, qq = A.get((lay, fov, b)), A.get((lay, fov, q))
                if bb and qq:
                    c[verdict(bb, qq)] += 1
        out[(b, q)] = c
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--force", action="store_true",
                   help="tally drifted layouts too. The number will not mean "
                        "what it looks like it means.")
    a = p.parse_args(argv)
    O, so = load(a.old)
    N, sn = load(a.new)

    same, drift, unverified = comparable(so, sn)
    # Every baseline cell that moved, checked BEFORE anything is tallied. The
    # nominal baselines share no code with the filter, so each one is proof that
    # something other than the filter changed underneath these two grids.
    bd = baseline_drift(O, N, LAYOUTS)
    moved = {lay for lay, _, _, _, _ in bd}
    demoted = [lay for lay in unverified if lay in moved]
    drift += demoted
    unverified = [lay for lay in unverified if lay not in moved]

    if drift:
        print("!! LAYOUT DRIFT -- not the same kitchen in both grids:")
        for lay in drift:
            why = ("baselines moved, and no fingerprint to check" if lay in demoted
                   else "%s -> %s" % (so.get(lay), sn.get(lay)))
            print("     %-13s %s" % (lay, why))
        print("   EXCLUDED from the tallies below."
              if not a.force else "   --force: included anyway. Read with care.")
    if unverified:
        print("?? UNVERIFIED (no layout_sha in one grid, but baselines match): %s"
              % ", ".join(unverified))
    if same:
        print("ok fingerprint-identical in both grids: %s" % ", ".join(same))

    if bd:
        print("\n!! BASELINE DRIFT on %d cells that should be identical." % len(bd))
        for lay, fov, b, o, n in bd[:12]:
            print("     %-13s fov%-4d %-8s %.1f/%dt -> %.1f/%dt"
                  % (lay, fov, b, o[0], round(o[1]), n[0], round(n[1])))
        if len(bd) > 12:
            print("     ... and %d more" % (len(bd) - 12))

    layouts = LAYOUTS if a.force else (same + unverified)
    if not layouts:
        print("\nNothing is comparable between these two grids. Re-run the old "
              "configuration against the CURRENT layouts if you need the diff.")
        return 1
    to, tn = tally(O, layouts), tally(N, layouts)

    print("\ncomparing over %d layout(s): %s" % (len(layouts), ", ".join(layouts)))
    print("%-12s %-22s %s" % ("pairing", "OLD  win/tie/loss", "NEW  win/tie/loss"))
    print("-" * 62)
    for b, q in PAIRS:
        o, n = to[(b, q)], tn[(b, q)]
        print("%-12s %-22s %d/%d/%d" % (
            q, "%d/%d/%d" % (o["win"], o["tie"], o["loss"]),
            n["win"], n["tie"], n["loss"]))

    print("\nCells whose verdict CHANGED:")
    any_ = False
    for b, q in PAIRS:
        for lay in layouts:
            for fov in FOVS:
                ob, oq = O.get((lay, fov, b)), O.get((lay, fov, q))
                nb, nq = N.get((lay, fov, b)), N.get((lay, fov, q))
                if not (ob and oq and nb and nq):
                    continue
                vo, vn = verdict(ob, oq), verdict(nb, nq)
                if vo != vn:
                    any_ = True
                    print("  %-12s %-13s fov%-4d %-5s -> %-5s   %.1f/%dt -> %.1f/%dt"
                          % (q, lay, fov, vo, vn, oq[0], round(oq[1]), nq[0], round(nq[1])))
    if not any_:
        print("  (none -- the change was behaviourally inert on this grid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
