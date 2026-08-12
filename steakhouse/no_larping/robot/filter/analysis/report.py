"""Read evaluate.py's JSONL and answer the one question: did we clear the gate?

    python -m robot.filter.analysis.report runs.jsonl
    python -m robot.filter.analysis.report runs.jsonl --pairs qmdp=handoff

THE GATE, stated once: the comparison is PAIRED. Each filter is measured against
the baseline it actually wraps -- qmdp over handoff against handoff, not
against whichever baseline happens to be strongest on that layout. That is the
comparison the design can actually make a claim about: the layer re-ranks its own
baseline's jobs and degrades to its own baseline's action, so its own baseline is
the thing it must beat. Better means more deliveries; at equal deliveries it
means fewer ticks.

Per-cell and not pooled, because a method that wins big on two layouts and loses
quietly on four would pool to a win and still be the wrong policy. Episodes are
averaged over seeds first, so one lucky seed cannot carry a cell.
"""
import argparse
import collections
import json
import sys

# filter -> the baseline it wraps. This IS the gate's definition, and each filter
# is scored against the EXACT object it wraps -- the same instance kind, drawing
# from the same pi. There is only one kind of baseline now, so this is simply the
# row above each filter in robot/methods.py.
#
# The two CONTROLS pair against handoff too, and that is deliberate rather than
# sloppy: qmdp-base must TIE handoff in every cell (it is the parity check), and
# qmdp-fixed vs handoff is not the interesting comparison -- the interesting one
# is qmdp vs qmdp-fixed, which --pairs qmdp=qmdp-fixed asks for directly.
PAIRS = {
    "qmdp": "handoff",
    "qmdp-greedy": "greedy",
    "qmdp-solo": "solo",
    "qmdp-bayes": "bayes",
    "qmdp-base": "handoff",
    "qmdp-fixed": "handoff",
    # Pre-rename spellings. Grids written while the baselines were briefly
    # called `-stoch` / `bayes-post` still report, and so do the older
    # deterministic-baseline grids: in both cases the JSONL names whatever
    # actually ran, and it is paired against whatever it actually wrapped.
    "qmdp-exec": "handoff",
    "qmdp-exec-solo": "solo",
    "qmdp-exec-greedy": "greedy",
    "qmdp-exec-bayes": "bayes",
    "qmdp-exec-base": "handoff",
    "qmdp-exec-fixed": "handoff",
}

# method-name synonyms, applied to every row as it is loaded, so one grid can be
# read whatever vocabulary it was written with.
NAME_FIX = {"greedy-stoch": "greedy", "solo-stoch": "solo",
            "handoff-stoch": "handoff", "bayes-post": "bayes",
            "bayes-prior": "bayes-noip",
            # Retired filter spellings. There is ONE filter class now; these
            # were stages of it, and a grid written under any of them is a
            # grid of that stage. They map onto the surviving names so the
            # saved grids keep reporting -- check `layout_sha` before
            # comparing two of them, because the stages differ.
            "qmdp-exec": "qmdp", "qmdp-exec-solo": "qmdp-solo",
            "qmdp-exec-greedy": "qmdp-greedy",
            "qmdp-exec-bayes": "qmdp-bayes",
            "qmdp-exec-base": "qmdp-base",
            "qmdp-exec-fixed": "qmdp-fixed",
            "qmdp-enum": "qmdp", "qmdp-enum-solo": "qmdp-solo",
            "qmdp-enum-greedy": "qmdp-greedy",
            "qmdp-enum-bayes": "qmdp-bayes",
            "qmdp-mask": "qmdp", "qmdp-mask-solo": "qmdp-solo",
            "qmdp-mask-greedy": "qmdp-greedy",
            "qmdp-mask-bayes": "qmdp-bayes"}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["method"] = NAME_FIX.get(r["method"], r["method"])
                    rows.append(r)
    return rows


def agg(rows):
    """(layout, fov, method) -> {delivered, ticks, n, ...} averaged over seeds.

    `ticks` uses the episode length, which for a finished episode is when the
    last order landed and for an unfinished one is the horizon. That makes the
    two directly comparable without a special case: failing to finish is simply
    the worst possible time.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["layout"], r["fov"], r["method"])].append(r)
    out = {}
    for k, rs in by.items():
        n = len(rs)
        out[k] = {
            "delivered": sum(r["delivered"] for r in rs) / n,
            "ticks": sum(r["ticks"] for r in rs) / n,
            "idle": sum(r["idle_frac"] for r in rs) / n,
            "dev": sum(r.get("deviated_frac", 0) for r in rs) / n,
            "ms": sum(r["ms_per_tick"] for r in rs) / n,
            "n": n,
            "seeds": sorted(r["seed"] for r in rs),
        }
    return out


def better(a, b):
    """Is cell-result `a` at least as good as `b`? (deliveries, then speed)"""
    if a["delivered"] != b["delivered"]:
        return a["delivered"] > b["delivered"]
    return a["ticks"] <= b["ticks"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--pairs", default="",
                   help="comma-sep filter=baseline overrides; default is PAIRS")
    p.add_argument("--only", default="", help="restrict to these filters, comma-sep")
    a = p.parse_args(argv)

    rows = load(a.paths)
    A = agg(rows)
    pairs = dict(PAIRS)
    present = {k[2] for k in A}
    # `exec:<baseline>:<knobs>` names its own baseline, so the sweep spellings
    # pair themselves and no --pairs argument is needed for them.
    for m in present:
        if m.startswith("exec:"):
            pairs[m] = m.split(":")[1]
    for kv in [s for s in a.pairs.split(",") if s]:
        k, v = kv.rsplit("=", 1)     # knob values contain '=' too
        pairs[k] = v
    only = [s for s in a.only.split(",") if s]
    tested = [f for f in pairs if f in present and (not only or f in only)]
    tested.sort(key=lambda f: list(pairs).index(f))
    cells = sorted({(k[0], k[1]) for k in A})

    overall_ok = True
    for filt in tested:
        base = pairs[filt]
        print("\n=== %s  vs its baseline  %s ===" % (filt, base))
        print("%-13s %5s | %-14s %-14s | %-7s %s"
              % ("layout", "fov", base, filt, "delta", "GATE"))
        print("-" * 74)
        won = lost = missing = 0
        losses = []
        for lay, fov in cells:
            b, m = A.get((lay, fov, base)), A.get((lay, fov, filt))
            if b is None or m is None:
                missing += 1
                continue
            ok = better(m, b)
            if ok:
                won += 1
            else:
                lost += 1
                losses.append((lay, fov, b, m))
            print("%-13s %5d | %-14s %-14s | d%+.1f t%+-5d %s"
                  % (lay, fov,
                     "%.1f/%dt" % (b["delivered"], round(b["ticks"])),
                     "%.1f/%dt" % (m["delivered"], round(m["ticks"])),
                     m["delivered"] - b["delivered"],
                     round(m["ticks"] - b["ticks"]),
                     "PASS" if ok else "FAIL"))
        tot = won + lost
        print("  -> %d/%d cells pass%s" % (won, tot, "" if not missing
                                           else " (%d missing)" % missing))
        if lost:
            overall_ok = False
            print("  worst first:")
            for lay, fov, b, m in sorted(
                    losses, key=lambda x: (x[2]["delivered"] - x[3]["delivered"],
                                           x[3]["ticks"] - x[2]["ticks"]),
                    reverse=True)[:12]:
                print("    %-13s fov%-4d  base %.1f/%dt  ours %.1f/%dt  (d%+.1f t%+d)"
                      % (lay, fov, b["delivered"], round(b["ticks"]),
                         m["delivered"], round(m["ticks"]),
                         m["delivered"] - b["delivered"],
                         round(m["ticks"] - b["ticks"])))
        mine = [v for k, v in A.items() if k[2] == filt]
        if mine:
            print("  %s: mean %.0f ms/tick, deviates on %.0f%% of ticks"
                  % (filt, sum(r["ms"] for r in mine) / len(mine),
                     100 * sum(r["dev"] for r in mine) / len(mine)))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
