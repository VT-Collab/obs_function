"""Turn evaluate.py's JSONL into the paired tables.

    python aggregate.py results.jsonl                    # by fov
    python aggregate.py results.jsonl --by layout,fov
    python aggregate.py results.jsonl --arms lam         # what separates arms

PAIRED. Every arm plays the same (layout, fov, seed) cells, so the statistic
that matters is the per-cell DIFFERENCE against the beta=0 arm, not the two
marginal means. A paired sign test on those differences is reported alongside,
because with ~30 episodes per cell the marginal means move around a lot and an
unpaired difference is easy to over-read.
"""
import argparse
import collections
import json
import math
import sys


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_derive(json.loads(line)))
                except ValueError:
                    #a snapshot taken while a job was mid-write leaves one
                    #truncated last line. Skip it rather than lose the file.
                    continue
    return rows


def _derive(r):
    """Per-step rates. The raw human counters (h_explore, h_checks, h_wasted)
    are TOTALS, so a condition that simply finishes sooner lowers all of them
    without the human having behaved any differently. The rate is the honest
    version of "did the human waste less of its time"."""
    n = max(int(r.get("steps") or 0), 1)
    for c in ("h_explore", "h_checks", "h_wasted", "h_abandoned"):
        if c in r and r[c] is not None:
            r["rate_" + c[2:]] = float(r[c]) / n
    return r


def arm_key(row, arm_fields):
    return tuple(row.get(f) for f in arm_fields)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sign_test_p(wins, losses):
    """Two-sided binomial p under H0: P(win) = P(loss) = 1/2.

    Exact for small n; for large n the exact sum overflows a float, so fall
    back to the normal approximation with a continuity correction (n is in the
    hundreds to thousands here, where the approximation is very tight)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    if n <= 900:
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / float(2 ** n)
        return min(1.0, 2.0 * tail)
    mu = n / 2.0
    sd = math.sqrt(n) / 2.0
    z = (k + 0.5 - mu) / sd
    #two-sided tail of the standard normal
    return min(1.0, 2.0 * 0.5 * math.erfc(-z / math.sqrt(2.0)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", type=str, default="fov",
                    help="comma separated grouping columns for the table")
    ap.add_argument("--arms", type=str, default="lam,module_mode",
                    help="columns that identify an arm; the arm with lam==0 is "
                         "the reference")
    ap.add_argument("--cell", type=str, default="layout,fov,seed",
                    help="columns that identify one paired cell")
    ap.add_argument("--metric", type=str, default="completion_time")
    ap.add_argument("--extra", type=str,
                    default="rate_explore,rate_checks,rate_wasted,p_true_fov",
                    help="human-side / filter diagnostics to difference too. "
                         "MECHANISM evidence: if the module works for the "
                         "reason claimed, the human's wasted trips should fall")
    ap.add_argument("--ref", type=str, default=None,
                    help="explicit reference arm as a python-ish tuple string, "
                         "e.g. \"(4.0, 'noise')\". Default: the arm with lam==0")
    args = ap.parse_args(argv)

    rows = load(args.files)
    by = [c for c in args.by.split(",") if c]
    arm_fields = [c for c in args.arms.split(",") if c]
    cell_fields = [c for c in args.cell.split(",") if c]

    arms = sorted({arm_key(r, arm_fields) for r in rows}, key=str)
    ref = None
    if args.ref:
        for a in arms:
            if str(a) == args.ref:
                ref = a
                break
        if ref is None:
            print("no arm matches --ref %r; arms are %s" % (args.ref, arms))
            return 1
    else:
        for a in arms:
            if float(dict(zip(arm_fields, a)).get("lam", 1.0)) == 0.0:
                ref = a
                break
    print("arms: %s   reference: %s" % (arms, ref))
    print()

    # ---------------- marginal table
    hdr = "%-28s %-22s %7s %7s %7s %7s %6s" % (
        "|".join(by), "|".join(arm_fields), "n", "deliv", "comp", "steps",
        "fin")
    print(hdr)
    print("-" * len(hdr))
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g = tuple(r.get(c) for c in by)
        groups[g][arm_key(r, arm_fields)].append(r)
    for g in sorted(groups, key=str):
        for a in arms:
            v = groups[g][a]
            if not v:
                continue
            print("%-28s %-22s %7d %7.2f %7.1f %7.1f %6.2f" % (
                "|".join(str(x) for x in g), "|".join(str(x) for x in a),
                len(v), mean(x["deliveries"] for x in v),
                mean(x["completion_time"] for x in v),
                mean(x["steps"] for x in v),
                mean(float(x["finished"]) for x in v)))
        print()

    if ref is None:
        return 0

    # ---------------- paired differences vs the beta=0 arm
    extra = [c for c in args.extra.split(",") if c]
    print("PAIRED vs reference   (negative %s = the module is FASTER)"
          % args.metric)
    hdr = "%-28s %-22s %6s %8s %8s %6s %6s %6s %8s" % (
        "|".join(by), "arm", "n", "d_" + args.metric[:6], "d_deliv",
        "win", "loss", "tie", "p_sign")
    hdr += "".join("%10s" % ("d_" + c[:8]) for c in extra)
    print(hdr)
    print("-" * len(hdr))
    index = {}
    for r in rows:
        index[(arm_key(r, arm_fields), tuple(r.get(c) for c in cell_fields))] = r

    for g in sorted(groups, key=str):
        for a in arms:
            if a == ref:
                continue
            diffs, ddel, win, loss, tie = [], [], 0, 0, 0
            dextra = collections.defaultdict(list)
            for r in groups[g][a]:
                cell = tuple(r.get(c) for c in cell_fields)
                base = index.get((ref, cell))
                if base is None:
                    continue
                d = r[args.metric] - base[args.metric]
                diffs.append(d)
                ddel.append(r["deliveries"] - base["deliveries"])
                for c in extra:
                    if c in r and c in base and r[c] is not None \
                            and base[c] is not None:
                        dextra[c].append(r[c] - base[c])
                #a WIN is strictly faster with no loss of deliveries -- the
                #same rule fov/robot/policy/old/RESULTS.md used
                if r["deliveries"] > base["deliveries"] or (
                        r["deliveries"] == base["deliveries"] and d < 0):
                    win += 1
                elif r["deliveries"] < base["deliveries"] or d > 0:
                    loss += 1
                else:
                    tie += 1
            if not diffs:
                continue
            line = "%-28s %-22s %6d %8.1f %8.2f %6d %6d %6d %8.4f" % (
                "|".join(str(x) for x in g), "|".join(str(x) for x in a),
                len(diffs), mean(diffs), mean(ddel), win, loss, tie,
                sign_test_p(win, loss))
            line += "".join("%10.2f" % mean(dextra[c]) if dextra[c] else
                            "%10s" % "-" for c in extra)
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
