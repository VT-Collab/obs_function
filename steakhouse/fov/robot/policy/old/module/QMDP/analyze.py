"""Campaign-level readouts: pick a config on TRAIN seeds, score it on TEST.

    python analyze.py beta      /scratch1/$USER/steakhouse_qmdp/beta/*.jsonl
    python analyze.py weights   .../weights/*.jsonl
    python analyze.py ablate    .../ablate/*.jsonl
    python analyze.py decompose .../decompose/*.jsonl
    python analyze.py final     .../final/*.jsonl

Each mode prints one table. The SELECTION rule is stated in each so it cannot
drift: the number the sweep is ranked on is always the paired win rate against
the beta=0 arm within the SAME file (so the two arms saw the same layouts, the
same FOVs and the same seeds), and selection only ever runs on TRAIN seeds.
"""
import argparse
import collections
import json
import sys

from aggregate import load, mean, sign_test_p

DNF = None  # unused; kept so the import list is obvious


def paired(rows, arm_of, ref_pred, cell_fields=("layout", "fov", "seed"),
           metric="completion_time"):
    """{arm: {n, d_metric, d_deliv, win, loss, tie, p}} against its reference.

    The reference is matched WITHIN the same tag, so an arm is only ever
    compared to a baseline that played the identical cells in the identical
    process.
    """
    ref_index = {}
    for r in rows:
        if ref_pred(r):
            ref_index[(r.get("tag"),
                       tuple(r.get(c) for c in cell_fields))] = r
    out = collections.defaultdict(
        lambda: {"n": 0, "dm": [], "dd": [], "win": 0, "loss": 0, "tie": 0})
    for r in rows:
        if ref_pred(r):
            continue
        base = ref_index.get((r.get("tag"),
                              tuple(r.get(c) for c in cell_fields)))
        if base is None:
            continue
        a = arm_of(r)
        s = out[a]
        d = r[metric] - base[metric]
        s["n"] += 1
        s["dm"].append(d)
        s["dd"].append(r["deliveries"] - base["deliveries"])
        if r["deliveries"] > base["deliveries"] or (
                r["deliveries"] == base["deliveries"] and d < 0):
            s["win"] += 1
        elif r["deliveries"] < base["deliveries"] or d > 0:
            s["loss"] += 1
        else:
            s["tie"] += 1
    return out


def show(table, label, sort_by="winrate"):
    print("%-34s %6s %9s %8s %7s %7s %7s %9s" % (
        label, "n", "d_comp", "d_deliv", "win", "loss", "tie", "p_sign"))
    print("-" * 92)
    rows = []
    for a, s in table.items():
        if not s["n"]:
            continue
        wr = s["win"] / float(max(s["win"] + s["loss"], 1))
        rows.append((wr, mean(s["dm"]), a, s))
    rows.sort(key=(lambda t: -t[0]) if sort_by == "winrate"
              else (lambda t: t[1]))
    for wr, dm, a, s in rows:
        print("%-34s %6d %9.1f %8.2f %7d %7d %7d %9.4f" % (
            str(a), s["n"], dm, mean(s["dd"]), s["win"], s["loss"], s["tie"],
            sign_test_p(s["win"], s["loss"])))
    print()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["beta", "weights", "ablate", "decompose",
                                     "final"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by_fov", action="store_true")
    args = ap.parse_args(argv)
    rows = load(args.files)
    print("%d episodes loaded\n" % len(rows))

    is_ref = lambda r: float(r.get("lam", 0.0)) == 0.0

    if args.mode == "beta":
        print("SELECT: the beta with the highest paired win rate vs beta=0, "
              "pooled over the 11 usable layouts, TRAIN seeds.\n")
        show(paired(rows, lambda r: "beta=%.1f" % r["lam"], is_ref), "arm")
        if args.by_fov:
            for f in sorted({r["fov"] for r in rows}):
                sub = [r for r in rows if r["fov"] == f]
                show(paired(sub, lambda r: "beta=%.1f" % r["lam"], is_ref),
                     "fov=%d" % f)

    elif args.mode == "weights":
        print("SELECT: one-axis deviations from the shipped defaults. "
              "`w:centre` is the reference point, TRAIN seeds.\n")
        show(paired(rows, lambda r: r.get("tag", "?"), is_ref), "weight axis")

    elif args.mode == "ablate":
        print("CONTROLS. `real` is the module; the rest replace its output "
              "while changing nothing else.\n"
              "  uniform  must be exactly 0/0/all-tie -- proves the blend is inert\n"
              "  noise    same perturbation SIZE, random direction\n"
              "  shuffle  the real scores, permuted over the actions\n"
              "  fixed:F  the module run on a cone pinned to F instead of the "
              "inferred posterior\n")
        #the baseline arm lives in its own tag, so pair on (layout, fov, seed)
        #across tags instead
        base = {}
        for r in rows:
            if float(r.get("lam", 0.0)) == 0.0:
                base[(r["layout"], r["fov"], r["seed"])] = r
        tab = collections.defaultdict(
            lambda: {"n": 0, "dm": [], "dd": [], "win": 0, "loss": 0, "tie": 0})
        for r in rows:
            if float(r.get("lam", 0.0)) == 0.0:
                continue
            b = base.get((r["layout"], r["fov"], r["seed"]))
            if b is None:
                continue
            s = tab[r.get("module_mode", "real")]
            d = r["completion_time"] - b["completion_time"]
            s["n"] += 1
            s["dm"].append(d)
            s["dd"].append(r["deliveries"] - b["deliveries"])
            if r["deliveries"] > b["deliveries"] or (
                    r["deliveries"] == b["deliveries"] and d < 0):
                s["win"] += 1
            elif r["deliveries"] < b["deliveries"] or d > 0:
                s["loss"] += 1
            else:
                s["tie"] += 1
        show(tab, "module_mode")
        if args.by_fov:
            for f in sorted({r["fov"] for r in rows}):
                sub = collections.defaultdict(
                    lambda: {"n": 0, "dm": [], "dd": [], "win": 0, "loss": 0,
                             "tie": 0})
                for r in rows:
                    if r["fov"] != f or float(r.get("lam", 0.0)) == 0.0:
                        continue
                    b = base.get((r["layout"], r["fov"], r["seed"]))
                    if b is None:
                        continue
                    s = sub[r.get("module_mode", "real")]
                    d = r["completion_time"] - b["completion_time"]
                    s["n"] += 1
                    s["dm"].append(d)
                    s["dd"].append(r["deliveries"] - b["deliveries"])
                    if r["deliveries"] > b["deliveries"] or (
                            r["deliveries"] == b["deliveries"] and d < 0):
                        s["win"] += 1
                    elif r["deliveries"] < b["deliveries"] or d > 0:
                        s["loss"] += 1
                    else:
                        s["tie"] += 1
                show(sub, "fov=%d mode" % f)

    elif args.mode == "decompose":
        print("Which HALF of the cost function is doing the work.\n"
              "  kb_only          the eight knowledge-base terms, collisions off\n"
              "  collision_only   interference + self_block only\n"
              "  full             both\n")
        show(paired(rows, lambda r: r["tag"].rsplit(":", 1)[-1], is_ref),
             "half")
        if args.by_fov:
            for f in sorted({r["fov"] for r in rows}):
                sub = [r for r in rows if r["fov"] == f]
                show(paired(sub, lambda r: r["tag"].rsplit(":", 1)[-1], is_ref),
                     "fov=%d half" % f)

    elif args.mode == "final":
        print("HELD-OUT SEEDS. Nothing was selected on these.\n")
        show(paired(rows, lambda r: "module", is_ref), "overall")
        for f in sorted({r["fov"] for r in rows}):
            sub = [r for r in rows if r["fov"] == f]
            show(paired(sub, lambda r: "fov=%d" % f, is_ref), "by fov")
        for L in sorted({r["layout"] for r in rows}):
            sub = [r for r in rows if r["layout"] == L]
            show(paired(sub, lambda r, L=L: L, is_ref), "by layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
