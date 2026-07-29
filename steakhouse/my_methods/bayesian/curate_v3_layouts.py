"""
MISHA NEW CHANGE - curate fov/layouts_v3 by SIGNAL-TO-NOISE, then rank.

WHY THIS EXISTS. A layout passes the v3 search on absolute divergence counts, but
what actually predicts good inference is how far the FOV effect rises above that
layout's OWN RNG noise floor. This was confirmed accidentally and then
deliberately:

    layout        late acc   clean signal / rng noise
    keep01          1.000     11 / 2.9  = 3.8x
    keep02          1.000      6 / 0.0  = inf
    keep03          1.000     19 / 6.8  = 2.8x
    keep04          1.000      7 / 4.0  = 1.75x
    r1_idx0002      0.927     35 / 31.0 = 1.1x     <- only sub-margin layout,
                                                      only one below 1.000

r1_idx0002 has the LARGEST absolute divergence of the five and the WORST
inference accuracy, because its noise floor is nearly as large. Ranking on raw
counts would have put it first. That is a smaller version of the exact error that
made v1's rankings meaningless, so the search now enforces NOISE_MARGIN=1.5 - but
layouts found before that landed still need filtering, and some batches were
launched under the old criteria.

Usage:
    python -m my_methods.bayesian.curate_v3_layouts              # report only
    python -m my_methods.bayesian.curate_v3_layouts --apply      # move rejects aside
    python -m my_methods.bayesian.curate_v3_layouts --apply --margin 2.0
"""
import argparse
import os
import re
import shutil

LAYOUTS_V3_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fov", "layouts_v3")
REJECT_DIR = os.path.join(LAYOUTS_V3_DIR, "rejected")

DEFAULT_MARGIN = 1.5


def read_scores(path):
    with open(path) as f:
        head = f.read()
    # Two header formats exist: the worker-side writer (write_layout_file_v3,
    # "clean pairwise...") and the earlier end-of-run writer ("min pairwise...").
    total = re.search(r"(?:clean|min) pairwise subtask disagreement, whole episode:\s*(\d+)", head)
    late = re.search(r"(?:clean|min) pairwise subtask disagreement, latter half\s*:\s*(\d+)", head)
    noise = re.search(r"RNG noise floor \(same FOV, different seed\)\s*:\s*([\d.]+)", head)
    fov = re.search(r"# FOV triple \(deg\): (\d+), (\d+), (\d+)", head)
    kbd = re.search(r"kb_update_delay:\s*(\d+)", head)
    subs = re.search(r"distinct subtasks \(worst rollout\):\s*(\d+)", head)
    if not (total and noise and fov):
        return None
    t, n = int(total.group(1)), float(noise.group(1))
    return dict(
        path=path, name=os.path.basename(path), total=t, noise=n,
        late=int(late.group(1)) if late else 0,
        # inf when the noise floor is exactly zero: every disagreeing step is pure
        # FOV signal, which is the strongest possible result, not a divide-by-zero.
        ratio=(float("inf") if n == 0 else t / n),
        fov=tuple(int(x) for x in fov.groups()),
        kbd=int(kbd.group(1)) if kbd else None,
        subtasks=int(subs.group(1)) if subs else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="move sub-margin layouts into rejected/ (default: report only)")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help=f"minimum signal/noise ratio to keep (default {DEFAULT_MARGIN})")
    args = ap.parse_args()

    if not os.path.isdir(LAYOUTS_V3_DIR):
        print(f"no {LAYOUTS_V3_DIR}")
        return
    rows = []
    for fn in sorted(os.listdir(LAYOUTS_V3_DIR)):
        if not fn.endswith(".layout"):
            continue
        r = read_scores(os.path.join(LAYOUTS_V3_DIR, fn))
        if r:
            rows.append(r)
        else:
            print(f"  (unparsed header, skipped: {fn})")
    if not rows:
        print("no parsable layouts")
        return

    rows.sort(key=lambda r: (-r["ratio"], -r["late"]))
    keep = [r for r in rows if r["ratio"] >= args.margin]
    drop = [r for r in rows if r["ratio"] < args.margin]

    print(f"{len(rows)} layouts; margin={args.margin}x -> keep {len(keep)}, drop {len(drop)}\n")
    print(f"{'layout':<28} {'signal':>6} {'noise':>6} {'ratio':>7} {'late':>5} {'kbd':>4} {'subt':>5}  fov")
    for r in rows:
        mark = "keep" if r["ratio"] >= args.margin else "DROP"
        ratio = "inf" if r["ratio"] == float("inf") else f"{r['ratio']:.2f}"
        print(f"{r['name']:<28} {r['total']:>6} {r['noise']:>6.1f} {ratio:>7} "
              f"{r['late']:>5} {str(r['kbd']):>4} {str(r['subtasks']):>5}  {r['fov']}  {mark}")

    if not args.apply:
        print("\n(report only - pass --apply to move rejects into rejected/)")
        return

    if drop:
        os.makedirs(REJECT_DIR, exist_ok=True)
        for r in drop:
            shutil.move(r["path"], os.path.join(REJECT_DIR, r["name"]))
        print(f"\nmoved {len(drop)} sub-margin layouts to {REJECT_DIR}")
        print("(moved, not deleted - they are still valid divergence findings, "
              "just too close to their own noise floor to be useful for inference)")
    print(f"{len(keep)} layouts kept in {LAYOUTS_V3_DIR}")


if __name__ == "__main__":
    main()
