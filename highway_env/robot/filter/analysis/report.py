"""Reads a JSONL file of episode results (harness/evaluate.py's own output)
and prints a pairwise comparison: each FOV-aware method against the plain
baseline it's meant to improve on. Never touches core/ or harness/ --
mirrors steakhouse/no_larping (and misha)'s own robot/filter/analysis/
report.py: PAIRS hardcodes filter -> baseline, agg() averages over seeds
per (scene, fov, method) cell, better() picks a winner, output is a plain
console table plus a win/loss tally -- no file/plot output.

    python report.py runs.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict

PAIRS = {
    "fov_aware": "nominal",
    "cautious": "nominal",
}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def agg(rows):
    """Average every numeric metric over seeds, grouped by (scene, fov,
    robot_vehicle, method) -- robot_vehicle (idm/linear/aggressive/
    defensive, robot/nominal_policy/vehicles.py) is a separate axis from
    method (the policy wrapper), so a pairwise comparison always holds the
    underlying vehicle dynamics fixed and varies only the wrapper.
    r.get("robot_vehicle", "idm") defaults older JSONL (pre-dating this
    field, always idm back then) so old and new runs can still be
    compared."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scene"], r["fov"], r.get("robot_vehicle", "idm"), r["method"])].append(r)
    out = {}
    for key, rs in groups.items():
        n = len(rs)

        def avg(field):
            vals = [r[field] for r in rs if r.get(field) is not None]
            return sum(vals) / len(vals) if vals else None

        out[key] = {
            "n": n,
            "human_crash_rate": sum(r["human_crashed"] for r in rs) / n,
            "robot_crash_rate": sum(r["robot_crashed"] for r in rs) / n,
            "avg_scene_crashes": avg("scene_crash_count"),
            "avg_human_progress": avg("human_progress_frac"),
            "avg_robot_progress": avg("robot_progress_frac"),
            "avg_min_gap": avg("min_gap_human_robot"),
            "avg_map_fov_accuracy": avg("map_fov_accuracy"),
        }
    return out


def better(a, b):
    """True if cell `a` beats cell `b`: fewer crashes first (safety is the
    whole point of a filter), then more robot progress (a filter that's
    merely more cautious without ever finishing anything isn't a win),
    matching the reference's own "more deliveries, tie-break fewer ticks"
    ordering adapted to this domain's own metrics."""
    crash_a = a["human_crash_rate"] + a["robot_crash_rate"]
    crash_b = b["human_crash_rate"] + b["robot_crash_rate"]
    if crash_a != crash_b:
        return crash_a < crash_b
    return (a["avg_robot_progress"] or 0) > (b["avg_robot_progress"] or 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", help="path to evaluate.py's --out file")
    args = parser.parse_args()

    rows = load(args.jsonl)
    if not rows:
        sys.exit(f"no rows in {args.jsonl}")
    cells = agg(rows)

    cases = sorted({(r["scene"], r["fov"], r.get("robot_vehicle", "idm")) for r in rows})
    wins = defaultdict(int)
    losses = defaultdict(int)

    for method, baseline in PAIRS.items():
        print(f"\n=== {method} vs {baseline} ===")
        header = (f"{'scene':<12}{'fov':>6}  {'vehicle':<10}  {'human_crash%':>12}  {'robot_crash%':>12}  "
                  f"{'human_prog':>10}  {'robot_prog':>10}  {'min_gap':>8}  {'winner':>8}")
        print(header)
        print("-" * len(header))
        for scene, fov, robot_vehicle in cases:
            a = cells.get((scene, fov, robot_vehicle, method))
            b = cells.get((scene, fov, robot_vehicle, baseline))
            if a is None or b is None:
                continue
            winner = method if better(a, b) else (baseline if better(b, a) else "tie")
            if winner == method:
                wins[method] += 1
            elif winner == baseline:
                losses[method] += 1
            print(f"{scene:<12}{fov:>6.0f}  {robot_vehicle:<10}  "
                  f"{a['human_crash_rate']*100:>6.0f} / {b['human_crash_rate']*100:<5.0f}  "
                  f"{a['robot_crash_rate']*100:>6.0f} / {b['robot_crash_rate']*100:<5.0f}  "
                  f"{(a['avg_human_progress'] or 0)*100:>9.0f}%  "
                  f"{(a['avg_robot_progress'] or 0)*100:>9.0f}%  "
                  f"{a['avg_min_gap'] or float('nan'):>8.1f}  "
                  f"{winner:>8}")
        print(f"{method}: {wins[method]} wins, {losses[method]} losses, "
              f"{len(cases) - wins[method] - losses[method]} ties/missing")

    if any(r.get("map_fov_accuracy") is not None for r in rows):
        print("\n=== FOV posterior accuracy (fov_aware method only, by true fov) ===")
        by_fov = defaultdict(list)
        for r in rows:
            if r["method"] == "fov_aware" and r.get("map_fov_accuracy") is not None:
                by_fov[r["fov"]].append(r["map_fov_accuracy"])
        for fov in sorted(by_fov):
            accs = by_fov[fov]
            print(f"  true_fov={fov:>5.0f}: avg_map_accuracy={sum(accs)/len(accs):.2f} (n={len(accs)})")


if __name__ == "__main__":
    main()
