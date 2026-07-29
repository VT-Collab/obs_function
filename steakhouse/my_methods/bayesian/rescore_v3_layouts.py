"""
MISHA NEW CHANGE - re-score the existing fov/layouts_v3 set with the CORRECTED
divergence metric, and report which layouts still qualify.

WHY. The layouts in fov/layouts_v3 were selected with an index-aligned
disagreement count. That metric counts a pure TIMING OFFSET as disagreement: if
two FOVs execute the identical ordered sequence of subtasks but one runs a step
ahead, every index near a transition mismatches even though both humans made the
same decisions.

Measured on keep01, pair (128,178): identical ordered prefix of 9 subtask runs,
178 simply ~1 step ahead. 7 of 11 counted disagreements were phase artifacts.
Phase-corrected keep01 scores 4, which FAILS MIN_PAIR_TOTAL=6 and FAILS the
1.5x noise margin (4 < 1.5 * 2.9 = 4.35). The inflation lands specifically on the
pair that sets the scored minimum, so it is not a wash - it admitted layouts.

pair_disagreement now run-length compresses each sequence to its ordered
DECISIONS and counts only Needleman-Wunsch SUBSTITUTIONS (indels = timing).
This script re-runs the same rollouts the search ran and re-derives every
layout's score under that metric, so the set can be re-curated honestly.

It also reports LIVENESS, which the original scoring never checked: how many
rollouts end DONE (task completed) versus STUCK. An audit found 28/36 rollouts
ending STUCK and delivering nothing, so headers claiming the human "completes
real steak orders" overstate what these layouts show.

Run with: python -m my_methods.bayesian.rescore_v3_layouts [n_workers] [n_seeds]
"""
import multiprocessing as mp
import os
import sys

from my_methods.bayesian.fov_subtask_divergence_test import build_mdp_and_mlp
from my_methods.bayesian.fov_divergence_search_v2 import (
    rollout, pair_disagreement, discard_planner_cache,
    MIN_PAIR_TOTAL, MIN_PAIR_LATE, NOISE_MARGIN,
    MIN_DISTINCT_SUBTASKS, MIN_DISTINCT_POSITIONS, DEAD,
)
from my_methods.bayesian.fov_inference_v3 import parse_v3_layout, LAYOUTS_V3_DIR


def rescore(args):
    path, n_seeds = args
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        cfg = parse_v3_layout(path)
    except Exception as e:
        return dict(name=name, error=f"parse: {type(e).__name__}: {e}")

    # rollout() needs these keys; layouts record hide_pos in the header but the
    # working teammate ignores it.
    cfg = dict(cfg, robot_mode=cfg.get("robot_mode", "work"), human_mode="plain",
               m_pos=(0, 0), hide_pos=(0, 0))
    try:
        mdp, mlp = build_mdp_and_mlp(name, cfg["grid"], order_list=cfg["order_list"],
                                     cook_time=cfg["cook_time"], chop_time=cfg["chop_time"],
                                     wash_time=cfg["wash_time"],
                                     num_items_for_steak=cfg["num_items_for_steak"])
    except Exception as e:
        return dict(name=name, error=f"build: {type(e).__name__}: {e}")

    fovs = list(cfg["fov_triple"])
    try:
        rolls = {s: {f: rollout(mdp, mlp, cfg, f, s) for f in fovs} for s in range(n_seeds)}
    except Exception as e:
        discard_planner_cache(name)
        return dict(name=name, error=f"sim: {type(e).__name__}: {e}")
    discard_planner_cache(name)

    per_seed = []
    for s in range(n_seeds):
        pairs = []
        for i in range(len(fovs)):
            for j in range(i + 1, len(fovs)):
                pairs.append(pair_disagreement(rolls[s][fovs[i]][0], rolls[s][fovs[j]][0]))
        per_seed.append(dict(
            raw_total=min(p[0] for p in pairs),
            clean_total=min(p[2] for p in pairs),
            clean_late=min(p[3] for p in pairs),
            distinct=len({tuple(rolls[s][f][0]) for f in fovs}) == 3,
        ))

    rng_pairs = []
    for f in fovs:
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                rng_pairs.append(pair_disagreement(rolls[i][f][0], rolls[j][f][0])[2])
    # MISHA NEW CHANGE - compare like with like. The old gate took the MEAN over
    # all seed-pairs as the noise floor while the signal is a MIN over FOV pairs.
    # The noise distribution is strongly bimodal, so the mean flattered layouts
    # whose chaos is concentrated in one FOV (on keep01 that FOV alone scores 9
    # against a min signal of 11 - a 1.2x margin, not the advertised 3.8x).
    # Report both; gate on the max.
    rng_mean = sum(rng_pairs) / len(rng_pairs) if rng_pairs else 0.0
    rng_max = max(rng_pairs) if rng_pairs else 0.0

    all_rolls = [rolls[s][f] for s in range(n_seeds) for f in fovs]
    min_subtasks = min(len({x for x in r[0] if x not in DEAD}) for r in all_rolls)
    min_positions = min(len(set(r[1])) for r in all_rolls)
    n_done = sum(1 for r in all_rolls if r[0] and r[0][-1] == 'DONE')
    n_stuck = sum(1 for r in all_rolls if r[0] and r[0][-1] == 'STUCK')

    sig = min(ps["clean_total"] for ps in per_seed)
    late = min(ps["clean_late"] for ps in per_seed)
    passes = (all(ps["distinct"] for ps in per_seed)
              and sig >= MIN_PAIR_TOTAL and late >= MIN_PAIR_LATE
              and sig >= NOISE_MARGIN * rng_max
              and min_subtasks >= MIN_DISTINCT_SUBTASKS
              and min_positions >= MIN_DISTINCT_POSITIONS)
    return dict(name=name, fov=tuple(fovs), kbd=cfg["kb_update_delay"],
                raw_total=min(ps["raw_total"] for ps in per_seed),
                clean_total=sig, clean_late=late,
                rng_mean=rng_mean, rng_max=rng_max,
                min_subtasks=min_subtasks, min_positions=min_positions,
                n_done=n_done, n_stuck=n_stuck, n_rolls=len(all_rolls), passes=passes)


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    files = sorted(os.path.join(LAYOUTS_V3_DIR, f) for f in os.listdir(LAYOUTS_V3_DIR)
                   if f.endswith(".layout") and not f.startswith("._"))
    print(f"re-scoring {len(files)} layouts with PHASE-CORRECTED divergence, "
          f"{n_seeds} seeds, {n_workers} workers", flush=True)
    print(f"gate: clean>={MIN_PAIR_TOTAL}, late>={MIN_PAIR_LATE}, "
          f"clean>={NOISE_MARGIN}x rng_MAX, subtasks>={MIN_DISTINCT_SUBTASKS}, "
          f"tiles>={MIN_DISTINCT_POSITIONS}\n", flush=True)

    results = []
    with mp.Pool(n_workers) as pool:
        for r in pool.imap_unordered(rescore, [(f, n_seeds) for f in files]):
            results.append(r)
            if "error" in r:
                print(f"{r['name']:<30} ERROR {r['error']}", flush=True)
            else:
                print(f"{r['name']:<30} raw={r['raw_total']:>3} clean={r['clean_total']:>3} "
                      f"late={r['clean_late']:>3} rngmean={r['rng_mean']:>5.1f} "
                      f"rngmax={r['rng_max']:>4.0f} subt={r['min_subtasks']:>2} "
                      f"done={r['n_done']}/{r['n_rolls']} stuck={r['n_stuck']} "
                      f"{'PASS' if r['passes'] else 'fail'}", flush=True)

    valid = [r for r in results if "error" not in r]
    keep = [r for r in valid if r["passes"]]
    print(f"\n=== {len(keep)}/{len(valid)} still qualify under the corrected metric ===")
    if valid:
        infl = [r for r in valid if r["raw_total"] > r["clean_total"]]
        print(f"phase inflation present in {len(infl)}/{len(valid)} layouts; "
              f"mean raw={sum(r['raw_total'] for r in valid)/len(valid):.1f} vs "
              f"mean clean={sum(r['clean_total'] for r in valid)/len(valid):.1f}")
        tot_done = sum(r["n_done"] for r in valid)
        tot_roll = sum(r["n_rolls"] for r in valid)
        print(f"LIVENESS: {tot_done}/{tot_roll} rollouts completed the task (DONE), "
              f"{sum(r['n_stuck'] for r in valid)}/{tot_roll} ended STUCK")
    if keep:
        print("\nsurvivors, best first:")
        for r in sorted(keep, key=lambda r: -(r["clean_total"] / max(r["rng_max"], 0.5))):
            print(f"  {r['name']:<30} clean={r['clean_total']:>3} late={r['clean_late']:>3} "
                  f"rngmax={r['rng_max']:>4.0f} done={r['n_done']}/{r['n_rolls']} {r['fov']}")


if __name__ == "__main__":
    main()
