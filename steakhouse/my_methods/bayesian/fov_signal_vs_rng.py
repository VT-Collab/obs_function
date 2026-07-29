"""
MISHA NEW CHANGE - separates the FOV EFFECT from RNG NOISE in the divergence
measurement that selected the 23 curated layouts.

WHY THIS EXISTS. fov_parallel_layout_search.py scored a layout by running one
independent rollout per FOV candidate and counting steps where the subtask
sequences disagreed. That comparison silently assumed the simulation is
deterministic given the layout - fov_search_results.md explicitly says so, and
flags as "unexplained" that two jobs with identical layout parameters produced
different divergence numbers.

The simulation is NOT deterministic. GreedyHumanModel's auto_unstuck branch
picks its unblocking move with np.random.choice (agent.py:449), and it fires
constantly on these layouts (the human oscillates against the parked robot). So
each per-FOV rollout drew from a DIFFERENT point in the global RNG stream, and
the "disagreement" between them mixed together two completely different things:

    (a) the FOV effect      - what we actually want to measure
    (b) RNG divergence      - pure noise from auto_unstuck's random choice

Locally on rank01 (the top-ranked layout, recorded as maximal divergence with
pairs_late_half=(60,60,60)) the split was:
    FOV effect at fixed seed : 0 / 120 steps   <- nothing
    seed-to-seed at fixed FOV: 107 / 120 steps <- everything
i.e. that layout's entire recorded "divergence" was RNG.

WHAT THIS SCRIPT MEASURES. For every layout, for each of several seeds:
  * fov_effect  - reseed to the SAME seed before each FOV's rollout, then
                  compare subtask sequences ACROSS FOVs. Same RNG stream, so
                  any disagreement is genuinely caused by field of view.
  * rng_effect  - hold FOV fixed and vary the seed, then compare. This is the
                  noise floor.
A layout only carries real FOV signal if fov_effect > 0, and it is only worth
using for Bayesian inference if fov_effect is large relative to rng_effect.

Run with: python -m my_methods.bayesian.fov_signal_vs_rng [n_workers] [n_seeds] [n_steps]
"""
import os
import random
import sys
import multiprocessing as mp

import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from my_methods.bayesian.fov_subtask_divergence_test import build_mdp_and_mlp
from my_methods.bayesian.sticky_subtask_human import StickySubtaskHumanModel
from my_methods.bayesian.fov_sustained_batch import robot_next_action
from my_methods.bayesian.fov_inference_batch import parse_layout_file, HIDE_POS, LAYOUTS_DIR

N_STEPS = 120
N_SEEDS = 3


def rollout(mdp, mlp, cfg, hide_pos, fov, seed, n_steps):
    """One independent rollout, with the RNG reset to `seed` first so that two
    rollouts differing only in `fov` see the identical random stream."""
    np.random.seed(seed)
    random.seed(seed)
    setup_state = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300).state.deepcopy()
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300)
    human = StickySubtaskHumanModel(mlp, setup_state, vision_limit=True,
                                    vision_bound=fov, debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(setup_state)
    subs = []
    for _ in range(n_steps):
        state = env.state
        a_robot = robot_next_action(mlp, state, 0, cfg["m_pos"], hide_pos)
        try:
            a_human, _ = human.action(state)
        except AssertionError:
            subs.append("STUCK")
            break
        subs.append(human.prev_chosen_subtask)
        _, _, done, _ = env.step((a_robot, a_human))
        if done:
            subs.append("DONE")
            break
    return subs


def disagree(a, b):
    """Steps where two subtask sequences differ (compared over their overlap)."""
    return sum(1 for x, y in zip(a, b) if x != y)


def run_layout(args):
    rank, n_seeds, n_steps = args
    name = f"fov_search_rank{rank:02d}"
    base = dict(rank=rank)
    try:
        cfg = parse_layout_file(os.path.join(LAYOUTS_DIR, f"{name}.layout"))
        mdp, mlp = build_mdp_and_mlp(name, cfg["grid"], order_list=cfg["order_list"],
                                     cook_time=cfg["cook_time"], chop_time=cfg["chop_time"],
                                     wash_time=cfg["wash_time"],
                                     num_items_for_steak=cfg["num_items_for_steak"])
    except Exception as e:
        return dict(base, error=f"build failed: {type(e).__name__}: {e}")

    hide_pos = HIDE_POS[rank]
    fovs = list(cfg["fov_triple"])
    try:
        # rolls[seed][fov] - every rollout for a given seed starts from the same
        # RNG state, so across-FOV comparison isolates the FOV effect.
        rolls = {s: {f: rollout(mdp, mlp, cfg, hide_pos, f, s, n_steps) for f in fovs}
                 for s in range(n_seeds)}
    except Exception as e:
        return dict(base, error=f"sim failed: {type(e).__name__}: {e}")

    # FOV effect: all 3 pairs, at each fixed seed.
    fov_pairs = []
    for s in range(n_seeds):
        for i in range(len(fovs)):
            for j in range(i + 1, len(fovs)):
                fov_pairs.append(disagree(rolls[s][fovs[i]], rolls[s][fovs[j]]))
    # RNG effect: same FOV, different seeds.
    rng_pairs = []
    for f in fovs:
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                rng_pairs.append(disagree(rolls[i][f], rolls[j][f]))

    lens = [len(rolls[s][f]) for s in range(n_seeds) for f in fovs]
    uniq = sorted({x for s in range(n_seeds) for f in fovs for x in rolls[s][f]})
    return dict(base, fov_triple=tuple(fovs), n_seeds=n_seeds, mean_len=sum(lens) / len(lens),
                fov_effect_mean=sum(fov_pairs) / len(fov_pairs) if fov_pairs else 0,
                fov_effect_max=max(fov_pairs) if fov_pairs else 0,
                rng_effect_mean=sum(rng_pairs) / len(rng_pairs) if rng_pairs else 0,
                rng_effect_max=max(rng_pairs) if rng_pairs else 0,
                n_distinct_subtasks=len(uniq), subtasks=uniq[:8])


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else N_SEEDS
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else N_STEPS

    jobs = [(rank, n_seeds, n_steps) for rank in range(1, 24)]
    print(f"FOV signal vs RNG noise: 23 layouts x {n_seeds} seeds x 3 FOVs, "
          f"{n_steps} steps, {n_workers} workers", flush=True)
    print("fov_effect = across-FOV disagreement at FIXED seed (real signal)", flush=True)
    print("rng_effect = across-seed disagreement at FIXED fov (noise floor)\n", flush=True)

    results = []
    with mp.Pool(n_workers) as pool:
        for r in pool.imap_unordered(run_layout, jobs):
            results.append(r)
            if "error" in r:
                print(f"rank{r['rank']:02d} ERROR: {r['error']}", flush=True)
            else:
                print(f"rank{r['rank']:02d} fov={r['fov_triple']} len={r['mean_len']:.0f} "
                      f"FOV_effect={r['fov_effect_mean']:6.1f} (max {r['fov_effect_max']:>3}) "
                      f"RNG_noise={r['rng_effect_mean']:6.1f} (max {r['rng_effect_max']:>3}) "
                      f"n_subtasks={r['n_distinct_subtasks']}", flush=True)

    valid = [r for r in results if "error" not in r]
    print(f"\n=== SUMMARY ({len(valid)}/23) ===")
    if not valid:
        return
    real = [r for r in valid if r["fov_effect_mean"] > 0]
    beats = [r for r in valid if r["fov_effect_mean"] > r["rng_effect_mean"]]
    print(f"layouts with ANY FOV effect at fixed seed : {len(real)}/{len(valid)}")
    print(f"layouts where FOV effect > RNG noise      : {len(beats)}/{len(valid)}")
    print(f"mean FOV effect across layouts            : "
          f"{sum(r['fov_effect_mean'] for r in valid)/len(valid):.2f} steps")
    print(f"mean RNG noise across layouts             : "
          f"{sum(r['rng_effect_mean'] for r in valid)/len(valid):.2f} steps")
    print(f"layouts where human only ever does 1 subtask (livelocked): "
          f"{sum(1 for r in valid if r['n_distinct_subtasks'] <= 1)}/{len(valid)}")
    if real:
        print("\nlayouts WITH real FOV signal, best first:")
        for r in sorted(real, key=lambda r: -r["fov_effect_mean"]):
            print(f"  rank{r['rank']:02d} fov={r['fov_triple']} "
                  f"FOV_effect={r['fov_effect_mean']:.1f} RNG={r['rng_effect_mean']:.1f} "
                  f"subtasks={r['subtasks']}")


if __name__ == "__main__":
    main()
