"""
MISHA NEW CHANGE - Bayesian FOV inference accuracy on the REAL hand-designed
steak layouts, using the new minigrid-style log-space filter
(fov_bayes_filter.SteakBayesFOVInference).

WHY THIS, AND NOT THE fov_search_rank* LAYOUTS
----------------------------------------------
The 23 curated fov_search_rank* layouts turned out to carry no usable FOV
signal. Two independent problems, both verified on rank01 (their TOP-ranked
layout, recorded as maximal divergence at pairs_late_half=(60,60,60)):

  1. Their divergence score was an RNG artifact. v1's search rolled out one
     episode per FOV sequentially without resetting the RNG, and
     auto_unstuck picks its unblocking move with np.random.choice
     (agent.py:449). Holding the seed fixed across FOVs gives 0/120 steps of
     disagreement; varying only the seed gives 107/120.
  2. The human cannot do the task on those generated rooms anyway - it visits
     4 distinct tiles in 120 steps and never gets past 'drop_meat'. v1's
     liveness check (`len(set(positions)) <= 3`, fov_parallel_layout_search
     .py:166) was permissive enough to call that "not stuck".

Meanwhile CARC_NOTES.md records that this project ALREADY got 88.3% accuracy
with 100% informative episodes on the hand-designed `steak_island`, using
evaluate_bayesian_lightweight.py. So the signal was in the real layouts the
whole time. This script keeps that validated episode setup verbatim -
SteakLimitVisionHumanModel ground truth, full-vision GreedySteakHumanModel
teammate, randomized 4-8 steak orders, horizon 200 - and changes exactly one
thing: the filter.

WHY FOV SO OFTEN DOES NOTHING (from evaluate_bayesian_lightweight.py's notes)
----------------------------------------------------------------------------
in_bound() treats any tile immediately beside the player as visible regardless
of FOV, and the human is essentially always standing next to the station it is
about to act on. So FOV only matters for facts learned at a DISTANCE. That is
why geometric divergence scans mispredicted, and why live behavioural testing -
does the belief ever leave the uniform prior - is the only trustworthy screen.
This script reports exactly that as `informative`.

Run with:
    python -m my_methods.bayesian.evaluate_bayesian_v2 [n_workers] [mode] [n_episodes]
"""
import multiprocessing as mp
import os
import sys

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel, GreedySteakHumanModel
from my_methods.bayesian.fov_bayes_filter import SteakBayesFOVInference

FOV_CANDIDATES = (60, 120, 180)
EPISODE_LEN = 200
MIN_ORDERS, MAX_ORDERS = 4, 8

# Every hand-designed steak layout with a cached MediumLevelPlanner. steak_island
# is the one already validated at 88.3%; the rest are the comparison set.
LAYOUTS = [
    "steak_island", "steak_island2", "steak", "steak_api", "steak_mid_1",
    "steak_mid_2", "steak_none_3", "steak_parrallel", "steak_side_2",
    "steak_side_3", "steak_side_4", "steak_test", "steak_tshape", "10x15_steak",
]

# What the human/hypotheses know at t=0. "omniscient" reproduces the validated
# 88.3% baseline exactly; "fov" makes narrow hypotheses start more ignorant.
INITIAL_KB = os.environ.get("FOV_INITIAL_KB", "omniscient")


def run_episode(mlp, layout_name, true_fov, mode, seed):
    """One episode. RNG is seeded up front so that episodes differing only in
    `true_fov` or `mode` are otherwise identical - without this, auto_unstuck's
    np.random.choice makes every comparison noisy (the exact flaw that
    invalidated the v1 layout search)."""
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    n_orders = rng.randint(MIN_ORDERS, MAX_ORDERS + 1)
    mdp = SteakHouseGridworld.from_layout_name(layout_name,
                                               start_order_list=['steak'] * n_orders)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=EPISODE_LEN)

    human = SteakLimitVisionHumanModel(mlp, env.state, vision_limit=True,
                                       vision_bound=true_fov, debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(env.state)

    teammate = GreedySteakHumanModel(mlp)
    teammate.set_agent_index(0)

    inf = SteakBayesFOVInference(mlp, env.state, candidate_fovs=FOV_CANDIDATES,
                                 human_agent_index=1, likelihood=mode,
                                 agent_cls=SteakLimitVisionHumanModel,
                                 initial_kb=INITIAL_KB)

    uniform = 1.0 / len(FOV_CANDIDATES)
    correct = total = 0
    ever_informative = False
    estimates = []
    for _ in range(EPISODE_LEN):
        state = env.state
        try:
            a_teammate = teammate.action(state)[0]
            a_human = human.action(state)[0]
            inf.update(state, a_human)
        except (AssertionError, IndexError, NameError):
            # Known library edge case in the greedy subtask logic - end the
            # episode like a natural completion rather than crashing the batch.
            break

        belief = np.array([inf.posterior()[f] for f in FOV_CANDIDATES])
        if np.max(np.abs(belief - uniform)) > 0.01:
            ever_informative = True
        total += 1
        est = inf.map_fov()
        estimates.append(est)
        if est == true_fov:
            correct += 1

        _, _, done, _ = env.step((a_teammate, a_human))
        if done:
            break

    if total == 0:
        return None
    half = total // 2
    return dict(
        layout=layout_name, true_fov=true_fov, mode=mode, seed=seed, steps=total,
        step_accuracy=correct / total,
        late_accuracy=sum(1 for e in estimates[half:] if e == true_fov) / max(1, total - half),
        final_correct=inf.map_fov() == true_fov,
        informative=ever_informative,
        p_true=inf.posterior()[true_fov],
        entropy=inf.entropy(),
        goal_divergence_rate=(inf.n_goal_divergent_steps / inf.n_steps_seen
                              if inf.n_steps_seen else 0.0),
    )


def run_layout(args):
    layout_name, mode, n_episodes = args
    try:
        mdp = SteakHouseGridworld.from_layout_name(layout_name)
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    except Exception as e:
        return dict(layout=layout_name, mode=mode,
                    error=f"planner failed: {type(e).__name__}: {e}")

    eps = []
    for true_fov in FOV_CANDIDATES:
        for i in range(n_episodes):
            try:
                r = run_episode(mlp, layout_name, true_fov, mode, seed=i * 7 + true_fov)
            except Exception as e:
                r = None
                if i == 0:
                    print(f"  [{layout_name}/{mode}/fov{true_fov}] episode error: "
                          f"{type(e).__name__}: {e}", flush=True)
            if r:
                eps.append(r)
    if not eps:
        return dict(layout=layout_name, mode=mode, error="no episodes completed")

    n = len(eps)
    informative = [e for e in eps if e["informative"]]
    return dict(
        layout=layout_name, mode=mode, n_episodes=n,
        step_accuracy=sum(e["step_accuracy"] for e in eps) / n,
        late_accuracy=sum(e["late_accuracy"] for e in eps) / n,
        final_accuracy=sum(1 for e in eps if e["final_correct"]) / n,
        informative_rate=len(informative) / n,
        informed_final_accuracy=(sum(1 for e in informative if e["final_correct"]) / len(informative)
                                 if informative else 0.0),
        p_true=sum(e["p_true"] for e in eps) / n,
        entropy=sum(e["entropy"] for e in eps) / n,
        mean_steps=sum(e["steps"] for e in eps) / n,
        divergence=sum(e["goal_divergence_rate"] for e in eps) / n,
    )


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    modes = ["greedy", "boltzmann"] if len(sys.argv) <= 2 or sys.argv[2] == "all" else [sys.argv[2]]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    jobs = [(lay, mode, n_episodes) for lay in LAYOUTS for mode in modes]
    print(f"evaluating {len(LAYOUTS)} hand-designed steak layouts x {len(modes)} mode(s) "
          f"x 3 FOVs x {n_episodes} episodes  (initial_kb={INITIAL_KB})", flush=True)
    print(f"baseline to beat: steak_island 88.3% final accuracy, 100% informative "
          f"(old linear-space filter)\n", flush=True)

    results = []
    with mp.Pool(n_workers) as pool:
        for r in pool.imap_unordered(run_layout, jobs):
            results.append(r)
            if "error" in r:
                print(f"{r['layout']:<16} {r['mode']:<9} ERROR: {r['error']}", flush=True)
            else:
                print(f"{r['layout']:<16} {r['mode']:<9} final={r['final_accuracy']*100:5.1f}% "
                      f"step={r['step_accuracy']*100:5.1f}% late={r['late_accuracy']*100:5.1f}% "
                      f"informative={r['informative_rate']*100:5.1f}% "
                      f"informed_final={r['informed_final_accuracy']*100:5.1f}% "
                      f"P(true)={r['p_true']:.3f} H={r['entropy']:.3f} "
                      f"steps={r['mean_steps']:.0f} div={r['divergence']:.2f}", flush=True)

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nno valid results")
        return
    print(f"\n=== RANKED BY final accuracy on informative episodes ===")
    for r in sorted(valid, key=lambda r: -r["informed_final_accuracy"]):
        print(f"  {r['layout']:<16} {r['mode']:<9} informed_final={r['informed_final_accuracy']*100:5.1f}% "
              f"final={r['final_accuracy']*100:5.1f}% informative={r['informative_rate']*100:5.1f}%")
    best = max(valid, key=lambda r: r["informed_final_accuracy"])
    print(f"\nBEST: {best['layout']} / {best['mode']} -> "
          f"{best['informed_final_accuracy']*100:.1f}% on informative episodes, "
          f"{best['informative_rate']*100:.1f}% of episodes informative")


if __name__ == "__main__":
    main()
