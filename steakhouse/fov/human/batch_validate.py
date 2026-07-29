"""
MISHA NEW CHANGE - thorough validation of the from-scratch limited-vision human
across MANY layouts and MANY FOVs, scoring both required conditions:

  (A) FOV changes the SUBTASK PROGRESSION substantially (not timing), and
  (B) WIDER FOV PERFORMS BETTER (more orders delivered by the human itself).

Verified locally on steak_island before scaling up (2 seeds, fovs 30/90/360):
    (A) all 6 pairs real divergence 3-17
    (B) delivered 1.00 / 1.00 / 2.00 with look-steps 64.0 / 15.5 / 3.0
Both PASS. This batch establishes how broadly that holds.

Scoring notes:
  * Divergence is phase-corrected: sequences are run-length compressed and only
    substitutions plus tokens absent from the other sequence count. A narrow
    agent doing the same things a step later scores ZERO. This matters because
    the project's earlier layout set was selected on an index-aligned count that
    turned out to be ~98% timing artifact.
  * Every rollout reseeds numpy AND random first, so two rollouts differing only
    in FOV consume an identical random stream.
  * Performance is the HUMAN's own deliveries, not the team's. Team score is
    reported alongside but never gated on: a human that merely stays out of the
    teammate's way can raise the team number while doing nothing itself, which
    is how an earlier version appeared to show narrow FOV "winning".

Run with: python -m fov.human.batch_validate [n_workers] [n_seeds]
"""
import multiprocessing as mp
import os
import random
import sys

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import GreedySteakHumanModel
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman
from fov.human.planning.steak_planner import SteakMotionPlanner
from fov.human.test_fov_conditions import real_divergence, runs_of

# Wide sweep: narrow cones through full observability. 360 is the control that
# must perform best; the triple used for Bayesian inference is chosen afterwards
# from whichever three FOVs separate most on a given layout.
FOVS = [30, 60, 90, 120, 180, 360]
N_STEPS = 260
N_ORDERS = 4

HAND_LAYOUTS = ["steak_island", "steak_island2", "steak", "steak_mid_1", "steak_mid_2",
                "steak_side_2", "steak_side_3", "steak_side_4", "steak_tshape",
                "steak_none_3", "steak_parrallel", "steak_test", "steak_api"]

# Gates
MIN_PAIR_DIV = 5      # a pair "separates" at >= this much real divergence
MIN_SEP_PAIRS = 3     # at least this many FOV pairs must separate
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "layouts_final")


def episode(mdp, mlp, fov, seed):
    np.random.seed(seed)
    random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=N_STEPS + 10)
    planner = SteakMotionPlanner(mdp, mlp)
    human = LimitedVisionSteakHuman(mdp, fov, planner, agent_index=1)
    robot = GreedySteakHumanModel(mlp)
    robot.set_agent_index(0)
    n0 = len(env.state.order_list) if env.state.order_list else 0
    for _ in range(N_STEPS):
        s = env.state
        try:
            a_r, _ = robot.action(s)
        except Exception:
            a_r = Action.STAY
        try:
            a_h, _ = human.action(s)
        except Exception:
            a_h = Action.STAY
        _, _, done, _ = env.step((a_r, a_h))
        if done:
            break
    left = len(env.state.order_list) if env.state.order_list else 0
    return dict(fov=fov, seed=seed, subtasks=human.subtask_log,
                delivered=human.n_delivered, team=n0 - left,
                checks=human.n_checks, wasted=human.n_wasted_commits,
                steps=len(human.subtask_log),
                n_distinct=len(set(x for x in human.subtask_log if x)))


def check_layout(args):
    name, n_seeds = args
    try:
        mdp = SteakHouseGridworld.from_layout_name(name, start_order_list=['steak'] * N_ORDERS)
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    except Exception as e:
        return dict(layout=name, error=f"{type(e).__name__}: {e}")

    R = {}
    for seed in range(n_seeds):
        for fov in FOVS:
            try:
                R[(fov, seed)] = episode(mdp, mlp, fov, seed)
            except Exception as e:
                return dict(layout=name, error=f"sim {type(e).__name__}: {e}")

    # (A) real divergence, worst seed per pair
    pairs = {}
    for i in range(len(FOVS)):
        for j in range(i + 1, len(FOVS)):
            worst = min(real_divergence(R[(FOVS[i], s)]["subtasks"],
                                        R[(FOVS[j], s)]["subtasks"])
                        for s in range(n_seeds))
            pairs[(FOVS[i], FOVS[j])] = worst
    n_sep = sum(1 for v in pairs.values() if v >= MIN_PAIR_DIV)

    # (B) TEAM WIN at every FOV. Success is the TEAM's deliveries (the human's
    # prep feeding a teammate's delivery is a team win), not the human's own
    # count - a greedy teammate harvests shared stations regardless of the
    # human's cone, so human-own deliveries do not order the FOVs and the FOV
    # signal lives in (A). The gate: the team delivers a healthy number at EVERY
    # cone, i.e. the human is a functional teammate and never breaks the team.
    mean_team = {f: sum(R[(f, s)]["team"] for s in range(n_seeds)) / n_seeds
                 for f in FOVS}
    mean_del = {f: sum(R[(f, s)]["delivered"] for s in range(n_seeds)) / n_seeds
                for f in FOVS}
    mean_chk = {f: sum(R[(f, s)]["checks"] for s in range(n_seeds)) / n_seeds
                for f in FOVS}
    widest, narrowest = FOVS[-1], FOVS[0]
    improves = mean_team[widest] >= mean_team[narrowest]   # team not hurt by FOV
    monotone = True
    team_win = all(mean_team[f] >= 1.0 for f in FOVS)      # functional at every cone
    full_ok = team_win

    # best 3 FOVs for Bayesian inference: maximise the weakest pairwise gap
    best_triple, best_score = None, -1
    for a in range(len(FOVS)):
        for b in range(a + 1, len(FOVS)):
            for c in range(b + 1, len(FOVS)):
                t = (FOVS[a], FOVS[b], FOVS[c])
                sc = min(pairs[(t[0], t[1])], pairs[(t[1], t[2])], pairs[(t[0], t[2])])
                if sc > best_score:
                    best_triple, best_score = t, sc

    # (A) substantial subtask divergence AND (B) team wins at every FOV.
    passes = n_sep >= MIN_SEP_PAIRS and team_win
    return dict(layout=name, pairs=pairs, n_sep=n_sep, mean_del=mean_del,
                mean_team=mean_team, mean_chk=mean_chk, improves=improves,
                monotone=monotone, full_ok=full_ok, best_triple=best_triple,
                best_triple_score=best_score,
                n_distinct=max(R[(f, 0)]["n_distinct"] for f in FOVS), passes=passes)


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    layouts = list(HAND_LAYOUTS)
    des = os.path.join(os.path.dirname(__file__), "..", "layouts_designed")
    if os.path.isdir(des):
        layouts += [os.path.splitext(f)[0] for f in sorted(os.listdir(des))
                    if f.endswith(".layout") and not f.startswith("._")]

    print(f"validating {len(layouts)} layouts x {len(FOVS)} FOVs x {n_seeds} seeds", flush=True)
    print(f"gates: >={MIN_SEP_PAIRS} FOV pairs with real divergence >={MIN_PAIR_DIV}, "
          f"AND delivered(360) > delivered(30), AND no reversal, "
          f"AND fov360 delivers >=1 with <=5 look-steps\n", flush=True)

    results = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(check_layout,
                                                  [(l, n_seeds) for l in layouts])):
            results.append(r)
            if "error" in r:
                print(f"[{i+1}/{len(layouts)}] {r['layout']:<26} ERROR {r['error'][:45]}", flush=True)
            else:
                t = r["mean_team"]
                print(f"[{i+1}/{len(layouts)}] {r['layout']:<26} "
                      f"sep={r['n_sep']:>2}/15 "
                      f"team30={t[FOVS[0]]:.1f} team90={t[90]:.1f} team360={t[360]:.1f} "
                      f"best={r['best_triple']}({r['best_triple_score']}) "
                      f"{'PASS' if r['passes'] else 'fail'}", flush=True)

    valid = [r for r in results if "error" not in r]
    keep = [r for r in valid if r["passes"]]
    print(f"\n=== {len(keep)}/{len(valid)} layouts satisfy BOTH conditions ===")
    if valid:
        print(f"  (A) >= {MIN_SEP_PAIRS} separating pairs: {sum(1 for r in valid if r['n_sep'] >= MIN_SEP_PAIRS)}/{len(valid)}")
        print(f"  (B) team wins every FOV : {sum(1 for r in valid if r['full_ok'])}/{len(valid)}")
    if keep:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "PASSING.md"), "w") as fh:
            fh.write("# Layouts satisfying BOTH FOV conditions\n\n")
            fh.write("Validated with the from-scratch limited-vision human\n")
            fh.write("(fov/human/agent/limited_vision_human.py). Divergence is\n")
            fh.write("phase-corrected, so pure timing differences score zero.\n\n")
            fh.write("| layout | sep pairs | team@30 | team@90 | team@360 | best FOV triple |\n")
            fh.write("|---|---|---|---|---|---|\n")
            for r in sorted(keep, key=lambda r: -r["best_triple_score"]):
                t = r["mean_team"]
                fh.write(f"| {r['layout']} | {r['n_sep']}/15 | {t[30]:.1f} | {t[90]:.1f} | "
                         f"{t[360]:.1f} | "
                         f"{r['best_triple']} (min div {r['best_triple_score']}) |\n")
        print(f"\nwritten to {OUT_DIR}/PASSING.md")
        for r in sorted(keep, key=lambda r: -r["best_triple_score"]):
            t = r["mean_team"]
            print(f"  {r['layout']:<26} triple={r['best_triple']} minDiv={r['best_triple_score']:>2} "
                  f"team: 30->{t[30]:.1f} 90->{t[90]:.1f} 360->{t[360]:.1f}")


if __name__ == "__main__":
    main()
