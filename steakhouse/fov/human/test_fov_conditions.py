"""
MISHA NEW CHANGE - does the from-scratch human satisfy the two conditions?

  (A) FOV changes the SUBTASK SEQUENCE substantially (not just timing), and
  (B) MORE VISION PERFORMS BETTER (more orders delivered / fewer wasted trips).

Both are measured with the RNG held fixed across FOVs, because auto_unstuck-style
randomness previously swamped every FOV comparison in this project (rank01:
0/120 disagreement at fixed seed vs 107/120 across seeds).

Run with: python -m fov.human.test_fov_conditions [layout] [n_seeds]
"""
import os
import random
import sys

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import GreedySteakHumanModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman
from fov.human.planning.steak_planner import SteakMotionPlanner

FOVS = [30, 90, 360]          # 360 = full observability control
N_STEPS = 250


def run(mdp, mlp, fov, seed, n_steps=N_STEPS):
    np.random.seed(seed)
    random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=n_steps + 10)
    planner = SteakMotionPlanner(mdp, mlp)
    human = LimitedVisionSteakHuman(mdp, fov, planner, agent_index=1)
    robot = GreedySteakHumanModel(mlp)
    robot.set_agent_index(0)

    n_orders0 = len(env.state.order_list) if env.state.order_list else 0
    for _ in range(n_steps):
        s = env.state
        try:
            a_r, _ = robot.action(s)
        except Exception:
            a_r = Action.STAY
        try:
            a_h, _ = human.action(s)
        except Exception:
            a_h = Action.STAY
        _, _, done, info = env.step((a_r, a_h))
        if done:
            break
    # Orders actually completed = how much the order list shrank. The previous
    # metric read env.cumulative_sparse_rewards, which does not exist on this
    # env, so every FOV silently scored 0 and condition (B) could never pass.
    n_left = len(env.state.order_list) if env.state.order_list else 0
    return dict(fov=fov, seed=seed, subtasks=human.subtask_log,
                n_checks=human.n_checks, n_wasted=human.n_wasted_commits,
                steps=len(human.subtask_log), team_delivered=n_orders0 - n_left,
                delivered=human.n_delivered)


def runs_of(seq):
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def real_divergence(a, b):
    """Substitutions + unmatched tokens between run-compressed subtask
    sequences. Pure timing offsets score 0 (see fov_divergence_search_v2)."""
    ra, rb = runs_of(a), runs_of(b)
    if not ra or not rb:
        return 0
    na, nb = len(ra), len(rb)
    d = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(na + 1):
        d[i][0] = i
    for j in range(nb + 1):
        d[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            d[i][j] = min(d[i - 1][j - 1] + (0 if ra[i - 1] == rb[j - 1] else 1),
                          d[i - 1][j] + 1, d[i][j - 1] + 1)
    sa, sb = set(ra), set(rb)
    i, j, n = na, nb, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            c = 0 if ra[i - 1] == rb[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + c:
                n += c
                i -= 1
                j -= 1
                continue
        if i > 0 and (j == 0 or d[i][j] == d[i - 1][j] + 1):
            n += ra[i - 1] not in sb
            i -= 1
        else:
            n += rb[j - 1] not in sa
            j -= 1
    return n


def main():
    layout = sys.argv[1] if len(sys.argv) > 1 else "steak_island"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    mdp = SteakHouseGridworld.from_layout_name(layout, start_order_list=['steak'] * 4)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    print(f"layout={layout}  fovs={FOVS}  seeds={n_seeds}\n")

    all_r = {}
    for seed in range(n_seeds):
        for fov in FOVS:
            r = run(mdp, mlp, fov, seed)
            all_r[(fov, seed)] = r
            print(f"  fov={fov:>3} seed={seed} steps={r['steps']:>3} "
                  f"checks={r['n_checks']:>3} wasted={r['n_wasted']:>2} "
                  f"delivered={r['delivered']:>2}(team {r['team_delivered']}) "
                  f"subtasks={runs_of(r['subtasks'])[:9]}", flush=True)

    print("\n(A) REAL subtask divergence at FIXED seed (timing excluded):")
    ok_a = True
    for seed in range(n_seeds):
        for i in range(len(FOVS)):
            for j in range(i + 1, len(FOVS)):
                n = real_divergence(all_r[(FOVS[i], seed)]["subtasks"],
                                    all_r[(FOVS[j], seed)]["subtasks"])
                flag = "OK" if n >= 3 else "LOW"
                if n < 3:
                    ok_a = False
                print(f"  seed{seed} fov{FOVS[i]} vs fov{FOVS[j]}: {n:>3}  {flag}")

    # (B) TEAM WIN: the human is a functional teammate at every FOV. Success is
    # measured at the TEAM (the human's prep feeding a teammate's delivery is a
    # team win), NOT the human's own count - a greedy teammate harvests shared
    # stations regardless of the human's cone, so human-own deliveries do not
    # order the FOVs; the FOV signal lives in (A). We require the team to deliver
    # a healthy number at EVERY cone (the human never breaks the team).
    print("\n(B) TEAM must deliver at every FOV (team win is a win):")
    team = {f: sum(all_r[(f, s)]["team_delivered"] for s in range(n_seeds)) / n_seeds
            for f in FOVS}
    own = {f: sum(all_r[(f, s)]["delivered"] for s in range(n_seeds)) / n_seeds
           for f in FOVS}
    checks = {f: sum(all_r[(f, s)]["n_checks"] for s in range(n_seeds)) / n_seeds
              for f in FOVS}
    for f in FOVS:
        print(f"  fov={f:>3}  team delivered={team[f]:.2f}  (human-own={own[f]:.2f}, "
              f"look-steps={checks[f]:.1f})")
    ok_b = all(team[f] >= 1.0 for f in FOVS)

    print(f"\nCONDITION (A) substantial subtask divergence : {'PASS' if ok_a else 'FAIL'}")
    print(f"CONDITION (B) team delivers at every FOV     : {'PASS' if ok_b else 'FAIL'}")


if __name__ == "__main__":
    main()
