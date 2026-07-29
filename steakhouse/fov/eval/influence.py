"""ROBOT-IN-VIEW INFLUENCE: does SEEING the robot change what the human does?

The FOV-aware robot's only lever (action-only domain) is being in/out of the
human's cone, so this is THE metric. At each tick the robot is visible, we compare
the human's policy WITH the sighting vs the counterfactual WITHOUT it (clear the
robot belief + seen-pose, recompute subtask_distribution). Reports:

  seen_frac      : fraction of ticks the robot is in the human's cone (grows w/ FOV)
  flip_on_seen   : of ticks the robot is seen, fraction where seeing it CHANGES the
                   human's top subtask (the decision would differ if blind)
  flip_on_choice : same but restricted to ticks where the human is actually making
                   a free sampling CHOICE (excludes forced held-object / look ticks)
  tv_on_seen     : mean total-variation shift of the subtask distribution when seen
  influence/ep   : seen_frac * flip_on_seen = fraction of ALL ticks a sighting flips

Usage: influence.py <layout> [seeds=6] [N=12] [H=300]
"""
import sys, os, json, random, contextlib
import numpy as np
sys.path.insert(0, "/Users/mishafu/Desktop/obs_function/steakhouse")
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import GreedySteakHumanModel
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman, SAMPLING_SUBTASKS
from fov.human.planning.steak_planner import SteakMotionPlanner

FOVS = [30, 60, 90, 120, 180, 360]
DEV = open(os.devnull, "w")


def top(d):
    return max(d, key=d.get) if d else None


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def episode(mdp, mlp, fov, seed, N, H, suppress=None):
    np.random.seed(seed); random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=H + 10)
    human = LimitedVisionSteakHuman(mdp, fov, SteakMotionPlanner(mdp, mlp), agent_index=1)
    if suppress is not None:
        human.ROBOT_SUPPRESS = suppress
    robot = GreedySteakHumanModel(mlp); robot.set_agent_index(0)
    n0 = len(env.state.order_list) if env.state.order_list else 0
    ticks = seen = flips = tv_sum = 0
    choice_ticks = choice_flips = 0
    with contextlib.redirect_stdout(DEV):
        for _ in range(H):
            s = env.state
            try: ar, _ = robot.action(s)
            except Exception: ar = Action.STAY
            human.observe(s)                                   # set this tick's beliefs
            rb = human.beliefs.get(human.ROBOT)
            robot_seen = rb is not None and rb.seen_at == human.t
            ticks += 1
            if robot_seen:
                seen += 1
                d1 = human.subtask_distribution(s, human._sampled)
                saved_rb = human.beliefs.pop(human.ROBOT, None); saved_rs = human._robot_seen
                human._robot_seen = None
                d0 = human.subtask_distribution(s, human._sampled)  # counterfactual: blind to robot
                if saved_rb is not None: human.beliefs[human.ROBOT] = saved_rb
                human._robot_seen = saved_rs
                flipped = top(d1) != top(d0)
                flips += 1 if flipped else 0
                tv_sum += tv(d1, d0)
                # is this a genuine free CHOICE (support has >1 sampling option)?
                if len(set(d1) & SAMPLING_SUBTASKS) > 1 or len(set(d0) & SAMPLING_SUBTASKS) > 1:
                    choice_ticks += 1
                    choice_flips += 1 if flipped else 0
            try: ah, _ = human.action(s)
            except Exception: ah = Action.STAY
            _, _, d, _ = env.step((ar, ah))
            if d or (env.state.order_list is not None and len(env.state.order_list) == 0):
                break
    left = len(env.state.order_list) if env.state.order_list else 0
    return dict(ticks=ticks, seen=seen, flips=flips, tv=tv_sum,
                cticks=choice_ticks, cflips=choice_flips, team=n0 - left)


def main():
    layout = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    try:
        mdp = SteakHouseGridworld.from_layout_name(layout, start_order_list=['steak'] * N)
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    except Exception as e:
        print("JSON " + json.dumps({"layout": layout, "error": str(e)})); return
    suppress = float(sys.argv[5]) if len(sys.argv) > 5 else None
    per = {}
    for f in FOVS:
        rs = [episode(mdp, mlp, f, s, N, H, suppress) for s in range(seeds)]
        T = sum(r["ticks"] for r in rs); S = sum(r["seen"] for r in rs)
        F = sum(r["flips"] for r in rs); TVs = sum(r["tv"] for r in rs)
        C = sum(r["cticks"] for r in rs); CF = sum(r["cflips"] for r in rs)
        per[str(f)] = {
            "seen_frac": round(S / max(1, T), 2),
            "flip_on_seen": round(F / max(1, S), 2),
            "flip_on_choice": round(CF / max(1, C), 2),
            "influence_per_ep": round(F / max(1, T), 3),
            "team": round(float(np.mean([r["team"] for r in rs])), 1),
        }
    print("JSON " + json.dumps({"layout": layout, "suppress": suppress, "per_fov": per}))


if __name__ == "__main__":
    main()
