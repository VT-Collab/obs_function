"""Does the human's reaction to the robot have PERFORMANCE VALUE? Compare the
finalized (robot-aware) human vs a robot-BLIND human (ignores the teammate) on
the SAME episodes:
  team    : team throughput (deliveries / fixed H, large order list)
  wasted  : human's redundant trips (n_wasted_commits: fetch into an occupied
            station) - the concrete redundancy the reaction is meant to cut
  aband   : commitments abandoned on observing them pointless
If aware CUTS wasted work AND/OR lifts team over blind, the influence has positive
performance value even against the (reactive) greedy partner. If team is ~flat,
the influence is behaviourally real but its throughput payoff needs the AWARE ROBOT.
Usage: perf_value.py <layout> [seeds=6] [N=14] [H=300]
"""
import sys, os, json, random, contextlib
import numpy as np
sys.path.insert(0, "/Users/mishafu/Desktop/obs_function/steakhouse")
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import GreedySteakHumanModel
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman
from fov.human.planning.steak_planner import SteakMotionPlanner

FOVS = [30, 60, 90, 120, 180, 360]
DEV = open(os.devnull, "w")


class Blind(LimitedVisionSteakHuman):
    def observe(self, state):
        super().observe(state)
        self.beliefs.pop(self.ROBOT, None)
        self._robot_seen = None


def ep(mdp, mlp, Cls, fov, seed, N, H):
    np.random.seed(seed); random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=H + 10)
    human = Cls(mdp, fov, SteakMotionPlanner(mdp, mlp), agent_index=1)
    robot = GreedySteakHumanModel(mlp); robot.set_agent_index(0)
    n0 = len(env.state.order_list) if env.state.order_list else 0
    with contextlib.redirect_stdout(DEV):
        for _ in range(H):
            s = env.state
            try: ar, _ = robot.action(s)
            except Exception: ar = Action.STAY
            try: ah, _ = human.action(s)
            except Exception: ah = Action.STAY
            _, _, d, _ = env.step((ar, ah))
            if d or (env.state.order_list is not None and len(env.state.order_list) == 0):
                break
    left = len(env.state.order_list) if env.state.order_list else 0
    return n0 - left, human.n_wasted_commits, human.n_abandoned


def main():
    layout = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    try:
        mdp = SteakHouseGridworld.from_layout_name(layout, start_order_list=['steak'] * N)
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    except Exception as e:
        print("JSON " + json.dumps({"layout": layout, "error": str(e)[:100]})); return
    per = {}
    for f in FOVS:
        aw = [ep(mdp, mlp, LimitedVisionSteakHuman, f, s, N, H) for s in range(seeds)]
        bl = [ep(mdp, mlp, Blind, f, s, N, H) for s in range(seeds)]
        per[str(f)] = {
            "team_aware": round(float(np.mean([a[0] for a in aw])), 1),
            "team_blind": round(float(np.mean([b[0] for b in bl])), 1),
            "wasted_aware": round(float(np.mean([a[1] for a in aw])), 1),
            "wasted_blind": round(float(np.mean([b[1] for b in bl])), 1),
        }
    ta = float(np.mean([per[str(f)]["team_aware"] for f in FOVS]))
    tb = float(np.mean([per[str(f)]["team_blind"] for f in FOVS]))
    wa = float(np.mean([per[str(f)]["wasted_aware"] for f in FOVS]))
    wb = float(np.mean([per[str(f)]["wasted_blind"] for f in FOVS]))
    print("JSON " + json.dumps({
        "layout": layout, "per_fov": per,
        "team_aware_mean": round(ta, 2), "team_blind_mean": round(tb, 2),
        "team_gain_pct": round((ta - tb) / (tb or 1) * 100, 0),
        "wasted_aware_mean": round(wa, 1), "wasted_blind_mean": round(wb, 1),
        "wasted_cut_pct": round((wb - wa) / (wb or 1) * 100, 0),
    }))


if __name__ == "__main__":
    main()
