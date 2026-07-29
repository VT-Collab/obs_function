"""FULL per-layout test battery for the finalized human. One JSON verdict:
  team_ok  : team delivers >=1 at every FOV (team-win)
  P1       : cross-FOV subtask divergence (min over key pairs) -> P1_ok if >=3
  P2_ok    : blind spots grow as FOV narrows (blind@30 > blind@360)
  nocheat  : illegal belief writes (belief for a non-visible tile) -> must be 0
  infer    : exact Bayesian FOV final-correct (>=0.5 => well above 1/6 chance)
  INFL     : aware-vs-robot-blind subtask divergence = how much SEEING the robot
             changes behaviour (the collaboration lever). Reported + flagged high.
  PASS     : team_ok & P1_ok & P2_ok & nocheat_ok & infer_ok
Usage: fulltest_layout.py <layout> [seeds=4] [infer_seeds=3] [N=12] [H=300]
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
from fov.human.test_fov_conditions import real_divergence
from fov.robot.inference.evaluate_sampling_inference import run_episode as infer_ep

FOVS = [30, 60, 90, 120, 180, 360]
PREF = ("ready", "cooking", "chopping", "washing", "occupied", "empty")
DEV = open(os.devnull, "w")


class Blind(LimitedVisionSteakHuman):
    """robot-blind control: ignores the teammate entirely."""
    def observe(self, state):
        super().observe(state)
        self.beliefs.pop(self.ROBOT, None)
        self._robot_seen = None


def tstate(h, s, locs):
    vals = [h._true_state_of(s, l) for l in locs]
    for p in PREF:
        if p in vals:
            return p
    return "unknown"


def run(mdp, mlp, Cls, fov, seed, ap, N, H):
    np.random.seed(seed); random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=H + 10)
    human = Cls(mdp, fov, SteakMotionPlanner(mdp, mlp), agent_index=1)
    robot = GreedySteakHumanModel(mlp); robot.set_agent_index(0)
    n0 = len(env.state.order_list) if env.state.order_list else 0
    blind = illegal = ticks = 0
    with contextlib.redirect_stdout(DEV):
        for _ in range(H):
            s = env.state
            try: ar, _ = robot.action(s)
            except Exception: ar = Action.STAY
            human.observe(s)
            for loc, bel in human.beliefs.items():          # no-cheat: fresh belief => tile visible
                if isinstance(loc, tuple) and bel.seen_at == human.t and not human.visible(s, loc):
                    illegal += 1
            tp = tstate(human, s, ap["pot"]); tb = tstate(human, s, ap["board"]); ts = tstate(human, s, ap["sink"])
            blind += len(set(human._available_advancing(tp, tb, ts)) -
                         set(human._available_advancing(human.believed("pot"), human.believed("board"), human.believed("sink"))))
            ticks += 1
            try: ah, _ = human.action(s)
            except Exception: ah = Action.STAY
            _, _, d, _ = env.step((ar, ah))
            if d or (env.state.order_list is not None and len(env.state.order_list) == 0):
                break
    left = len(env.state.order_list) if env.state.order_list else 0
    return human.subtask_log, n0 - left, blind / max(1, ticks), illegal


def main():
    layout = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    iseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    H = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    try:
        mdp = SteakHouseGridworld.from_layout_name(layout, start_order_list=['steak'] * N)
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    except Exception as e:
        print("JSON " + json.dumps({"layout": layout, "error": str(e)[:120]})); return
    tpd = mdp.terrain_pos_dict
    ap = {"pot": tpd.get("P", []), "board": tpd.get("B", []), "sink": tpd.get("W", [])}
    logs, team, blind, illegal, infl = {}, {}, {}, 0, {}
    for f in FOVS:
        aw = [run(mdp, mlp, LimitedVisionSteakHuman, f, s, ap, N, H) for s in range(seeds)]
        bl = [run(mdp, mlp, Blind, f, s, ap, N, H) for s in range(seeds)]
        logs[f] = [a[0] for a in aw]
        team[f] = float(np.mean([a[1] for a in aw]))
        blind[f] = float(np.mean([a[2] for a in aw]))
        illegal += sum(a[3] for a in aw)
        infl[f] = float(np.mean([real_divergence(aw[s][0], bl[s][0]) for s in range(seeds)]))
    p1 = min(min(real_divergence(logs[a][i], logs[b][i]) for i in range(seeds))
             for a, b in [(30, 90), (90, 360), (30, 360)])
    inf = []
    with contextlib.redirect_stdout(DEV):
        for f in FOVS:
            for s in range(iseeds):
                r = infer_ep(mdp, mlp, f, FOVS, s)
                if r: inf.append(r["final_correct"])
    infer = float(np.mean(inf)) if inf else 0.0
    infl_wide = (infl[180] + infl[360]) / 2
    verdict = {
        "layout": layout,
        "team_ok": bool(all(team[f] >= 1.0 for f in FOVS)),
        "team_min": round(min(team.values()), 1),
        "P1": int(p1), "P1_ok": bool(p1 >= 3),
        "P2_ok": bool(blind[30] > blind[360]), "blind30": round(blind[30], 2), "blind360": round(blind[360], 2),
        "nocheat_ok": bool(illegal == 0), "illegal": int(illegal),
        "infer": round(infer, 3), "infer_ok": bool(infer >= 0.5),
        "INFL_mean": round(float(np.mean(list(infl.values()))), 1),
        "INFL_wide": round(infl_wide, 1),
        "influential": bool(infl_wide >= 8.0),
        "team_by_fov": {str(f): round(team[f], 1) for f in FOVS},
    }
    verdict["PASS"] = bool(verdict["team_ok"] and verdict["P1_ok"] and verdict["P2_ok"]
                           and verdict["nocheat_ok"] and verdict["infer_ok"])
    print("JSON " + json.dumps(verdict))


if __name__ == "__main__":
    main()
