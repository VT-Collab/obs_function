"""Evaluate EXACT Bayesian FOV inference against the sampling human.

Run the TRUE human at a known FOV; each tick feed its observed SUBTASK to
SamplingBayesFOVInference (one shadow per candidate FOV, exact known likelihood).
Report how fast/accurately the posterior concentrates on the true FOV.

Metrics (averaged over true FOVs x seeds):
  final_correct : MAP == true at the last step
  late_acc      : fraction of the LAST HALF of steps with MAP == true
  acc           : fraction of ALL steps with MAP == true
  P(true)       : posterior mass on the true FOV (mean over steps; and final)
  entropy       : final posterior entropy (0 = certain; log|cands| = uniform)

Run: python -m fov.robot.inference.evaluate_sampling_inference [layout] [n_seeds]
"""
import os, sys, random, contextlib
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import GreedySteakHumanModel
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman
from fov.human.planning.steak_planner import SteakMotionPlanner
from steakhouse.fov.robot.policy.old.inference.bayes_fov_sampling import SamplingBayesFOVInference

CANDS = [30, 60, 90, 120, 180, 360]
HORIZON, ORDERS = 250, 8
DEVNULL = open(os.devnull, "w")


def run_episode(mdp, mlp, true_fov, cands, seed):
    np.random.seed(seed); random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=HORIZON + 10)
    planner = SteakMotionPlanner(mdp, mlp)
    human = LimitedVisionSteakHuman(mdp, true_fov, planner, agent_index=1)
    robot = GreedySteakHumanModel(mlp); robot.set_agent_index(0)
    filt = SamplingBayesFOVInference(mdp, mlp, cands, human_agent_index=1)

    correct, ptrue = [], []
    with contextlib.redirect_stdout(DEVNULL):
        for _ in range(HORIZON):
            s = env.state
            try: a_r, _ = robot.action(s)
            except Exception: a_r = Action.STAY
            try:
                a_h, info = human.action(s)
                sub = info.get("subtask")
            except Exception:
                a_h, sub = Action.STAY, None
            if sub is not None:
                filt.update(s, sub)
                correct.append(1.0 if filt.map_fov() == true_fov else 0.0)
                ptrue.append(filt.p_true(true_fov))
            _, _, done, _ = env.step((a_r, a_h))
            if done:
                break
    n = len(correct)
    if n == 0:
        return None
    half = n // 2
    return dict(final_correct=correct[-1], acc=float(np.mean(correct)),
                late_acc=float(np.mean(correct[half:])),
                ptrue=float(np.mean(ptrue)), ptrue_final=ptrue[-1],
                entropy=filt.entropy(), informative=filt.n_informative, steps=n)


def main():
    layouts = (sys.argv[1].split(",") if len(sys.argv) > 1
               else ["steak_side_2", "steak_test", "steak_mid_2"])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    import math
    print(f"EXACT sampling-FOV inference | cands={CANDS} (chance={1/len(CANDS):.3f}, "
          f"uniform entropy={math.log(len(CANDS)):.3f})")
    print(f"seeds={n_seeds}  H={HORIZON}  orders={ORDERS}\n")

    grand = []
    for name in layouts:
        try:
            mdp = SteakHouseGridworld.from_layout_name(name, start_order_list=['steak'] * ORDERS)
            mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
        except Exception as e:
            print(f"### {name}: ERROR {type(e).__name__}: {e}\n"); continue
        print(f"### {name}")
        print(f"  {'trueFOV':>7} {'final_ok':>9} {'late_acc':>9} {'acc':>6} "
              f"{'P(true)':>8} {'Pfin':>6} {'entropy':>8} {'steps':>6}")
        layout_rows = []
        for tf in CANDS:
            rs = [run_episode(mdp, mlp, tf, CANDS, s) for s in range(n_seeds)]
            rs = [r for r in rs if r]
            if not rs:
                print(f"  {tf:>7}  (no informative steps)"); continue
            m = lambda k: float(np.mean([r[k] for r in rs]))
            print(f"  {tf:>7} {m('final_correct'):>9.2f} {m('late_acc'):>9.3f} "
                  f"{m('acc'):>6.3f} {m('ptrue'):>8.3f} {m('ptrue_final'):>6.2f} "
                  f"{m('entropy'):>8.3f} {m('steps'):>6.0f}")
            layout_rows.extend(rs); grand.extend(rs)
        if layout_rows:
            print(f"  {'ALL':>7} {np.mean([r['final_correct'] for r in layout_rows]):>9.2f} "
                  f"{np.mean([r['late_acc'] for r in layout_rows]):>9.3f} "
                  f"{np.mean([r['acc'] for r in layout_rows]):>6.3f} "
                  f"{np.mean([r['ptrue'] for r in layout_rows]):>8.3f}")
        print()
    if grand:
        print(f"=== OVERALL: final_correct={np.mean([r['final_correct'] for r in grand]):.3f}  "
              f"late_acc={np.mean([r['late_acc'] for r in grand]):.3f}  "
              f"P(true)={np.mean([r['ptrue'] for r in grand]):.3f} ===")


if __name__ == "__main__":
    main()
