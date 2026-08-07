"""
MISHA NEW CHANGE - measure Bayesian FOV-inference accuracy on the validated
layouts, using the from-scratch human.

For each validated (layout, FOV triple) from fov/layouts_final/README.md, and
each candidate FOV taken as ground truth, run an episode with that human and let
BayesFOVInference watch it online. Reports per-step accuracy, late-episode
accuracy, final-estimate accuracy, and the posterior mass on the true FOV.

Late accuracy is the number that matters: the filter starts from a uniform prior
and needs a few observations before it can commit, so per-step accuracy is
diluted by the unavoidable early phase.

Run with: python -m fov.robot.inference.evaluate_inference [n_workers] [n_seeds]
"""
import multiprocessing as mp
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
from steakhouse.fov.robot.policy.old.inference.bayes_fov import BayesFOVInference

N_STEPS = 260
N_ORDERS = 4

# (layout, FOV triple) from the CARC validation sweep, job 10487033.
FULL_SWEEP = [(l, (30, 60, 90, 120, 180, 360)) for l, _ in [
    ("steak_side_2", None), ("steak_island", None), ("steak_api", None),
    ("steak_test", None), ("steak_none_3", None), ("steak_parrallel", None),
    ("steak_side_3", None)]]

VALIDATED = [
    ("steak_side_2", (30, 90, 180)),
    ("steak_island", (30, 90, 180)),
    ("steak_api", (30, 90, 360)),
    ("steak_test", (30, 60, 90)),
    ("steak_none_3", (30, 90, 360)),
    ("steak_parrallel", (30, 60, 90)),
    ("steak_side_3", (30, 60, 90)),
]


def trial(args):
    layout, triple, true_fov, seed = args
    base = dict(layout=layout, triple=triple, true_fov=true_fov, seed=seed)
    try:
        mdp = SteakHouseGridworld.from_layout_name(layout, start_order_list=['steak'] * N_ORDERS)
        mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    except Exception as e:
        return dict(base, error=f"{type(e).__name__}: {e}")

    np.random.seed(seed)
    random.seed(seed)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=N_STEPS + 10)
    planner = SteakMotionPlanner(mdp, mlp)
    human = LimitedVisionSteakHuman(mdp, true_fov, planner, agent_index=1)
    robot = GreedySteakHumanModel(mlp)
    robot.set_agent_index(0)
    inf = BayesFOVInference(mdp, mlp, triple, human_agent_index=1)

    ests = []
    try:
        for _ in range(N_STEPS):
            s = env.state
            try:
                a_r, _ = robot.action(s)
            except Exception:
                a_r = Action.STAY
            a_h, _ = human.action(s)
            # update BEFORE stepping: shadows must see the state the human acted on
            inf.update(s, a_h)
            ests.append(inf.map_fov())
            _, _, done, _ = env.step((a_r, a_h))
            if done:
                break
    except Exception as e:
        return dict(base, error=f"sim {type(e).__name__}: {e}")

    if not ests:
        return dict(base, error="no steps")
    n = len(ests)
    half = n // 2
    post = inf.posterior()
    return dict(base, steps=n,
                acc=sum(1 for e in ests if e == true_fov) / n,
                late_acc=sum(1 for e in ests[half:] if e == true_fov) / max(1, n - half),
                final_correct=ests[-1] == true_fov,
                p_true=post[true_fov], entropy=inf.entropy(),
                div_rate=inf.n_divergent_steps / max(1, inf.n_steps),
                crash_rate=inf.n_crash_steps / max(1, inf.n_steps))


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    # argv[3]="full" evaluates all 6 FOVs as a single 6-way hypothesis set,
    # which is strictly harder than the recommended 3-way triple.
    table = FULL_SWEEP if (len(sys.argv) > 3 and sys.argv[3] == "full") else VALIDATED
    jobs = [(lay, tri, f, s) for lay, tri in table for f in tri for s in range(n_seeds)]
    print(f"Bayesian FOV inference: {len(VALIDATED)} layouts x 3 FOVs x {n_seeds} seeds "
          f"= {len(jobs)} trials\n", flush=True)

    res = []
    with mp.Pool(n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(trial, jobs)):
            res.append(r)
            if "error" in r:
                print(f"[{i+1}/{len(jobs)}] {r['layout']:<18} fov={r['true_fov']:>3} "
                      f"ERROR {r['error'][:40]}", flush=True)
            else:
                print(f"[{i+1}/{len(jobs)}] {r['layout']:<18} fov={r['true_fov']:>3} "
                      f"s{r['seed']} acc={r['acc']:.2f} late={r['late_acc']:.2f} "
                      f"final={str(r['final_correct']):<5} P(true)={r['p_true']:.3f} "
                      f"H={r['entropy']:.2f} div={r['div_rate']:.2f}", flush=True)

    ok = [r for r in res if "error" not in r]
    if not ok:
        print("\nno valid trials")
        return
    n = len(ok)
    chance = 1.0 / len(ok[0]["triple"])
    print(f"\n=== ACCURACY over {n} trials (chance = {chance:.3f}) ===")
    print(f"  mean per-step accuracy : {sum(r['acc'] for r in ok)/n:.3f}")
    print(f"  mean LATE accuracy     : {sum(r['late_acc'] for r in ok)/n:.3f}")
    print(f"  final-estimate correct : {sum(1 for r in ok if r['final_correct'])/n:.3f}")
    print(f"  mean P(true FOV)       : {sum(r['p_true'] for r in ok)/n:.3f}")
    print(f"  mean entropy           : {sum(r['entropy'] for r in ok)/n:.3f} "
          f"(uniform = {__import__('math').log(len(ok[0]['triple'])):.3f})")
    print(f"  mean shadow divergence : {sum(r['div_rate'] for r in ok)/n:.3f}")
    print(f"  mean crash-skip rate   : {sum(r['crash_rate'] for r in ok)/n:.3f}")

    print("\nper-layout LATE accuracy:")
    by = {}
    for r in ok:
        by.setdefault(r["layout"], []).append(r)
    for lay, rs in sorted(by.items(), key=lambda kv: -sum(x["late_acc"] for x in kv[1]) / len(kv[1])):
        la = sum(x["late_acc"] for x in rs) / len(rs)
        fa = sum(1 for x in rs if x["final_correct"]) / len(rs)
        print(f"  {lay:<18} late={la:.3f} final={fa:.3f} triple={rs[0]['triple']}")

    print("\nper-true-FOV LATE accuracy (is any hypothesis systematically missed?):")
    byf = {}
    for r in ok:
        byf.setdefault(r["true_fov"], []).append(r)
    for f, rs in sorted(byf.items()):
        print(f"  fov={f:>3}: late={sum(x['late_acc'] for x in rs)/len(rs):.3f} "
              f"({len(rs)} trials)")


def math_log3():
    import math
    return math.log(3)


if __name__ == "__main__":
    main()
