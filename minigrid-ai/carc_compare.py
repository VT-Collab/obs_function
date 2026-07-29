"""Final comparison: no-assist / baseline / module / module(mute) / static_120 / dynamic.

All six conditions on IDENTICAL layouts (run_episode rebuilds the env from
seed+fov), reporting success rate, adjusted steps, raw steps and reveals per FOV.

  python carc_compare.py --ckpt <baseline.pt> --seeds 50
"""
from __future__ import annotations
import argparse, os, statistics as stt, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import robot.policy.deterministic.eval_three_way as e3w
from robot.policy.deterministic.eval_three_way import run_episode
from robot.policy.deterministic.no_assist import NoAssist
from robot.policy.deterministic.static_assist import StaticAssist
from robot.policy.deterministic.dynamic_assist import DynamicAssist
from robot.policy.neural.module.fov_module import FovModule
from robot.policy.neural.module.compare_module import BaselineRobot

e3w.RANDOM_WALLS = True
e3w.MAX_STEPS = 190
FOVS = [60, 120, 180]


def run(make, seeds):
    torch.manual_seed(0)
    per = []
    for f in FOVS:
        r = make()
        for s in seeds:
            per.append((f, run_episode(s, f, r)))
    return per


def agg(per, f, key, pct=False):
    xs = [r for ff, r in per if ff == f]
    if pct:
        return 100.0 * sum(1 for r in xs if r[key]) / len(xs)
    return stt.mean([r[key] for r in xs])


def overall(per, key, pct=False):
    if pct:
        return 100.0 * sum(1 for _, r in per if r[key]) / len(per)
    return stt.mean([r[key] for _, r in per])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--comm-cost", type=float, default=0.005)
    p.add_argument("--conf-switch", type=float, default=0.25)
    a = p.parse_args()
    seeds = list(range(a.seeds))

    conds = [
        ("no-assist",    lambda: NoAssist()),
        ("baseline",     lambda: BaselineRobot(a.ckpt)),
        ("static_120",   lambda: StaticAssist(patience=1)),
        ("dynamic",      lambda: DynamicAssist(patience=1)),
        ("module",       lambda: FovModule(a.ckpt, method="rec_ppo",
                                           comm_cost=a.comm_cost,
                                           conf_switch=a.conf_switch)),
        ("module(mute)", lambda: FovModule(a.ckpt, method="mute",
                                           comm_cost=a.comm_cost,
                                           conf_switch=a.conf_switch)),
    ]

    res = {}
    for name, mk in conds:
        res[name] = run(mk, seeds)
        print(f"  {name} done", flush=True)

    n = a.seeds * len(FOVS)
    W = 15 + 8 * len(FOVS) + 9

    def block(title, key, pct=False, sub=""):
        print("\n" + "=" * W)
        print(f"{title}" + (f"   ({sub})" if sub else ""))
        print("=" * W)
        print(f"{'condition':<15}" + "".join(f"{'fov'+str(f):>8}" for f in FOVS) + f"{'ALL':>9}")
        print("-" * W)
        for name, _ in conds:
            per = res[name]
            print(f"{name:<15}" + "".join(f"{agg(per,f,key,pct):>8.1f}" for f in FOVS)
                  + f"{overall(per,key,pct):>9.1f}")

    block("SUCCESS RATE  %", "success", True,
          f"n={a.seeds} seeds x {len(FOVS)} FOV = {n} episodes per condition")
    block("ADJUSTED STEPS  (steps + reveals; lower better)", "adjusted_steps", False,
          "truncation counts as 190")
    block("RAW STEPS", "steps", False)
    block("REVEALS PER EPISODE", "n_assists", False)

    print("\n" + "=" * W)
    print("DELTAS vs baseline")
    print("=" * W)
    b = res["baseline"]
    for name in ("module", "module(mute)", "static_120", "dynamic"):
        m = res[name]
        print(f"{name:<15} success {overall(m,'success',True)-overall(b,'success',True):+6.1f} pts | "
              f"adjsteps {overall(m,'adjusted_steps')-overall(b,'adjusted_steps'):+7.1f} | "
              f"reveals {overall(m,'n_assists'):6.1f} vs {overall(b,'n_assists'):.1f}")


if __name__ == "__main__":
    main()
