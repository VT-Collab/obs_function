# ═══════════════════════════════════════════════════════════════════════════
# robot/policy/neural/module/ - reproduces the baseline-vs-module comparison.
#
# Runs the frozen no_fov rec_ppo baseline and the FOV module through the SAME
# eval_three_way harness (identical layouts per seed+FOV), broken down by the
# human's true FOV, reporting success % and adjusted steps (steps + reveals).
#
#   python -m robot.policy.neural.module.compare_module              # 50 seeds
#   python -m robot.policy.neural.module.compare_module --seeds 100  # tighter
#   python -m robot.policy.neural.module.compare_module --seeds 50
#
# Self-contained: no /tmp, no scratchpad. Uses the checkpoint saved alongside
# this module (checkpoints/rec_ppo_baseline.pt). The module's decision knobs are
# its own defaults (conf_switch=0.25, gate=0.5, beta=0.5); see FovModule.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse, os, statistics as st, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np
import torch

import robot.policy.deterministic.eval_three_way as e3w
from robot.policy.deterministic.eval_three_way import run_episode
from robot.policy.neural.baseline.no_fov.features import ACTIONS, REVEAL_KEYS, encode_state
from robot.policy.neural.baseline.no_fov.actor_critic import NoFovRecAC
from robot.policy.neural.module.candidates import CandidateFinder
from robot.policy.neural.module.fov_module import FovModule

# the env the baseline was trained in - must match or the policy sees OOD input.
e3w.RANDOM_WALLS = True
e3w.MAX_STEPS = 190

CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "rec_ppo_baseline.pt")


class BaselineRobot:
    """Frozen no_fov rec_ppo behind the eval_three_way robot interface. Nearest
    key, no FOV reasoning - exactly the baseline the module wraps."""

    def __init__(self, ckpt=CKPT):
        self.net = NoFovRecAC()
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net.eval()

    def reset(self, state):
        self.finder = CandidateFinder(patience=1); self.finder.reset(state)
        self.told = {k: False for k in REVEAL_KEYS}
        self.n_assists = 0
        self.memory = torch.zeros(1, self.net.memory_size)

    def observe(self, state, action):
        pass

    def step(self, state, human_kb):
        obs = encode_state(state, self.told, e3w.MAX_STEPS)
        with torch.no_grad():
            dist, _, self.memory = self.net(torch.tensor(obs).unsqueeze(0), self.memory)
            kind = ACTIONS[int(dist.probs.argmax())]
        if kind == "wait":
            return None
        t = self._resolve(kind, state)
        if t is None:
            return None
        c, loc = t
        self.finder._write_reveal(human_kb, (kind, c, loc), state)
        self.told[kind] = True
        self.n_assists += 1
        return (kind, c, loc)

    def _resolve(self, kind, state):
        sol = self.finder.solution
        ax, ay = state.agent_pos
        car = getattr(state, "carrying", None)
        hc = getattr(car, "color", None)
        if kind == "key":
            live = [(c, l) for c, l in sol["keys"].items()
                    if getattr(state.grid.get(*l), "type", None) == "key"]
            return min(live, key=lambda c: abs(c[1][0]-ax)+abs(c[1][1]-ay)) if live else None
        if kind == "door":
            l = sol["doors"].get(hc); return (hc, l) if l else None
        if kind == "goal":
            l = sol.get("goal"); return (None, l) if l else None
        if kind == "dead_room":
            return (hc, (ax, ay)) if hc else None
        if kind == "empty_room":
            return (None, (ax, ay))
        return None


def evaluate(make_robot, seeds, fovs):
    torch.manual_seed(0)          # policies act greedily, but pin RNG for exactness
    per = []
    for fov in fovs:
        robot = make_robot()
        for s in seeds:
            per.append((fov, run_episode(s, fov, robot)))
    return per


def succ(per, fov):
    x = [r for f, r in per if f == fov]
    return 100.0 * sum(1 for r in x if r["success"]) / len(x)


def adj(per, fov):
    return st.mean([r["adjusted_steps"] for f, r in per if f == fov])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--fovs", type=int, nargs="+", default=[60, 120, 180])
    p.add_argument("--conf-switch", type=float, default=0.25)
    a = p.parse_args()
    seeds = list(range(a.seeds))

    print(f"baseline: {os.path.relpath(CKPT)}   module conf_switch={a.conf_switch}", flush=True)
    bp = evaluate(lambda: BaselineRobot(), seeds, a.fovs); print("  baseline done", flush=True)
    mp = evaluate(lambda: FovModule(CKPT, method="rec_ppo", comm_cost=0.02,
                                    conf_switch=a.conf_switch),
                  seeds, a.fovs); print("  module done", flush=True)

    fs = a.fovs
    print("\n" + "=" * 72)
    print(f"{'':<10}" + "".join(f"{'f'+str(f):>7}" for f in fs) + f"{'ALL':>7}   "
          + "".join(f"{'adj'+str(f):>8}" for f in fs) + f"{'rev':>6}")
    print("-" * 72)
    for name, per in (("baseline", bp), ("module", mp)):
        alls = 100.0 * sum(1 for _, r in per if r["success"]) / len(per)
        rev = st.mean([r["n_assists"] for _, r in per])
        print(f"{name:<10}" + "".join(f"{succ(per,f):>7.1f}" for f in fs) + f"{alls:>7.1f}   "
              + "".join(f"{adj(per,f):>8.1f}" for f in fs) + f"{rev:>6.1f}")
    print("\ndelta module - baseline (success %):  "
          + "  ".join(f"f{f}: {succ(mp,f)-succ(bp,f):+.1f}" for f in fs))
    da = (100.0*sum(1 for _,r in mp if r["success"])/len(mp)
          - 100.0*sum(1 for _,r in bp if r["success"])/len(bp))
    print(f"delta overall: {da:+.1f}   "
          f"reveals: {st.mean([r['n_assists'] for _,r in mp]):.1f} vs "
          f"{st.mean([r['n_assists'] for _,r in bp]):.1f}")
    print(f"n = {a.seeds} seeds x {len(fs)} FOV = {a.seeds*len(fs)} episodes/condition, "
          f"identical layouts.")


if __name__ == "__main__":
    main()
