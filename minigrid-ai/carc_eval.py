"""Evaluate one trained baseline checkpoint on the eval_three_way harness.

Prints ONE tab-separated line so an array job can collate results:
  name  mode  succ_all  succ60  succ120  succ180  adj_all  reveals

Both action-selection modes are reported, because they are different policies:
  greedy   argmax  - what compare_module.BaselineRobot uses
  sampled  sample  - what the stochastic policy actually is

  python carc_eval.py <ckpt.pt> [--seeds 25]
"""
from __future__ import annotations
import argparse, os, statistics as stt, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import robot.policy.deterministic.eval_three_way as e3w
from robot.policy.deterministic.eval_three_way import run_episode
from robot.policy.deterministic.no_assist import NoAssist
from robot.policy.neural.baseline.no_fov.actor_critic import NoFovRecAC
from robot.policy.neural.baseline.no_fov.features import ACTIONS, REVEAL_KEYS, encode_state
from robot.policy.neural.module.candidates import CandidateFinder

e3w.RANDOM_WALLS = True
e3w.MAX_STEPS = 190
FOVS = [60, 120, 180]


class Robot:
    def __init__(self, ckpt, greedy=True):
        self.net = NoFovRecAC()
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net.eval()
        self.greedy = greedy

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
            a = int(dist.probs.argmax()) if self.greedy else int(dist.sample())
        kind = ACTIONS[a]
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
        hc = getattr(getattr(state, "carrying", None), "color", None)
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


def run(make, seeds):
    torch.manual_seed(0)
    per = []
    for f in FOVS:
        r = make()
        for s in seeds:
            per.append((f, run_episode(s, f, r)))
    return per


def stats(per, seeds):
    def sc(f):
        return 100.0 * sum(1 for ff, x in per if ff == f and x["success"]) / len(seeds)
    allsucc = 100.0 * sum(1 for _, x in per if x["success"]) / len(per)
    adj = stt.mean([x["adjusted_steps"] for _, x in per])
    rev = stt.mean([x["n_assists"] for _, x in per])
    return allsucc, sc(60), sc(120), sc(180), adj, rev


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--seeds", type=int, default=25)
    p.add_argument("--with-noassist", action="store_true")
    a = p.parse_args()
    seeds = list(range(a.seeds))
    name = os.path.splitext(os.path.basename(a.ckpt))[0]

    if a.with_noassist:
        s = stats(run(lambda: NoAssist(), seeds), seeds)
        print("no-assist\t-\t" + "\t".join(f"{v:.2f}" for v in s), flush=True)

    for mode, greedy in (("greedy", True), ("sampled", False)):
        s = stats(run(lambda: Robot(a.ckpt, greedy=greedy), seeds), seeds)
        print(f"{name}\t{mode}\t" + "\t".join(f"{v:.2f}" for v in s), flush=True)
