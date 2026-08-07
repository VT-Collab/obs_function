"""Which cost term is doing the work, at which FOV.

    python diagnose.py --layouts steak_gc00 --fovs 30,90,360 --seeds 0-3

For every tick it records, per term, the SPREAD across the 6 candidate actions
(max - min of that term's contribution). A term whose spread is zero cannot
change the decision no matter what weight it is given, so this is the map of
where the module has leverage and where it is silent -- which is what a weight
sweep should be aimed at rather than searched blindly.

It also reports `flip`: how often the pooled argmax differs from the baseline
argmax. That is the module's actual bite; everything else is potential.
"""
import argparse
import collections
import sys

import numpy as np

import _paths  # noqa: F401
from _paths import checkpoint_path, stage_layouts

from baseline import load_policy, BaselineActor
from cost import DEFAULT_WEIGHTS, TERM_NAMES
from env import make_human, HUMAN_INDEX
from human_model import NON_ADVANCING
from policy import BlendedRobotPolicy
from rollout import make_env


def run(layout, fov, seed, lam, weights, cand, horizon, actor, sample=True):
    env = make_env(layout, 4, horizon)
    env.reset()
    human = make_human(env.mdp, fov, seed, agent_index=HUMAN_INDEX)
    pol = BlendedRobotPolicy(actor, env.mdp, cand, lam=lam, weights=weights,
                             rng=np.random.RandomState(10000 + seed),
                             sample=sample)
    pol.reset()

    spread = collections.defaultdict(list)
    defers, flips, ticks = [], 0, 0
    subtask_mass = collections.defaultdict(float)

    while env.t < horizon and not env.mdp.is_terminal(env.state):
        s = env.state
        a, idx = pol.act(env)
        tr = pol.trace[-1]
        if tr["q"] is not None:
            ticks += 1
            terms = pol.module.last.get("terms") or {}
            for k in TERM_NAMES:
                v = terms.get(k)
                if v is not None:
                    spread[k].append(float(np.max(v) - np.min(v)))
            defers.append(tr["defer"] or 0.0)
            if int(np.argmax(tr["p_base"])) != idx:
                flips += 1
        hyps, _hp = pol.hm.predict(s)
        for h in hyps:
            subtask_mass[h["subtask"]] += h["prob"]
        h_act, _info = human.action(s)
        pol.observe_human(s, h_act)
        _sp, done, _t = env.step(a, h_act)
        if done:
            break

    n = max(ticks, 1)
    tot = sum(subtask_mass.values()) or 1.0
    return {
        "spread": {k: float(np.mean(v)) if v else 0.0
                   for k, v in spread.items()},
        "active": {k: float(np.mean([x > 1e-9 for x in v])) if v else 0.0
                   for k, v in spread.items()},
        "defer": float(np.mean(defers)) if defers else 0.0,
        "flip": flips / float(n),
        "ticks": ticks,
        "nonadv": sum(v for k, v in subtask_mass.items()
                      if k in NON_ADVANCING) / tot,
        "top_subtasks": sorted(((v / tot, k) for k, v in subtask_mass.items()),
                               reverse=True)[:4],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=str, default="steak_gc00")
    ap.add_argument("--fovs", type=str, default="30,90,180,360")
    ap.add_argument("--seeds", type=str, default="0-3")
    ap.add_argument("--candidate_fovs", type=str, default="30,60,90,120,180,360")
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--horizon", type=int, default=400)
    args = ap.parse_args(argv)
    stage_layouts()

    a, b = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(a), int(b) + 1)) if b else [int(a)]
    fovs = [int(x) for x in args.fovs.split(",")]
    cand = [int(x) for x in args.candidate_fovs.split(",")]

    for layout in args.layouts.split(","):
        env0 = make_env(layout, 4, args.horizon)
        net, net_args = load_policy(layout, env0.obs_shape,
                                    checkpoint_path(layout))
        print("\n=== %s ===" % layout)
        hdr = "%5s %6s %6s %6s  " % ("fov", "defer", "nonadv", "flip")
        hdr += " ".join("%14s" % k for k in TERM_NAMES)
        print(hdr)
        for fov in fovs:
            acc = collections.defaultdict(list)
            for sd in seeds:
                r = run(layout, fov, sd, args.lam, dict(DEFAULT_WEIGHTS), cand,
                        args.horizon, BaselineActor(net, net_args))
                for k in TERM_NAMES:
                    acc[k].append(r["spread"].get(k, 0.0))
                    acc["act_" + k].append(r["active"].get(k, 0.0))
                acc["defer"].append(r["defer"])
                acc["flip"].append(r["flip"])
                acc["nonadv"].append(r["nonadv"])
            line = "%5d %6.2f %6.2f %6.2f  " % (
                fov, np.mean(acc["defer"]), np.mean(acc["nonadv"]),
                np.mean(acc["flip"]))
            line += " ".join("%6.3f/%5.2f" % (np.mean(acc[k]),
                                              np.mean(acc["act_" + k]))
                             for k in TERM_NAMES)
            print(line)
        print("  (each cell: mean SPREAD over the 6 actions / fraction of "
              "ticks the term is non-zero)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
