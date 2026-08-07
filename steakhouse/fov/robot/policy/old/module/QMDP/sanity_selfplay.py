"""HARNESS SANITY CHECK — the one test that validates every other number here.

The claim in RESULTS.md §6b is strong: paired with the limited-vision human,
the trained self-play robot is beaten by a near-random one. Before believing
that, you have to rule out the boring explanation -- that this harness loads or
drives the policy wrongly, which would cripple the trained agent while leaving a
random agent untouched.

So: run the SAME checkpoint through the SAME env, the SAME observation builder
and the SAME action path, but with BOTH chefs driven by the policy -- i.e. the
exact condition it was trained in. `CARC_RUNS.md` §4 says these checkpoints
score 50-60 out of a perfect 60 there (5 of them a flat 60/60 under argmax).

    if self-play here reproduces ~60      -> harness is sound, the ZSC gap is real
    if self-play here is much worse       -> the harness is broken, ignore §6b

It also measures a TRUE uniform-random robot (numpy, 1/6 each) next to the
`--base_temperature 1000` proxy used elsewhere, so "random" means what it says.

    python sanity_selfplay.py --layouts steak_gc00,steak_gc03,steak_gc04 --seeds 0-4
"""
import argparse
import sys

import numpy as np
import torch

import _paths  # noqa: F401
from _paths import checkpoint_path, stage_layouts

from baseline import load_policy, BaselineActor
from env import make_human, HUMAN_INDEX, ROBOT_INDEX
from rollout import make_env, DNF_PENALTY

from overcooked_ai_py.mdp.overcooked_mdp import Action
from utils.features import build_full_state


def _obs(env, idx):
    o = build_full_state(env.mdp, env.state, agent_index=idx, t=env.t,
                         horizon=env.horizon)
    return o.astype(np.float32)[None, ...]


def play(env, pick0, pick1, horizon):
    """pick_i(env) -> primitive action for chef i."""
    env.reset()
    deliveries, ticks = 0, []
    while env.t < horizon and not env.mdp.is_terminal(env.state):
        a0, a1 = pick0(env), pick1(env)
        sparse, done, _ = env.step(a0, a1)
        if sparse > 0:
            for _ in range(max(1, int(round(sparse / float(env.mdp.delivery_reward))))):
                deliveries += 1
                ticks.append(env.t)
        if done:
            break
    finished = len(ticks) >= env.n_orders - 1
    return {"deliveries": deliveries,
            "sparse": deliveries * env.mdp.delivery_reward,
            "completion": ticks[-1] if finished else horizon + DNF_PENALTY,
            "finished": finished}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=str,
                    default="steak_gc00,steak_gc03,steak_gc04,steak_gc07")
    ap.add_argument("--seeds", type=str, default="0-4")
    ap.add_argument("--horizon", type=int, default=400)
    args = ap.parse_args(argv)
    stage_layouts()
    torch.set_num_threads(1)
    a, b = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(a), int(b) + 1)) if b else [int(a)]

    print("%-12s %-26s %7s %8s %8s %7s" % (
        "layout", "condition", "sparse", "deliv", "comp", "fin"))
    print("-" * 74)

    for layout in args.layouts.split(","):
        env = make_env(layout, 4, args.horizon)
        net, na = load_policy(layout, env.obs_shape, checkpoint_path(layout))

        def run(label, make_pickers, seeds=seeds):
            rows = []
            for sd in seeds:
                np.random.seed(sd)
                p0, p1 = make_pickers(sd)
                rows.append(play(env, p0, p1, args.horizon))
            print("%-12s %-26s %7.2f %8.3f %8.1f %7.2f" % (
                layout, label,
                np.mean([r["sparse"] for r in rows]),
                np.mean([r["deliveries"] for r in rows]),
                np.mean([r["completion"] for r in rows]),
                np.mean([float(r["finished"]) for r in rows])))

        # ---------- 1. SELF-PLAY, sampled: the training condition
        def sp_sampled(sd):
            rng = np.random.RandomState(sd)
            acts = [BaselineActor(net, na), BaselineActor(net, na)]
            for x in acts:
                x.reset()

            def pick(i):
                def f(env):
                    p = acts[i].probs(_obs(env, i))
                    return Action.INDEX_TO_ACTION[int(rng.choice(6, p=p))]
                return f
            return pick(0), pick(1)
        run("self-play (sampled)", sp_sampled)

        # ---------- 2. SELF-PLAY, argmax: what CARC_RUNS.md section 4 measures
        def sp_argmax(sd):
            acts = [BaselineActor(net, na), BaselineActor(net, na)]
            for x in acts:
                x.reset()

            def pick(i):
                def f(env):
                    p = acts[i].probs(_obs(env, i))
                    return Action.INDEX_TO_ACTION[int(np.argmax(p))]
                return f
            return pick(0), pick(1)
        run("self-play (argmax)", sp_argmax)

        # ---------- 3. the policy + the limited-vision human, fov 90
        def cross(sd, fov=90):
            rng = np.random.RandomState(sd)
            act = BaselineActor(net, na)
            act.reset()
            human = make_human(env.mdp, fov, sd, agent_index=HUMAN_INDEX)

            def p0(env):
                p = act.probs(_obs(env, ROBOT_INDEX))
                return Action.INDEX_TO_ACTION[int(rng.choice(6, p=p))]

            def p1(env):
                a, _info = human.action(env.state)
                return a
            return p0, p1
        run("policy + human fov90", cross)

        # ---------- 4. TRUE uniform-random robot + the same human
        def rnd(sd, fov=90):
            rng = np.random.RandomState(sd)
            human = make_human(env.mdp, fov, sd, agent_index=HUMAN_INDEX)

            def p0(env):
                return Action.INDEX_TO_ACTION[int(rng.randint(6))]

            def p1(env):
                a, _info = human.action(env.state)
                return a
            return p0, p1
        run("UNIFORM random + human", rnd)

        # ---------- 5. a robot that does NOTHING + the human. This is the
        # decisive interpretive condition: if the human finishes the task on its
        # own, then "random beats trained" does not mean random is good, it
        # means the trained robot INTERFERES and the random one stays out of
        # the way.
        def idle(sd, fov=90):
            human = make_human(env.mdp, fov, sd, agent_index=HUMAN_INDEX)

            def p0(env):
                return Action.STAY

            def p1(env):
                a, _info = human.action(env.state)
                return a
            return p0, p1
        run("IDLE robot + human", idle)

        # ---------- 6. two random robots, no human: the absolute floor
        def rnd2(sd):
            rng = np.random.RandomState(sd)

            def f(env):
                return Action.INDEX_TO_ACTION[int(rng.randint(6))]
            return f, f
        run("random + random (floor)", rnd2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
