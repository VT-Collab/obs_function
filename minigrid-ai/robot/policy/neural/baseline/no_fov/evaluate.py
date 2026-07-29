# ═══════════════════════════════════════════════════════════════════════════
# baseline/no_fov/ - the RAW-STATE baseline. See features.py for the design.
#
# THIS FILE - scoring a trained no_fov checkpoint.
#
# The bar to clear is NOT zero, it is the MUTE ROBOT: a policy that never speaks
# earns exactly the no-assist success rate (measured +0.608 at any comm_cost,
# since it pays nothing). Any assistance policy that scores below that has
# learned something worse than silence, and at comm_cost=0.05 every condition in
# this project - including the hand-coded ones - does exactly that. Always print
# the mute baseline next to the policy, at the same comm_cost, or the number is
# uninterpretable.
#
#   python -m robot.policy.neural.baseline.no_fov.evaluate \
#          --checkpoint .../ppo/model_s1.pt --method ppo --comm-cost 0.02
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

import numpy as np
import torch

from robot.policy.neural.baseline.no_fov.my_env_wrapper import NoFovAssistEnv
from robot.policy.neural.baseline.no_fov.actor_critic import NoFovAC, NoFovRecAC
from robot.policy.neural.baseline.no_fov.features import ACTIONS

# sac.py is not in the tree. Lazy import so evaluating ppo/rec_ppo still works;
# restoring the file re-enables --method sac with no edit here.
try:
    from robot.policy.neural.baseline.no_fov.sac import DiscreteSAC
    HAVE_SAC = True
except ImportError:
    DiscreteSAC = None
    HAVE_SAC = False


def load(method: str, path: str):
    sd = torch.load(path, map_location="cpu")
    if method == "sac":
        if not HAVE_SAC:
            raise RuntimeError(
                "--method sac needs robot/policy/neural/baseline/no_fov/sac.py, which is "
                "not in the tree. Restore it, or evaluate ppo / rec_ppo instead.")
        ag = DiscreteSAC()
        ag.actor.load_state_dict(sd["actor"]); ag.q.load_state_dict(sd["q"])
        ag.actor.eval()
        return ag
    m = NoFovRecAC() if method == "rec_ppo" else NoFovAC()
    m.load_state_dict(sd); m.eval()
    return m


def rollout(policy, method, env, episodes, seed0, greedy):
    succ = words = steps = 0
    rets = []
    for e in range(episodes):
        obs, _ = env.reset(seed=seed0 + e)
        mem = torch.zeros(1, policy.memory_size) if method == "rec_ppo" else None
        done, R, w = False, 0.0, 0
        while not done:
            if method == "sac":
                a = policy.act(obs, greedy=greedy)
            else:
                with torch.no_grad():
                    ob = torch.tensor(obs).unsqueeze(0)
                    if method == "rec_ppo":
                        dist, _, mem = policy(ob, mem)
                    else:
                        dist, _ = policy(ob)
                    a = int(dist.probs.argmax()) if greedy else int(dist.sample())
            if ACTIONS[a] != "wait":
                w += 1
            obs, r, term, trunc, _ = env.step(a)
            R += r; steps += 1; done = term or trunc
        rets.append(R); succ += term; words += w
    n = episodes
    # TRUE objective, not the shaped training return: success minus words spoken.
    # Training adds dense progress shaping + a correct-key milestone to make the
    # signal learnable, but those are training aids - the policy is judged on the
    # real task only.
    true_ret = succ / n - env.comm_cost * (words / n)
    return dict(succ=succ / n, words=words / n, ret=true_ret, steps=steps / n)


def mute(env, episodes, seed0):
    succ, rets, steps = 0, [], 0
    for e in range(episodes):
        obs, _ = env.reset(seed=seed0 + e)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(0)
            R += r; steps += 1; done = term or trunc
        rets.append(R); succ += term
    return dict(succ=succ / episodes, words=0.0, ret=succ / episodes, steps=steps / episodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--method", choices=["ppo", "rec_ppo", "sac"], required=True)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--comm-cost", type=float, default=0.02)
    p.add_argument("--eval-seed", type=int, default=0,
                   help="eval layouts start here - keep clear of training's 1000+")
    p.add_argument("--greedy", action="store_true")
    a = p.parse_args()

    torch.manual_seed(12345)   # policies sample; pin it so reruns match
    env = NoFovAssistEnv(comm_cost=a.comm_cost)
    pol = load(a.method, a.checkpoint)

    r = rollout(pol, a.method, env, a.episodes, a.eval_seed, a.greedy)
    m = mute(env, a.episodes, a.eval_seed)
    print(f"{'':<14}{'success':>9}{'words':>8}{'steps':>8}{'reward':>9}")
    print(f"{a.method:<14}{r['succ']:>9.1%}{r['words']:>8.2f}{r['steps']:>8.1f}{r['ret']:>+9.3f}")
    print(f"{'mute robot':<14}{m['succ']:>9.1%}{m['words']:>8.2f}{m['steps']:>8.1f}{m['ret']:>+9.3f}")
    d = r["ret"] - m["ret"]
    print(f"\n  vs mute: {d:+.3f}  ->  {'BEATS silence' if d > 0 else 'WORSE THAN SILENCE'}")


if __name__ == "__main__":
    main()
