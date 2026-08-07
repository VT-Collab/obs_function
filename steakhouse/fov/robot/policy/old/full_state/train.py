"""
PPO trainer for the full-state baselines. The SAME loop trains both encodings -
the net is passed in (MLP for flat, CNN for grid). Same hyper-params as the
original baseline so the ONLY variable is the observation. Rollouts silenced;
progress -> stderr.

Run: python -m fov.robot.policy.full_state.train <flat|grid> [iters] [layout] [out.pt]
"""
import sys

import numpy as np
import torch

from steakhouse.fov.robot.policy.old.baseline.policy import ActorCritic
from steakhouse.fov.robot.policy.old.full_state.policy_cnn import CNNActorCritic
from steakhouse.fov.robot.policy.old.full_state.env import FlatEnv, GridEnv
from steakhouse.fov.robot.policy.old.full_state.features_flat import OBS_DIM_FULL
from steakhouse.fov.robot.policy.old.override_v2._util import quiet

GAMMA, LAM, CLIP, LR = 0.99, 0.95, 0.2, 3e-4
STEPS_PER_ITER, EPOCHS = 2048, 4


def rollout(env, net, steps):
    ob, ac, lp, vl, rw, dn = [], [], [], [], [], []
    rets, dels = [], []
    with quiet():
        o = env.reset()
        ep = 0.0
        for _ in range(steps):
            x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a, l, _, v = net.act(x)
            o2, r, d, info = env.step(int(a.item()))
            ob.append(o); ac.append(int(a.item())); lp.append(float(l.item()))
            vl.append(float(v.item())); rw.append(r); dn.append(d)
            ep += r; o = o2
            if d:
                rets.append(ep); dels.append(info["delivered"]); ep = 0.0
                o = env.reset()
    return (np.array(ob, dtype=np.float32), np.array(ac), np.array(lp, dtype=np.float32),
            np.array(vl, dtype=np.float32), np.array(rw, dtype=np.float32),
            np.array(dn), rets, dels)


def gae(rew, val, done):
    adv = np.zeros_like(rew); last = 0.0
    for t in reversed(range(len(rew))):
        nv = 0.0 if (t == len(rew) - 1 or done[t]) else val[t + 1]
        delta = rew[t] + GAMMA * nv - val[t]
        last = delta + GAMMA * LAM * (0.0 if done[t] else last)
        adv[t] = last
    return adv, adv + val


def train(env, net, iters, log_every=5, tag=""):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for it in range(iters):
        ob, ac, lp, vl, rw, dn, rets, dels = rollout(env, net, STEPS_PER_ITER)
        adv, ret = gae(rw, vl, dn)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ob_t = torch.as_tensor(ob); ac_t = torch.as_tensor(ac); lp_t = torch.as_tensor(lp)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        for _ in range(EPOCHS):
            logits, v = net(ob_t)
            d = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(d.log_prob(ac_t) - lp_t)
            loss = (-torch.min(ratio * adv_t,
                               torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_t).mean()
                    + 0.5 * ((v - ret_t) ** 2).mean()
                    - 0.01 * d.entropy().mean())
            opt.zero_grad(); loss.backward(); opt.step()
        if rets and (it % log_every == 0 or it == iters - 1):
            print(f"  [{tag}] iter {it+1:>4}/{iters}  return={np.mean(rets):7.2f}  "
                  f"delivered={np.mean(dels):.2f}", file=sys.stderr, flush=True)
    return net


def build(kind, layout, horizon=260):
    if kind == "flat":
        return FlatEnv(layout=layout, horizon=horizon), ActorCritic(obs_dim=OBS_DIM_FULL)
    if kind == "grid":
        return GridEnv(layout=layout, horizon=horizon), CNNActorCritic()
    raise ValueError(f"unknown kind {kind}")


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "flat"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    layout = sys.argv[3] if len(sys.argv) > 3 else "steak_gc00"
    out = sys.argv[4] if len(sys.argv) > 4 else f"{kind}_{layout}.pt"
    env, net = build(kind, layout)
    print(f"[full_state:{kind}] layout={layout} iters={iters}", file=sys.stderr)
    train(env, net, iters, tag=kind)
    torch.save(net.state_dict(), out)
    print(f"[full_state:{kind}] saved -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
