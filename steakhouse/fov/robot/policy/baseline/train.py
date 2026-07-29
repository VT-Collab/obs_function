"""
MISHA NEW CHANGE - PPO training for the FOV-BLIND baseline robot.

THIS FILE MUST NEVER IMPORT ANYTHING FOV-RELATED. No inference, no posterior,
no entropy, no candidate list. The baseline has to be runnable with the whole
of fov/robot/inference/ and fov/robot/policy/module/ deleted from disk - that is
the control condition, and an import here would quietly compromise it.

The robot sees: station states (it has full vision of the WORLD), the human's
position and whether they are carrying something, and the clock. It never learns
how much the human can see. Against a partner whose FOV is resampled every
episode, the best it can do is one averaged strategy - which is exactly the floor
the module has to beat.

Run: python -m fov.robot.policy.baseline.train [iters] [layout] [out.pt]
"""
import sys

import numpy as np
import torch

from fov.robot.policy.baseline.env_wrapper import RobotAssistEnv
from fov.robot.policy.baseline.policy import ActorCritic

GAMMA, LAM, CLIP, LR = 0.99, 0.95, 0.2, 3e-4
STEPS_PER_ITER, EPOCHS = 2048, 4


def rollout(env, net, steps):
    """FOV-blind rollout. No logit bias is ever applied here."""
    ob, ac, lp, vl, rw, dn = [], [], [], [], [], []
    rets, dels = [], []
    o = env.reset()
    ep = 0.0
    for _ in range(steps):
        x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a, l, _, v = net.act(x)                 # note: no bias argument
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


def train(env, iters, log_every=5):
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for it in range(iters):
        ob, ac, lp, vl, rw, dn, rets, dels = rollout(env, net, STEPS_PER_ITER)
        adv, ret = gae(rw, vl, dn)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ob_t, ac_t = torch.as_tensor(ob), torch.as_tensor(ac)
        lp_t = torch.as_tensor(lp)
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
            print(f"  iter {it+1:>4}/{iters}  return={np.mean(rets):7.2f}  "
                  f"delivered={np.mean(dels):.2f}  eps={len(rets)}", flush=True)
    return net


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    layout = sys.argv[2] if len(sys.argv) > 2 else "steak_side_2"
    out = sys.argv[3] if len(sys.argv) > 3 else f"baseline_{layout}.pt"
    env = RobotAssistEnv(layout=layout)
    print(f"BASELINE (FOV-blind) | layout={layout} | human FOV resampled per episode")
    net = train(env, iters)
    torch.save(net.state_dict(), out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
