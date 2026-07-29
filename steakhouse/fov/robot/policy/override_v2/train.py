"""
Train a FOV-BLIND baseline for the override experiment. Self-contained PPO (same
hyper-params as baseline/train.py) so the new set stands alone, but it trains the
SAME frozen ActorCritic on the SAME validated RobotAssistEnv - it is the honest
control the override module sits on top of. Rollouts are silenced; progress goes
to stderr so it survives the quiet() wrapper.

Run: python -m fov.robot.policy.override_v2.train [iters] [layout] [out.pt]
"""
import sys

import numpy as np
import torch

from fov.robot.policy.baseline.env_wrapper import RobotAssistEnv
from fov.robot.policy.baseline.policy import ActorCritic
from fov.robot.policy.override_v2._util import quiet

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
                a, l, _, v = net.act(x)            # FOV-blind: no bias, no override
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
                  f"delivered={np.mean(dels):.2f}  eps={len(rets)}",
                  file=sys.stderr, flush=True)
    return net


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    layout = sys.argv[2] if len(sys.argv) > 2 else "steak_side_2"
    out = sys.argv[3] if len(sys.argv) > 3 else f"baseline_{layout}.pt"
    env = RobotAssistEnv(layout=layout)
    print(f"[train] FOV-blind baseline | layout={layout} | iters={iters}", file=sys.stderr)
    net = train(env, iters)
    torch.save(net.state_dict(), out)
    print(f"[train] saved -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
