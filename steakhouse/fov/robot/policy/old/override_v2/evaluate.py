"""
Baseline vs FOV-override, the honest comparison.

For each true FOV and seed we run the SAME frozen baseline net twice, PAIRED on
seed (same layout init, same human FOV realisation):
    A) baseline   - RobotAssistEnv, net.act, no bias, no override  (the control)
    B) override   - OverrideEnv, net SUGGESTS, module may override anything

We report, per FOV:
    delivered      mean orders delivered (the NO-WORSE-DELIVERY constraint)
    t_complete     mean steps to finish all N_ORDERS (censored at horizon if not)
    override%      fraction of steps the module overrode the baseline
    in_cone%       fraction of steps the robot sat in the inferred human cone
A win = delivered NOT worse AND t_complete lower.

Run: python -m fov.robot.policy.override_v2.evaluate baseline_<layout>.pt [layout] [n_eps]
"""
import sys

import numpy as np
import torch

from steakhouse.fov.robot.policy.old.baseline.policy import ActorCritic
from steakhouse.fov.robot.policy.old.baseline.env_wrapper import (
    RobotAssistEnv, CANDIDATE_FOVS, N_ORDERS, HORIZON)
from steakhouse.fov.robot.policy.old.override_v2.env import OverrideEnv
from steakhouse.fov.robot.policy.old.override_v2._util import quiet


def _episode(env, net, horizon):
    """Returns (delivered, t_complete). t_complete = step of the last delivery
    once all N_ORDERS are out, else `horizon` (censored)."""
    with quiet():
        o = env.reset()
        done = False
        delivered, t, t_complete = 0, 0, horizon
        while not done:
            x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a, _, _, _ = net.act(x)
            o, r, done, info = env.step(int(a.item()))
            t += 1
            if info["delivered"] > delivered:
                delivered = info["delivered"]
                if delivered >= N_ORDERS:
                    t_complete = t
                    break
    return delivered, t_complete


def evaluate(ckpt, layout="steak_side_2", n_eps=8, knob=3.0, horizon=HORIZON):
    net = ActorCritic()
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()

    print(f"layout={layout}  n_eps={n_eps}  knob={knob}  ckpt={ckpt}\n", file=sys.stderr)
    header = (f"{'fov':>4} | {'base_deliv':>10} {'base_t':>7} | "
              f"{'ovr_deliv':>9} {'ovr_t':>6} {'ovr%':>5} {'cone%':>5} | {'verdict':>10}")
    print(header)
    print("-" * len(header))

    for fov in CANDIDATE_FOVS:
        bd, bt, od, ot, ov, cone = [], [], [], [], [], []
        for s in range(n_eps):
            benv = RobotAssistEnv(layout=layout, fovs=[fov], horizon=horizon, seed=s)
            d, t = _episode(benv, net, horizon)
            bd.append(d); bt.append(t)

            oenv = OverrideEnv(layout=layout, fovs=[fov], horizon=horizon, seed=s,
                               knob=knob, candidate_fovs=CANDIDATE_FOVS)
            d2, t2 = _episode(oenv, net, horizon)
            od.append(d2); ot.append(t2)
            ov.append(oenv.n_override / max(1, oenv.n_step))
            cone.append(oenv.n_in_cone / max(1, oenv.n_step))

        bd_m, bt_m = np.mean(bd), np.mean(bt)
        od_m, ot_m = np.mean(od), np.mean(ot)
        no_worse = od_m >= bd_m - 1e-9
        faster = ot_m < bt_m - 1e-9
        verdict = "WIN" if (no_worse and faster) else ("tie" if no_worse else "WORSE")
        print(f"{fov:>4} | {bd_m:>10.2f} {bt_m:>7.1f} | "
              f"{od_m:>9.2f} {ot_m:>6.1f} {np.mean(ov):>4.0%} {np.mean(cone):>4.0%} | "
              f"{verdict:>10}")


def main():
    if len(sys.argv) < 2:
        print("usage: python -m fov.robot.policy.override_v2.evaluate baseline.pt "
              "[layout] [n_eps]", file=sys.stderr)
        sys.exit(1)
    ckpt = sys.argv[1]
    layout = sys.argv[2] if len(sys.argv) > 2 else "steak_side_2"
    n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    evaluate(ckpt, layout, n_eps)


if __name__ == "__main__":
    main()
