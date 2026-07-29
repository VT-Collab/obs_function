"""
Mechanism tests for the override experiment (no trained net needed - a random
ActorCritic is enough to exercise the plumbing). Checks:

  1. override_task=False, knob=0 -> behaves like the plain baseline (0 overrides)
  2. narrow FOV (30) + confident -> the module TAKES OVER (n_override > 0)
  3. wide   FOV (360)            -> the VISIBLE branch can fire; robot spends time
                                    in-cone; nothing crashes
  4. knob=0 vs knob>0           -> in-cone occupancy does not DROP with the reroute

Run: python -m fov.robot.policy.override_v2.tests [layout]
"""
import sys

import torch

from fov.robot.policy.baseline.policy import ActorCritic
from fov.robot.policy.override_v2.env import OverrideEnv
from fov.robot.policy.override_v2._util import quiet

LAYOUT = sys.argv[1] if len(sys.argv) > 1 else "steak_side_2"
HORIZON = 120


def run(net, fov, override_task, reroute, knob, seed=0):
    env = OverrideEnv(layout=LAYOUT, fovs=[fov], horizon=HORIZON, seed=seed,
                      knob=knob, override_task=override_task, reroute=reroute)
    with quiet():
        o = env.reset()
        done = False
        while not done:
            x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a, _, _, _ = net.act(x)
            o, r, done, info = env.step(int(a.item()))
    return env


def main():
    torch.manual_seed(0)
    net = ActorCritic()
    print(f"layout={LAYOUT} horizon={HORIZON} (random baseline)\n")

    e1 = run(net, 30, override_task=False, reroute=False, knob=0.0)
    assert e1.n_override == 0, "control must never override"
    print(f"[1] control              overrides={e1.n_override}  -> OK (0)")

    e2 = run(net, 30, override_task=True, reroute=True, knob=3.0)
    print(f"[2] narrow fov=30        overrides={e2.n_override} "
          f"(takeover={e2.n_takeover} visible={e2.n_visible})  "
          f"in_cone={e2.n_in_cone}/{e2.n_step}")
    assert e2.n_override > 0, "should take over a blind human once confident"

    e3 = run(net, 360, override_task=True, reroute=True, knob=3.0)
    print(f"[3] wide fov=360         overrides={e3.n_override} "
          f"(takeover={e3.n_takeover} visible={e3.n_visible})  "
          f"in_cone={e3.n_in_cone}/{e3.n_step}")

    e4a = run(net, 360, override_task=True, reroute=False, knob=0.0)
    e4b = run(net, 360, override_task=True, reroute=True, knob=5.0)
    print(f"[4] wide fov=360 in_cone reroute-off={e4a.n_in_cone}/{e4a.n_step} "
          f"vs reroute-on={e4b.n_in_cone}/{e4b.n_step}")
    assert e4b.n_in_cone >= e4a.n_in_cone, "reroute must not reduce in-cone time"

    print("\nALL MECHANISM TESTS PASSED")


if __name__ == "__main__":
    main()
