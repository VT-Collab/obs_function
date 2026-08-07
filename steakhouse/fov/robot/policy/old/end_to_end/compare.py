"""
MISHA NEW CHANGE - evaluate a TRAINED FOV-blind baseline with and without the
FOV module, and report the difference.

The weights are identical in both conditions; only the module's logit bias
differs. Train first with:
    python -m fov.robot.policy.baseline.train 120 steak_side_2 base.pt
then:
    python -m fov.robot.policy.end_to_end.compare base.pt steak_side_2 30
"""
import sys

import numpy as np
import torch

from steakhouse.fov.robot.policy.old.baseline.env_wrapper import RobotAssistEnv
from steakhouse.fov.robot.policy.old.baseline.policy import ActorCritic
from steakhouse.fov.robot.policy.old.module.fov_module import FOVModule


def run(env, net, module, n_ep, tag):
    by_fov = {}
    for i in range(n_ep):
        o = env.reset(seed=5000 + i)
        if module:
            module.reset()
        done, delivered = False, 0
        while not done:
            x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            b = module.bias().unsqueeze(0) if module else None
            with torch.no_grad():
                a, _, _, _ = net.act(x, b)
            o, r, done, info = env.step(int(a.item()))
            if module:
                # PRE-step state + the human's SUBTASK the filter scores
                module.observe(info["obs_state"], info.get("human_subtask"))
            delivered = info["delivered"]
        by_fov.setdefault(info["true_fov"], []).append(delivered)
    return by_fov


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else "base.pt"
    layout = sys.argv[2] if len(sys.argv) > 2 else "steak_side_2"
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
    fovs = [30, 60, 90, 120, 180, 360]

    env = RobotAssistEnv(layout=layout, fovs=fovs)
    net = ActorCritic()
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()

    base = run(env, net, None, n_ep, "baseline")
    mod = run(env, net, FOVModule(env.mdp, env.mlp, fovs, strength=strength), n_ep, "module")

    print(f"\nlayout={layout}  weights={weights}  episodes={n_ep}  strength={strength}")
    print(f"{'true FOV':>9} {'baseline':>10} {'module':>10} {'delta':>8}")
    ab, am = [], []
    for f in sorted(set(list(base) + list(mod))):
        b, m = base.get(f, []), mod.get(f, [])
        ab += b; am += m
        print(f"{f:>9} {np.mean(b or [0]):>10.2f} {np.mean(m or [0]):>10.2f} "
              f"{np.mean(m or [0]) - np.mean(b or [0]):>+8.2f}")
    print(f"{'ALL':>9} {np.mean(ab):>10.2f} {np.mean(am):>10.2f} "
          f"{np.mean(am)-np.mean(ab):>+8.2f}")


if __name__ == "__main__":
    main()
