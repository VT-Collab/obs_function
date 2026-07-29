"""Ablation: is the FOV-module's speedup driven by the INFERRED fov, or just a
generic 'cook more' bias? At a single true FOV, compare:
  baseline           - no module
  module INFERRED    - real SamplingBayesFOVInference
  module ORACLE      - force_fov = the true fov (upper bound on inference)
  module WRONG(360)  - force_fov = 360 (assume fully-sighted) -> blindness 0 ->
                       ~no bias -> should collapse to baseline
If inferred ~= oracle >> baseline ~= wrong, the win REQUIRES knowing the fov.
Usage: ablate.py <layout> <weights.pt> <true_fov> <K> <n_ep> <strength> <kw>
"""
import sys, os, json, contextlib
import numpy as np
import torch
sys.path.insert(0, "/Users/mishafu/Desktop/obs_function/steakhouse")
from fov.robot.policy.baseline.env_wrapper import RobotAssistEnv
from fov.robot.policy.baseline.policy import ActorCritic
from fov.robot.policy.module.fov_module import FOVModule

DEV = open(os.devnull, "w")


def episode(env, net, module, seed, K, mode):
    o = env.reset(seed=seed)
    if module:
        module.reset()
        module.force_fov = (env.true_fov if mode == "oracle" else
                            360 if mode == "wrong" else None)
    done, t, tK, delivered = False, 0, None, 0
    with contextlib.redirect_stdout(DEV):
        while not done:
            x = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            b = module.bias().unsqueeze(0) if module else None
            with torch.no_grad():
                a, _, _, _ = net.act(x, b)
            o, r, done, info = env.step(int(a.item()))
            if module:
                module.observe(info["obs_state"], info.get("human_subtask"))
            t += 1
            delivered = info["delivered"]
            if tK is None and delivered >= K:
                tK = t
    return (tK if tK is not None else env.horizon + 100), delivered


def run(env, net, module, n_ep, K, mode):
    ts, ds = [], []
    for i in range(n_ep):
        tK, d = episode(env, net, module, 8000 + i, K, mode)
        ts.append(tK); ds.append(d)
    return round(float(np.mean(ts)), 1), round(float(np.mean(ds)), 2)


def main():
    layout, weights = sys.argv[1], sys.argv[2]
    true_fov = int(sys.argv[3])
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    n_ep = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    strength = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0
    kw = float(sys.argv[7]) if len(sys.argv) > 7 else 3.0
    env = RobotAssistEnv(layout=layout, fovs=[true_fov])
    net = ActorCritic(); net.load_state_dict(torch.load(weights, map_location="cpu")); net.eval()
    mod = FOVModule(env.mdp, env.mlp, [30, 60, 90, 120, 180, 360], strength=strength, kw=kw, ks=0.0, kt=0.0)
    out = {"layout": layout, "true_fov": true_fov, "K": K, "n_ep": n_ep}
    out["baseline"] = run(env, net, None, n_ep, K, "none")
    out["inferred"] = run(env, net, mod, n_ep, K, "inferred")
    out["oracle"] = run(env, net, mod, n_ep, K, "oracle")
    out["wrong360"] = run(env, net, mod, n_ep, K, "wrong")
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
