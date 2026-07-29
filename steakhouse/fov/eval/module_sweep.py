"""Sweep the FOV-module strength against a fixed trained baseline: does ANY
strength give a positive delta, or is the bias direction net-harmful?
Usage: module_sweep.py <layout> <weights.pt> [n_ep=36]"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/Users/mishafu/Desktop/obs_function/steakhouse")
from fov.robot.policy.baseline.env_wrapper import RobotAssistEnv
from fov.robot.policy.baseline.policy import ActorCritic
from fov.robot.policy.module.fov_module import FOVModule
from fov.robot.policy.end_to_end.compare import run

layout = sys.argv[1]
weights = sys.argv[2]
n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 36
FOVS = [30, 60, 90, 120, 180, 360]

env = RobotAssistEnv(layout=layout, fovs=FOVS)
net = ActorCritic()
net.load_state_dict(torch.load(weights, map_location="cpu"))
net.eval()

base = run(env, net, None, n_ep, "base")
allb = np.mean([d for v in base.values() for d in v])
print(f"\n### {layout}  baseline ALL = {allb:.2f}  (n_ep={n_ep}/condition)")
print(f"{'strength':>9} {'module ALL':>11} {'delta':>8}   per-FOV delta")
for s in [0.0, 0.5, 1.0, 2.0, 3.5]:
    mod = run(env, net, FOVModule(env.mdp, env.mlp, FOVS, strength=s), n_ep, "mod")
    allm = np.mean([d for v in mod.values() for d in v])
    pf = " ".join(f"{f}:{np.mean(mod.get(f, [0])) - np.mean(base.get(f, [0])):+.1f}" for f in FOVS)
    print(f"{s:>9.2f} {allm:>11.2f} {allm - allb:>+8.2f}   {pf}", flush=True)
