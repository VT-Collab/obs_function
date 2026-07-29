"""Baseline vs baseline+FOV-module on the THESIS metric: TIME to complete K
deliveries (fewer steps = faster) with the no-worse-delivery constraint.

Evaluates on a chosen FOV SET (the human's FOV is drawn from it each episode), so
we can find the layout x fov-combo where the module (which INFERS the fov and
adapts) beats the FOV-blind baseline (a single averaged strategy).

Usage: compare_time.py <layout> <weights.pt> <fov_csv> <K> <n_ep> <strength> <kw> <ks> <kt>
Prints one JSON verdict.
"""
import sys, os, json, contextlib
import numpy as np
import torch
sys.path.insert(0, os.environ.get("STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
from fov.robot.policy.baseline.env_wrapper import RobotAssistEnv
from fov.robot.policy.baseline.policy import ActorCritic
from fov.robot.policy.module.fov_module import FOVModule

DEV = open(os.devnull, "w")


def episode(env, net, module, seed, K):
    o = env.reset(seed=seed)
    if module:
        module.reset()
    done, t, time_to_K, delivered = False, 0, None, 0
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
            if time_to_K is None and delivered >= K:
                time_to_K = t
    if time_to_K is None:
        time_to_K = env.horizon + 100        # DNF penalty
    return time_to_K, delivered, info["true_fov"]


def run(env, net, module, n_ep, K, seed0):
    time_by, del_by = {}, {}
    for i in range(n_ep):
        tK, d, fov = episode(env, net, module, seed0 + i, K)
        time_by.setdefault(fov, []).append(tK)
        del_by.setdefault(fov, []).append(d)
    return time_by, del_by


def main():
    layout = sys.argv[1]
    weights = sys.argv[2]
    fovs = [int(x) for x in sys.argv[3].split(",")]
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    n_ep = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    strength = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5
    kw = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    ks = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    kt = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    try:
        env = RobotAssistEnv(layout=layout, fovs=fovs)
        net = ActorCritic()
        net.load_state_dict(torch.load(weights, map_location="cpu"))
        net.eval()
    except Exception as e:
        print("JSON " + json.dumps({"layout": layout, "error": str(e)[:120]})); return

    bt, bd = run(env, net, None, n_ep, K, 7000)
    mod = FOVModule(env.mdp, env.mlp, fovs, strength=strength, kw=kw, ks=ks, kt=kt)
    mt, md = run(env, net, mod, n_ep, K, 7000)     # same seeds -> paired

    def agg(d):
        allv = [x for v in d.values() for x in v]
        return float(np.mean(allv)) if allv else 0.0

    b_time, m_time = agg(bt), agg(mt)
    b_del, m_del = agg(bd), agg(md)
    per_fov = {}
    for f in fovs:
        per_fov[str(f)] = {
            "b_time": round(float(np.mean(bt.get(f, [0]))), 1),
            "m_time": round(float(np.mean(mt.get(f, [0]))), 1),
            "b_del": round(float(np.mean(bd.get(f, [0]))), 2),
            "m_del": round(float(np.mean(md.get(f, [0]))), 2),
        }
    out = {
        "layout": layout, "fovs": fovs, "K": K, "n_ep": n_ep,
        "cfg": {"strength": strength, "kw": kw, "ks": ks, "kt": kt},
        "b_time": round(b_time, 1), "m_time": round(m_time, 1),
        "time_delta": round(m_time - b_time, 1),          # negative = module FASTER
        "b_del": round(b_del, 2), "m_del": round(m_del, 2),
        "del_delta": round(m_del - b_del, 2),             # >=0 = no worse
        "module_wins": bool(m_time < b_time - 1.0 and m_del >= b_del - 0.05),
        "per_fov": per_fov,
    }
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
