"""Per-layout search: for one trained baseline, sweep FOV-SETS x module CONFIGS on
the TIME-to-complete metric (baseline run ONCE per fov-set, reused across configs).
Prints one JSON with every (fov_set, config) result + win flag.
Usage: search_layout.py <layout> <weights.pt> [K=2] [n_ep=24]
"""
import sys, os, json, contextlib
import numpy as np
import torch
sys.path.insert(0, os.environ.get("STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
from steakhouse.fov.robot.policy.old.baseline.env_wrapper import RobotAssistEnv
from steakhouse.fov.robot.policy.old.baseline.policy import ActorCritic
from steakhouse.fov.robot.policy.old.module.fov_module import FOVModule

DEV = open(os.devnull, "w")
FOV_SETS = [[30], [60], [90], [120], [180], [360], [30, 360], [30, 60, 90, 120, 180, 360]]
CONFIGS = [(2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0), (2.0, 4.0, 0.0, 0.0),
           (1.5, 2.0, 0.0, 0.0), (2.0, 2.0, 0.0, 1.0)]
SEED0 = 7000


def episode(env, net, module, seed, K):
    o = env.reset(seed=seed)
    if module:
        module.reset()
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


def run(env, net, module, n_ep, K):
    ts, ds = [], []
    for i in range(n_ep):
        tK, d = episode(env, net, module, SEED0 + i, K)
        ts.append(tK); ds.append(d)
    return float(np.mean(ts)), float(np.mean(ds))


def main():
    layout = sys.argv[1]
    weights = sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    n_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    try:
        env = RobotAssistEnv(layout=layout)
        net = ActorCritic()
        net.load_state_dict(torch.load(weights, map_location="cpu"))
        net.eval()
    except Exception as e:
        print("JSON " + json.dumps({"layout": layout, "error": str(e)[:120]})); return
    results = []
    for fov_set in FOV_SETS:
        env.fovs = list(fov_set)
        b_time, b_del = run(env, net, None, n_ep, K)
        for (s, kw, ks, kt) in CONFIGS:
            mod = FOVModule(env.mdp, env.mlp, fov_set, strength=s, kw=kw, ks=ks, kt=kt)
            m_time, m_del = run(env, net, mod, n_ep, K)
            results.append({
                "fov_set": ",".join(map(str, fov_set)),
                "cfg": f"s{s}kw{kw}ks{ks}kt{kt}",
                "b_time": round(b_time, 1), "m_time": round(m_time, 1),
                "time_delta": round(m_time - b_time, 1),
                "b_del": round(b_del, 2), "m_del": round(m_del, 2),
                "del_delta": round(m_del - b_del, 2),
                "wins": bool(m_time < b_time - 1.0 and m_del >= b_del - 0.05),
            })
    print("JSON " + json.dumps({"layout": layout, "results": results}))


if __name__ == "__main__":
    main()
