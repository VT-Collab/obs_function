"""Screen a checkpoint for STATE-DEPENDENCE, the thing every earlier config failed.

A policy that emits one action at prob ~1.000 everywhere has learned a constant and
is worthless regardless of its return (this is what d12/d04 did). Reports:
  dominant   most-frequent greedy action and its share
  max_prob   highest action-probability seen (1.000 => fully committed constant)
  spread     std of P(speak) across states (0.000 => ignores the observation)
"""
import sys, collections
sys.path.append(sys.path[0] or ".")
import numpy as np, torch
from robot.policy.neural.baseline.no_fov.actor_critic import NoFovAC, NoFovRecAC
from robot.policy.neural.baseline.no_fov.my_env_wrapper import NoFovAssistEnv
from robot.policy.neural.baseline.no_fov.features import ACTIONS

for path in sys.argv[1:]:
    name = path.split("/")[-1].replace(".pt", "")
    sd = torch.load(path, map_location="cpu")
    rec = any(k.startswith("memory_rnn") or "rnn" in k for k in sd)
    m = NoFovRecAC() if rec else NoFovAC()
    try:
        m.load_state_dict(sd)
    except Exception:
        m = NoFovAC() if rec else NoFovRecAC()
        m.load_state_dict(sd)
        rec = not rec
    m.eval()
    env = NoFovAssistEnv(); cnt = collections.Counter(); ps = []
    for ep in range(6):
        obs, _ = env.reset(seed=900 + ep)
        mem = torch.zeros(1, m.memory_size) if rec else None
        while True:
            with torch.no_grad():
                if rec: d, _, mem = m(torch.tensor(obs).unsqueeze(0), mem)
                else:   d, _ = m(torch.tensor(obs).unsqueeze(0))
            pr = d.probs[0].numpy(); ps.append(pr)
            a = int(pr.argmax()); cnt[ACTIONS[a]] += 1
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc: break
    P = np.array(ps); tot = sum(cnt.values())
    top, n = cnt.most_common(1)[0]
    spread = float((1 - P[:, 0]).std())
    verdict = "CONSTANT" if (P.max(1).mean() > 0.97 and n / tot > 0.97) else \
              "weak" if spread < 0.05 else "STATE-DEPENDENT"
    print(f"{name:<6} {'rec' if rec else 'ff ':<4} dominant={top:<11}{100*n/tot:5.1f}%  "
          f"max_prob={P.max():.3f}  P(speak)spread={spread:.4f}  -> {verdict}")
