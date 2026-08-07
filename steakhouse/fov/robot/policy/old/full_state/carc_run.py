"""
CARC driver for the OBSERVATION comparison. For one layout, train three FOV-blind
baselines - old 17-dim, full-state flat (48), full-state grid (CNN) - then
evaluate each across all FOVs. Writes one CSV per layout: layout,kind,fov,deliv,t.

The question: does giving the FOV-blind baseline the FULL state (instead of the
coarse 17-dim) improve delivery? (Follow-up: does the FOV module still help on
top of the strongest observation.)

Run: python -m fov.robot.policy.full_state.carc_run <layout> [iters] [n_eps] [outdir] [horizon]
"""
import csv
import os
import sys

import numpy as np
import torch

from steakhouse.fov.robot.policy.old.baseline.policy import ActorCritic
from steakhouse.fov.robot.policy.old.baseline.env_wrapper import RobotAssistEnv, CANDIDATE_FOVS
from steakhouse.fov.robot.policy.old.full_state.env import FlatEnv, GridEnv
from steakhouse.fov.robot.policy.old.full_state.train import train, build
from steakhouse.fov.robot.policy.old.override_v2.evaluate import _episode

KINDS = ["old", "flat", "grid"]


def make_train(kind, layout, horizon):
    if kind == "old":
        return RobotAssistEnv(layout=layout, horizon=horizon), ActorCritic()
    return build(kind, layout, horizon)


def make_eval(kind, layout, fov, horizon, seed):
    if kind == "old":
        return RobotAssistEnv(layout=layout, fovs=[fov], horizon=horizon, seed=seed)
    if kind == "flat":
        return FlatEnv(layout=layout, fovs=[fov], horizon=horizon, seed=seed)
    return GridEnv(layout=layout, fovs=[fov], horizon=horizon, seed=seed)


def main():
    layout = sys.argv[1] if len(sys.argv) > 1 else "steak_gc00"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    outdir = sys.argv[4] if len(sys.argv) > 4 else "carc_results_fs"
    horizon = int(sys.argv[5]) if len(sys.argv) > 5 else 450
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for kind in KINDS:
        print(f"[{layout}:{kind}] train {iters} iters (horizon={horizon})",
              file=sys.stderr, flush=True)
        env, net = make_train(kind, layout, horizon)
        train(env, net, iters, tag=f"{layout}:{kind}")
        torch.save(net.state_dict(), os.path.join(outdir, f"{kind}_{layout}.pt"))
        net.eval()
        for fov in CANDIDATE_FOVS:
            ds, ts = [], []
            for s in range(n_eps):
                d, t = _episode(make_eval(kind, layout, fov, horizon, s), net, horizon)
                ds.append(d); ts.append(t)
            rows.append(dict(layout=layout, kind=kind, fov=fov,
                             deliv=round(float(np.mean(ds)), 3),
                             t=round(float(np.mean(ts)), 2)))
        print(f"[{layout}:{kind}] eval done", file=sys.stderr, flush=True)

    out = os.path.join(outdir, f"fullstate_{layout}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layout", "kind", "fov", "deliv", "t"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[{layout}] wrote {out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
