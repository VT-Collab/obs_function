"""
CARC driver: for ONE layout, train a FOV-blind baseline, then evaluate the
override module vs that baseline across all FOVs and a knob sweep. Writes one CSV
per layout (rows = layout,fov,config,deliv,t,override,cone) to <outdir>.

config values: baseline | override_k0 (task-override only, no reroute)
                       | override_k3 | override_k6 (task-override + reroute knob)

Run (on a compute node):
  python -m fov.robot.policy.override_v2.carc_run <layout> [iters] [n_eps] [outdir]
"""
import csv
import os
import sys

import numpy as np
import torch

from steakhouse.fov.robot.policy.old.baseline.policy import ActorCritic
from steakhouse.fov.robot.policy.old.baseline.env_wrapper import (
    RobotAssistEnv, CANDIDATE_FOVS, HORIZON)
from steakhouse.fov.robot.policy.old.override_v2.env import OverrideEnv
from steakhouse.fov.robot.policy.old.override_v2.train import train
from steakhouse.fov.robot.policy.old.override_v2.evaluate import _episode

KNOBS = [0.0, 3.0, 6.0]


def main():
    layout = sys.argv[1] if len(sys.argv) > 1 else "steak_gc00"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    n_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    outdir = sys.argv[4] if len(sys.argv) > 4 else "carc_results"
    mode = sys.argv[5] if len(sys.argv) > 5 else "conservative"
    horizon = int(sys.argv[6]) if len(sys.argv) > 6 else HORIZON
    os.makedirs(outdir, exist_ok=True)

    print(f"[{layout}] training FOV-blind baseline: {iters} iters "
          f"(mode={mode} horizon={horizon})", file=sys.stderr, flush=True)
    net = train(RobotAssistEnv(layout=layout, horizon=horizon), iters)
    torch.save(net.state_dict(), os.path.join(outdir, f"baseline_{layout}.pt"))
    net.eval()

    rows = []
    for fov in CANDIDATE_FOVS:
        # baseline: once per FOV (FOV-blind net, shared across knobs)
        bd, bt = [], []
        for s in range(n_eps):
            d, t = _episode(RobotAssistEnv(layout=layout, fovs=[fov],
                                           horizon=horizon, seed=s), net, horizon)
            bd.append(d); bt.append(t)
        rows.append(dict(layout=layout, fov=fov, config="baseline",
                         deliv=round(float(np.mean(bd)), 3),
                         t=round(float(np.mean(bt)), 2), override=0.0, cone=0.0))
        # override: knob sweeps the trajectory reroute (task authority per `mode`)
        for knob in KNOBS:
            od, ot, ov, cone = [], [], [], []
            for s in range(n_eps):
                oe = OverrideEnv(layout=layout, fovs=[fov], horizon=horizon, seed=s,
                                 knob=knob, candidate_fovs=CANDIDATE_FOVS, mode=mode)
                d, t = _episode(oe, net, horizon)
                od.append(d); ot.append(t)
                ov.append(oe.n_override / max(1, oe.n_step))
                cone.append(oe.n_in_cone / max(1, oe.n_step))
            rows.append(dict(layout=layout, fov=fov, config=f"override_k{int(knob)}",
                             deliv=round(float(np.mean(od)), 3),
                             t=round(float(np.mean(ot)), 2),
                             override=round(float(np.mean(ov)), 3),
                             cone=round(float(np.mean(cone)), 3)))
        print(f"[{layout}] fov={fov} done", file=sys.stderr, flush=True)

    out = os.path.join(outdir, f"override_v2_{layout}_{mode}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layout", "fov", "config", "deliv", "t",
                                          "override", "cone"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[{layout}] wrote {out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
