"""
WATCH ONE EPISODE TWICE -- baseline on the left, baseline+module on the right.

    python watch.py --layout steak_gc00 --fov 30 --seed 0
    python watch.py --layout steak_cram --fov 30 --seed 3 --only-diverged
    python watch.py --layout steak_gc00 --fov 90 --seed 0 --out replay.txt

Both arms get the SAME checkpoint, the SAME human seed and the SAME action rule,
so any difference on screen is the module and nothing else. The two are played
independently and then shown side by side, tick by tick, with the divergence
point marked -- after it, the two kitchens are genuinely different worlds, so
"the same tick" stops meaning the same situation. That is expected, not a bug.

LEGEND
    R  the robot (chef 0)        H  the human (chef 1)
    P  pot        B  board       W  sink
    M  meat       O  onion       D  dish        S  serve
    X  counter    .  floor
    A carried item is shown next to the chef, e.g. R:meat
    ^v<>  the direction a chef is facing

The header line per tick shows what each arm did, and marks OVERRIDE whenever
the module chose something the baseline would not have.
"""
import argparse
import sys

import numpy as np
import torch

import baseline  # noqa: F401   sys.path shim
from baseline import Kitchen, Actor, make_human, resolve_device, stage_layouts
from inference import SamplingBayesFOVInference                 # noqa: E402
from qmdp import QMDPModule, sync_shadows                       # noqa: E402
from overcooked_ai_py.mdp.actions import Action, Direction      # noqa: E402

ARROW = {(0, -1): "^", (0, 1): "v", (1, 0): ">", (-1, 0): "<", (0, 0): "o"}
NAME = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W",
        (0, 0): "stay", "interact": "INTERACT"}


def render(mdp, state, fov_human=None):
    """The kitchen as a list of text rows."""
    grid = [list(row) for row in mdp.terrain_mtx]
    out = []
    for y, row in enumerate(grid):
        line = []
        for x, c in enumerate(row):
            line.append("." if c == " " else c)
        out.append(line)
    for i, p in enumerate(state.players):
        x, y = p.position
        out[y][x] = "R" if i == 0 else "H"
    rows = ["".join(r) for r in out]
    #what each chef is holding and facing
    info = []
    for i, p in enumerate(state.players):
        held = p.held_object.name if p.held_object else "-"
        info.append("%s%s:%s" % ("R" if i == 0 else "H",
                                 ARROW.get(tuple(p.orientation), "?"), held))
    rows.append(" ".join(info))
    return rows


def play(layout, fov, seed, use_module, k, depth, alpha, device, horizon):
    """One episode; returns the per-tick trace."""
    kit = Kitchen(layout, horizon=horizon)
    act = Actor(layout, kit.obs_shape, device=device)
    kit.reset()
    hum = make_human(kit.mdp, fov, seed)
    act.reset()
    filt = SamplingBayesFOVInference(kit.mdp, None, [30, 60, 90, 120, 180, 360],
                                     human_agent_index=1)
    mod = QMDPModule(kit.mdp, act, k=k, depth=depth, alpha=alpha,
                     horizon=kit.horizon)
    trace = []
    while kit.t < kit.horizon and not kit.mdp.is_terminal(kit.state):
        s = kit.state
        p = act.probs(kit.robot_obs())
        want = int(np.argmax(p))
        if use_module:
            idx = mod.choose(s, p, filt, seed=seed)
        else:
            sync_shadows(filt, s)
            idx = want
        a = Action.INDEX_TO_ACTION[int(idx)]
        h, info = hum.action(s)              # info["subtask"] shown, never used
        filt.update(s, h)
        trace.append({
            "t": kit.t, "grid": render(kit.mdp, s),
            "robot": NAME.get(a, str(a)), "human": NAME.get(h, str(h)),
            "subtask": info.get("subtask"), "override": idx != want,
            "map_fov": filt.map_fov(),
        })
        sparse, done = kit.step(a, h)
        if sparse > 0:
            trace[-1]["delivery"] = True
        if done:
            break
    return trace, kit.t


def main(argv=None):
    p = argparse.ArgumentParser("watch baseline vs baseline+module, side by side")
    p.add_argument("--layout", default="steak_gc00")
    p.add_argument("--fov", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=None,
                   help="override qmdp.TAU_EFFORT, the evidence gate. 0 = ungated "
                        "(the module always acts on its ranking)")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--ckpt", default=None, help="explicit checkpoint .pt to load")
    p.add_argument("--only-diverged", action="store_true",
                   help="skip ticks where both arms did the same thing")
    p.add_argument("--max-ticks", type=int, default=120)
    p.add_argument("--out", default=None, help="write to a file instead of stdout")
    args = p.parse_args(argv)

    stage_layouts()
    if args.tau is not None:
        import qmdp
        qmdp.TAU_EFFORT = args.tau
    dev = resolve_device("cpu")
    torch.set_num_threads(1)

    A, ta = play(args.layout, args.fov, args.seed, False, args.k, args.depth,
                 args.alpha, dev, args.horizon)
    B, tb = play(args.layout, args.fov, args.seed, True, args.k, args.depth,
                 args.alpha, dev, args.horizon)

    L = []
    L.append("=" * 88)
    L.append("%s   fov=%d   seed=%d   k=%d depth=%d alpha=%.2f   (collisions OFF)"
             % (args.layout, args.fov, args.seed, args.k, args.depth, args.alpha))
    L.append("BASELINE finished at tick %d        MODULE finished at tick %d   -> %+d"
             % (ta, tb, tb - ta))
    L.append("overrides: %d of %d ticks" % (sum(1 for x in B if x["override"]), len(B)))
    L.append("=" * 88)

    #first tick where the two worlds differ
    div = None
    for i in range(min(len(A), len(B))):
        if A[i]["grid"] != B[i]["grid"]:
            div = i
            break

    w = max(len(r) for x in (A + B) for r in x["grid"]) + 2
    for i in range(min(len(A), len(B), args.max_ticks)):
        a, b = A[i], B[i]
        same = a["grid"] == b["grid"] and a["robot"] == b["robot"]
        if args.only_diverged and same and not b["override"]:
            continue
        mark = "  <<< OVERRIDE" if b["override"] else ""
        if i == div:
            L.append("-" * 88)
            L.append(">>> WORLDS DIVERGE HERE - after this the same tick is a "
                     "different situation in each arm")
        L.append("-" * 88)
        L.append("t=%-4d  BASELINE robot=%-8s | MODULE robot=%-8s  human=%-8s "
                 "subtask=%-14s mapfov=%s%s"
                 % (a["t"], a["robot"], b["robot"], b["human"],
                    b.get("subtask"), b.get("map_fov"), mark))
        for ra, rb in zip(a["grid"], b["grid"]):
            L.append("   %-*s   |   %s" % (w, ra, rb))
        if a.get("delivery") or b.get("delivery"):
            L.append("   *** DELIVERY   baseline=%s  module=%s"
                     % (bool(a.get("delivery")), bool(b.get("delivery"))))

    text = "\n".join(L)
    if args.out:
        open(args.out, "w").write(text)
        print("wrote %s (%d lines)" % (args.out, len(L)))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
