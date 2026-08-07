"""The experiment: frozen self-play baseline vs the same baseline + QMDP module.

    python evaluate.py --layouts steak_gc00 --fovs 30,90,180,360 \
                       --lams 0.0,0.5 --seeds 0-19 --out results.jsonl

PAIRED BY CONSTRUCTION. Every (layout, fov, seed) is played once per lam, and
`lam=0.0` is the baseline -- not a re-implementation of it, the identical code
path with the pooling weight set to zero, so a paired difference is exactly the
effect of the module. The human is rebuilt from the same seed in every arm, so
both arms start from the same sampler draws and diverge only because the world
diverged.

One row of JSONL per episode. aggregate.py turns them into the table.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import _paths
from _paths import checkpoint_path, stage_layouts, USABLE_LAYOUTS

from baseline import load_policy, BaselineActor
from policy import BlendedRobotPolicy
from rollout import play_episode, make_env
from cost import DEFAULT_WEIGHTS, PRESETS


def parse_seeds(spec):
    out = []
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def parse_floats(spec):
    return [float(x) for x in str(spec).split(",") if str(x).strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser("QMDP FOV module vs self-play baseline")
    p.add_argument("--layouts", type=str, default="steak_gc00",
                   help="comma separated, or 'usable' for the 11 layouts whose "
                        "self-play specialist actually delivers")
    p.add_argument("--fovs", type=str, default="30,60,90,120,180,360",
                   help="TRUE cone of the human, one episode per value")
    p.add_argument("--candidate_fovs", type=str, default="30,60,90,120,180,360",
                   help="the filter's hypothesis set. The true fov need not be "
                        "in it -- see --held_out_fov")
    p.add_argument("--lams", type=str, default="0.0,1.0",
                   help="blend weights (beta). 0.0 IS the baseline arm; with "
                        "--blend bias, beta scales log p_mod as a logit bias")
    p.add_argument("--seeds", type=str, default="0-9")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--n_orders", type=int, default=4)
    p.add_argument("--ckpt_seed", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.5,
                   help="the human's subtask sampler temperature")
    p.add_argument("--module_temperature", type=float, default=1.0)
    p.add_argument("--base_temperature", type=float, default=1.0,
                   help="temperature on the BASELINE's own distribution, "
                        "applied in both arms. 1.0 = the network as trained. "
                        "Sweep at --lams 0.0 to find the baseline's best "
                        "operating point. A LARGE value (1000) flattens it to "
                        "uniform, i.e. a RANDOM ROBOT -- which is the floor "
                        "this experiment turned out to need, because the "
                        "self-play policy paired with this human does not "
                        "beat it (RESULTS.md section 0)")
    p.add_argument("--blend", type=str, default="bias",
                   choices=["bias", "loglinear", "linear"])
    p.add_argument("--module_mode", type=str, default="real",
                   help="real | uniform | noise | shuffle | fixed:<fov>  "
                        "(ablations, see policy.py)")
    p.add_argument("--topk", type=int, default=2,
                   help="how many candidate human actions to integrate over")
    p.add_argument("--weights", type=str, default=None,
                   help="JSON dict overriding cost.DEFAULT_WEIGHTS, or one of "
                        "the names in cost.PRESETS (full / collision / kb)")
    p.add_argument("--normalize_q", action="store_true",
                   help="rescale the module scores to unit spread before the "
                        "softmax, so beta means the same strength for any "
                        "weight vector (needed for a fair decomposition)")
    p.add_argument("--sample", action="store_true",
                   help="sample the robot action instead of argmax")
    p.add_argument("--max_ticks", type=int, default=None)
    p.add_argument("--out", type=str, default="results.jsonl")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--progress", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stage_layouts()

    layouts = (USABLE_LAYOUTS if args.layouts.strip() == "usable"
               else [s.strip() for s in args.layouts.split(",") if s.strip()])
    fovs = [int(x) for x in parse_floats(args.fovs)]
    cand = [int(x) for x in parse_floats(args.candidate_fovs)]
    lams = parse_floats(args.lams)
    seeds = parse_seeds(args.seeds)
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        if args.weights in PRESETS:
            weights = dict(PRESETS[args.weights])
        else:
            weights.update(json.loads(args.weights))

    torch.set_num_threads(1)          # the bottleneck is the python env loop
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    print("[setup] layouts=%s fovs=%s lams=%s seeds=%d cand=%s"
          % (layouts, fovs, lams, len(seeds), cand), flush=True)
    print("[setup] weights=%s blend=%s Tmod=%.2f topk=%d"
          % (weights, args.blend, args.module_temperature, args.topk),
          flush=True)

    t_start = time.time()
    n_done = 0
    total = len(layouts) * len(fovs) * len(seeds) * len(lams)

    with open(args.out, "a") as fh:
        for layout in layouts:
            env = make_env(layout, args.n_orders, args.horizon)
            ck = checkpoint_path(layout, args.ckpt_seed)
            policy_net, net_args = load_policy(layout, env.obs_shape, ck)
            actor = BaselineActor(policy_net, net_args)
            print("[layout] %s obs=%s ckpt=%s" % (layout, env.obs_shape, ck),
                  flush=True)

            for fov in fovs:
                for seed in seeds:
                    for lam in lams:
                        def factory(mdp, lam=lam):
                            return BlendedRobotPolicy(
                                actor, mdp, cand, lam=lam, weights=weights,
                                module_temperature=args.module_temperature,
                                blend=args.blend, human_action_topk=args.topk,
                                rng=np.random.RandomState(10000 + seed),
                                sample=args.sample,
                                module_mode=args.module_mode,
                                base_temperature=args.base_temperature,
                                normalize_q=args.normalize_q)

                        row = play_episode(env, actor, factory, fov, seed,
                                           temperature=args.temperature,
                                           max_ticks=args.max_ticks)
                        row.update({"lam": lam, "tag": args.tag,
                                    "blend": args.blend,
                                    "module_mode": args.module_mode,
                                    "sample": bool(args.sample),
                                    "module_temperature": args.module_temperature,
                                    "base_temperature": args.base_temperature,
                                    "normalize_q": bool(args.normalize_q),
                                    "topk": args.topk,
                                    "candidate_fovs": cand,
                                    "weights": weights})
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        n_done += 1
                        if args.progress and n_done % args.progress == 0:
                            el = time.time() - t_start
                            eta = el / max(n_done, 1) * (total - n_done)
                            print("  [%4d/%4d] %s fov=%3d seed=%2d lam=%.2f "
                                  "-> del=%d t=%d comp=%d (%.1fs, eta %.0fs)"
                                  % (n_done, total, layout, fov, seed, lam,
                                     row["deliveries"], row["steps"],
                                     row["completion_time"], row["wall_s"], eta),
                                  flush=True)

    print("[done] %d episodes in %.0fs -> %s"
          % (n_done, time.time() - t_start, args.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
