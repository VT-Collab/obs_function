"""Headless episode runner. One JSON line per episode, no window, no pygame.

    python -m robot.filter.harness.evaluate --layouts divide --fovs 30 --seeds 0 --methods handoff
    python -m robot.filter.harness.evaluate --all --out runs.jsonl        the full grid

watch.py cannot be reused for this. It imports play.py for the renderer, play.py
imports pygame, and pygame needs a display -- so on a compute node the whole
import chain dies before a single tick runs. What watch.py DOES own that we must
copy exactly is two things: the layouts-dir override (overcooked_ai_py reads
LAYOUTS_DIR at call time, so pointing it at ours is what makes --layout divide
resolve) and the tick order. Both are reproduced below and nothing else is.

THE TICK ORDER IS LOAD-BEARING, not a detail. Both agents decide on the SAME
un-mutated state, and the robot decides BEFORE post.update folds in the human's
action. Any FOV-aware policy therefore acts on a posterior that has observed
through t-1, which is correct -- at tick t the human's action has not happened
yet -- and it is also why a shadow cloned during the robot's decision has not yet
seen this tick's state, so the clone's first simulated action is exactly that
shadow's prediction for this tick. Swap the two lines and every rollout observes
the same tick twice, inflating seen_count, which feeds the human's explore rule.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))            # .../no_larping
# Derived from __file__ so this runs unchanged on a compute node; STEAK_ROOT
# still wins if the two trees are ever split.
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(ROOT)))
sys.path.insert(0, ROOT)

import overcooked_ai_py                                            # noqa: E402
LAYOUTS_DIR = os.path.join(ROOT, "layout", "layouts")
overcooked_ai_py.LAYOUTS_DIR = LAYOUTS_DIR   # read at call time; see play.py:168

from overcooked_ai_py.mdp.actions import Action                    # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv      # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from human.limited_vision_human import (LimitedVisionHuman,        # noqa: E402
                                        FORGET_HORIZON)
from robot.methods import (make_robot, resolve, METHOD_KEYS,       # noqa: E402
                           METHODS, DEFAULTS, Drivers)
from robot.filter.core.fov_posterior import FOVPosterior                # noqa: E402
from robot.filter.core.qmdp import QMDPFilter                  # noqa: E402


def _parse_kw(s):
    """'T=30|eps=2.5|cell_k=6' -> {'T': 30, 'eps': 2.5, 'cell_k': 6}

    `|` separates the pairs, not `,` -- --methods is itself a comma-separated
    list, so a comma inside one method name would split the method in half.
    """
    kw = {}
    for part in [p for p in s.replace(",", "|").split("|") if p]:
        k, v = part.split("=")
        if v in ("True", "False"):
            kw[k] = (v == "True")
        elif v == "None":
            kw[k] = None
        else:
            kw[k] = float(v) if "." in v else int(v)
    return kw


_SHA = {}


def layout_sha(layout):
    """First 10 hex of sha256 of the .layout file. Cheap, cached, per row.

    Not a version number: nobody remembers to bump one of those. This changes on
    its own whenever the file does, which is the property that matters.
    """
    if layout not in _SHA:
        import hashlib
        p = os.path.join(LAYOUTS_DIR, layout + ".layout")
        try:
            with open(p, "rb") as f:
                _SHA[layout] = hashlib.sha256(f.read()).hexdigest()[:10]
        except OSError:
            _SHA[layout] = "missing"
    return _SHA[layout]


def build_robot(method, mdp, robot_idx, human_idx, seed, top_k, depth):
    """make_robot, plus an ad-hoc `exec:<baseline>:<k=v,...>` spelling.

    Sweeping T or eps through the registry would mean a new _TABLE row per
    value; this lets a sweep name the configuration on the command line and
    keeps the registry for configurations that earned a name.
    """
    if not method.startswith("exec:"):
        return make_robot(method, mdp, robot_idx, human_idx, seed, top_k, depth)
    parts = method.split(":")
    base_key = parts[1]
    kw = _parse_kw(parts[2] if len(parts) > 2 else "")
    cfg = dict(DEFAULTS, seed=seed, top_k=top_k, depth=depth)
    base, base_drv = METHODS[base_key].build(mdp, robot_idx, human_idx, cfg)
    post = FOVPosterior(mdp, human_index=human_idx, seed=seed)
    bot = QMDPFilter(mdp, base, post, agent_index=robot_idx, **kw)
    return bot, Drivers(base_drv.members + [post], cone=post)

FOVS = [30, 60, 90, 180, 360]
# The six that layout/layouts holds. Anything under layouts/untested is excluded
# on purpose: those are generated candidates that have never been played through,
# so a number from one of them is not comparable with the rest of the suite.
LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table",
           "divide", "pantry"]


def layout_names():
    return sorted(f[:-len(".layout")] for f in os.listdir(LAYOUTS_DIR)
                  if f.endswith(".layout"))


def run_episode(layout, fov, seed, method, horizon=400, forget=FORGET_HORIZON,
                top_k=3, depth=40, human_index=1, trace=False):
    """One episode. Returns a dict of per-episode metrics.

    `idle_ticks` counts ticks where the robot reported no subtask at all. That is
    the marker for a policy whose candidate list came back empty -- baselines
    return (STAY, {"subtask": None}) in that case -- and it is reported
    separately from plain STAY, because standing still while holding a subtask is
    a choice and standing still with nothing to do is a failure.
    """
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env.reset()
    human_idx, robot_idx = human_index, 1 - human_index

    human = LimitedVisionHuman(mdp, fov, agent_index=human_idx,
                               forget_horizon=forget, seed=seed)
    robot, post = build_robot(method, mdp, robot_idx, human_idx, seed, top_k, depth)

    orders_total = len(env.state.order_list or [])
    delivered_at, idle, stay, deviated = [], 0, 0, 0
    reward, t0, tick_times = 0, time.time(), []
    tr = []

    done = False
    while not done:
        state = env.state
        before = len(state.order_list or [])
        ts = time.time()
        r_act, r_info = robot.action(state)
        tick_times.append(time.time() - ts)
        h_act, _ = human.action(state)
        if post is not None:
            post.update(state, h_act)

        joint = [None, None]
        joint[robot_idx], joint[human_idx] = r_act, h_act
        nxt, rew, done, _ = env.step(tuple(joint))
        reward += rew
        done = done or mdp.is_terminal(nxt)

        if r_info.get("subtask") is None:
            idle += 1
        if r_act == Action.STAY:
            stay += 1
        if r_info.get("deviated"):
            deviated += 1
        if len(nxt.order_list or []) < before:
            delivered_at.append(env.t)
        if trace:
            tr.append({"t": env.t, "r": str(r_act), "h": str(h_act),
                       "sub": str(r_info.get("subtask")),
                       "p": r_info.get("fov_post")})

    n = max(len(tick_times), 1)
    return {
        "layout": layout, "fov": fov, "seed": seed, "method": method,
        # THE FINGERPRINT OF THE KITCHEN THIS WAS PLAYED IN. Four .layout files
        # were edited between two grids, changing which stations each agent can
        # reach, and the two grids were then diffed as though the only
        # difference were the code -- 47 of 120 BASELINE cells moved, which no
        # code change could explain. The layouts are experimental material and
        # will be edited again, so the row carries what it was run against and
        # compare_grids refuses to diff across a mismatch.
        "layout_sha": layout_sha(layout),
        "horizon": horizon, "forget": forget,
        "delivered": orders_total - len(env.state.order_list or []),
        "orders_total": orders_total,
        "delivered_at": delivered_at,
        "ticks": env.t,
        "first_delivery": delivered_at[0] if delivered_at else None,
        "last_delivery": delivered_at[-1] if delivered_at else None,
        "reward": reward,
        "idle_ticks": idle, "idle_frac": idle / n,
        "stay_ticks": stay, "stay_frac": stay / n,
        "deviated_ticks": deviated, "deviated_frac": deviated / n,
        "ms_per_tick": 1000.0 * sum(tick_times) / n,
        "wall_s": time.time() - t0,
        **({"trace": tr} if trace else {}),
    }


def _expand(spec, whole):
    """'a,b' -> [a,b]; 'all' -> whole; '0-9' -> [0..9] for the numeric ones."""
    if spec in (None, "all"):
        return list(whole)
    out = []
    for part in str(spec).split(","):
        if "-" in part and part.replace("-", "").isdigit():
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            out.append(int(part))
        else:
            out.append(part)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layouts", default="divide")
    p.add_argument("--fovs", default="30")
    p.add_argument("--seeds", default="0")
    p.add_argument("--methods", default="handoff")
    p.add_argument("--all", action="store_true",
                   help="the full grid: 6 layouts x 5 cones x 10 seeds")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--forget", type=int, default=FORGET_HORIZON)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--depth", type=int, default=40)
    p.add_argument("--out", default=None, help="JSONL path (default: stdout)")
    p.add_argument("--trace", action="store_true", help="per-tick trace in the record")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    if a.all:
        layouts, fovs, seeds = LAYOUTS, FOVS, list(range(10))
    else:
        layouts = _expand(a.layouts, LAYOUTS)
        fovs = _expand(a.fovs, FOVS)
        seeds = _expand(a.seeds, [0])
    methods = [m for m in a.methods.split(",") if m] if a.methods != "all" \
        else list(METHOD_KEYS)
    for m in methods:
        if not m.startswith("exec:") and resolve(m) is None:
            raise SystemExit("no robot method %r" % m)

    sink = open(a.out, "a") if a.out else sys.stdout
    try:
        for layout in layouts:
            for fov in fovs:
                for seed in seeds:
                    for m in methods:
                        r = run_episode(layout, fov, seed, m, a.horizon, a.forget,
                                        a.top_k, a.depth, trace=a.trace)
                        sink.write(json.dumps(r) + "\n")
                        sink.flush()
                        if not a.quiet:
                            sys.stderr.write(
                                "%-13s fov%-4d s%-2d %-16s -> %d/%d in %d ticks "
                                "(idle %.0f%%, %.0f ms/tick)\n"
                                % (layout, fov, seed, m, r["delivered"],
                                   r["orders_total"], r["ticks"],
                                   100 * r["idle_frac"], r["ms_per_tick"]))
    finally:
        if a.out:
            sink.close()


if __name__ == "__main__":
    main()
