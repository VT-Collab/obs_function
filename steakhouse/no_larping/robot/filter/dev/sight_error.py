"""How far ahead can the forecast be trusted about WHEN the partner will look?

    python -m robot.filter.dev.sight_error --layouts back_bar,pantry,divide --seeds 0-2

THE ASYMMETRY THIS EXISTS TO MEASURE. Two ways to hand something over:

  CAUSED SIGHT   press while their cone is already on the counter. `sight == t_end`
                 BY CONSTRUCTION -- there is nothing to predict, the belief is
                 created by the act. Forecast error cannot touch it.
  PREDICTED SIGHT  press on a counter they are not looking at, and rely on their
                 cone sweeping it later. Now the estimate depends on knowing where
                 they will be looking k ticks from now, and on the press landing at
                 the right tick -- place it earlier or later and the sweep catches
                 it or misses it entirely.

`_collect` in core/qmdp.py reads both off the same forecast and treats them as the
same kind of number. If forecast error grows with k, they are not the same kind of
number at all, and a distant predicted sweep should be discounted against a sight
the robot can simply cause. This measures that error directly:

  predicted   the sight tick the filter's own forecast gives for a counter
  actual      the sight tick in a real rollout from the same state
  error       actual - predicted, bucketed by how far ahead the prediction was

A flat error across buckets means the forecast is reliable at any horizon and no
discount is warranted. Error growing with the horizon is the quantitative form of
"you have to reason about the tick you place it, not just the place".

The rollout drives the ROBOT with its real policy, so the human's trajectory is the
one the robot's own behaviour induces -- the forecast's job is to predict exactly
that, and grading it against a robot standing still would be grading a different
question.
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(ROOT)))
sys.path.insert(0, ROOT)

import overcooked_ai_py                                            # noqa: E402
overcooked_ai_py.LAYOUTS_DIR = os.path.join(ROOT, "layout", "layouts")

from overcooked_ai_py.mdp.actions import Action                    # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv      # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from common import geometry as geo                                 # noqa: E402
from common.views import TruthView                                 # noqa: E402
from human.limited_vision_human import (LimitedVisionHuman,        # noqa: E402
                                        FORGET_HORIZON)
from robot.filter.core.fov_posterior import FOVPosterior           # noqa: E402
from robot.filter.core.qmdp import fast_clone_human                # noqa: E402
from robot.methods import make_robot                               # noqa: E402

LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table", "divide", "pantry"]
BUCKETS = [(0, 0), (1, 3), (4, 7), (8, 15), (16, 30)]


def first_sight(seq, cells, fov, terrain):
    """{cell: first index in `seq` whose cone covers it}. seq = [(pos, orient)]."""
    out = {}
    for k, (pos, orient) in enumerate(seq):
        vis = geo.visible_cells(terrain, pos, orient, fov)
        for c in cells:
            if c not in out and c in vis:
                out[c] = k
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", default="back_bar,pantry,divide")
    p.add_argument("--fov", type=int, default=90)
    p.add_argument("--seeds", default="0-2")
    p.add_argument("--robot", default="qmdp-greedy")
    p.add_argument("--horizon", type=int, default=160)
    p.add_argument("--look", type=int, default=30, help="forecast length graded")
    p.add_argument("--every", type=int, default=6, help="grade every Nth tick")
    a = p.parse_args(argv)

    lays = a.layouts.split(",") if a.layouts != "all" else LAYOUTS
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))

    err = collections.defaultdict(list)
    missed = collections.Counter()
    total = collections.Counter()
    for lay in lays:
        for seed in seeds:
            mdp = SteakHouseGridworld.from_layout_name(lay)
            env = OvercookedEnv.from_mdp(mdp, horizon=a.horizon, info_level=0)
            env.reset()
            human = LimitedVisionHuman(mdp, a.fov, agent_index=1,
                                       forget_horizon=FORGET_HORIZON, seed=seed)
            bot, drv = make_robot("handoff", mdp, 0, 1, seed)
            post = FOVPosterior(mdp, human_index=1, seed=seed)
            terrain = mdp.terrain_mtx
            cells = sorted(TruthView(mdp, env.state).free_counters())
            real, snaps = [], {}
            done = False
            while not done:
                st = env.state
                hp = st.players[1]
                real.append((tuple(hp.position), tuple(hp.orientation)))
                r_act, _ = bot.action(st)
                if env.t % a.every == 0 and post.p:
                    f = max(post.p, key=post.p.get)
                    snaps[env.t] = (st.deepcopy(), post.shadows[f], f,
                                    bot.last_subtask)
                ha, _ = human.action(st)
                post.update(st, ha)
                if drv is not None:
                    drv.update(st, ha)
                nxt, _, done, _ = env.step((r_act, ha))
                done = done or mdp.is_terminal(nxt)

            for t0, (st, shadow, fov, sub_) in snaps.items():
                # The forecast, reproduced as core/qmdp._forecast builds it: step the
                # MAP shadow forward while the robot beelines to its baseline's
                # realised cell. Pure geo.step_towards, no side effects on either.
                sim = st.deepcopy()
                h = fast_clone_human(shadow)
                walk = set(TruthView(mdp, st).walkable)
                ref = sub_[2] if sub_ else None
                pred_seq = []
                for _k in range(a.look):
                    hp = sim.players[1]
                    pred_seq.append((tuple(hp.position), tuple(hp.orientation)))
                    ah, _ = h.action(sim)
                    me = sim.players[0]
                    if ref is not None:
                        mv, arr = geo.step_towards(walk | {tuple(me.position)},
                                                   tuple(me.position),
                                                   tuple(me.orientation), ref)
                        ar = Action.INTERACT if arr else (mv or Action.STAY)
                    else:
                        ar = Action.STAY
                    sim, _, _, _ = mdp.get_state_transition(sim, (ar, ah))
                    if mdp.is_terminal(sim):
                        break
                pred = first_sight(pred_seq, cells, fov, terrain)
                act = first_sight(real[t0:t0 + a.look], cells, a.fov, terrain)
                for c, k in pred.items():
                    b = next(((lo, hi) for lo, hi in BUCKETS if lo <= k <= hi), None)
                    if b is None:
                        continue
                    total[b] += 1
                    if c in act:
                        err[b].append(act[c] - k)
                    else:
                        missed[b] += 1

    print("Forecast of WHEN the partner looks at a counter, graded against a real "
          "rollout.\nfov %d, forecast %d ticks, robot %s, layouts %s, seeds %s\n"
          % (a.fov, a.look, a.robot, ",".join(lays), a.seeds))
    print("%-14s %7s %8s %8s %8s %9s %s"
          % ("predicted at", "n", "mean err", "median", "|err|>3", "never", "note"))
    for b in BUCKETS:
        n = total[b]
        if not n:
            continue
        e = err[b]
        m = sum(e) / len(e) if e else float("nan")
        med = sorted(e)[len(e) // 2] if e else float("nan")
        bad = sum(1 for x in e if abs(x) > 3) / len(e) if e else float("nan")
        note = "CAUSED -- nothing to predict" if b == (0, 0) else ""
        print("t+%-11s %7d %8.2f %8.1f %7.0f%% %8.0f%% %s"
              % ("%d-%d" % b, n, m, med, 100 * bad,
                 100.0 * missed[b] / n, note))
    print("\n'never' = the forecast said they would look and in the rollout they "
          "never did\nwithin the same window. That is the failure a caused sight "
          "cannot have.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
