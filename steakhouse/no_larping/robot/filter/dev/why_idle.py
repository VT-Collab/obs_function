"""Why does the baseline return NO subtask on this tick?

    python -m robot.filter.dev.why_idle --layout chefs_table --fov 360 \
        --robot qmdp-bayes --at 53

A baseline that hands back `subtask=None` makes the filter a pass-through by
contract, so the robot stands still and the trace shows a freeze with no Q and no
gain. That looks like a filter bug and is not one. But it is not necessarily a
baseline bug either: the baseline is being asked about a state the FILTER created,
and if the same baseline never freezes when it drives itself, the interesting
question is which of its own gates is closing in this particular state.

So this walks the ladder from the outside in:

  1. rank_subtasks           empty or not
  2. legal_subtasks WITH the reachability predicate the baseline passes
  3. legal_subtasks WITHOUT it -- if candidates appear here, REACHABILITY is the
     gate, meaning the filter parked the robot somewhere its own ladder cannot be
     served from
  4. saturated()             which verbs the surplus veto is blocking
  5. what the robot holds, where it stands, and which room that is

Then it re-runs the same episode with the BASELINE driving and reports whether the
baseline ever idles on its own. That comparison is the point: idling in a state the
filter chose is a filter problem wearing a baseline's clothes.
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

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv      # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from common import geometry as geo                                 # noqa: E402
from common import tasks                                           # noqa: E402
from common.views import TruthView                                 # noqa: E402
from human.limited_vision_human import (LimitedVisionHuman,        # noqa: E402
                                        FORGET_HORIZON)
from robot.methods import make_robot                               # noqa: E402


def dissect(mdp, state, bot, ri):
    me = state.players[ri]
    pos = tuple(me.position)
    held = me.held_object.name if me.held_object else None
    view = TruthView(mdp, state)
    walk = set(view.walkable) | {pos}
    field = geo.dist_field(walk, pos)

    def ok(c):
        return geo.path_len_in(field, walk, c) is not None

    inner = getattr(bot, "inner", bot)
    ranked = bot.rank_subtasks(state) if hasattr(bot, "rank_subtasks") else []
    with_ok = tasks.legal_subtasks(view, held, ok)
    without = tasks.legal_subtasks(view, held, None)
    sat = tasks.saturated(view, ok)

    print("  robot at %-8s holding %-12s  ranked=%d" % (pos, held, len(ranked)))
    print("  legal_subtasks WITH reachability : %d" % len(with_ok))
    print("  legal_subtasks WITHOUT it        : %d" % len(without))
    if not with_ok and without:
        print("  -> REACHABILITY IS THE GATE. Verbs that exist but cannot be served")
        print("     from here: %s" % sorted({v for _t, v, _c in without})[:10])
        unreach = sorted({c for _t, _v, c in without})[:6]
        print("     e.g. cells %s all unreachable from %s" % (unreach, pos))
    elif not without:
        print("  -> NOTHING IS LEGAL AT ALL, reachability aside.")
        print("     saturated() blocks: %s" % sorted(sat))
    if sat:
        print("  saturated() blocks %d verbs: %s" % (len(sat), sorted(sat)))
    return len(ranked)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="chefs_table")
    p.add_argument("--fov", type=int, default=360)
    p.add_argument("--robot", default="qmdp-bayes")
    p.add_argument("--baseline", default="bayes",
                   help="the same robot without the filter, for the comparison")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--at", type=int, default=53)
    p.add_argument("--horizon", type=int, default=300)
    a = p.parse_args(argv)

    for label, robot in ((a.robot, a.robot), ("%s (no filter)" % a.baseline, a.baseline)):
        mdp = SteakHouseGridworld.from_layout_name(a.layout)
        env = OvercookedEnv.from_mdp(mdp, horizon=a.horizon, info_level=0)
        env.reset()
        human = LimitedVisionHuman(mdp, a.fov, agent_index=1,
                                   forget_horizon=FORGET_HORIZON, seed=a.seed)
        bot, drv = make_robot(robot, mdp, 0, 1, a.seed)
        idle = 0
        runs, cur = [], 0
        print("\n=== %s" % label)
        done = False
        while not done:
            st = env.state
            r_act, info = bot.action(st)
            none_ = info.get("subtask") is None
            idle += none_
            cur = cur + 1 if none_ else 0
            if cur:
                runs.append(cur)
            if none_ and env.t == a.at:
                print("  --- tick %d: baseline returned NO subtask" % env.t)
                base = getattr(bot, "baseline", bot)
                dissect(mdp, st, base, 0)
            ha, _ = human.action(st)
            if drv is not None:
                drv.update(st, ha)
            nxt, _, done, _ = env.step((r_act, ha))
            done = done or mdp.is_terminal(nxt)
        print("  idle ticks: %d/%d (%.0f%%)   longest idle run: %d"
              % (idle, env.t, 100.0 * idle / max(env.t, 1), max(runs) if runs else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
