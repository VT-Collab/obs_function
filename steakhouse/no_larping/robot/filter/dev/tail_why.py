"""Why did the tail prefer THAT job? Decompose it.

    python -m robot.filter.dev.tail_why --layout back_bar --fov 60 \
        --robot qmdp-greedy --at 40

For every verb except `stash`, `avail` is just `t_end`, so the ENTIRE difference
between one job and another is the tail. `C` alone cannot tell you whether a
candidate won on being quick to reach or on leaving the kitchen in a better shape,
and those have opposite fixes. This prints:

  ORDERS      what `progress.orders_remaining` thinks is still owed. The tail
              prices ONLY THE NEXT DISH plus a constant for the rest, so if a leg
              looks wrongly valued the first question is whether the tail can even
              see the thing you think makes it unnecessary. "There are plenty of
              steak dishes already" is invisible to it if those dishes are not
              counted as progress on the next order.

  LEGS        the recipe legs the tail believes remain, from value_tail._legs, with
              the work each carries. A leg missing here can never be scored; a leg
              present here that you believe is already done is the bug.

  CANDIDATES  per candidate: t_end, the tail AFTER the press, and dTAIL against the
              tail of the CURRENT state. dTAIL is the number that actually decides
              a non-stash choice -- how much closer this press leaves the kitchen
              to a finished dish. A press with dTAIL 0 bought nothing and is being
              chosen only because t_end is small.
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
from human.limited_vision_human import (LimitedVisionHuman,        # noqa: E402
                                        FORGET_HORIZON)
from robot.filter.core import value_tail as vt                     # noqa: E402
from robot.filter.core.progress import orders_remaining            # noqa: E402
from robot.methods import make_robot                               # noqa: E402


def dump(bot, state, info, fov):
    print("=" * 78)
    me = state.players[bot.agent_index]
    held = me.held_object.name if me.held_object else "-"
    print("robot at %s holding %-12s   baseline wants %s"
          % (tuple(me.position), held, info.get("base_subtask")))
    print("\nORDERS  remaining=%s   order_list=%s"
          % (orders_remaining(bot.mdp, state), list(state.order_list or [])))

    # what the tail thinks is left to do
    g = vt._geo(bot.mdp, state)
    heldmap = {i: (state.players[i].held_object.name
                   if state.players[i].held_object else None) for i in (0, 1)}
    legs = vt._legs(bot.mdp, g, state, heldmap)
    print("\nLEGS the tail believes remain (%d):" % len(legs))
    if not legs:
        print("   NONE -- tail returns its cap, so every candidate scores alike")
    for l in legs:
        print("   %-22s src=%-9s dst=%-9s work=%-5.1f ready=%-5.1f after=%s"
              % (l.key, str(l.src), str(l.dst), l.work, l.ready,
                 ",".join(map(str, l.after)) or "-"))

    shadow = bot.post.shadows[fov]
    now = vt.tail_ticks(bot.mdp, state, human_view=shadow.view,
                        human_index=bot.other_index, blind=bot.blind)
    print("\ntail of the CURRENT state = %.1f" % now)

    cands = info.get("cands") or []
    if not cands:
        print("\nCANDIDATES none (gated=%s no_reach=%s)"
              % (info.get("gated"), info.get("no_reach")))
        return
    best = {}
    for verb, cell, t_end, c, k, pk, av, tl in cands:
        if cell not in best or c < best[cell][1]:
            best[cell] = (verb, c, t_end, tl, av, k, pk)
    print("\nCANDIDATES, best C first.  dTAIL = tail_after - tail_now")
    print("  %-10s %-9s %-7s %-8s %-8s %-8s %s"
          % ("verb", "cell", "press@", "tail", "dTAIL", "avail", "C"))
    for cell, (verb, c, t_end, tl, av, k, pk) in sorted(
            best.items(), key=lambda kv: kv[1][1]):
        print("  %-10s %-9s t+%-5d %-8.1f %-+8.1f %-8.1f %.1f"
              % (verb, str(cell), t_end, tl, tl - now, av, c))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="back_bar")
    p.add_argument("--fov", type=int, default=60)
    p.add_argument("--robot", default="qmdp-greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--at", type=int, default=40, help="dump at this tick")
    p.add_argument("--held", default=None,
                   help="dump at the first tick the robot holds this instead")
    a = p.parse_args(argv)

    mdp = SteakHouseGridworld.from_layout_name(a.layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=max(a.at + 1, 400), info_level=0)
    env.reset()
    human = LimitedVisionHuman(mdp, a.fov, agent_index=1,
                               forget_horizon=FORGET_HORIZON, seed=a.seed)
    bot, drv = make_robot(a.robot, mdp, 0, 1, a.seed)
    done = False
    while not done:
        st = env.state
        r_act, info = bot.action(st)
        hit = (a.held is not None
               and st.players[bot.agent_index].held_object is not None
               and a.held in st.players[bot.agent_index].held_object.name)
        if (a.held is None and env.t == a.at) or hit:
            fov = bot._map_cone()[0]
            if fov is not None:
                dump(bot, st, info, fov)
                return 0
        ha, _ = human.action(st)
        if drv is not None:
            drv.update(st, ha)
        nxt, _, done, _ = env.step((r_act, ha))
        done = done or mdp.is_terminal(nxt)
    print("never reached that tick / never held that item")
    return 1


if __name__ == "__main__":
    sys.exit(main())
