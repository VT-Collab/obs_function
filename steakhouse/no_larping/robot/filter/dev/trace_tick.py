"""Per-tick trace of what the filter chose and why. The debugging entry point.

    python -m robot.filter.dev.trace_tick --layout back_bar --fov 60 \
        --robot qmdp-greedy --ticks 60 --held meat

Prints one line per tick: where the robot is, what it holds, the plan it committed
to (target cell and the tick it means to press), the action it emitted, and whether
the target MOVED since last tick. A target that changes every tick with no INTERACT
in between is a thrash, and this is the readout that shows it as a thrash rather
than as "the robot looks stuck".

`--held X` narrows to ticks where the robot is carrying X, which is how you isolate
"it will not put the meat down" from everything else happening in the episode.

Why a tracer rather than a debugger: the interesting event is a RELATION between
consecutive ticks (the target moved, and nothing was pressed), and a breakpoint
shows you one tick at a time. Set `--pdb N` to break at tick N once the trace has
told you which tick to look at.
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
from human.limited_vision_human import (LimitedVisionHuman,        # noqa: E402
                                        FORGET_HORIZON)
from robot.methods import make_robot                               # noqa: E402

ARROW = {(0, -1): "^", (0, 1): "v", (1, 0): ">", (-1, 0): "<", (0, 0): "."}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="back_bar")
    p.add_argument("--fov", type=int, default=60)
    p.add_argument("--robot", default="qmdp-greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ticks", type=int, default=80)
    p.add_argument("--held", default=None,
                   help="only print ticks where the robot holds this object")
    p.add_argument("--pdb", type=int, default=None, help="break at this tick")
    a = p.parse_args(argv)

    mdp = SteakHouseGridworld.from_layout_name(a.layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=a.ticks, info_level=0)
    env.reset()
    human = LimitedVisionHuman(mdp, a.fov, agent_index=1,
                               forget_horizon=FORGET_HORIZON, seed=a.seed)
    bot, drv = make_robot(a.robot, mdp, 0, 1, a.seed)

    print("%-4s %-9s %-12s %-22s %-22s %-7s %5s %5s %s"
          % ("t", "pos/or", "held", "plan (cell @ tick)", "baseline job",
             "action", "Q", "gain", "flags"))
    print("-" * 118)
    prev_target = None
    since_press = 0
    switches = collections.Counter()
    done = False
    while not done:
        st = env.state
        if a.pdb is not None and env.t == a.pdb:
            import pdb
            pdb.set_trace()
        r_act, info = bot.action(st)
        me = st.players[0]
        held = me.held_object.name if me.held_object else "-"
        plan = info.get("plan")
        target = plan[1] if plan else None
        flags = []
        if target is not None and prev_target is not None and target != prev_target:
            flags.append("TARGET-MOVED")
            if since_press > 0:
                switches["abandoned"] += 1
        if info.get("gated"):
            flags.append("gated")
        if info.get("no_reach"):
            flags.append("no_reach")
        if info.get("deviated"):
            flags.append("dev")
        if r_act == Action.INTERACT:
            flags.append("PRESS")

        show = (a.held is None) or (a.held in held)
        if show:
            print("%-4d %-9s %-12s %-22s %-22s %-7s %5s %5s %s"
                  % (env.t,
                     "%s%s" % (tuple(me.position), ARROW.get(tuple(me.orientation), "?")),
                     held,
                     "%s @ t+%s" % (plan[1], info.get("plan_t", "?")) if plan else "-",
                     str(info.get("base_subtask")),
                     str(r_act),
                     ("%.1f" % info["Q"]) if "Q" in info else "-",
                     ("%+.1f" % info["gain"]) if "gain" in info else "-",
                     " ".join(flags)))
        if r_act == Action.INTERACT:
            since_press = 0
        else:
            since_press += 1
        prev_target = target

        ha, _ = human.action(st)
        if drv is not None:
            drv.update(st, ha)
        nxt, _, done, _ = env.step((r_act, ha))
        done = done or mdp.is_terminal(nxt)

    print("\ntarget changed with no INTERACT in between: %d times"
          % switches["abandoned"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
