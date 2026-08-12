"""Checks for the SEARCH: the mask, the clone, and the certainty gate. Run it directly.

    python -m robot.filter.tests.test_qmdp

Three things are checked, and the first is the one the speed rests on.

1. `fast_clone_human` == `clone_human`. The fast clone replaces copy.deepcopy with
   shallow dict copies, which is only sound because every value in the view's
   stores is immutable except `_stations`' sets. That is an argument; this is the
   check. Both clones are driven forward over a real rollout and every action and
   every internal field is compared tick by tick, following the rule the rest of
   this package already follows -- an optimisation is proven equivalent BEFORE it
   is relied on (RESULTS.md section 10).

2. The mask holds every stash counter. The whole point of the rewrite is that no
   cell is dropped, so on a tick where the baseline wants to stash, the number of
   stash cells in the mask must equal the number in the full ranking -- not
   `cell_k` of them.

3. The certainty gate. Below `certainty` the filter must return its baseline's
   action untouched and report `gated`.
"""
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
from robot.filter.core.qmdp import clone_human                # noqa: E402
from robot.filter.core.qmdp import fast_clone_human           # noqa: E402
from robot.methods import make_robot                               # noqa: E402

LAYOUTS = ["divide", "butchery", "pantry", "back_bar"]
FAIL = []


def check(ok, label):
    print("%-4s %s" % ("ok" if ok else "FAIL", label))
    if not ok:
        FAIL.append(label)


def human_fields(h):
    v = h.view
    return (h.t, h.explore_ticks, h.last_subtask, h._goal, tuple(h._recent),
            sorted(h.abandoned), v.robot, v.t, sorted(v.contents.items()),
            sorted(v.seen_count.items()), sorted(v.known_terrain.items()),
            sorted((k, sorted(s)) for k, s in v._stations.items()),
            h._rng.getstate())


def test_fast_clone():
    """Drive both clones through a rollout and compare everything, every tick."""
    for lay in LAYOUTS:
        for fov in (30, 90, 360):
            mdp = SteakHouseGridworld.from_layout_name(lay)
            env = OvercookedEnv.from_mdp(mdp, horizon=60, info_level=0)
            env.reset()
            human = LimitedVisionHuman(mdp, fov, agent_index=1,
                                       forget_horizon=FORGET_HORIZON, seed=0)
            bot, drv = make_robot("handoff", mdp, 0, 1, 0)
            slow = fast = None
            same = True
            done = False
            while not done and same:
                st = env.state
                # Fork both ways off the SAME live human, then step both against
                # the same states and require identical behaviour throughout.
                if slow is None:
                    slow, fast = clone_human(human), fast_clone_human(human)
                a_slow, _ = slow.action(st)
                a_fast, _ = fast.action(st)
                if a_slow != a_fast or human_fields(slow) != human_fields(fast):
                    same = False
                    break
                ra, _ = bot.action(st)
                ha, _ = human.action(st)
                if drv is not None:
                    drv.update(st, ha)
                nxt, _, done, _ = env.step((ra, ha))
                done = done or mdp.is_terminal(nxt)
            check(same, "fast_clone_human == clone_human  %s fov%d" % (lay, fov))


def test_mask_keeps_every_stash_cell():
    """No stash counter may be dropped. cell_k is gone; prove it."""
    for lay in LAYOUTS:
        mdp = SteakHouseGridworld.from_layout_name(lay)
        env = OvercookedEnv.from_mdp(mdp, horizon=120, info_level=0)
        env.reset()
        human = LimitedVisionHuman(mdp, 90, agent_index=1,
                                   forget_horizon=FORGET_HORIZON, seed=0)
        bot, drv = make_robot("qmdp", mdp, 0, 1, 0)
        seen_any = False
        ok = True
        done = False
        while not done:
            st = env.state
            ranked = bot.baseline.rank_subtasks(st)
            me = st.players[0]
            pos, orient = tuple(me.position), tuple(me.orientation)
            walk = bot.walkable(st) | {pos}
            stash_all = {c for _t, v, c in ranked if v == "stash"}
            if stash_all:
                mask, _jobs, owner = bot._mask(st, ranked, pos, orient, walk)
                in_mask = stash_all & mask
                # Either stash is not among the top-m jobs at all (then none of
                # its cells are in the mask), or it is and ALL of them are.
                if in_mask and in_mask != stash_all:
                    ok = False
                    print("     %s: %d/%d stash cells in mask"
                          % (lay, len(in_mask), len(stash_all)))
                    break
                if in_mask:
                    seen_any = True
            ra, _ = bot.action(st)
            ha, _ = human.action(st)
            if drv is not None:
                drv.update(st, ha)
            nxt, _, done, _ = env.step((ra, ha))
            done = done or mdp.is_terminal(nxt)
        check(ok, "mask keeps ALL stash cells  %s%s"
              % (lay, "" if seen_any else "  (no stash tick seen)"))


def test_certainty_gate():
    """certainty=1.01 can never be met, so the filter must be a pass-through."""
    mdp = SteakHouseGridworld.from_layout_name("divide")
    env = OvercookedEnv.from_mdp(mdp, horizon=40, info_level=0)
    env.reset()
    human = LimitedVisionHuman(mdp, 60, agent_index=1,
                               forget_horizon=FORGET_HORIZON, seed=0)
    bot, drv = make_robot("qmdp", mdp, 0, 1, 0)
    bot.certainty = 1.01
    gated = total = 0
    dev = 0
    done = False
    while not done:
        st = env.state
        ra, info = bot.action(st)
        total += 1
        gated += bool(info.get("gated"))
        dev += bool(info.get("deviated"))
        ha, _ = human.action(st)
        if drv is not None:
            drv.update(st, ha)
        nxt, _, done, _ = env.step((ra, ha))
        done = done or mdp.is_terminal(nxt)
    check(gated == total and dev == 0,
          "certainty gate: %d/%d ticks gated, %d deviations" % (gated, total, dev))


if __name__ == "__main__":
    test_fast_clone()
    test_mask_keeps_every_stash_cell()
    test_certainty_gate()
    print("\n%d failure(s)" % len(FAIL))
    sys.exit(1 if FAIL else 0)
