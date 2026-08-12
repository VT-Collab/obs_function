"""Does the ROUTE change what the human sees and does, before any INTERACT?

The deviation search is exponential because it branches on every action. That is
only worth paying for if the robot's PATH actually reaches the human. Two channels
exist and they are not equal:

  ROUTE     which cells the robot's body occupies, tick by tick. Reaches the human
            only through `view.robot`, and only when the robot is inside the
            human's cone -- so on layouts where the two never share sight lines it
            reaches them not at all.
  INTERACT  where and when the press lands, which rewrites counter contents and is
            what the tail prices.

If the route channel is empty before the first INTERACT, then every branch of the
search produces an IDENTICAL human, the only thing that distinguishes plans is
(cell, arrival tick), and the exhaustive action search is buying nothing that a
cheap enumeration over (cell, t) would not. That is the difference between an
exponential search and a linear one, so it is worth measuring rather than assuming.

Test 1 -- ROUTE. At each real tick, take every legal first action, then drive all
branches forward with the SAME subsequent actions, and compare the human's action
sequences. Any difference means the route reached the human.

Test 2 -- INTERACT. Press at each legal mask cell and check the world actually
changed, and that the tail values the presses differently. If every cell scored
the same, the layer would have nothing to choose between.
"""
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
from common.views import TruthView                                 # noqa: E402
from human.limited_vision_human import (LimitedVisionHuman,        # noqa: E402
                                        FORGET_HORIZON)
from robot.filter.core.qmdp import fast_clone_human, MOVES    # noqa: E402
from robot.filter.core.value_tail import tail_ticks                # noqa: E402
from robot.methods import make_robot                               # noqa: E402

LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table", "divide", "pantry"]
K = 6            # ticks rolled forward per branch


def probe(layout, fov, seed, horizon=120):
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
    env.reset()
    human = LimitedVisionHuman(mdp, fov, agent_index=1,
                               forget_horizon=FORGET_HORIZON, seed=seed)
    bot, drv = make_robot("qmdp", mdp, 0, 1, seed)
    r = collections.Counter()
    done = False
    while not done:
        st = env.state
        shadow = bot.post.shadows[bot._map_cone()[0]] if bot.post.p else None
        if shadow is not None:
            r["ticks"] += 1
            r["robot_seen"] += (shadow.view.robot is not None
                                and shadow.view.robot[0] is not None)
            # ROUTE: same continuation, different first action.
            seqs = {}
            for a0 in MOVES + [Action.STAY]:
                sim = st.deepcopy()
                h = fast_clone_human(shadow)
                acts = []
                for k in range(K):
                    ah, _ = h.action(sim)
                    acts.append(str(ah))
                    a = a0 if k == 0 else Action.STAY
                    sim, _, _, _ = mdp.get_state_transition(sim, (a, ah))
                seqs[str(a0)] = tuple(acts)
            r["route_diverged"] += (len(set(seqs.values())) > 1)

            # INTERACT: does pressing change the world, and do cells differ?
            me = st.players[0]
            pos, orient = tuple(me.position), tuple(me.orientation)
            walk = bot.walkable(st) | {pos}
            ranked = bot.baseline.rank_subtasks(st)
            if ranked:
                mask, _j, _o = bot._mask(st, ranked, pos, orient, walk)
                # every mask cell we could be standing at and facing
                tails, worlds = set(), set()
                for c in list(mask)[:12]:
                    for d in MOVES:
                        stand = (c[0] - d[0], c[1] - d[1])
                        if stand not in walk:
                            continue
                        sim = st.deepcopy()
                        p = sim.players[0]
                        p.position = stand
                        p.orientation = d
                        # PlayerState asserts held_object.position == position, so
                        # a teleport has to carry what the robot is holding.
                        if p.held_object is not None:
                            p.held_object.position = stand
                        h = fast_clone_human(shadow)
                        ah, _ = h.action(sim)
                        nxt, _, _, _ = mdp.get_state_transition(
                            sim, (Action.INTERACT, ah))
                        worlds.add(len(nxt.objects))
                        tails.add(round(tail_ticks(mdp, nxt, human_view=h.view,
                                                   human_index=1, blind=8.0), 1))
                        break
                if len(tails) > 1:
                    r["tail_varies"] += 1
                r["tail_levels"] += len(tails)
                r["world_levels"] += len(worlds)
                r["interact_ticks"] += 1
        ra, _ = bot.action(st)
        ha, _ = human.action(st)
        if drv is not None:
            drv.update(st, ha)
        nxt, _, done, _ = env.step((ra, ha))
        done = done or mdp.is_terminal(nxt)
    return r


def main():
    print("K=%d ticks rolled per branch. 'route diverged' = the human behaved "
          "differently\nfor at least one first action, with all else equal.\n" % K)
    print("%-13s %6s %11s %13s %12s %11s"
          % ("layout", "ticks", "robot seen", "route diverged", "tail varies",
             "tail levels"))
    tot = collections.Counter()
    for lay in LAYOUTS:
        r = probe(lay, 90, 0)
        n = max(r["ticks"], 1)
        ni = max(r["interact_ticks"], 1)
        print("%-13s %6d %10.1f%% %12.1f%% %11.1f%% %11.2f"
              % (lay, r["ticks"], 100.0 * r["robot_seen"] / n,
                 100.0 * r["route_diverged"] / n,
                 100.0 * r["tail_varies"] / ni, r["tail_levels"] / ni))
        for k in r:
            tot[k] += r[k]
    n = max(tot["ticks"], 1)
    ni = max(tot["interact_ticks"], 1)
    print("-" * 72)
    print("%-13s %6d %10.1f%% %12.1f%% %11.1f%% %11.2f"
          % ("ALL", tot["ticks"], 100.0 * tot["robot_seen"] / n,
             100.0 * tot["route_diverged"] / n,
             100.0 * tot["tail_varies"] / ni, tot["tail_levels"] / ni))


if __name__ == "__main__":
    main()
