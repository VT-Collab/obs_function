"""Which stations can each agent actually WALK to? Derived, never transcribed.

    python -m robot.filter.analysis.layout_facts

This table is quoted in DESIGN.md section 0 and RESULTS.md section 3, and it was
transcribed by hand once. THE LAYOUTS THEN CHANGED UNDER IT: four of the six
.layout files were edited mid-experiment, chefs_table alone losing the robot's
pot and its onion, and the transcribed table went on describing a kitchen that
no longer existed. It had been correct when written, which is worse than being
wrong -- nothing about it looked stale. A whole conclusion in RESULTS.md rested
on it and had to be withdrawn.

Reading it by eye is also just hard: a station in the dividing wall belongs to
BOTH rooms, a corridor that looks open is capped by a counter three rows down,
and the room an agent starts in is not always the room the nearest station
faces. So it is computed here, from the same TruthView and the same BFS the
policies themselves use, and the docs quote this output rather than a copy of it.

If a grid's numbers disagree with an earlier grid's, CHECK THIS FIRST. Every
JSONL row carries a `layout_sha` for the same reason (evaluate.py) -- two grids
with different fingerprints are measuring different kitchens and cannot be
diffed, however tempting the diff looks.

WHAT IT ANSWERS. Whether a dish is completable by one agent, which is the
premise the whole value function is built on. A dish needs a steak (pot + meat),
a garnish (board + onion) and a washed plate (sink + plate), then a serve hatch.
If nobody can do all of that alone, every dish crosses the divide through a
counter both agents can stand at, and a tail that cannot express a relay reports
"impossible" for every candidate and the layer goes inert.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../robot/filter/analysis
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../no_larping
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(ROOT)))
sys.path.insert(0, ROOT)

import overcooked_ai_py                                            # noqa: E402
overcooked_ai_py.LAYOUTS_DIR = os.path.join(ROOT, "layout", "layouts")

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv      # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld  # noqa: E402
from common import geometry as geo                                 # noqa: E402
from common.views import TruthView                                 # noqa: E402
from robot.filter.harness.evaluate import LAYOUTS                          # noqa: E402

KINDS = "PBWMODS"
NAMES = {"P": "pot", "B": "board", "W": "sink", "M": "meat",
         "O": "onion", "D": "plate", "S": "serve"}
# A dish, decomposed: everything except the hatch it is finally handed over at.
ASSEMBLE = "PBWMOD"


def reach(mdp, state, i):
    """The station kinds agent `i` can walk to, as a 'PBWMODS' mask string."""
    v = TruthView(mdp, state)
    walk = set(v.walkable)
    field = geo.dist_field(walk, tuple(state.players[i].position))
    out = ""
    for k in KINDS:
        ok = any(n in field for c in v.stations(k)
                 for n in geo.adjacent_standing_cells(walk, c))
        out += k if ok else "."
    return out


def facts(layout):
    mdp = SteakHouseGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=1, info_level=0)
    env.reset()
    # Agent 0 is the robot seat and agent 1 the human seat, matching
    # evaluate.py's human_index=1 default and watch.py's.
    return reach(mdp, env.state, 0), reach(mdp, env.state, 1)


def main():
    print("%-13s %-9s %-9s  %s" % ("layout", "robot", "human", "can assemble alone"))
    print("-" * 62)
    solo = []
    for lay in LAYOUTS:
        r, h = facts(lay)
        who = [n for n, m in (("robot", r), ("human", h))
               if all(c in m for c in ASSEMBLE)]
        solo.append((lay, who))
        print("%-13s %-9s %-9s  %s" % (lay, r, h, ", ".join(who) or "neither"))
    print("\n(P pot B board W sink M meat O onion D plate S serve; '.' = cannot reach)")
    print("A dish needs %s, then a serve hatch." % ASSEMBLE)
    none = [l for l, w in solo if not w]
    print("\nNeither agent can assemble alone on %d of %d layouts: %s"
          % (len(none), len(solo), ", ".join(none)))
    for lay, who in solo:
        if who:
            print("  %s: %s can assemble alone" % (lay, " and ".join(who)))
    # The premise, restated as a check rather than a claim.
    hatch = [lay for lay in LAYOUTS if "S" in facts(lay)[0]]
    print("\nRobot can reach a serve hatch on: %s"
          % (", ".join(hatch) if hatch else "NO layout -- every dish is handed over"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
