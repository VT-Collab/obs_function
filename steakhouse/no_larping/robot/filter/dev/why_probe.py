"""On the ticks it drops a rung, is the job it picks actually CLOSER?

The claim to test: the layer trades urgency for proximity. C is pure estimated
ticks (t_end + tail) with no tier term anywhere, so if that is what is happening
the chosen cell should be nearer than the baseline's on off-tier ticks. Measures
walking distance to both cells at the moment of the choice.
"""
import collections
import os
import sys

# Derived from __file__ so this runs unchanged on a compute node, the same
# reason evaluate.py does it that way.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
import overcooked_ai_py
overcooked_ai_py.LAYOUTS_DIR = os.path.join(ROOT, "layout", "layouts")

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld
from common import geometry as geo
from common.tasks import TIER_NAME
from common.views import TruthView
from human.limited_vision_human import LimitedVisionHuman, FORGET_HORIZON
from robot.filter.harness.evaluate import build_robot

TIER_IDX = {v: k for k, v in TIER_NAME.items()}
CELLS = [("divide", 60), ("butchery", 360), ("chefs_table", 60),
         ("banquet_pass", 360), ("back_bar", 90), ("pantry", 180)]

buckets = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])
for layout, fov in CELLS:
    for seed in range(3):
        mdp = SteakHouseGridworld.from_layout_name(layout)
        env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
        env.reset()
        human = LimitedVisionHuman(mdp, fov, agent_index=1,
                                   forget_horizon=FORGET_HORIZON, seed=seed)
        bot, drv = build_robot("qmdp", mdp, 0, 1, seed, 3, 40)
        done = False
        while not done:
            state = env.state
            ra, info = bot.action(state)
            q, b = info.get("subtask"), info.get("base_subtask")
            if "pair_idx" in info and q and b and q[:2] != b[:2]:
                me = state.players[0]
                pos = tuple(me.position)
                walk = TruthView(mdp, state).walkable | {pos}
                dq = geo.path_len(walk, pos, q[2])
                db = geo.path_len(walk, pos, b[2])
                dt = TIER_IDX[q[0]] - TIER_IDX[b[0]]
                if dq is not None and db is not None:
                    k = "off-tier (%+d rungs)" % dt if dt else "same tier, new verb"
                    e = buckets[k]
                    e[0] += 1
                    e[1] += dq
                    e[2] += db
                    e[3] += int(dq < db)
                    e[4] += info.get("gain") or 0.0
            # FEED THE POSTERIOR. Without this the cone belief sits at its prior
            # for the whole episode, which makes the thing being measured
            # theta-blind rather than qmdp -- and theta enters C, so it changes
            # which job wins and therefore every row of this table. The human's
            # action must also be taken exactly once per tick: it advances the
            # human's clock and its view, so calling it a second time inside
            # env.step would step the human twice per tick.
            ha, _ = human.action(state)
            drv.update(state, ha)
            nxt, _, done, _ = env.step((ra, ha))
            done = done or mdp.is_terminal(nxt)
    print("%-13s fov%-4d done" % (layout, fov), file=sys.stderr)

print("qmdp, 6 cells x 3 seeds. Distance to the CHOSEN cell vs the BASELINE's cell")
print("at the tick of the choice, on ticks where the job differs.")
print()
print("%-24s %6s  %8s %8s  %8s  %8s" % ("switch", "ticks", "chosen d",
                                        "base d", "closer%", "est gain"))
for k in sorted(buckets, key=lambda k: -buckets[k][0]):
    n, sq, sb, closer, g = buckets[k]
    print("%-24s %6d  %8.1f %8.1f  %7.1f%%  %8.2f"
          % (k, n, sq / n, sb / n, 100.0 * closer / n, g / n))
