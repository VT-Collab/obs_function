"""Does pair_k=1 actually collapse the search to ONE candidate?

The retired enumerating filter's action():
    pairs = all_pairs[:self.pair_k]                       # len 1 when pair_k=1
    for must in (self.committed, base_pair):
        if must is not None and must in all_pairs and must not in pairs:
            pairs = pairs[:max(1, self.pair_k - 1)] + [must]

With pair_k=1, max(1, 0) == 1, so that line is pairs[:1] + [must] -> LENGTH 2.
The must-keep guard cannot shrink below one element, so on any tick where the
committed pair is legal, in all_pairs, and not already pairs[0], the parity
control scores two candidates instead of one. Count it.
"""
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
from human.limited_vision_human import LimitedVisionHuman, FORGET_HORIZON
from robot.filter.harness.evaluate import build_robot

HORIZON = int(sys.argv[1]) if len(sys.argv) > 1 else 400
LAYOUTS = ["back_bar", "banquet_pass", "butchery", "chefs_table", "divide", "pantry"]
FOVS = [30, 60, 90, 180, 360]
SEEDS = range(3)

tot = {"ticks": 0, "multi": 0, "dev": 0, "multi_dev": 0}
per = {}
for layout in LAYOUTS:
    for fov in FOVS:
        for seed in SEEDS:
            mdp = SteakHouseGridworld.from_layout_name(layout)
            env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0)
            env.reset()
            human = LimitedVisionHuman(mdp, fov, agent_index=1,
                                       forget_horizon=FORGET_HORIZON, seed=seed)
            bot, drv = build_robot("qmdp-base", mdp, 0, 1, seed, 3, 40)
            done = False
            while not done:
                state = env.state
                ra, info = bot.action(state)
                ha, _ = human.action(state)
                if drv is not None:
                    drv.update(state, ha)
                if "n_pairs" in info:
                    tot["ticks"] += 1
                    multi = info["n_pairs"] > 1
                    dev = bool(info["deviated"])
                    tot["multi"] += multi
                    tot["dev"] += dev
                    tot["multi_dev"] += (multi and dev)
                    if multi or dev:
                        k = (layout, fov)
                        p = per.setdefault(k, [0, 0])
                        p[0] += multi
                        p[1] += dev
                nxt, _, done, _ = env.step((ra, ha))
                done = done or mdp.is_terminal(nxt)
        print("%-13s fov%-4d done" % (layout, fov), file=sys.stderr)

print("qmdp-base (only_base=True, cell_k=1, pair_k=1), horizon %d, seeds 0-2" % HORIZON)
print("  scored ticks                  %6d" % tot["ticks"])
print("  ticks with n_pairs > 1        %6d  %5.2f%%"
      % (tot["multi"], 100.0 * tot["multi"] / max(tot["ticks"], 1)))
print("  ticks that DEVIATED           %6d  %5.2f%%"
      % (tot["dev"], 100.0 * tot["dev"] / max(tot["ticks"], 1)))
print("  deviated AND n_pairs > 1      %6d" % tot["multi_dev"])
print()
print("cells where either fired (multi, dev):")
for k in sorted(per):
    print("  %-13s fov%-4d  multi=%-5d dev=%d" % (k[0], k[1], per[k][0], per[k][1]))
