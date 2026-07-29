"""
Build (or load, if already cached) the QMDP planner for steak_island in its
"unaware" configuration (robot's internal model assumes the human sees
everything - vision_bound=0).

MISHA NEW CHANGE - critical fix: steak_island.layout's "start_order_list" is
written as the raw string 'steak, steak' (12 characters), not a proper
Python list. State construction in planners.py's init_states() has a loop
over order_list that DOUBLES the state count per order - iterating over a
12-character string instead of a 2-element list inflates the state space by
2^12 / 2^2 = 1024x. Measured: this alone takes the QMDP state count from an
intended ~28,812 (a totally tractable ~6.6GB transition matrix) up to
38,569,664 (a literal 10.6 PETABYTE matrix - confirmed via a real
MemoryError on CARC). The fix is just to always pass start_order_list as an
explicit, correct list when building the mdp here - no changes needed to
planners.py itself, the bug is purely in how the .layout file's string gets
consumed if you don't override it.

Run with: python -m my_methods.bayesian.build_qmdp_steak_island
"""
import time
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, SteakKnowledgeBasePlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel

LAYOUT_NAME = "steak_island"
SEARCH_DEPTH = 5
KB_SEARCH_DEPTH = 2
# MISHA NEW CHANGE: must be a real list, not the layout file's raw comma-string default (see module docstring)
FIXED_ORDER_LIST = ['steak', 'steak']


def main():
    t0 = time.time()
    mdp = SteakHouseGridworld.from_layout_name(LAYOUT_NAME, start_order_list=FIXED_ORDER_LIST)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=200)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)
    print(f"mlp ready in {time.time() - t0:.1f}s", flush=True)

    non_limited_human = SteakLimitVisionHumanModel(
        mlp, env.state, vision_limit=False, vision_bound=0, kb_update_delay=1, debug=False,
    )
    non_limited_human.set_agent_index(1)

    t0 = time.time()
    SteakKnowledgeBasePlanner.from_pickle_or_compute(
        mdp, BASE_PARAMS, force_compute_all=False,
        jmp=mlp.ml_action_manager.joint_motion_planner,
        vision_limited_human=non_limited_human,
        search_depth=SEARCH_DEPTH, kb_search_depth=KB_SEARCH_DEPTH,
    )
    print(f"QMDP planner ready in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
