"""
Compare two QMDP robot teammates on steak_island:

  - static: the robot's internal model of the human never updates - it
    always assumes vision_bound=0 (sees everything). This is the "unaware"
    planner used everywhere else in this project so far.
  - dynamic: the robot's internal model gets updated every step to the
    current best guess from FOVBayesFilter (the exact Bayesian filter in
    bayesian_inference.py), so as the filter becomes confident about the
    human's real FOV, the robot's planning adapts to it in real time.
    Concretely this is just `mdp_planner.sim_human_model.vision_bound =
    bayes_filter.estimate()` after each filter update - the QMDP planner
    reads that attribute fresh on every belief_update() call, so no deeper
    surgery is needed (see planners.py's belief_update -> sim_human_model).

This tests the actual question the paper cares about: does knowing (or
inferring) the human's FOV help the robot collaborate better - measured by
how many steps it takes to finish the order list (fewer = better) and
whether it finishes within the horizon at all.

Requires the QMDP planner already built for steak_island - see
build_qmdp_steak_island.py (expensive, run that first / separately).

Run with: python -m my_methods.bayesian.qmdp_bayesian_vs_static
"""
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, SteakKnowledgeBasePlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel, MediumQMdpPlanningAgent
from my_methods.bayesian.bayesian_inference import FOVBayesFilter

LAYOUT_NAME = "steak_island"
FOV_CANDIDATES = (60, 120, 180)
EPISODE_LEN = 300
N_EPISODES_PER_FOV = 10
MIN_ORDERS, MAX_ORDERS = 4, 8
SEARCH_DEPTH = 5
KB_SEARCH_DEPTH = 2


def build_planners():
    mdp = SteakHouseGridworld.from_layout_name(LAYOUT_NAME)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=EPISODE_LEN)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)

    non_limited_human = SteakLimitVisionHumanModel(
        mlp, env.state, vision_limit=False, vision_bound=0, kb_update_delay=1, debug=False,
    )
    non_limited_human.set_agent_index(1)

    mdp_planner = SteakKnowledgeBasePlanner.from_pickle_or_compute(
        mdp, BASE_PARAMS, force_compute_all=False,
        jmp=mlp.ml_action_manager.joint_motion_planner,
        vision_limited_human=non_limited_human,
        search_depth=SEARCH_DEPTH, kb_search_depth=KB_SEARCH_DEPTH,
    )
    mdp_planner.debug = False
    return mlp, mdp_planner


def run_episode(mlp, mdp_planner, true_fov, mode, seed=None):
    """mode: 'static' (robot never updates its FOV belief) or 'dynamic'
    (robot's belief is driven by FOVBayesFilter's live estimate)."""
    rng = np.random.RandomState(seed)
    n_orders = rng.randint(MIN_ORDERS, MAX_ORDERS + 1)
    mdp = SteakHouseGridworld.from_layout_name(LAYOUT_NAME, start_order_list=['steak'] * n_orders)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=EPISODE_LEN)

    human = SteakLimitVisionHumanModel(mlp, env.state, vision_limit=True, vision_bound=true_fov, debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(env.state)

    # mdp_planner is shared across all episodes (expensive to (re)build), but its
    # internal belief-about-the-human must be reset fresh each episode, and both
    # modes start "unaware" (vision_bound=0) - dynamic mode then updates it live.
    mdp_planner.sim_human_model.init_knowledge_base(env.state)
    mdp_planner.sim_human_model.vision_bound = 0

    teammate = MediumQMdpPlanningAgent(mdp_planner, greedy=True, auto_unstuck=True, low_level_action_flag=True)
    teammate.set_agent_index(0)

    bayes_filter = None
    if mode == 'dynamic':
        bayes_filter = FOVBayesFilter(mlp, env.state, fov_candidates=FOV_CANDIDATES, human_agent_index=1)

    total_reward = 0.0
    done = False
    steps_taken = 0
    for t in range(EPISODE_LEN):
        state = env.state
        try:
            a_teammate = teammate.action(state)[0]
            a_human = human.action(state)[0]
        except AssertionError:
            # pre-existing edge case in the library's greedy subtask logic - see
            # evaluate_bayesian_lightweight.py's docstring. End episode early.
            break

        if mode == 'dynamic':
            bayes_filter.step(state, a_human)
            mdp_planner.sim_human_model.vision_bound = bayes_filter.estimate()

        _, reward, done, _ = env.step((a_teammate, a_human))
        total_reward += reward
        steps_taken = t + 1
        if done:
            break

    return steps_taken, total_reward, done


def main():
    print("building/loading steak_island QMDP planner (run build_qmdp_steak_island.py "
          "first if this is the first time - this can take a long time uncached)...", flush=True)
    mlp, mdp_planner = build_planners()

    print(f"{'mode':>8} | {'true FOV':>8} | {'avg steps':>10} | {'avg reward':>10} | {'completion rate':>16}")
    results = {}
    for mode in ('static', 'dynamic'):
        for true_fov in FOV_CANDIDATES:
            steps_list, reward_list, done_list = [], [], []
            for ep in range(N_EPISODES_PER_FOV):
                steps, reward, done = run_episode(mlp, mdp_planner, true_fov, mode, seed=ep * 11 + true_fov)
                steps_list.append(steps)
                reward_list.append(reward)
                done_list.append(done)
            results[(mode, true_fov)] = (steps_list, reward_list, done_list)
            print(f"{mode:>8} | {true_fov:>8} | {np.mean(steps_list):>10.1f} | "
                  f"{np.mean(reward_list):>10.1f} | {np.mean(done_list) * 100:>15.1f}%", flush=True)

    print("\n--- static vs dynamic, averaged over all FOVs ---")
    for mode in ('static', 'dynamic'):
        all_steps = [s for fov in FOV_CANDIDATES for s in results[(mode, fov)][0]]
        all_reward = [r for fov in FOV_CANDIDATES for r in results[(mode, fov)][1]]
        all_done = [d for fov in FOV_CANDIDATES for d in results[(mode, fov)][2]]
        print(f"{mode:>8}: avg steps={np.mean(all_steps):.1f}  avg reward={np.mean(all_reward):.1f}  "
              f"completion rate={np.mean(all_done) * 100:.1f}%")


if __name__ == "__main__":
    main()
