"""
MISHA NEW CHANGE - diagnostic only, not part of the main experiment.

The full static-vs-dynamic exact-subtask run (results_exact_subtask.md)
produced numerically IDENTICAL static/dynamic rows at every FOV level. This
script checks why: it runs one 'dynamic' episode and prints, every step, the
Bayesian filter's live FOV estimate (mdp_planner.sim_human_model.vision_bound)
alongside the deduced subtask, so we can see directly whether the estimate
ever leaves 0 (full vision) and whether that ever changes the deduced
subtask - rather than guessing from the aggregate numbers alone.

Run with: python -m my_methods.bayesian.diagnose_dynamic_fov
"""
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel
from my_methods.bayesian.bayesian_inference import FOVBayesFilter
from my_methods.bayesian.qmdp_bayesian_vs_static_exact_subtask import (
    build_planners_fixed, ExactSubtaskQMdpAgent, FIXED_ORDER_LIST,
)
from my_methods.bayesian.qmdp_bayesian_vs_static import LAYOUT_NAME, EPISODE_LEN, FOV_CANDIDATES

N_STEPS = 100
TRUE_FOV = 60  # the condition where static mode failed (0% completion)


def main():
    mlp, mdp_planner = build_planners_fixed()

    mdp = SteakHouseGridworld.from_layout_name(LAYOUT_NAME, start_order_list=FIXED_ORDER_LIST)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=EPISODE_LEN)

    human = SteakLimitVisionHumanModel(mlp, env.state, vision_limit=True, vision_bound=TRUE_FOV, debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(env.state)

    mdp_planner.sim_human_model.init_knowledge_base(env.state)
    mdp_planner.sim_human_model.vision_bound = 0

    teammate = ExactSubtaskQMdpAgent(mdp_planner, greedy=True, auto_unstuck=True, low_level_action_flag=True)
    teammate.set_agent_index(0)

    bayes_filter = FOVBayesFilter(mlp, env.state, fov_candidates=FOV_CANDIDATES, human_agent_index=1)

    prev_estimate = None
    for t in range(N_STEPS):
        state = env.state
        try:
            a_teammate = teammate.action(state)[0]
        except AssertionError as e:
            print(f"t={t}: CRASH in teammate.action: {e}", flush=True)
            break
        deduced_subtask = mdp_planner.sim_human_model.prev_chosen_subtask

        try:
            a_human = human.action(state)[0]
        except AssertionError as e:
            print(f"t={t}: CRASH in human.action: {e}", flush=True)
            break

        try:
            bayes_filter.step(state, a_human)
            estimate = bayes_filter.estimate()
        except AssertionError as e:
            print(f"t={t}: CRASH in bayes_filter.step: {e}", flush=True)
            break

        flag = " <-- ESTIMATE CHANGED" if estimate != prev_estimate else ""
        print(f"t={t}: robot_vision_bound_used={mdp_planner.sim_human_model.vision_bound} "
              f"new_estimate={estimate} deduced_subtask={deduced_subtask}{flag}", flush=True)
        prev_estimate = estimate

        mdp_planner.sim_human_model.vision_bound = estimate

        _, reward, done, _ = env.step((a_teammate, a_human))
        if reward > 0:
            print(f"  -> REWARD {reward} at t={t}", flush=True)
        if done:
            print(f"  -> DONE at t={t}", flush=True)
            break


if __name__ == "__main__":
    main()
