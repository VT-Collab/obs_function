"""
MISHA NEW CHANGE - diagnostic only, not part of the main experiment.

Round 1: robot froze solid at step 14 (fixed with a hand-rolled unstuck).
Round 2: fixed the freeze, but exposed the human getting stuck in a 2-cycle
oscillation for 44+ steps trying to route around the now-idling robot.
Round 3 (current): qmdp_bayesian_vs_static_exact_subtask.py was rearchitected
to use ExactSubtaskQMdpAgent (a MediumQMdpPlanningAgent subclass that keeps
the real resolve_stuck-backed .action() pipeline, plus a position-history
check for oscillations) instead of calling mdp_planner.step() directly. This
re-runs the same trace with that fix to confirm the team can now actually
deliver an order.

Run with: python -m my_methods.bayesian.diagnose_exact_subtask
"""
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel
from my_methods.bayesian.qmdp_bayesian_vs_static_exact_subtask import (
    build_planners_fixed, ExactSubtaskQMdpAgent, FIXED_ORDER_LIST,
)
from my_methods.bayesian.qmdp_bayesian_vs_static import LAYOUT_NAME, EPISODE_LEN

N_STEPS = 100
TRUE_FOV = 120


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

    for t in range(N_STEPS):
        state = env.state
        robot_pos = state.players[0].pos_and_or
        human_pos = state.players[1].pos_and_or
        robot_held = state.players[0].held_object.name if state.players[0].held_object else None
        human_held = state.players[1].held_object.name if state.players[1].held_object else None

        try:
            a_teammate = teammate.action(state)[0]
        except Exception as e:
            print(f"t={t}: CRASH in teammate.action: {type(e).__name__}: {e}", flush=True)
            break
        a_human = human.action(state)[0]

        print(f"t={t}: robot@{robot_pos} held={robot_held} action={a_teammate} | "
              f"human@{human_pos} held={human_held} action={a_human} | "
              f"deduced_subtask={mdp_planner.sim_human_model.prev_chosen_subtask}",
              flush=True)

        _, reward, done, _ = env.step((a_teammate, a_human))
        if reward > 0:
            print(f"  -> REWARD {reward} at t={t}", flush=True)
        if done:
            print(f"  -> DONE at t={t}", flush=True)
            break


if __name__ == "__main__":
    main()
