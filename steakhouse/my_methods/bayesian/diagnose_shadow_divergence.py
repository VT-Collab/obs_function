"""
MISHA NEW CHANGE - diagnostic for WHY the Bayes filter's posterior never moves.

The inference batch reports goal_divergence_rate = 0.000 and final entropy
exactly ln(3) = 1.099: all three FOV shadows choose the SAME goal at every step,
so all three likelihoods are identical and the posterior stays at the prior.

This is a different measurement from the one that validated these layouts.
fov_parallel_layout_search.py measured divergence between INDEPENDENT ROLLOUTS -
each FOV shadow drove its own simulation, so their states drifted apart and the
subtask sequences diverged. The Bayes filter instead runs all shadows as
SPECTATORS on ONE real trajectory: same pose, same world, differing only in what
each has been able to SEE. Divergence-under-independent-rollout does not imply
divergence-under-shared-spectating, and this script measures the latter directly.

Per step it prints, for the real human and each shadow:
  - the chosen subtask
  - the chosen motion goal
  - the likelihood assigned to the action the human actually took
  - the number of objects in that shadow's knowledge base (a cheap proxy for
    "how much of the world has this FOV seen")
so we can see whether the KBs differ at all, whether differing KBs ever produce
differing subtasks, and at what step that starts.

Only rank01's MediumLevelPlanner cache exists locally, so that is the default.

Run with: python -m my_methods.bayesian.diagnose_shadow_divergence [rank] [true_fov] [n_steps]
"""
import os
import sys

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action
from my_methods.bayesian.fov_subtask_divergence_test import build_mdp_and_mlp
from my_methods.bayesian.sticky_subtask_human import StickySubtaskHumanModel
from my_methods.bayesian.fov_sustained_batch import robot_next_action
from my_methods.bayesian.fov_bayes_filter import SteakBayesFOVInference, apply_initial_kb
from my_methods.bayesian.fov_inference_batch import (
    parse_layout_file, HIDE_POS, LAYOUTS_DIR, ROBOT_MODE, INITIAL_KB)
from overcooked_ai_py.agents.agent import GreedySteakHumanModel


def kb_key(agent):
    """The game-relevant summary of this shadow's knowledge base:
    'num_item_in_pot.chop_time.wash_time.robot_held_object' (agent.py:1118).

    These four fields are exactly what FOV can make hypotheses disagree about,
    and exactly what ml_action() branches on - so if these are identical across
    shadows, the subtask choice CANNOT diverge, no matter the layout. An earlier
    version of this diagnostic counted KB entries instead, which excluded
    pot/chop/sink/other_player and therefore measured the wrong thing entirely.
    """
    try:
        return agent.get_kb_key(agent.knowledge_base)
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def main():
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 120

    name = f"fov_search_rank{rank:02d}"
    cfg = parse_layout_file(os.path.join(LAYOUTS_DIR, f"{name}.layout"))
    true_fov = int(sys.argv[2]) if len(sys.argv) > 2 else cfg["fov_triple"][1]
    hide_pos = HIDE_POS[rank]
    robot_idx, human_idx = 0, 1

    print(f"=== {name}  fov_triple={cfg['fov_triple']}  true_fov={true_fov}  "
          f"n_steps={n_steps} ===", flush=True)

    mdp, mlp = build_mdp_and_mlp(name, cfg["grid"], order_list=cfg["order_list"],
                                 cook_time=cfg["cook_time"], chop_time=cfg["chop_time"],
                                 wash_time=cfg["wash_time"],
                                 num_items_for_steak=cfg["num_items_for_steak"])

    setup_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300)
    setup_state = setup_env.state.deepcopy()
    sim_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300)

    real_human = StickySubtaskHumanModel(mlp, setup_state, vision_limit=True,
                                         vision_bound=true_fov, debug=False)
    real_human.set_agent_index(human_idx)
    apply_initial_kb(real_human, setup_state, INITIAL_KB)

    robot = None
    if ROBOT_MODE == "work":
        robot = GreedySteakHumanModel(mlp)
        robot.set_agent_index(robot_idx)
    print(f"robot={ROBOT_MODE}  initial_kb={INITIAL_KB}", flush=True)

    inf = SteakBayesFOVInference(mlp, setup_state, candidate_fovs=cfg["fov_triple"],
                                 human_agent_index=human_idx, likelihood="greedy",
                                 initial_kb=INITIAL_KB)

    fovs = list(inf.candidate_fovs)
    hdr = "  ".join(f"fov{f:<3}(sub/goal/p/kb)" for f in fovs)
    print(f"{'t':>3} {'human_sub':<18} {'act':<10} {hdr}", flush=True)

    n_subtask_div = 0
    n_goal_div = 0
    n_kb_div = 0
    first_div = None

    for t in range(n_steps):
        state = sim_env.state
        if robot is not None:
            try:
                a_robot, _ = robot.action(state)
            except Exception:
                a_robot = Action.STAY
        else:
            a_robot = robot_next_action(mlp, state, robot_idx, cfg["m_pos"], hide_pos)
        try:
            a_human, _ = real_human.action(state)
        except AssertionError:
            print(f"{t:>3} real human STUCK - ending", flush=True)
            break

        inf.update(state, a_human)

        # Read per-hypothesis detail recorded during update() - do NOT re-probe,
        # that would advance each shadow's knowledge base twice on one state.
        cells = []
        subs, goals, kbs = [], [], []
        for fov in fovs:
            d = inf.last_step_detail[fov]
            p, goal, sub = d["p"], d["goal"], d["subtask"]
            k = kb_key(inf.hypothesis_agents[fov])
            subs.append(sub)
            goals.append(str(goal))
            kbs.append(k)
            pstr = "None" if p is None else f"{p:.3f}"
            cells.append(f"{str(sub)[:13]:<13}/{pstr}/{k}")

        if len(set(subs)) > 1:
            n_subtask_div += 1
            if first_div is None:
                first_div = t
        if len(set(goals)) > 1:
            n_goal_div += 1
        if len(set(kbs)) > 1:
            n_kb_div += 1

        post = inf.posterior()
        pstr = " ".join(f"{f}:{post[f]:.3f}" for f in fovs)
        rp = state.players[robot_idx]
        robot_obj = rp.held_object.name if rp.held_object is not None else "-"
        # The TRUE knowledge base, i.e. what a full-vision observer would know.
        # If this never changes, nothing is happening in the world for any FOV
        # to have a differing opinion about.
        truth = kb_key(real_human)
        print(f"{t:>3} {str(real_human.prev_chosen_subtask)[:17]:<17} "
              f"{str(a_human):<9} R@{str(rp.position):<8}{robot_obj:<7} "
              f"truth={truth:<14} " + "  ".join(cells) + f"  | post {pstr}", flush=True)

        joint = [Action.STAY, Action.STAY]
        joint[robot_idx], joint[human_idx] = a_robot, a_human
        _, _, done, _ = sim_env.step(tuple(joint))
        if done:
            print(f"{t:>3} episode DONE", flush=True)
            break

    seen = inf.n_steps_seen
    print(f"\n=== divergence summary over {seen} steps ===")
    print(f"  steps where shadow KB SIZES differ  : {n_kb_div}")
    print(f"  steps where shadow SUBTASKS differ  : {n_subtask_div}  (first at t={first_div})")
    print(f"  steps where shadow GOALS differ     : {n_goal_div}")
    print(f"  filter's own goal-divergent count   : {inf.n_goal_divergent_steps}")
    print(f"  shadow crashes                      : {inf.n_crashes}")
    print(f"  final posterior : {inf.posterior()}")
    print(f"  MAP={inf.map_fov()}  true={true_fov}  entropy={inf.entropy():.4f}")


if __name__ == "__main__":
    main()
