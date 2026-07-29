"""
Evaluate FOVBayesFilter.

Setup matches the paper's actual scenario (same as agents_play() in
scripts/steak_study.py): one FOV-limited human + one robot teammate running
the SteakKnowledgeBasePlanner QMDP planner. We use the "unaware" planner
variant here (robot assumes the human sees everything) since we're only
testing the raw accuracy of the FOV filter itself, not yet feeding its
estimate back into the robot's own planning.

For each candidate FOV, we simulate an episode where a real
SteakLimitVisionHumanModel with that exact FOV plays the human role (so we
know ground truth), and feed the resulting (state, human_action) pairs into
the filter step by step. Then we check how often the filter's guess matches
the ground-truth FOV.

The QMDP planner is loaded once (~5.9GB, reused from
overcooked_ai_py/data/planners/) and shared across all episodes - it's an
expensive one-time load, not something we want to repeat per episode.

Run with: python -m my_methods.bayesian.evaluate_bayesian
"""
import numpy as np
from overcooked_ai_py.helpers import init_steak_env, BASE_PARAMS
from overcooked_ai_py.planning.planners import MediumLevelPlanner, SteakKnowledgeBasePlanner
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel, MediumQMdpPlanningAgent
from my_methods.bayesian.bayesian_inference import FOVBayesFilter

LVL_STR = """XXXXXXXXXXXXXXX
XXXOXMXXXPXXDXX
XX           XX
XX 2         XX
XX        1  XX
XX   XBWXX   XX
XX   XXSSX   XX
XX           XX
XX           XX
XX           XX
XXXXXXXXXXXXXXX"""

FOV_CANDIDATES = (60, 120, 180)
EPISODE_LEN = 100
N_EPISODES_PER_FOV = 10

# the FOV the cached "unaware" QMDP planner was built to believe the human has
# (vision_limit=False -> it doesn't actually use this, it assumes full vision)
SEARCH_DEPTH = 5
KB_SEARCH_DEPTH = 2


def build_unaware_teammate_planner():
    """Build (or, in practice, load from the cached pickle) the robot's QMDP
    planner in its "unaware" configuration: it plans assuming the human can
    see everything, regardless of what the human's actual FOV is."""
    env = init_steak_env(LVL_STR, horizon=EPISODE_LEN)
    mlp = MediumLevelPlanner.from_pickle_or_compute(env.mdp, BASE_PARAMS, force_compute=False)

    non_limited_human = SteakLimitVisionHumanModel(
        mlp, env.state, vision_limit=False, vision_bound=0, kb_update_delay=1, debug=False,
    )
    non_limited_human.set_agent_index(1)

    mdp_planner = SteakKnowledgeBasePlanner.from_pickle_or_compute(
        env.mdp, BASE_PARAMS, force_compute_all=False,
        jmp=mlp.ml_action_manager.joint_motion_planner,
        vision_limited_human=non_limited_human,
        search_depth=SEARCH_DEPTH, kb_search_depth=KB_SEARCH_DEPTH,
    )
    # the cached pickle was saved with debug=True baked in, which makes every
    # .step() call print a huge dump - silence it now that it's loaded
    mdp_planner.debug = False
    return mlp, mdp_planner


def run_episode(mlp, mdp_planner, true_fov, horizon=EPISODE_LEN, seed=None):
    """Play one episode with a human whose real FOV is `true_fov`, run the
    Bayes filter alongside it, and report how accurate the filter was."""
    if seed is not None:
        np.random.seed(seed)

    env = init_steak_env(LVL_STR, horizon=horizon)

    # ground truth: a human that really does have vision_bound == true_fov
    human = SteakLimitVisionHumanModel(mlp, env.state, vision_limit=True, vision_bound=true_fov, debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(env.state)

    # the robot teammate: fixed QMDP planner, doesn't know the human's real FOV.
    # mdp_planner is shared across episodes (it's expensive to load), but it keeps
    # its own internal "belief" human model (mdp_planner.sim_human_model) that must
    # be reset to the fresh starting state each episode, or it carries over stale
    # knowledge from the previous episode and crashes.
    mdp_planner.sim_human_model.init_knowledge_base(env.state)
    teammate = MediumQMdpPlanningAgent(mdp_planner, greedy=True, auto_unstuck=True, low_level_action_flag=True)
    teammate.set_agent_index(0)

    bayes_filter = FOVBayesFilter(mlp, env.state, fov_candidates=FOV_CANDIDATES, human_agent_index=1)
    uniform_prob = 1.0 / len(FOV_CANDIDATES)

    correct_steps = 0
    total_steps = 0
    ever_informative = False  # did the human ever take an action that actually
    # distinguished between the FOV hypotheses? If not, the filter has nothing to
    # go on and its guess is just argmax's tie-break, not a real inference - worth
    # tracking separately from "wrong despite evidence"
    for _ in range(horizon):
        state = env.state
        try:
            a_teammate = teammate.action(state)[0]
            a_human = human.action(state)[0]
            belief = bayes_filter.step(state, a_human)
        except AssertionError:
            # pre-existing edge case in the library's greedy subtask logic
            # (agent.py's ml_action can find zero valid motion goals on some
            # longer/unusual order sequences) - can hit the ground-truth
            # human, teammate, or any of the filter's shadow models. Not
            # something to patch in shared library code here, just end the
            # episode early like a natural completion
            break

        if np.max(np.abs(belief - uniform_prob)) > 0.01:
            ever_informative = True
        total_steps += 1
        if bayes_filter.estimate() == true_fov:
            correct_steps += 1

        _, _, done, _ = env.step((a_teammate, a_human))
        if done:
            break

    step_accuracy = correct_steps / total_steps if total_steps > 0 else 0.0
    final_correct = bayes_filter.estimate() == true_fov
    return step_accuracy, final_correct, ever_informative


def main():
    print("loading QMDP teammate planner (one-time, reused across all episodes)...")
    mlp, mdp_planner = build_unaware_teammate_planner()

    print(f"{'true FOV':>10} | {'final-guess accuracy':>21} | {'avg step accuracy':>18} | {'episodes w/ any evidence':>24}")
    all_finals, all_informative = [], []
    for true_fov in FOV_CANDIDATES:
        finals, step_accs, informative = [], [], []
        for ep in range(N_EPISODES_PER_FOV):
            step_acc, final_correct, ever_informative = run_episode(mlp, mdp_planner, true_fov, seed=ep)
            finals.append(final_correct)
            step_accs.append(step_acc)
            informative.append(ever_informative)
        all_finals.extend(finals)
        all_informative.extend(informative)
        print(f"{true_fov:>10} | {np.mean(finals) * 100:>20.1f}% | {np.mean(step_accs) * 100:>17.1f}% "
              f"| {sum(informative)}/{len(informative):>22}")

    print(f"\noverall final-guess accuracy: {np.mean(all_finals) * 100:.1f}% over {len(all_finals)} episodes")
    if sum(all_informative) > 0:
        informed_acc = np.mean([f for f, i in zip(all_finals, all_informative) if i])
        print(f"accuracy restricted to episodes with any real evidence: {informed_acc * 100:.1f}% "
              f"({sum(all_informative)}/{len(all_informative)} episodes had evidence)")
    else:
        print("WARNING: no episode ever produced evidence that distinguished the FOV hypotheses - "
              "the human's actions never depended on FOV in this scenario, so the accuracy numbers "
              "above are meaningless (pure argmax tie-break, not real inference). Try a layout/spawn "
              "where the human has to notice something outside a narrow forward cone.")


if __name__ == "__main__":
    main()
