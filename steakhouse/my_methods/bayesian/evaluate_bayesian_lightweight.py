"""
Evaluate FOVBayesFilter on a layout empirically confirmed to make FOV
actually matter during real play - "steak_island" (an existing layout
already in the repo, not one we invented).

IMPORTANT LESSON (see fov_divergence_scan.py for the full story): raw
"can the cone geometrically disagree" metrics turned out to be a weak
predictor of real divergence. steak_tshape scored highest on that geometric
scan (42.9% at-station disagreement) but produced ZERO informative episodes
across 45 live runs - the human's greedy subtask logic just never happened
to hinge on a fact it could only get from a divergence-sensitive glance,
mainly because in_bound() has a special case that any tile immediately
beside the player is always visible regardless of FOV, and the human is
always standing right next to whatever station it's about to act on. What
actually predicts real divergence is *live behavioral testing*: run the
human+filter loop and check whether belief ever leaves the uniform prior.
By that empirical test, steak_island hit 100% informative episodes (60/60)
with 88.3% final-guess accuracy (see this file's docstring history / rerun
main() to reproduce) - steak_side_4 also hit 100% informative but only
37.5% accuracy on a small sample, worth another look before trusting it.

Uses a lightweight, non-QMDP teammate (GreedySteakHumanModel, full vision)
instead of the paper's full QMDP-planning robot - steak_island has no cached
QMDP planner, and computing one from scratch is a much bigger value-iteration
job (that's what makes evaluate_bayesian.py's ~6GB cache so precious). The
motion-level planner (MediumLevelPlanner) it does need is cheap (~12s,
one-time, then cached) - that's all a lightweight teammate needs. This
script validates the filter's raw accuracy given a scenario that actually
contains evidence; evaluate_bayesian.py is for the more paper-faithful (but
currently divergence-poor, since it's pinned to a pre-cached layout) full
QMDP-teammate scenario.

Each episode gets a longer, randomized order list (4-8 steaks) instead of
the fixed 2 orders evaluate_bayesian.py's layout has - more running time
means more chances for a real "the human wasn't looking" moment.

Run with: python -m my_methods.bayesian.evaluate_bayesian_lightweight
"""
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel, GreedySteakHumanModel
from my_methods.bayesian.bayesian_inference import FOVBayesFilter

LAYOUT_NAME = "steak_island"
FOV_CANDIDATES = (60, 120, 180)
EPISODE_LEN = 200
N_EPISODES_PER_FOV = 20
MIN_ORDERS, MAX_ORDERS = 4, 8


def build_mlp():
    """One-time (cheap, ~13s) planner build, shared across all episodes."""
    mdp = SteakHouseGridworld.from_layout_name(LAYOUT_NAME)
    return MediumLevelPlanner.from_pickle_or_compute(mdp, BASE_PARAMS, force_compute=False)


def run_episode(mlp, true_fov, horizon=EPISODE_LEN, seed=None):
    rng = np.random.RandomState(seed)
    n_orders = rng.randint(MIN_ORDERS, MAX_ORDERS + 1)
    mdp = SteakHouseGridworld.from_layout_name(LAYOUT_NAME, start_order_list=['steak'] * n_orders)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=horizon)

    # ground truth: a human that really does have vision_bound == true_fov
    human = SteakLimitVisionHumanModel(mlp, env.state, vision_limit=True, vision_bound=true_fov, debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(env.state)

    # lightweight full-vision teammate, no QMDP planner needed
    teammate = GreedySteakHumanModel(mlp)
    teammate.set_agent_index(0)

    bayes_filter = FOVBayesFilter(mlp, env.state, fov_candidates=FOV_CANDIDATES, human_agent_index=1)
    uniform_prob = 1.0 / len(FOV_CANDIDATES)

    correct_steps = 0
    total_steps = 0
    ever_informative = False
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
    print(f"loading planner for {LAYOUT_NAME} (one-time, ~12s, then cached)...")
    mlp = build_mlp()

    print(f"{'true FOV':>10} | {'final-guess accuracy':>21} | {'avg step accuracy':>18} | {'episodes w/ any evidence':>24}")
    all_finals, all_informative = [], []
    for true_fov in FOV_CANDIDATES:
        finals, step_accs, informative = [], [], []
        for ep in range(N_EPISODES_PER_FOV):
            step_acc, final_correct, ever_informative = run_episode(mlp, true_fov, seed=ep * 7 + true_fov)
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
        print("WARNING: still zero informative episodes - see fov_divergence_scan.py for layout diagnostics.")


if __name__ == "__main__":
    main()
