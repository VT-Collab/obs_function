"""
MISHA NEW CHANGE - exploratory/diagnostic only, not part of any pipeline.

Full-episode (120 step) test on the 3 validated sustained-divergence layouts
(fov3090180_D1_v2, D2_v2, D6_v2 - see fov_sustained_batch.py and fov/layouts/).
Two things, both over the SAME longer horizon (the earlier batch test only
ran 40 steps):

  (A) Divergence: same idea as fov_sustained_batch.py's run_layout() - 3
      counterfactual StickySubtaskHumanModel shadows (one per FOV candidate),
      each independently driving its own full simulation alongside the same
      scripted robot, subtask sequence compared against the FOV=180 baseline.

  (B) Bayesian inference: does FOVBayesFilter-style inference (adapted here
      to use StickySubtaskHumanModel shadows - see StickyFOVBayesFilter -
      since that's what's actually driving ground truth now, not the plain
      SteakLimitVisionHumanModel the original bayesian_inference.py assumes)
      correctly identify the human's TRUE FOV by watching their actions,
      and how fast/reliably, on layouts where FOV genuinely changes subtask
      behavior (unlike steak_island, where inference was validated before).

Run with: python -m my_methods.bayesian.fov_full_episode_test [layout_name]
    (optional arg to run just one layout, e.g. fov3090180_D1_v2)
"""
import sys
import numpy as np
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action
from my_methods.bayesian.fov_subtask_divergence_test import build_grid, build_mdp_and_mlp
from my_methods.bayesian.sticky_subtask_human import StickySubtaskHumanModel
from my_methods.bayesian.fov_sustained_batch import robot_next_action

FOV_CANDIDATES = (30, 90, 180)
N_STEPS = 120


class StickyFOVBayesFilter:
    """MISHA NEW CHANGE - same Bayes update as bayesian_inference.py's
    FOVBayesFilter, but shadows are StickySubtaskHumanModel (matching what's
    actually driving ground truth in this diagnostic) instead of the plain
    SteakLimitVisionHumanModel the original class hardcodes.

    MISHA NEW CHANGE (soft shadows): shadows are now constructed with
    ll_boltzmann_rational=True instead of the default fully-greedy policy.
    The greedy policy makes action_probs near one-hot (~0.95 on its single
    chosen action, ~noise-floor on everything else) - comparing that against
    a real observed action is effectively a binary hit/miss, which measured
    at-chance accuracy across all 23 validated-divergence layouts (see
    fov_search_results.md / the inference batch run). Boltzmann-rational
    action selection instead scores EVERY action by its one-step-ahead plan
    cost toward the shadow's current goal and softmaxes over that
    (agent.py:539 boltzmann_rational_ll_action) - a real action that's
    slightly-suboptimal under a given FOV hypothesis now registers as
    "plausible but not the best" instead of "basically impossible", giving
    the Bayes update a continuous, informative gradient instead of a
    near-binary one. Ground truth (the real human in fov_inference_batch.py)
    is untouched - only the shadows (the observer's internal hypothesis
    models) change, keeping the "one shadow per FOV hypothesis" structure
    exactly as before."""

    def __init__(self, mlam, start_state, fov_candidates, human_agent_index, action_noise=0.05,
                 belief_floor=0.01, ll_temp=1):
        self.fov_candidates = list(fov_candidates)
        self.action_noise = action_noise
        # MISHA NEW CHANGE: without this, a candidate's belief can shrink
        # multiplicatively over many disagreeing steps until it underflows to
        # computational zero - at which point NO amount of later favorable
        # evidence can revive it (0 * anything = 0), permanently locking the
        # filter onto whichever candidate happened to win early. The
        # action_noise floor on the LIKELIHOOD alone doesn't prevent this,
        # since the POSTERIOR itself can still underflow after enough steps.
        self.belief_floor = belief_floor
        self.belief = np.ones(len(self.fov_candidates)) / len(self.fov_candidates)
        self.shadows = []
        self.ll_temp = ll_temp
        # MISHA NEW CHANGE: diagnostic only - counts how many steps the
        # shadows' chosen GOALS actually disagree, separate from the belief
        # itself, so batch results distinguish "no divergent signal reached
        # this filter" from "signal present but too weak to move belief".
        self.n_steps_seen = 0
        self.n_goal_divergent_steps = 0
        for fov in self.fov_candidates:
            # MISHA NEW CHANGE: ll_boltzmann_rational=True is set here for
            # documentation/clarity, but note it does NOT by itself soften
            # the general per-step likelihood - agent.py:418 only invokes
            # boltzmann_rational_ll_action() when chosen_goal[0] ==
            # start_pos_and_or[0] (i.e. already standing at the goal), which
            # is a narrow edge case. step() below calls
            # boltzmann_rational_ll_action() directly and unconditionally
            # instead of going through shadow.action(), so this flag isn't
            # actually load-bearing - kept for clarity/consistency only.
            shadow = StickySubtaskHumanModel(mlam, start_state, vision_limit=True, vision_bound=fov, debug=False,
                                              ll_boltzmann_rational=True, ll_temp=ll_temp)
            shadow.set_agent_index(human_agent_index)
            shadow.init_knowledge_base(start_state)
            self.shadows.append(shadow)

    def step(self, state, human_action):
        action_idx = Action.ACTION_TO_INDEX[human_action]
        n_actions = len(Action.ALL_ACTIONS)
        likelihood = []
        goals_seen = []
        for shadow in self.shadows:
            try:
                # MISHA NEW CHANGE: orchestrate ml_action -> goal selection ->
                # boltzmann_rational_ll_action directly, instead of
                # shadow.action(state), so the SOFT (cost-based softmax)
                # likelihood is used for every step, not just the narrow
                # already-at-goal case shadow.action() would apply it to.
                # Goal SELECTION (which subtask/goal to pursue) stays
                # deterministic - that's the validated divergence signal -
                # only the low-level "how to get there" becomes soft.
                start_pos_and_or = state.players_pos_and_or[shadow.agent_index]
                motion_goals = shadow.ml_action(state)
                chosen_goal, _ = shadow.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
                _, action_probs = shadow.boltzmann_rational_ll_action(start_pos_and_or, chosen_goal)
                likelihood.append(action_probs[action_idx])
                goals_seen.append(chosen_goal)
            except Exception:
                likelihood.append(0.0)  # this shadow hit an impossible state - zero evidence, not a crash
                goals_seen.append('EXC')
        self.n_steps_seen += 1
        if len(set(goals_seen)) > 1:
            self.n_goal_divergent_steps += 1
        likelihood = np.array(likelihood)
        likelihood = (1 - self.action_noise) * likelihood + self.action_noise / n_actions
        posterior = self.belief * likelihood
        total = posterior.sum()
        if total > 0:
            self.belief = posterior / total
        # MISHA NEW CHANGE: floor the belief as a CLAMP (only lifts values that
        # would fall below the floor), not a constant per-step dilution toward
        # uniform. The dilution version (self.belief * (1-floor) + floor/n
        # applied every step) was fine against the old near-binary greedy
        # likelihoods (which needed a floor rarely, only to survive
        # underflow), but with the softer Boltzmann likelihoods below, the
        # per-step signal is weaker and 60 steps of constant ~1% dilution
        # toward uniform was enough to erase it entirely (belief stayed at
        # exactly [0.333,0.333,0.333] for every trial - the clamp form fixes
        # this since it's a no-op once a belief is already above the floor).
        n = len(self.belief)
        self.belief = np.maximum(self.belief, self.belief_floor / n)
        self.belief = self.belief / self.belief.sum()
        return self.belief

    def estimate(self):
        return self.fov_candidates[int(np.argmax(self.belief))]


LAYOUTS = {}

features1 = {(9, 5): '1', (7, 3): '2', (9, 6): 'M',
             (12, 1): 'O', (2, 1): 'D', (10, 1): 'P', (7, 6): 'B', (1, 7): 'W', (13, 7): 'S'}
LAYOUTS["fov3090180_D1_v2"] = (build_grid(15, 9, features1), (7, 3), (9, 5), (9, 6), (5, 4))

features2 = {(5, 5): '1', (7, 3): '2', (5, 6): 'M',
             (2, 1): 'O', (12, 1): 'D', (4, 1): 'P', (7, 6): 'B', (13, 7): 'W', (1, 7): 'S'}
LAYOUTS["fov3090180_D2_v2"] = (build_grid(15, 9, features2), (7, 3), (5, 5), (5, 6), (9, 4))

features6 = {(10, 6): '1', (8, 4): '2', (10, 7): 'M',
             (12, 1): 'O', (2, 1): 'D', (7, 1): 'P', (2, 4): 'B', (13, 4): 'W', (1, 7): 'S'}
LAYOUTS["fov3090180_D6_v2"] = (build_grid(15, 9, features6), (8, 4), (10, 6), (10, 7), (6, 5))


def run_layout_full_episode(name, grid, human_pos, robot_start, m_pos, hide_pos):
    mdp, mlp = build_mdp_and_mlp(name, grid)
    robot_idx, human_idx = 0, 1
    setup_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300)
    setup_state = setup_env.state.deepcopy()

    print(f"\n=== {name}: full-episode test ({N_STEPS} steps) ===", flush=True)

    # (A) divergence: 3 counterfactual shadows, independent full trajectories
    divergence_progressions = {}
    for fov in FOV_CANDIDATES:
        sim_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300)
        shadow = StickySubtaskHumanModel(mlp, setup_state, vision_limit=True, vision_bound=fov, debug=False)
        shadow.set_agent_index(human_idx)
        shadow.init_knowledge_base(setup_state)
        subtasks = []
        for t in range(N_STEPS):
            state = sim_env.state
            a_robot = robot_next_action(mlp, state, robot_idx, m_pos, hide_pos)
            try:
                a_human, _ = shadow.action(state)
            except AssertionError:
                subtasks.append('STUCK')
                break
            subtasks.append(shadow.prev_chosen_subtask)
            joint = [Action.STAY, Action.STAY]
            joint[robot_idx], joint[human_idx] = a_robot, a_human
            _, _, done, _ = sim_env.step(tuple(joint))
            if done:
                subtasks.append('DONE')
                break
        divergence_progressions[fov] = subtasks

    ref = divergence_progressions[180]
    disagreement = {fov: sum(1 for a, b in zip(seq, ref) if a != b) for fov, seq in divergence_progressions.items()}
    print(f"  DIVERGENCE over {N_STEPS} steps -> disagreement vs FOV=180: {disagreement}", flush=True)
    for fov in FOV_CANDIDATES:
        print(f"    FOV={fov} (len={len(divergence_progressions[fov])}): {divergence_progressions[fov]}", flush=True)

    # (B) Bayesian inference: for each TRUE fov, ground-truth human + live filter watching
    inference_results = {}
    for true_fov in FOV_CANDIDATES:
        sim_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=300)
        real_human = StickySubtaskHumanModel(mlp, setup_state, vision_limit=True, vision_bound=true_fov, debug=False)
        real_human.set_agent_index(human_idx)
        real_human.init_knowledge_base(setup_state)
        bayes_filter = StickyFOVBayesFilter(mlp, setup_state, FOV_CANDIDATES, human_idx)

        estimates = []
        for t in range(N_STEPS):
            state = sim_env.state
            a_robot = robot_next_action(mlp, state, robot_idx, m_pos, hide_pos)
            try:
                a_human, _ = real_human.action(state)
            except AssertionError:
                break
            bayes_filter.step(state, a_human)
            estimates.append(bayes_filter.estimate())
            joint = [Action.STAY, Action.STAY]
            joint[robot_idx], joint[human_idx] = a_robot, a_human
            _, _, done, _ = sim_env.step(tuple(joint))
            if done:
                break

        correct = sum(1 for e in estimates if e == true_fov)
        first_correct = next((i for i, e in enumerate(estimates) if e == true_fov), None)
        ever_wrong_after_first_correct = False
        if first_correct is not None:
            ever_wrong_after_first_correct = any(e != true_fov for e in estimates[first_correct:])
        inference_results[true_fov] = {
            "estimates": estimates,
            "final_belief": bayes_filter.belief.tolist(),
            "accuracy": correct / len(estimates) if estimates else 0,
            "first_correct_step": first_correct,
            "unstable_after_converging": ever_wrong_after_first_correct,
        }
        print(f"  INFERENCE true_fov={true_fov}: accuracy={inference_results[true_fov]['accuracy']:.2f} "
              f"first_correct_step={first_correct} unstable_after_converging={ever_wrong_after_first_correct} "
              f"final_belief={[round(b, 3) for b in bayes_filter.belief]}", flush=True)
        print(f"    estimates: {estimates}", flush=True)

    return disagreement, inference_results


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    names = [only] if only else list(LAYOUTS.keys())
    for name in names:
        grid, human_pos, robot_start, m_pos, hide_pos = LAYOUTS[name]
        try:
            run_layout_full_episode(name, grid, human_pos, robot_start, m_pos, hide_pos)
        except Exception as e:
            import traceback
            print(f"\n=== {name} ===\n  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
