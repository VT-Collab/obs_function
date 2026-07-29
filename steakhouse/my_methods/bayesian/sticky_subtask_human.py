"""
MISHA NEW CHANGE - exploratory/diagnostic only, not part of any pipeline.

Fallback human agent for sustained FOV divergence, per the user's explicit
direction: pure layout geometry only ever DELAYS convergence to the correct
task order, it can't change it - because SteakLimitVisionHumanModel.ml_action()
(agent.py:1472) recomputes "which subtask is objectively best right now" from
scratch on almost every call. The ONLY subtasks that get any commitment in the
base class are chop_onion/heat_hot_plate (agent.py:1579-1584, via a
`self.prev_chosen_subtask in [...]` check) - pickup_meat/pickup_onion/
pickup_plate/etc all get re-derived fresh every step from current (possibly
now-updated) knowledge.

Concretely: once a pickup_X subtask is explicitly requested (the
`chosen_subtask is not None` branch, agent.py:1523-1567), the underlying
am.pickup_meat_actions(counter_objects) etc. calls are UNCONDITIONAL - they
don't re-check other_has_meat/other_has_onion/etc at all. So forcing
`chosen_subtask=self.prev_chosen_subtask` when we're already mid-pickup
routes the human all the way to the dispenser and lets them grab the item
regardless of whether it's still needed, only re-evaluating once they're
holding it (has_object() becomes True) and something else must be done with
it. That turns a 1-step "wrong guess, immediately corrected" into a genuine
multi-step wasted round trip (walk to dispenser, grab, walk to drop-off,
drop) before the human ever reconsiders - real, sustained divergence instead
of a timing shift.

Run with: python -m my_methods.bayesian.sticky_subtask_human
"""
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel

STICKY_SUBTASKS = {
    'pickup_meat', 'pickup_onion', 'pickup_plate',
    'pickup_hot_plate', 'pickup_steak', 'pickup_garnish',
}


class StickySubtaskHumanModel(SteakLimitVisionHumanModel):
    """Commits to self.prev_chosen_subtask until its own motion goals run out
    (i.e. it's been completed or become physically impossible), instead of
    re-deriving the "objectively best" subtask fresh every single step."""

    def ml_action(self, state, chosen_subtask=None):
        player = state.players[self.agent_index]
        if chosen_subtask is None and not player.has_object() and self.prev_chosen_subtask in STICKY_SUBTASKS:
            try:
                motion_goals = super().ml_action(state, chosen_subtask=self.prev_chosen_subtask)
            except Exception:
                motion_goals = []
            if len(motion_goals) > 0:
                return motion_goals
            # committed subtask has no valid moves left (done / impossible) - fall through to re-evaluation
        return super().ml_action(state, chosen_subtask=chosen_subtask)


if __name__ == "__main__":
    import sys
    from my_methods.bayesian.fov_subtask_divergence_test import (
        build_grid, build_mdp_and_mlp, render_layout_image, FOV_CANDIDATES, N_FORWARD_STEPS,
    )
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import Action

    # reuse the close-cluster layout (fast - already cached) rather than
    # waiting on the far-cluster build; stickiness should sustain divergence
    # even WITHOUT needing the robot to be spatially hidden from the redundant path.
    features = {
        (4, 5): '1', (7, 3): '2',
        (7, 6): 'B', (4, 6): 'M',
        (12, 1): 'O', (2, 1): 'D', (10, 1): 'P', (1, 7): 'W', (13, 7): 'S',
    }
    grid = build_grid(15, 9, features)
    name = "fov_meat_dx3_dy2"  # reuses the already-built pickle cache
    mdp, mlp = build_mdp_and_mlp(name, grid)
    robot_idx, human_idx = 0, 1

    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)
    setup_state = env.state.deepcopy()
    joint = [Action.STAY, Action.STAY]
    joint[robot_idx] = Action.INTERACT
    env.step(tuple(joint))
    reveal_state = env.state.deepcopy()
    human_pos = setup_state.players[human_idx].position
    human_ori = setup_state.players[human_idx].orientation
    print(f"human@{human_pos}/{human_ori}  robot@{reveal_state.players[robot_idx].position} "
          f"held={reveal_state.players[robot_idx].held_object}")

    progressions = {}
    for fov in FOV_CANDIDATES:
        sim_env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=100)
        sim_env.state = reveal_state.deepcopy()
        shadow = StickySubtaskHumanModel(mlp, setup_state, vision_limit=True, vision_bound=fov, debug=False)
        shadow.set_agent_index(human_idx)
        shadow.init_knowledge_base(setup_state)

        subtasks = []
        for t in range(N_FORWARD_STEPS):
            try:
                a_human, _ = shadow.action(sim_env.state)
            except AssertionError:
                subtasks.append('STUCK')
                break
            subtasks.append(shadow.prev_chosen_subtask)
            joint = [Action.STAY, Action.STAY]
            joint[human_idx] = a_human
            sim_env.step(tuple(joint))
        progressions[fov] = subtasks
        print(f"FOV={fov:>3}: {subtasks}")

    ref = progressions[180]
    disagreement = {fov: sum(1 for a, b in zip(seq, ref) if a != b) for fov, seq in progressions.items()}
    print(f"steps disagreeing with FOV=180: {disagreement}")
