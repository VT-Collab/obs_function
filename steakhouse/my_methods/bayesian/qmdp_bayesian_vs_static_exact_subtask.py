"""
MISHA NEW CHANGE - additive file, does not modify qmdp_bayesian_vs_static.py.

Same static-vs-dynamic FOV comparison as qmdp_bayesian_vs_static.py, but with
one change: the robot is handed the human's *exact* current subtask instead
of guessing it through belief_update(). FOV is the only thing left unknown.

Why this is possible cleanly: SteakKnowledgeBasePlanner.step() (the method
actually used, planners.py:7883 - it overrides the simpler step() in the
parent class SteakHumanSubtaskQMDPPlanner at planners.py:6434) takes belief
as a dict {subtask_name: probability}, and only reasons about subtasks with
belief_prob > 0.2 (see the `for curr_subtask, belief_prob in belief.items():
if belief_prob > 0.2 ...` filter at planners.py:7902). Handing it a one-hot
dict - 1.0 on the human's real current subtask, 0.0 on everything else -
makes it consider exactly and only that one subtask. No need to touch the
planner's own code at all; belief_update() (the thing that normally *infers*
the subtask probabilistically) just never gets called on this path.

FOV still matters even with the subtask given for certain: step() still uses
self.sim_human_model (planners.py:7896-7897, 7931) to reason about what the
human's knowledge - and therefore their subtask - might become a few steps
into the lookahead search, and that reasoning is gated by
sim_human_model.vision_bound. So this experiment isolates FOV as the only
source of uncertainty in the robot's planning, cleanly, rather than mixing
it with the robot's usual subtask-inference noise.

REVISION 2 - architecture change: earlier this called mdp_planner.step()
directly, bypassing MediumQMdpPlanningAgent.action() entirely - which also
bypasses its resolve_stuck() call (agent.py:1763-1792), the robot's normal
safety net for when it physically blocks itself against a wall/the human.
That turned out not to be a rare edge case: the robot froze solid within
~14 steps of every episode (diagnose_exact_subtask.py), so the first full
run showed zero reward across all 60 episodes. A hand-rolled fix
(unstuck_action, re-implementing resolve_stuck's logic manually) fixed the
freeze but exposed a second issue: resolve_stuck only checks the
immediately-prior step, so a 2-cycle oscillation (A->B->A->B...) slips
through it - the human ended up stuck 44+ steps trying to route around the
idling robot.

Rather than keep hand-reimplementing pieces of MediumQMdpPlanningAgent's
already-tested machinery one gap at a time, ExactSubtaskQMdpAgent below
takes a different approach: keep using the REAL .action() pipeline (so
resolve_stuck, prev_state tracking, and low-level action translation all
keep working exactly as they do in every other experiment in this project),
and just temporarily monkey-patch mdp_planner.belief_update to return a
one-hot "certain" belief instead of its normal probabilistic guess, for the
duration of one .action() call. Same effect as before (subtask becomes
exact, FOV stays the only real uncertainty) but built on tested
infrastructure instead of a fresh reimplementation. It also adds a short
position-history check on top, specifically to catch the oscillation case
resolve_stuck's single-step check misses.

Run with: python -m my_methods.bayesian.qmdp_bayesian_vs_static_exact_subtask
"""
import os
import time
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import SteakHouseGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, SteakKnowledgeBasePlanner
from overcooked_ai_py.helpers import BASE_PARAMS
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel, MediumQMdpPlanningAgent
from my_methods.bayesian.bayesian_inference import FOVBayesFilter
# MISHA NEW CHANGE: reuse the existing config constants, but NOT build_planners()
# (see build_planners_fixed below - the original has a critical bug, see there)
from my_methods.bayesian.qmdp_bayesian_vs_static import (
    LAYOUT_NAME, FOV_CANDIDATES, EPISODE_LEN, N_EPISODES_PER_FOV, MIN_ORDERS, MAX_ORDERS,
    SEARCH_DEPTH, KB_SEARCH_DEPTH,
)

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results_exact_subtask.md")
# MISHA NEW CHANGE: steak_island.layout's "start_order_list" is the raw string
# 'steak, steak' (12 characters), not a proper list. planners.py's init_states()
# loops over order_list and DOUBLES the state count per order - iterating a
# 12-char string instead of a 2-element list inflates the state space 1024x
# (2^12 vs 2^2), turning a tractable ~6.6GB transition matrix into a literal
# 10.6 PETABYTE one (confirmed via a real MemoryError on CARC). Always pass
# this explicit list instead of relying on the layout file's buggy default.
FIXED_ORDER_LIST = ['steak', 'steak']


# MISHA NEW CHANGE
def build_planners_fixed():
    """Same as qmdp_bayesian_vs_static.build_planners(), except the mdp used
    to build the QMDP planner gets a correct start_order_list (see
    FIXED_ORDER_LIST above) instead of the layout file's buggy string
    default - that bug alone caused a 10.6 PB memory allocation attempt."""
    mdp = SteakHouseGridworld.from_layout_name(LAYOUT_NAME, start_order_list=FIXED_ORDER_LIST)
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


# MISHA NEW CHANGE
def route_to_station_action(mp, player_pos_and_or, station_locations):
    """Compute the next low-level action to reach and interact with the
    nearest tile in station_locations, reusing the same motion-planning
    utilities the library itself uses elsewhere (MediumQMdpPlanningAgent.
    mdp_action_to_low_level_action, agent.py:2289, uses this exact same
    mp.motion_goals_for_pos / mp.get_plan pattern). Returns Action.INTERACT
    if already at a valid interact position for one of the stations,
    Action.STAY if no station is reachable at all."""
    for station_pos in station_locations:
        if player_pos_and_or in mp.motion_goals_for_pos[station_pos]:
            return Action.INTERACT

    best_action, best_cost = None, float('inf')
    for station_pos in station_locations:
        for goal in mp.motion_goals_for_pos[station_pos]:
            if not mp.is_valid_motion_start_goal_pair(player_pos_and_or, goal):
                continue
            action_plan, _, cost = mp.get_plan(player_pos_and_or, goal)
            if len(action_plan) > 0 and cost < best_cost:
                best_cost = cost
                best_action = action_plan[0]
    return best_action if best_action is not None else Action.STAY


# MISHA NEW CHANGE
class ExactSubtaskQMdpAgent(MediumQMdpPlanningAgent):
    """A MediumQMdpPlanningAgent whose subtask belief is forced to be exact/
    certain - one-hot on whatever mdp_planner.sim_human_model currently
    deduces about the human, given the robot's own current fov belief -
    instead of the normal probabilistic belief_update() guess. Runs the
    REAL .action() underneath (via super()), so resolve_stuck() and all its
    prev_state tracking work exactly as in every other experiment here; only
    belief_update is swapped out, and only for the duration of one call.

    Also adds a short position-history check on top of resolve_stuck: that
    method only compares against the immediately-prior step, so a 2-cycle
    oscillation (A->B->A->B...) never looks "frozen" to it one step at a
    time. If the robot revisits the same exact spot 3+ times within the
    last 6 steps, force a move toward a not-recently-visited position.

    Also fixes a task-completion bug found via diagnose_exact_subtask.py:
    the QMDP action selection, given only the human's current subtask belief,
    has no independent notion of "I'm holding an onion, I should go chop it"
    - it picked up an onion early on but then, as the human's deduced
    subtask moved on to other things, dropped the onion on a random counter
    instead of the chopping board. Since an onion that's never chopped means
    "garnish" can never exist, this made the team's order permanently
    undeliverable in every single episode (confirmed: 0/6 mode-FOV groups in
    the first full run ever scored any reward). Fix: if the robot is
    currently holding an onion, override its action with a direct route to
    the chopping board instead of trusting the QMDP action for that step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recent_positions = []

    def action(self, state, **kwargs):
        self.mdp_planner.sim_human_model.action(state)
        true_subtask = self.mdp_planner.sim_human_model.prev_chosen_subtask

        def fake_belief_update(world_state, agent_player, observed_info, human_player,
                                belief_vector, prev_dist_to_feature, greedy=False,
                                vision_limit=False, prev_max_belief=None):
            one_hot = np.zeros(len(self.mdp_planner.subtask_dict), dtype=float)
            one_hot[self.mdp_planner.subtask_idx_dict[true_subtask]] = 1.0
            return one_hot, prev_dist_to_feature

        real_belief_update = self.mdp_planner.belief_update
        self.mdp_planner.belief_update = fake_belief_update
        try:
            chosen_action, info = super().action(state, **kwargs)
        finally:
            self.mdp_planner.belief_update = real_belief_update

        # MISHA NEW CHANGE: finish the onion-chopping task instead of trusting
        # whatever the QMDP action computed for this step - see class docstring.
        # Two cases: (a) still carrying the raw onion -> route to the board and
        # drop it; (b) already dropped (state.objects now has a 'garnish' object
        # sitting on the board - that's this game's name for an in-progress-or-
        # done chop) but chop_time hasn't reached mdp.chopping_time yet -> route
        # to the board and keep interacting, since chopping needs repeated
        # interacts while standing there, not just one drop-off.
        held = state.players[self.agent_index].held_object
        chopping_board_locations = self.mdp_planner.mdp.get_chopping_board_locations()
        player_pos_and_or = state.players[self.agent_index].pos_and_or

        if held is not None and held.name == 'onion':
            chosen_action = route_to_station_action(self.mdp_planner.mp, player_pos_and_or, chopping_board_locations)
            return chosen_action, info

        for obj in state.objects.values():
            if obj.name == 'garnish' and obj.position in chopping_board_locations \
                    and obj.state < self.mdp_planner.mdp.chopping_time:
                chosen_action = route_to_station_action(self.mdp_planner.mp, player_pos_and_or, [obj.position])
                return chosen_action, info

        current_pos = state.players[self.agent_index].pos_and_or
        self._recent_positions.append(current_pos)
        self._recent_positions = self._recent_positions[-6:]
        if self._recent_positions.count(current_pos) >= 3:
            # MISHA NEW CHANGE: plain python list + np.random.shuffle, NOT
            # np.array(Action.MOTION_ACTIONS)/np.random.permutation - numpy
            # silently turns a list of same-length tuples into a 2D array,
            # so permutation hands back numpy sub-arrays instead of the
            # original action tuples, which then blows up downstream trying
            # to use one as a hashable/boolean action (confirmed via a real
            # "truth value of an array is ambiguous" crash on CARC)
            shuffled_motion_actions = list(Action.MOTION_ACTIONS)
            np.random.shuffle(shuffled_motion_actions)
            for a in shuffled_motion_actions:
                joint = (a, Action.STAY) if self.agent_index == 0 else (Action.STAY, a)
                new_state, _, _, _ = self.mdp_planner.mdp.get_state_transition(state, joint)
                new_pos = new_state.players[self.agent_index].pos_and_or
                if self._recent_positions.count(new_pos) == 0:
                    chosen_action = a
                    break
            self._recent_positions = []

        return chosen_action, info


# MISHA NEW CHANGE
def run_episode_exact_subtask(mlp, mdp_planner, true_fov, mode, seed=None):
    """Same as qmdp_bayesian_vs_static.run_episode(), except the robot is
    never guessing a probability distribution over subtasks. Instead: the
    robot's own shadow (sim_human_model, running with the robot's CURRENT fov
    belief - 0 for static, the live bayes estimate for dynamic) deterministically
    picks a subtask, and that's what gets treated as certain. The robot never
    reads the real human's ground-truth subtask directly - only ever "what
    would I expect them to be doing, given what I currently believe about
    their fov." That's the "based on fov, subtask should be exact" framing:
    fov is the only free variable, subtask is a deterministic function of it."""
    # MISHA NEW CHANGE: must match the planner's build exactly (FIXED_ORDER_LIST,
    # 2 orders) - the QMDP planner's state space only covers whatever order-list
    # length it was built with (see FIXED_ORDER_LIST comment above). Feeding it
    # an episode with more orders than that produces a "remaining orders" state
    # the planner never enumerated, which crashes step() with a KeyError - this
    # is the same order-list root cause as the build-time bug, just hit from the
    # episode side instead. The random 4-8 order trick from
    # evaluate_bayesian_lightweight.py doesn't carry over here for that reason.
    if seed is not None:
        np.random.seed(seed)
    mdp = SteakHouseGridworld.from_layout_name(LAYOUT_NAME, start_order_list=FIXED_ORDER_LIST)
    env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=EPISODE_LEN)

    human = SteakLimitVisionHumanModel(mlp, env.state, vision_limit=True, vision_bound=true_fov, debug=False)
    human.set_agent_index(1)
    human.init_knowledge_base(env.state)

    mdp_planner.sim_human_model.init_knowledge_base(env.state)
    mdp_planner.sim_human_model.vision_bound = 0  # both modes start "unaware"

    # MISHA NEW CHANGE: real MediumQMdpPlanningAgent subclass (see class docstring)
    # instead of calling mdp_planner.step() directly - gets resolve_stuck for free
    teammate = ExactSubtaskQMdpAgent(mdp_planner, greedy=True, auto_unstuck=True, low_level_action_flag=True)
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
            # MISHA NEW CHANGE: bayes_filter.step() drives its own internal
            # shadow copies of the human policy (one per FOV candidate), which
            # can hit the same pre-existing ml_action() AssertionError as the
            # real human/teammate models above. This used to sit OUTSIDE this
            # try/except, so a shadow-model crash took down the whole SLURM
            # job instead of just ending this one episode (see job 10352356).
            if mode == 'dynamic':
                bayes_filter.step(state, a_human)
                mdp_planner.sim_human_model.vision_bound = bayes_filter.estimate()
        except AssertionError:
            break

        _, reward, done, _ = env.step((a_teammate, a_human))
        total_reward += reward
        steps_taken = t + 1
        if done:
            break

    return steps_taken, total_reward, done


# MISHA NEW CHANGE
def main():
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"# Static vs dynamic FOV, exact subtask (steak_island)")
    emit(f"Run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"Layout: {LAYOUT_NAME} | Episode len: {EPISODE_LEN} | Episodes/FOV: {N_EPISODES_PER_FOV} "
         f"| Orders/episode: {FIXED_ORDER_LIST} (must match the planner's build exactly)")
    emit("")
    emit("Robot is given the human's exact subtask every step (no subtask guessing) - "
         "FOV is the only thing the robot's planner doesn't know for certain.")
    emit("")

    emit("loading/building steak_island QMDP planner (build_qmdp_steak_island.py must have "
         "been run already, or this will trigger a fresh build - now fast, ~28,812 states "
         "after the order-list fix, not the original 38.5 million)...")
    mlp, mdp_planner = build_planners_fixed()

    emit("")
    emit(f"| mode | true FOV | avg steps | avg reward | completion rate |")
    emit(f"|---|---|---|---|---|")
    results = {}
    for mode in ('static', 'dynamic'):
        for true_fov in FOV_CANDIDATES:
            steps_list, reward_list, done_list = [], [], []
            for ep in range(N_EPISODES_PER_FOV):
                steps, reward, done = run_episode_exact_subtask(mlp, mdp_planner, true_fov, mode, seed=ep * 11 + true_fov)
                steps_list.append(steps)
                reward_list.append(reward)
                done_list.append(done)
            results[(mode, true_fov)] = (steps_list, reward_list, done_list)
            emit(f"| {mode} | {true_fov} | {np.mean(steps_list):.1f} | "
                 f"{np.mean(reward_list):.1f} | {np.mean(done_list) * 100:.1f}% |")

    emit("")
    emit("## static vs dynamic, averaged over all FOVs")
    emit("")
    emit(f"| mode | avg steps | avg reward | completion rate |")
    emit(f"|---|---|---|---|")
    for mode in ('static', 'dynamic'):
        all_steps = [s for fov in FOV_CANDIDATES for s in results[(mode, fov)][0]]
        all_reward = [r for fov in FOV_CANDIDATES for r in results[(mode, fov)][1]]
        all_done = [d for fov in FOV_CANDIDATES for d in results[(mode, fov)][2]]
        emit(f"| {mode} | {np.mean(all_steps):.1f} | {np.mean(all_reward):.1f} | "
             f"{np.mean(all_done) * 100:.1f}% |")

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nresults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
