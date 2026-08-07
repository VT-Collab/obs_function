"""The robot's model of its partner: a predictive wrapper over the FOV filter.

`SamplingBayesFOVInference` (fov/robot/policy/old/inference/bayes_fov_sampling.py)
already maintains the joint posterior b(theta, tau) over the human's FOV and
current subtask, from observed (state, action) pairs only. It is used here
UNCHANGED -- not subclassed, not edited, not copied. What this file adds is the
two things a PLANNER needs that a filter does not provide:

  1. PREDICT. The filter's job ends at "what were they doing". A robot that has
     to choose an action needs "what will they do next, and what would that do
     to what they know". `predict()` runs the filter's own transition kernel
     one step forward WITHOUT consuming evidence, and hands back the resulting
     hypotheses as data: (fov, subtask, probability, the primitive action that
     subtask would emit).

  2. COUNTERFACTUAL PERCEPTION. Given a hypothetical next state, what would a
     human with cone theta then believe? `beliefs_after()` is a faithful
     replica of LimitedVisionSteakHuman.observe(), restricted to cells the
     shadow has ALREADY discovered, and `scratch()` wraps the result in a
     throwaway copy of the shadow so the human's OWN decision code
     (subtask_distribution / _available_advancing / _weights) can be evaluated
     on it. Nothing is duplicated: the predicates are the human's.

===========================================================================
CAUSALITY -- THE ORDER THAT MAKES THIS NOT A CHEAT
===========================================================================
The robot and the human move SIMULTANEOUSLY. So at tick t the robot may
condition on states s_0..s_t and human actions h_0..h_{t-1}, and NOT on h_t.
The loop in rollout.py is therefore:

    sync_observe(s_t)        shadows perceive s_t. free -- looking is not
                             evidence about the cone, and a human with any
                             cone perceives whatever its cone allows.
    predict(s_t)             PREDICTIVE posterior: conditioned through t-1.
    robot picks a_t          <- uses only the above
    h_t = human.action(s_t)  the human moves
    update(s_t, h_t)         NOW the evidence is admissible
    env.step(a_t, h_t)

Reading h_t before choosing a_t would be peeking at a simultaneous move.
test_no_cheating.py pins this ordering.

The one subtlety: `update()` re-runs `observe(s_t)` internally. That is
idempotent -- same state, same clock, so every belief is rewritten with the
value and timestamp it already had.
"""
import copy

import _paths  # noqa: F401

from overcooked_ai_py.mdp.overcooked_mdp import Action

from fov.human.agent.limited_vision_human import (
    Belief, UNKNOWN, ROBOT_HELD_FETCH, SAMPLING_SUBTASKS)
from fov.robot.policy.old.inference.bayes_fov_sampling import (
    SamplingBayesFOVInference)

#Same preference order LimitedVisionSteakHuman.believed() uses to collapse
#several stations of one kind into a single value. Imported by value rather
#than re-derived so the two cannot disagree about what "the pot" is when a
#layout has two pots.
_BELIEF_PREFERENCE = ("ready", "cooking", "chopping", "washing",
                      "occupied", "empty")

#Subtasks that do NOT advance the recipe: they are what the human is forced
#into when its beliefs cannot offer it anything to do. This is a fact about the
#HUMAN MODEL (limited_vision_human._need_info_lookup emits exactly these), not
#about the steakhouse task -- no reward, no recipe step, no delivery is
#referenced anywhere in this package.
NON_ADVANCING = frozenset({"explore", "wait", "check_pot", "check_board",
                           "check_sink"})


def _install_explore_cache(shadow):
    """Memoise `_explore_action` per (shadow, state, tick).

    `_explore_action` is the most expensive call in a tick by a wide margin --
    it evaluates `_cone_gain` for all four orientations, each a full
    (2*SIGHT_RADIUS+1)^2 sweep with a cone test per cell. The filter already
    hoists it out of its tau loop for that reason; we call the filter and the
    planner in the same tick, so without this it runs twice.

    Safe because the function is PURE: it reads state / seen_cells /
    known_terrain, restores the orientation it borrows in `_cone_gain`, and
    writes nothing. The cache key holds a strong reference to the state object
    so its id() cannot be recycled under us.
    """
    original = shadow._explore_action
    cache = {"tick": None, "state": None, "value": None}

    def cached(state):
        if cache["tick"] != shadow.t or cache["state"] is not state:
            cache["tick"] = shadow.t
            cache["state"] = state
            cache["value"] = original(state)
        return cache["value"]

    shadow._explore_action = cached
    shadow._explore_action_raw = original


def believed_from(beliefs, station_cells):
    """Collapse per-cell beliefs about one station KIND into one value, using
    LimitedVisionSteakHuman.believed()'s own preference order."""
    if not station_cells:
        return UNKNOWN
    vals = [beliefs[c].value for c in station_cells if c in beliefs]
    for pref in _BELIEF_PREFERENCE:
        if pref in vals:
            return pref
    return UNKNOWN


class PredictiveHumanModel:
    """Filter + one-step prediction + counterfactual perception."""

    def __init__(self, mdp, candidate_fovs, human_index=1, robot_index=0):
        self.mdp = mdp
        self.human_index = human_index
        self.robot_index = robot_index
        self.candidate_fovs = list(candidate_fovs)
        #mlp is only ever stored by SteakMotionPlanner; the human routes with
        #its own BFS over seen floor, so None is correct and cannot leak a map.
        self.filter = SamplingBayesFOVInference(
            mdp, None, self.candidate_fovs, human_agent_index=human_index)
        for sh in self.filter.shadows.values():
            _install_explore_cache(sh)
        self.forget_horizon = next(iter(self.filter.shadows.values())).forget_horizon

    # -- filter passthrough --------------------------------------------------

    @property
    def tick(self):
        return self.filter.tick

    def posterior(self):
        return self.filter.posterior()

    def map_fov(self):
        return self.filter.map_fov()

    def entropy(self):
        return self.filter.entropy()

    def update(self, state, human_action):
        """Consume one (state, action) pair. The action is a physical event --
        the robot watched its partner move -- not a privileged label."""
        return self.filter.update(state, human_action)

    # -- perception ----------------------------------------------------------

    def sync_observe(self, state):
        """Let every shadow perceive `state` at the current clock.

        The filter does this at the top of update(), but update() cannot run
        until the human has acted, and the robot must decide first. Doing it
        here means the shadows' beliefs describe the state the robot is
        actually planning in. Idempotent with the filter's own call.
        """
        t = self.filter.tick
        for sh in self.filter.shadows.values():
            sh.t = t
            sh.observe(state)

    # -- prediction ----------------------------------------------------------

    def predict(self, state):
        """One predictive step of the filter's own kernel, WITHOUT evidence.

        Returns (hypotheses, human_action_probs) where each hypothesis is

            {fov, shadow, subtask, prob, action, arrived}

        `prob` is the JOINT weight of (fov, subtask) under the predictive
        posterior; the list sums to 1. This is exactly what QMDP integrates
        over -- b(theta) weighting a per-hypothesis value.
        """
        f = self.filter
        held = state.players[self.human_index].held_object
        held_name = held.name if held else None

        hypotheses = []
        for fov, shadow in f.shadows.items():
            mass = {tau: p for (f2, tau), p in f.b.items() if f2 == fov and p > 0.0}
            if not mass:
                continue
            try:
                explore_act = shadow._explore_action(state)
                forced = shadow._forced_held(held_name) is not None
                exec_memo = {}
                for tau, p_tau in mass.items():
                    next_subtasks, sampler_untouched = f._transition(
                        shadow, state, tau, forced)
                    for subtask, p_z in next_subtasks.items():
                        if p_z <= 0.0:
                            continue
                        if subtask not in exec_memo:
                            act, arrived, _ = shadow.execute(state, subtask,
                                                             explore_act)
                            exec_memo[subtask] = (act, arrived)
                        act, arrived = exec_memo[subtask]
                        #where the sampler's sticky draw lands NEXT tick --
                        #the same bookkeeping bayes_fov_sampling.update() does
                        #when it builds next_tau. The planner needs it because
                        #`subtask_distribution(state, committed)` is only
                        #correct when handed the line the human is still on.
                        next_sampled = (tau[1] if sampler_untouched
                                        else (subtask if subtask in SAMPLING_SUBTASKS
                                              else None))
                        hypotheses.append({
                            "fov": fov, "shadow": shadow, "subtask": subtask,
                            "prob": p_tau * p_z, "action": act,
                            "arrived": arrived, "next_sampled": next_sampled,
                        })
            except Exception:
                #a shadow that raises contributes nothing this tick; the filter
                #takes the same view (it discards the tick). Never fatal.
                continue

        total = sum(h["prob"] for h in hypotheses)
        if total <= 0.0:
            return [], {Action.STAY: 1.0}
        for h in hypotheses:
            h["prob"] /= total

        action_probs = {}
        for h in hypotheses:
            action_probs[h["action"]] = action_probs.get(h["action"], 0.0) + h["prob"]
        return hypotheses, action_probs

    # -- counterfactual perception ------------------------------------------

    def beliefs_after(self, shadow, next_state):
        """What shadow `shadow` would believe after perceiving `next_state`.

        A faithful replica of LimitedVisionSteakHuman.observe(), with ONE
        deliberate restriction: it does not add cells the human has not
        discovered yet. Discovery needs the full (2*SIGHT_RADIUS+1)^2 sweep,
        and -- decisively -- what it discovers does not depend on the ROBOT's
        choice of action, only on where the HUMAN ends up. Since every cost in
        cost.py is a DIFFERENCE between two robot actions at the same human
        action, the discovery term is identical in both and cancels exactly.

        Returns (beliefs, robot_seen).
        """
        t_next = shadow.t + 1
        horizon = shadow.forget_horizon
        out = {}
        for loc, bel in shadow.beliefs.items():
            if loc == shadow.ROBOT:
                continue
            if shadow.visible(next_state, loc):
                out[loc] = Belief(shadow._true_state_of(next_state, loc), t_next)
            elif t_next - bel.seen_at > horizon:
                out[loc] = Belief(UNKNOWN, bel.seen_at)
            else:
                out[loc] = bel

        #the teammate is perceived on exactly the same terms as a station
        rp = next_state.players[self.robot_index]
        prev = shadow.beliefs.get(shadow.ROBOT)
        robot_seen = shadow._robot_seen
        if shadow.visible(next_state, rp.position):
            out[shadow.ROBOT] = Belief(
                rp.held_object.name if rp.held_object else "none", t_next)
            robot_seen = (tuple(rp.position), tuple(rp.orientation), t_next)
        elif prev is not None:
            out[shadow.ROBOT] = (Belief(UNKNOWN, prev.seen_at)
                                 if t_next - prev.seen_at > horizon else prev)
        return out, robot_seen

    @staticmethod
    def scratch(shadow, beliefs, robot_seen):
        """A throwaway shadow carrying hypothetical beliefs.

        Shallow copy: `stations`, `known_terrain` and `seen_cells` are SHARED
        and only read; `beliefs`, `_robot_seen` and `t` are replaced. The point
        is to be able to call the human's OWN decision code
        (subtask_distribution, _available_advancing, _weights, believed) on a
        hypothetical knowledge base without re-implementing any of it, and
        without touching the live shadow the filter depends on.

        Safe against RNG contamination: none of those methods draws.
        """
        sc = copy.copy(shadow)
        sc.beliefs = beliefs
        sc._robot_seen = robot_seen
        sc.t = shadow.t + 1
        return sc

    # -- readouts used only for logging -------------------------------------

    def diagnostics(self):
        return {
            "map_fov": self.filter.map_fov(),
            "entropy": self.filter.entropy(),
            "n_informative": self.filter.n_informative,
            "n_skipped": self.filter.n_skipped,
            "posterior": dict(self.filter.posterior()),
        }


__all__ = ["PredictiveHumanModel", "believed_from", "NON_ADVANCING",
           "ROBOT_HELD_FETCH", "UNKNOWN"]
