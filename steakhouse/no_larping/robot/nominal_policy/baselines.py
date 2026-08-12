"""Two robot baselines. NEITHER models the human's field of view.

THETA-BLIND, and that is the entire specification. Both see the full state (per
the problem statement, the robot directly observes s) and both are deliberately
blind to theta: there is NO CONE ANYWHERE IN THIS FILE, not even an assumed one.
Grep it - no visible_cells, no assumed_fov, no posterior, no line of sight. An
earlier version broke ties in a corridor by guessing a 180 cone; that quietly
made the control theta-AWARE, and it never fired anyway, because the split
layouts leave the two agents no shared floor at all. These are the control
condition: any gain a later FOV-aware policy shows is measured against them, so
the ONLY thing they may lack is the human's observation model.

    B1  SoloRobot      runs the recipe itself and never stages anything for the
                       human. The floor: what the team gets with no cooperation.
    B2  HandoffRobot   does stage items on counters for the human, but picks the
                       counter by DISTANCE, having no idea what the human can see.

B2 is the interesting control. It performs the same helpful act as an FOV-aware
robot - put something where the human can pick it up - but chooses WHERE blindly.
The FOV-aware policy differs from it in exactly one decision, which is what makes
the comparison clean.

NO COLLISION HANDLING, HERE OR ANYWHERE IN THE PACKAGE. No yielding, no
right-of-way, no unstuck counter, no sidestep, and the human is not treated as an
obstacle - step_towards() plans straight through their tile. Every layout is two
rooms joined only by pass-through counters, so the two agents share no floor and
were measured adjacent on 0 of 4800 ticks. All of that machinery fired zero times
and was deleted rather than kept as decoration.

CONTENTION SURVIVES, AND IT IS NOT COLLISION HANDLING. What the two agents CAN
share is a station embedded in the dividing wall, reachable from both rooms -
they stand at it from opposite sides, and only one of them gets the job done.
_BaseRobot demotes such a subtask within its tier when the human would arrive
first. Measured: it fired 732 times, all of them on shared stations, which is
exactly what "reachable from both rooms" predicts. It is a partner model, not a
traffic rule, and it reads POSITION only - never the cone.

Both expose rank_subtasks(state) -> [(tier, verb, cell), ...] best first, so the
QMDP layer in robot/filter/ can read what this policy prefers without either
baseline needing to change. That layer does NOT simply re-order the list: it
aggregates the tuples into (tier, verb) JOBS, hands each job its whole legal cell
set, and searches over which cell and which first step -- precisely because the
cell chosen in here is chosen theta-blind, by distance from the ROBOT. See
robot/filter/DESIGN.md section 2 for the contract and
robot/nominal_policy/subtask_dist.py for the three accessors it reads through.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from overcooked_ai_py.mdp.actions import Action              # noqa: E402
from common import geometry as geo                            # noqa: E402
from common.tasks import legal_subtasks, TIER_NAME, T_STASH   # noqa: E402
from common.views import TruthView                            # noqa: E402


class _BaseRobot:
    """Shared machinery. Subclasses only change how subtasks are ranked."""

    def __init__(self, mdp, agent_index=0, seed=0):
        self.seed = seed
        self.mdp = mdp
        self.agent_index = agent_index
        self.other_index = 1 - agent_index
        self.reset()

    def reset(self):
        import random
        self._rng = random.Random(self.seed * 2)
        self.t = 0
        self.last_subtask = None
        self.log = []

    # -- ranking ------------------------------------------------------------
    def rank_subtasks(self, state):
        """[(tier, verb, cell)] best first. The hook the QMDP layer reads.

        Sorted by this policy's OWN key, so it IS the preference order -- but it
        is NOT what the policy does this tick, because action() applies
        within-tier stickiness on top. subtask_dist.true_ranking wraps this and
        true_subtask wraps action(); the two differ exactly when stickiness
        fires.
        """
        view = TruthView(self.mdp, state)
        me = state.players[self.agent_index]
        pos = tuple(me.position)
        held = me.held_object.name if me.held_object else None
        # the human's cell, used ONLY as the start of their hypothetical walk in
        # the contention test below. It is not removed from `walk`: nothing here
        # routes around them, and on these layouts nothing needs to.
        other = tuple(state.players[self.other_index].position)
        walk = view.walkable | {pos}

        # CONTENTION: give up a station the human would reach first, because the
        # job will be done by the time we arrive and we would have spent the walk
        # for nothing. Demoted within the tier, never across it, so if it is the
        # only thing in its tier we still do it.
        #
        # `hd` is measured over the whole walkable floor, which is what makes
        # this self-limiting: the two rooms are disconnected, so the human's path
        # to a robot-side station comes back None and the term cannot fire. It
        # only ever bites on a station embedded in the divide - measured 732
        # firings, all of them there.
        #
        # POSITION ONLY, NEVER THE CONE. This file is the theta-blind control, so
        # it may model where the human IS and what they are DOING, but never what
        # they can SEE. Note the asymmetry with the human's own contention check,
        # which asks the same question of a BELIEVED position and therefore
        # answers it worse the narrower the cone.
        # ONE memo, used BOTH to decide legality and to rank it -- the same rule
        # the human follows (limited_vision_human.rank). Filtering unreachable
        # candidates out AFTERWARDS, as this did, is not the same thing and it
        # broke two ways at once. legal_subtasks needs the predicate INSIDE it:
        # actionable() is the termination rule that kills the stash-and-take-back
        # loop (see tasks.py), and without `ok` it answers "could I ever use a
        # dish?" with yes on a serve hatch across the divide that this robot can
        # never reach. Measured on divide at fov 360: 115 stashes of a finished
        # dish and 114 take_dish of it straight back, 0 delivered. The other half
        # of the same bug: a non-empty legal list suppresses the stash fallback
        # (tasks.py `if not out`), then every candidate is dropped here for being
        # unreachable, the list comes back empty and the robot stands still --
        # 98% of ticks on back_bar and pantry.
        # ONE BFS sweep per start instead of one A* per candidate -- see
        # geometry.dist_field. `other` is a second start, so it gets its own.
        _field = geo.dist_field(walk, pos)
        _owalk = walk | {other}
        _ofield = geo.dist_field(_owalk, other)

        def dist(cell):
            return geo.path_len_in(_field, walk, cell)

        scored = []
        for tier, verb, cell in legal_subtasks(view, held, lambda c: dist(c) is not None):
            d = dist(cell)
            if d is None:
                continue
            hd = geo.path_len_in(_ofield, _owalk, cell)
            contested = int(hd is not None and hd < d)
            scored.append((tier, self._bias(view, state, verb, cell) + contested,
                           d, cell, verb))
        scored.sort()
        return [(t, verb, cell) for t, _, _, cell, verb in scored]

    def _bias(self, view, state, verb, cell):
        """Within-tier ordering hook. 0 for everything unless overridden."""
        return 0

    # -- acting -------------------------------------------------------------
    def action(self, state):
        """One env action: hold the current subtask if it is still among the best,
        then take one step of the walk towards it, or INTERACT on arrival.

        There is nothing after the step. No right-of-way clause, no unstuck
        counter, no sidestep - see the module docstring for why none of it can
        fire on these layouts.
        """
        ranked = self.rank_subtasks(state)
        if not ranked:
            self.t += 1
            return Action.STAY, {"subtask": None}
        tier, verb, cell = ranked[0]
        # sticky within a tier -- same reason as the human's decide(): a pure
        # argmax swaps between equally good targets as we walk. Cross-tier
        # preemption is untouched.
        if self.last_subtask is not None:
            for t2, v2, c2 in ranked:
                if t2 != tier:
                    break
                if (TIER_NAME[t2], v2, c2) == self.last_subtask:
                    tier, verb, cell = t2, v2, c2
                    break
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        view = TruthView(self.mdp, state)
        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = view.walkable | {pos}
        move, arrived = geo.step_towards(walk, pos, orient, cell)
        act = Action.INTERACT if arrived else (move or Action.STAY)


        self.t += 1
        self.log.append(self.last_subtask)
        return act, {"subtask": self.last_subtask}

    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp


class SoloRobot(_BaseRobot):
    """B1. Plays as if alone in the kitchen, with one exception.

    Same ladder as the human, evaluated on ground truth. It never stages anything
    for the teammate and never reads their intent. The exception is _BaseRobot's
    contention demotion, which it inherits: on a station in the divide it will
    step aside for a human who is closer. That is small but it is real
    coordination, which is why GreedyRobot exists in its own module without it -
    the difference between the two is the value of that one rule, and it is worth
    having as a number rather than folded into the floor.
    """
    name = "solo"


class HandoffRobot(_BaseRobot):
    """B2. Cooperative, but blind to what the human can see.

    Differs from SoloRobot in one way: when it is holding something the human
    could use and cannot immediately advance it itself, it STAGES the item on a
    counter instead of carrying it around - and it picks the counter NEAREST TO
    ITSELF, because it has no model of theta and therefore no way to prefer one
    the human is looking at.

    That single blind choice is the thing an FOV-aware policy improves on. Note
    what this baseline gets right: it does hand over, it does help. It just
    cannot tell a useful counter from one in the human's blind spot, so on the
    partitioned layouts its handoffs sometimes sit there until the human's belief
    about that counter has decayed and it stops being considered at all.
    """
    name = "handoff"

    #things worth leaving out for a teammate rather than hoarding
    SHAREABLE = {"washed_plate", "garnish", "steak_dish", "garnish_dish", "dish"}

    def _bias(self, view, state, verb, cell):
        #promote stashing a shareable item ahead of carrying it around, and
        #among counters prefer the closest -- distance is the ONLY criterion
        #available without a model of the human's cone.
        if verb == "stash":
            me = state.players[self.agent_index]
            held = me.held_object.name if me.held_object else None
            if held in self.SHAREABLE:
                return -1
        return 0

    def rank_subtasks(self, state):
        ranked = super().rank_subtasks(state)
        me = state.players[self.agent_index]
        held = me.held_object.name if me.held_object else None
        if held not in self.SHAREABLE:
            return ranked
        #if we are carrying something shareable and the only thing we can do with
        #it is hold on to it, put it down where the human might find it.
        if ranked and ranked[0][0] < T_STASH:
            return ranked
        view = TruthView(self.mdp, state)
        pos = tuple(me.position)
        walk = view.walkable | {pos}
        counters = []
        for c in view.free_counters():
            d = geo.path_len(walk, pos, c)
            if d is not None:
                counters.append((d, c))
        counters.sort()
        return [(T_STASH, "stash", c) for _, c in counters] + ranked


# The other two theta-blind controls live in their own modules because neither
# may inherit _BaseRobot, for opposite reasons: greedy is DEFINED by not having
# its contention demotion, and bayes replaces the whole ranking with a posterior
# over allocations. Both are still theta-blind - the thing that would disqualify
# a control is a cone, and neither has one. Imported here so BASELINES stays the
# single registry the harnesses read.
from robot.nominal_policy.greedy import GreedyRobot                    # noqa: E402
from robot.nominal_policy.bayesian_delegation import \
    BayesianDelegationRobot                                            # noqa: E402

BASELINES = {"solo": SoloRobot, "handoff": HandoffRobot,
             "greedy": GreedyRobot, "bayes": BayesianDelegationRobot}
