"""Two robot baselines. NEITHER models the human's field of view.

Both see the full state (per the problem statement: the robot directly observes
s), and both are deliberately blind to theta. They are the control condition: any
gain a later FOV-aware policy shows has to be measured against these, so it is
important that the ONLY thing they lack is the human's observation model.

    B1  SoloRobot      ignores the human entirely. Runs the recipe itself.
    B2  HandoffRobot   does stage items on counters for the human, but picks the
                       counter by DISTANCE, having no idea what the human can see.

B2 is the interesting control. It performs the same helpful act as an FOV-aware
robot - put something where the human can pick it up - but chooses WHERE blindly.
The FOV-aware policy differs from it in exactly one decision, which is what makes
the comparison clean.

Both expose rank_subtasks(state) -> [(tier, verb, cell), ...] best first, so the
planned QMDP filter can take the top-k, roll each out to delivery, and re-order
them without either baseline needing to change.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from overcooked_ai_py.mdp.actions import Action              # noqa: E402
from common import geometry as geo                            # noqa: E402
from common.tasks import legal_subtasks, TIER_NAME, T_STASH, COUNTER  # noqa: E402
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
        self._last_pos = None
        self._stuck = 0
        self.log = []

    # -- ranking ------------------------------------------------------------
    def rank_subtasks(self, state):
        """[(tier, verb, cell)] best first. The hook the QMDP filter will use."""
        view = TruthView(self.mdp, state)
        me = state.players[self.agent_index]
        pos = tuple(me.position)
        held = me.held_object.name if me.held_object else None
        # teammate's cell is not standable this tick -- see the human for why
        other = tuple(state.players[self.other_index].position)
        walk = (view.walkable | {pos}) - {other}

        scored = []
        for tier, verb, cell in legal_subtasks(view, held):
            d = geo.path_len(walk, pos, cell)
            if d is None:
                continue
            # CONTENTION: yield a station the human will reach first. The robot
            # always knows where the human is (it observes s fully), so it is
            # the one that can reliably do the yielding. This is coordination,
            # NOT theta-modelling -- it uses position only, never the cone.
            hd = geo.path_len(walk | {other}, other, cell)
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
        ranked = self.rank_subtasks(state)
        if not ranked:
            self.t += 1
            return Action.STAY, {"subtask": None}
        tier, verb, cell = ranked[0]
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        view = TruthView(self.mdp, state)
        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        other = tuple(state.players[self.other_index].position)
        walk = view.walkable | {pos}
        move, arrived = geo.step_towards(walk, pos, orient, cell, {other})
        act = Action.INTERACT if arrived else (move or Action.STAY)

        # RIGHT OF WAY: the robot gives way. If our next step is into the human,
        # or we failed to move last tick, dodge immediately rather than shoving.
        # The human holds station, so one of us moving is enough to resolve it.
        nxt = (pos[0] + act[0], pos[1] + act[1]) if isinstance(act, tuple) else None
        # A move into a NON-walkable cell is a TURN, not a step: position is
        # meant to stay the same. Counting that as "stuck" made the robot
        # sidestep away every time it lined up on a station, so it could never
        # complete the turn-then-interact that ends every single subtask.
        really_moving = nxt is not None and nxt in walk
        blocked_now = nxt == other
        if act is not Action.INTERACT and really_moving and \
                (blocked_now or pos == self._last_pos):
            self._stuck += 1
            if self._stuck >= 1:
                act = geo.sidestep(walk, pos, {other, nxt}, self._rng) \
                      or geo.sidestep(walk, pos, {other}, self._rng) or Action.STAY
                self._stuck = 0
        else:
            self._stuck = 0
        self._last_pos = pos

        self.t += 1
        self.log.append(self.last_subtask)
        return act, {"subtask": self.last_subtask}

    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp


class SoloRobot(_BaseRobot):
    """B1. Plays as if alone in the kitchen.

    Same ladder as the human, evaluated on ground truth. It never stages
    anything for the teammate and never reacts to them - the human is just a
    moving obstacle. This is the floor: whatever the team achieves here is what
    you get with zero coordination.
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


BASELINES = {"solo": SoloRobot, "handoff": HandoffRobot}
