"""B0  GreedyRobot -- the floor of the baseline set.

After Carroll, Shah, Ho, Griffiths, Seshia, Abbeel, Dragan (2019), "On the
Utility of Learning about Humans for Human-AI Coordination", and specifically
their GreedyHumanModel: at every step pick the highest-priority achievable
sub-task that advances the recipe, execute it by shortest path, and treat the
partner as part of the environment. No partner model, no yielding, no
coordination of any kind. It is the reference point the rest of the set is
measured from.

THETA-BLIND, like everything else in robot/nominal_policy/. It reads the full
state s through TruthView and it never touches the human's field of view: there
is no cone in this file, no visible_cells call, no assumed_fov, no posterior.
Grep it. Nor is there any partner model of any other kind -- no belief about
their intent, no reaction to what they are holding, no contention term.

WHY THIS IS NOT SoloRobot, which is the obvious objection
--------------------------------------------------------
"We already have a policy that ignores the human" is not quite true, and the gap
is the whole reason this file exists. SoloRobot inherits _BaseRobot, and
_BaseRobot coordinates in exactly one place:

  CONTENTION DEMOTION (_BaseRobot.rank_subtasks). Before committing to a station
  it asks whether the HUMAN would get there first, and if so pushes that sub-task
  down within its tier -- it gives up the shared board and goes and does
  something else. That is a partner model: it reasons about where the other agent
  is GOING, not merely about where it is standing.

GreedyRobot does not have it. It ranks purely by (tier, own distance) and it goes
where it wants to go. So SoloRobot is "does its own recipe but gives up a
contested station"; GreedyRobot is "does its own recipe". The measured difference
is the value of that one rule, and it is small and very unevenly distributed:
0.3% of actions on the main layouts, up to 47% on prep_room. That spread is the
argument for keeping them as two numbers -- a layout with a heavily-used middle
station is a different coordination problem from one without, and averaging the
two hides it.

WHAT IT SHARES WITH THE REST OF THE PACKAGE
-------------------------------------------
  * NO COLLISION HANDLING, and none needed. The partner is NOT an obstacle: walk
    is view.walkable, their tile is not removed from it, and step_towards()
    plans straight through them. There is no unstuck counter, no sidestep and no
    right-of-way clause in this file or any other. Every layout is two rooms
    joined only by pass-through counters, so the two agents share no floor and
    were measured adjacent on 0 of 4800 ticks. All of that machinery fired zero
    times and was deleted rather than left as decoration. "Treat the partner as
    part of the environment" costs nothing here because the partner is never in
    the way.
  * Stickiness WITHIN a tier. Among equally-ranked targets, keep doing what we
    were already doing. A pure argmax swaps between two equidistant pots as it
    walks and paces on the spot. Cross-tier preemption is untouched, so the
    ladder still snaps to a delivery the instant one is possible. This is not
    coordination -- it is a tie-break against our own footsteps.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from overcooked_ai_py.mdp.actions import Action              # noqa: E402
from common import geometry as geo                            # noqa: E402
from common.tasks import legal_subtasks, TIER_NAME            # noqa: E402
from common.views import TruthView                            # noqa: E402


class GreedyRobot:
    """Nearest-first on the shared ladder. No partner model at all."""

    name = "greedy"

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
        """[(tier, verb, cell)] best first: tier, then MY OWN distance, then cell.

        Compare _BaseRobot.rank_subtasks, which inserts a `contested` term
        between the first two -- that term is the coordination this baseline is
        defined by not having. Everything else about the two is identical, which
        is what makes the difference between them attributable to that one term.
        """
        view = TruthView(self.mdp, state)
        me = state.players[self.agent_index]
        pos = tuple(me.position)
        held = me.held_object.name if me.held_object else None
        walk = view.walkable | {pos}

        # One memo, feeding legality AND ranking -- see the long note in
        # _BaseRobot.rank_subtasks. Kept identical here on purpose: the whole
        # value of this baseline is that it differs from solo in the `contested`
        # term and in nothing else.
        _field = geo.dist_field(walk, pos)

        def dist(cell):
            return geo.path_len_in(_field, walk, cell)

        scored = []
        for tier, verb, cell in legal_subtasks(view, held, lambda c: dist(c) is not None):
            d = dist(cell)
            if d is None:
                continue
            scored.append((tier, d, cell, verb))
        scored.sort()
        return [(t, verb, cell) for t, _, cell, verb in scored]

    # -- acting -------------------------------------------------------------
    def action(self, state):
        ranked = self.rank_subtasks(state)
        if not ranked:
            self.t += 1
            return Action.STAY, {"subtask": None}
        tier, verb, cell = ranked[0]
        if self.last_subtask is not None:
            for t2, v2, c2 in ranked:
                if t2 != tier:
                    break                     # never sticky across tiers: a
                                              # delivery must preempt instantly
                if (TIER_NAME[t2], v2, c2) == self.last_subtask:
                    tier, verb, cell = t2, v2, c2
                    break
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        view = TruthView(self.mdp, state)
        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = view.walkable | {pos}
        # The walk is planned over the FULL walkable floor including whatever
        # tile the human is on. Nothing below fixes that up, because there is
        # nothing to fix: the two agents are in different rooms, so a path from
        # here can never contain their tile in the first place.
        move, arrived = geo.step_towards(walk, pos, orient, cell)
        act = Action.INTERACT if arrived else (move or Action.STAY)

        self.t += 1
        self.log.append(self.last_subtask)
        return act, {"subtask": self.last_subtask}

    # -- plumbing -----------------------------------------------------------
    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp
