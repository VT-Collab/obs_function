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
import math
import os
import random
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from overcooked_ai_py.mdp.actions import Action              # noqa: E402
from common import geometry as geo                            # noqa: E402
from common.tasks import legal_subtasks, TIER_NAME, T_EXPLORE  # noqa: E402
from common.views import TruthView                            # noqa: E402
# Imported, never redefined: the calibration argument for these two numbers is
# in bayesian_delegation.py and there must be exactly one copy of it. Standalone
# on purpose -- see baselines.py's module docstring for why this file does not
# share its sampling machinery with _BaseRobot's.
from robot.nominal_policy.bayesian_delegation import GAMMA, TIER_GAIN  # noqa: E402

BETA = 8.0        # sharpness of pi(tau) ~ value(tau) ** beta; see baselines.py
RHO = 0.95        # hold ~20 ticks between spontaneous re-draws


class GreedyRobot:
    """Nearest-first on the shared ladder. No partner model at all.

    action() DRAWS its sub-task from a distribution over rank_subtasks() rather
    than taking its head, the same way as every other baseline in
    robot/nominal_policy/ -- see baselines.py's module docstring for the
    distribution and the sticky kernel. This file's copy is standalone rather
    than shared, because greedy's whole reason to exist in its own module is
    that it must not inherit anything from _BaseRobot.
    """

    name = "greedy"

    def __init__(self, mdp, agent_index=0, seed=0, beta=BETA, rho=RHO):
        self.seed = seed
        self.mdp = mdp
        self.agent_index = agent_index
        self.other_index = 1 - agent_index
        self.beta, self.rho = beta, rho
        self.reset()

    def reset(self):
        self._rng = random.Random(self.seed * 2)
        self.t = 0
        self.last_subtask = None
        self.committed = None            # (verb, cell) -- NOT the tier, see _find
        self.last_pi = {}
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

    #---- NEW: one-line forward to baselines._remaining() -- the ticks-
    #     still-owed query robot/filter/core/my_fov_filter.py's safety
    #     filter reads. Its logic only ever touches mdp/agent_index/
    #     rank_subtasks, none of the contention/sampling machinery this
    #     file deliberately keeps standalone from _BaseRobot, so sharing
    #     it costs nothing this file's independence actually protects --
    #     imported lazily (not at module scope) to keep this file's own
    #     import list exactly what it was.
    #----------------------------------------------------------------------
    def remaining(self, state, subtask):
        """Ticks still owed to finish `subtask`=(tier, verb, cell) from
        `state`. See baselines._remaining() for the logic."""
        from robot.nominal_policy.baselines import _remaining
        return _remaining(self, state, subtask)

    # -- the distribution ----------------------------------------------------
    @staticmethod
    def _short(sub):
        """(tier, verb, cell) -> 'COMP plate_steak@12,5'. Same shape as
        baselines.short_subtask, kept as its own copy -- see the module
        docstring for why this file does not import it."""
        tier, verb, cell = sub
        return "%-4s %s@%d,%d" % (TIER_NAME[tier][:4], verb, cell[0], cell[1])

    @staticmethod
    def _steps(walk, pos, orient, cell):
        """Ticks to walk to `cell`, turn, and INTERACT. None if unreachable.

        Same shape as baselines.steps_to_finish, kept as its own copy here --
        see the module docstring for why this file does not import it.
        """
        d = geo.path_len(walk, pos, cell)
        if d is None:
            return None
        if d == 0:
            return 1 if (pos[0] + orient[0], pos[1] + orient[1]) == cell else 2
        return d + 2

    def _pi(self, ranked, pos, orient, walk):
        """{(tier, verb, cell): p} ~ value(tau) ** beta. No penalty term: this
        baseline's own key IS (tier, distance), nothing to reconcile."""
        cand = [s for s in ranked if self._steps(walk, pos, orient, s[2]) is not None]
        if not cand:
            return {}
        logv = {}
        for sub in cand:
            n = self._steps(walk, pos, orient, sub[2])
            logv[sub] = (math.log(TIER_GAIN) * (T_EXPLORE - sub[0])
                         + math.log(GAMMA) * n)
        if self.beta == float("inf"):
            top = max(logv.values())
            win = [s for s in cand if logv[s] == top]
            return {s: (1.0 / len(win) if s in win else 0.0) for s in cand}
        if self.beta == 0:
            return {s: 1.0 / len(cand) for s in cand}
        hi = max(logv.values())
        w = {s: math.exp(self.beta * (logv[s] - hi)) for s in cand}
        z = sum(w.values())
        return {s: x / z for s, x in w.items()}

    def _sample(self, pi):
        r, acc = self._rng.random(), 0.0
        for sub, p in pi.items():
            acc += p
            if r < acc:
                return sub
        return next(reversed(list(pi)))     # float slop on the last bucket

    def _find(self, ranked, committed):
        if committed is None:
            return None
        for sub in ranked:
            if (sub[1], sub[2]) == committed:
                return sub
        return None

    @staticmethod
    def top3(pi, held):
        out = []
        for s, p in sorted(pi.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
            out.append((GreedyRobot._short(s), float(p), s == held))
        return out

    def _info(self, held, pi, ranked, why):
        return {"subtask": self.last_subtask,
                "subtask_dist": {(TIER_NAME[s[0]], s[1], s[2]): p
                                 for s, p in pi.items()},
                "subtask_p": pi.get(held, 0.0) if held else 0.0,
                "subtask_rank": (ranked.index(held) if held and ranked
                                 and held in ranked else None),
                "subtask_redraw": why,
                "top3": self.top3(pi, held),
                "top3_kind": "p"}

    # -- acting -------------------------------------------------------------
    def action(self, state):
        """action() DRAWS its sub-task -- see the module docstring."""
        ranked = self.rank_subtasks(state)
        view = TruthView(self.mdp, state)
        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = view.walkable | {pos}

        if not ranked:
            self.committed = self.last_subtask = None
            self.last_pi = {}
            self.t += 1
            return Action.STAY, self._info(None, {}, None, "none")

        pi = self._pi(ranked, pos, orient, walk)
        self.last_pi = pi
        if not pi:
            self.committed = self.last_subtask = None
            self.t += 1
            return Action.STAY, self._info(None, {}, None, "none")

        held = self._find(ranked, self.committed)
        forced = held is None or ranked[0][0] < held[0]
        if forced:
            held, why = self._sample(pi), "forced"
        elif self._rng.random() > self.rho:
            held, why = self._sample(pi), "spontaneous"
        else:
            why = "held"

        tier, verb, cell = held
        self.committed = (verb, cell)
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        # The walk is planned over the FULL walkable floor including whatever
        # tile the human is on. Nothing below fixes that up, because there is
        # nothing to fix: the two agents are in different rooms, so a path from
        # here can never contain their tile in the first place.
        move, arrived = geo.step_towards(walk, pos, orient, cell)
        act = Action.INTERACT if arrived else (move or Action.STAY)

        self.t += 1
        self.log.append(self.last_subtask)
        return act, self._info(held, pi, ranked, why)

    # -- plumbing -----------------------------------------------------------
    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp
