"""A true distribution over sub-tasks, and a sticky draw from it.

Every policy in robot/nominal_policy/ is deterministic: rank_subtasks(state) then
ranked[0]. Given that choice the low-level action is a pure function -- all four
call the same geo.step_towards, no policy ever reads its own _rng, and A* breaks
ties by cell coordinate. So ALL the behavioural variation a nominal policy has
lives in which sub-task it picks, which makes that the only place worth putting
a distribution.

WHAT THE DISTRIBUTION IS, AND WHY IT IS NOT INVENTED HERE
--------------------------------------------------------
bayes already had to solve this. Bayesian delegation compares ALLOCATIONS, so it
cannot use the lexicographic (tier, distance) ordering the other three use -- it
needs sub-tasks commensurable on one scale. Its answer, from
bayesian_delegation.py:

    value(tau) = TIER_GAIN ** (T_EXPLORE - tier) * GAMMA ** steps

calibrated so TIER_GAIN**1 == GAMMA**-21: one rung of the ladder is worth about
21 tiles of walking, so tier dominance survives on kitchens this size but does so
by arithmetic rather than by construction. _Snapshot._prior() is exactly that
value, normalised.

So the distribution was already written; it was just trapped inside one policy.
subtask_pi() lifts it out and applies it to whatever candidates a baseline
proposed:

    pi(tau) ~ value(tau) ** beta

beta -> inf reproduces today's argmax exactly, beta == 1 IS bayes's team-value
prior restricted to a single agent, beta == 0 is uniform over legal sub-tasks.

PRIOR, NOT POSTERIOR. For greedy/solo/handoff this is a prior: a stochastic
policy proportional to how good each sub-task is by that policy's own lights,
conditioned on nothing. bayes's corresponding object is a genuine POSTERIOR,
P(tau_robot | history) -- that same prior updated by inverse planning from the
actions both agents were seen to take. The gap between them is what the evidence
buys, which is the comparison the baseline set exists to make, so this file is
careful never to call the first one a posterior.

STICKINESS THAT DOES NOT LIE ABOUT THE DISTRIBUTION
---------------------------------------------------
Drawing a fresh target every tick makes the robot oscillate: two steps toward the
sink, re-draw, turn around. Holding until a sub-task completes removes the
randomness. The way out is a sticky kernel -- hold with probability rho,
otherwise re-draw from pi:

    K(tau -> tau') = rho * delta(tau' = tau) + (1 - rho) * pi(tau')

which has pi as its EXACT stationary distribution:

    sum_tau pi(tau) K(tau -> tau')
        = rho * pi(tau') + (1 - rho) * pi(tau') * sum_tau pi(tau)
        = pi(tau')

That is the whole point: the occupancy distribution over ticks EQUALS the draw
distribution, so stickiness costs nothing in fidelity. It holds only under two
conditions, which are requirements on this code rather than preferences, and
test_subtask_dist.py asserts both by showing the check FAILS when either is
broken:

  * rho is UNIFORM across sub-tasks. Any per-sub-task hold probability -- "hold
    longer for distant targets" is the tempting one -- reintroduces dwell-time
    bias and pi stops being stationary.
  * a re-draw samples the FULL pi, INCLUDING the currently held sub-task.
    Excluding it and renormalising is the standard way to get this wrong; it
    makes the chain actively avoid its own mode.

Forced re-draws (the held sub-task finished or became illegal, or a strictly more
urgent tier appeared) are driven by the world rather than by the sampler, so they
are not pi-preserving in general -- but on those ticks the candidate set changed,
so pi is a different distribution anyway. The claim this file makes, and the one
the tests check, is per-tick: the draw is always from the CURRENT true pi, and
between forced events holding is pi-preserving.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from overcooked_ai_py.mdp.actions import Action                    # noqa: E402
from common import geometry as geo                                  # noqa: E402
from common.tasks import TIER_NAME, T_EXPLORE                       # noqa: E402
from common.views import TruthView                                  # noqa: E402
# Imported, never redefined: the calibration argument for these two numbers is
# in bayesian_delegation.py and there must be exactly one copy of it.
from robot.nominal_policy.bayesian_delegation import (              # noqa: E402
    GAMMA, TIER_GAIN, IDLE)

BETA = 8.0        # see _default_beta_note below
RHO = 0.95        # hold ~20 ticks between spontaneous re-draws

# WHY beta = 8 AND NOT 1. Two sub-tasks on the same tier five tiles apart differ
# in value by GAMMA**5 = 0.77, so beta = 1 -- bayes's own prior -- puts them at
# 56/44. That is too flat to be a policy: the robot would barely prefer the
# nearer of two identical jobs. beta = 8 puts that pair at roughly 90/10, while a
# one-tier gap sits at 3**8, so the ladder stays effectively deterministic and a
# possible DELIVER still preempts absolutely. It is a sharpness knob on a fixed
# distribution, not a second ranking rule.


def steps_to_finish(walk, pos, orient, cell):
    """Ticks to walk to `cell`, turn, and INTERACT. None if unreachable.

    Deliberately identical to _Snapshot.steps_from in bayesian_delegation.py:
    walk there (d), turn to face it (1), press (1); standing beside it already
    facing it costs the one press. geo.path_len returns the same d that file's
    backward BFS does, so the two agree cell for cell -- which is what lets
    subtask_pi at beta=1 reproduce _prior() exactly.
    """
    d = geo.path_len(walk, pos, cell)
    if d is None:
        return None
    if d == 0:
        return 1 if (pos[0] + orient[0], pos[1] + orient[1]) == cell else 2
    return d + 2


def within_tier_penalty(inner, state, sub, walk, pos):
    """The term the baseline sorts on BETWEEN tier and distance, or 0.

    _BaseRobot ranks on (tier, _bias + contested, distance): solo demotes a
    station the human would reach first, handoff additionally promotes stashing
    something shareable. greedy has neither and returns 0 here, which is why its
    ordering is (tier, distance) and matches the raw value function already.

    Without this term the value function has no way to express that middle key,
    so argmax(pi) is not the policy's own pick -- measured at 2.4% of ticks for
    solo and 1.2% for handoff. A distribution whose mode is not the deterministic
    choice is not the distribution that policy induces, so this is a correctness
    requirement rather than a refinement.

    One unit of penalty is worth one tier here. That is the closest this scale
    allows and it is not exact -- see HOW FAITHFUL, below.

    HOW FAITHFUL, MEASURED. Over 4 layouts x fov{30,360} x 400 ticks,
    argmax(pi) is the baseline's own pick on:

        greedy   99.9%    (its key IS (tier, distance) -- nothing to reconcile)
        solo     99.5%    (was 97.6% before this term existed)
        handoff  99.7%    (was 98.8%)

    The residue is NOT fixable on this scale, and the reason is worth stating
    rather than tuning against. The baselines are LEXICOGRAPHIC: tier wins
    absolutely, penalty orders within a tier, distance only breaks ties. The
    value function is deliberately NOT -- TIER_GAIN**1 == GAMMA**-21 makes one
    rung worth about 21 tiles, so a long enough walk can outweigh a rung.
    bayesian_delegation.py says so in as many words: "on a big enough map this
    robot will take the ready garnish under its nose over the finished dish
    across the room. That is a deliberate difference from solo/handoff/greedy,
    not a bug."

    So no penalty weight can work: it would have to be worth less than one tier
    (3x) and more than the within-tier distance span (GAMMA**-40, about 7.7x),
    and 3 < 7.7. Half the residue is plain distance ties, the other half is
    exactly the tier-vs-distance case above. Making pi lexicographic instead
    would buy the last 0.5% at the cost of the property that makes it principled
    -- that pi at beta=1 IS bayes's prior, which test_subtask_dist.py checks to
    machine precision. That trade is available but it is a different design.
    """
    if not hasattr(inner, "_bias"):
        return 0
    _, verb, cell = sub
    pen = inner._bias(TruthView(inner.mdp, state), state, verb, cell)
    other = tuple(state.players[inner.other_index].position)
    d = geo.path_len(walk, pos, cell)
    hd = geo.path_len(walk | {other}, other, cell)
    return pen + int(hd is not None and d is not None and hd < d)


def subtask_pi(ranked, pos, orient, walk, beta=BETA, penalty=None):
    """{(tier, verb, cell): p} over the candidates a baseline proposed.

    Computed in logs and softmaxed against the max, so beta can be pushed high
    enough to reproduce argmax without 3**(8*beta) overflowing on the way.
    beta=inf is handled exactly rather than by a large float.

    `penalty` is the per-sub-task within-tier term from within_tier_penalty();
    omit it and this is the bare value function, which is what makes the
    beta=1-equals-bayes's-prior check in the tests a real comparison.
    """
    cand = [s for s in ranked if steps_to_finish(walk, pos, orient, s[2]) is not None]
    if not cand:
        return {}
    penalty = penalty or {}

    logv = {}
    for sub in cand:
        n = steps_to_finish(walk, pos, orient, sub[2])
        logv[sub] = (math.log(TIER_GAIN) * (T_EXPLORE - sub[0] - penalty.get(sub, 0))
                     + math.log(GAMMA) * n)

    if beta == float("inf"):
        top = max(logv.values())
        win = [s for s in cand if logv[s] == top]
        return {s: (1.0 / len(win) if s in win else 0.0) for s in cand}
    if beta == 0:
        return {s: 1.0 / len(cand) for s in cand}

    hi = max(logv.values())
    w = {s: math.exp(beta * (logv[s] - hi)) for s in cand}
    z = sum(w.values())
    return {s: x / z for s, x in w.items()}


class StochasticSubtask:
    """Any nominal policy, with argmax replaced by a sticky draw from pi.

    Wraps rather than subclasses, so baselines.py / greedy.py /
    bayesian_delegation.py are untouched and this can be removed without trace.
    rank_subtasks is delegated verbatim, which is what keeps QMDPFilter able to
    wrap one of these; update is delegated, which is what keeps bayes fed by the
    Drivers fan-out in robot/methods.py.
    """

    def __init__(self, inner, beta=BETA, rho=RHO, seed=0, source="prior"):
        self.inner = inner
        self.beta, self.rho, self.source = beta, rho, source
        # Its own stream, distinct from the baselines' seed*2, so a stochastic
        # run is reproducible and the wrapped policy's stream is undisturbed.
        self._rng = random.Random(seed * 7 + 11)
        self.committed = None            # (verb, cell) -- NOT the tier, see _find
        self.last_subtask = None
        self.last_pi = {}

    # -- plumbing: everything except the choice is the inner policy's ---------
    @property
    def mdp(self):
        return self.inner.mdp

    @property
    def agent_index(self):
        return self.inner.agent_index

    @property
    def p(self):
        return self.inner.p if hasattr(self.inner, "p") else {}

    def rank_subtasks(self, state):
        return self.inner.rank_subtasks(state)

    def update(self, state, human_action):
        return self.inner.update(state, human_action)

    def reset(self):
        self.inner.reset()
        self.committed = self.last_subtask = None
        self.last_pi = {}

    def set_agent_index(self, i):
        self.inner.set_agent_index(i)
        self.committed = None

    def set_mdp(self, mdp):
        self.inner.set_mdp(mdp)

    # -- the distribution ----------------------------------------------------
    def _pi(self, ranked, pos, orient, walk, state):
        """pi over `ranked`. The posterior branch is bayes's and only bayes's."""
        if self.source == "posterior" and hasattr(self.inner, "_marginal"):
            marg = self.inner._marginal(0)
            # IDLE is dropped exactly as rank_subtasks already drops it, so the
            # candidate set matches the deterministic policy's and the DRAW is
            # the only difference between them.
            live = {s: p for s, p in marg.items() if s is not IDLE and p > 0}
            z = sum(live.values())
            if z > 0:
                return {s: p / z for s, p in live.items()}
            return {}                       # nothing but IDLE: caller will STAY
        pen = {s: within_tier_penalty(self.inner, state, s, walk, pos)
               for s in ranked}
        return subtask_pi(ranked, pos, orient, walk, self.beta, pen)

    def _sample(self, pi):
        r, acc = self._rng.random(), 0.0
        for sub, p in pi.items():
            acc += p
            if r < acc:
                return sub
        return next(reversed(list(pi)))     # float slop on the last bucket

    def _find(self, ranked, committed):
        """The current ranking's entry for a (verb, cell), or None if it is gone.

        Matched on (verb, cell) and NOT on the tier: a sub-task whose tier moved
        because the world moved is the same job, and treating it as gone would
        force a re-draw every time a pot finished somewhere.
        """
        if committed is None:
            return None
        for sub in ranked:
            if (sub[1], sub[2]) == committed:
                return sub
        return None

    # -- the tick ------------------------------------------------------------
    def action(self, state):
        ranked = self.inner.rank_subtasks(state)
        me = state.players[self.inner.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = TruthView(self.mdp, state).walkable | {pos}

        if not ranked:
            self.committed = self.last_subtask = None
            self.last_pi = {}
            return Action.STAY, self._info(None, {}, None, "none")

        pi = self._pi(ranked, pos, orient, walk, state)
        self.last_pi = pi
        if not pi:
            self.committed = self.last_subtask = None
            return Action.STAY, self._info(None, {}, None, "none")

        held = self._find(ranked, self.committed)
        # Forced when the job is finished or illegal, or when a strictly more
        # urgent tier has appeared -- the baselines never stick across tiers,
        # because a possible DELIVER has to preempt on the tick it becomes
        # possible. Stickiness must not weaken that.
        forced = held is None or ranked[0][0] < held[0]
        if forced:
            held, why = self._sample(pi), "forced"
        elif self._rng.random() > self.rho:
            # Re-draw from the FULL pi, current sub-task included. Excluding it
            # would make the chain avoid its own mode and pi would stop being
            # the stationary distribution -- see the module docstring.
            held, why = self._sample(pi), "spontaneous"
        else:
            why = "held"

        tier, verb, cell = held
        self.committed = (verb, cell)
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        move, arrived = geo.step_towards(walk, pos, orient, cell)
        act = Action.INTERACT if arrived else (move or Action.STAY)

        # bayes scores its OWN last action inside update() -- that is what keeps
        # its belief committed to the job it is walking towards. We bypassed
        # inner.action(), so hand the action back or that half of the inverse
        # planning silently stops working.
        if hasattr(self.inner, "last_action"):
            self.inner.last_action = act
        if hasattr(self.inner, "t"):
            self.inner.t += 1
        if hasattr(self.inner, "log"):
            self.inner.log.append(self.last_subtask)
        self.inner.last_subtask = self.last_subtask

        return act, self._info(held, pi, ranked, why)

    def _info(self, held, pi, ranked, why):
        out = {"subtask": self.last_subtask,
               "subtask_dist": {(TIER_NAME[s[0]], s[1], s[2]): p
                                for s, p in pi.items()},
               "subtask_p": pi.get(held, 0.0) if held else 0.0,
               "subtask_rank": (ranked.index(held) if held and ranked
                                and held in ranked else None),
               "subtask_redraw": why}
        # bayes's own readouts, reproduced rather than taken from inner.action()
        # because we never call it. Same keys, so both HUDs light up unchanged.
        if hasattr(self.inner, "partner_map"):
            name, prob = self.inner.partner_map()
            out.update({"partner_subtask": name, "partner_p": prob,
                        "partner_entropy": self.inner.partner_entropy(),
                        "n_alloc": len(self.inner.belief)})
        return out


def stochastic(cls, beta=BETA, rho=RHO, source="prior", **inner_kw):
    """A class with the standard (mdp, agent_index, seed) signature.

    QMDPFilter._rollout rebuilds its baseline with
    `type(self.baseline)(self.mdp, agent_index=...)`, so a wrapper whose __init__
    takes (inner, ...) would crash the moment anything tried to roll one out.
    Handing back a real class instead of a partial keeps that door open even
    though robot/methods.py does not currently walk through it.
    """
    class _Stochastic(StochasticSubtask):
        INNER = cls

        def __init__(self, mdp, agent_index=0, seed=0):
            StochasticSubtask.__init__(
                self, cls(mdp, agent_index=agent_index, seed=seed, **inner_kw),
                beta=beta, rho=rho, seed=seed, source=source)

    _Stochastic.__name__ = "Stochastic" + cls.__name__
    _Stochastic.__qualname__ = _Stochastic.__name__
    return _Stochastic
