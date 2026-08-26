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
FoV filter in robot/filter/ (my_fov_filter.py) can read what this policy
prefers without either baseline needing to change. That layer does NOT simply
re-order the list: it
aggregates the tuples into (tier, verb) JOBS, hands each job its whole legal cell
set, and searches over which cell and which first step -- precisely because the
cell chosen in here is chosen theta-blind, by distance from the ROBOT. See
robot/filter/DESIGN.md section 2 for the contract.

ACTION IS A STICKY DRAW, NOT AN ARGMAX. rank_subtasks() is still the exact,
deterministic preference order -- (tier, within-tier penalty, distance) -- but
action() does not simply take its head. It lifts that ranking into a
distribution, pi(tau) ~ value(tau) ** beta (see BETA below), and samples from it
with a sticky kernel: hold the current pick with probability rho, otherwise
re-draw from the FULL pi. That is what makes the low-level action the only place
a nominal policy's behavioural variation lives, and what makes the filter in
robot/filter/ able to consume a genuine DISTRIBUTION over sub-tasks rather than a
lifted reconstruction of a policy that never drew anything. See BETA/RHO below
for the calibration and the stationarity argument.

GreedyRobot (greedy.py) and BayesianDelegationRobot (bayesian_delegation.py) each
draw the same way but STANDALONE -- their own action(), their own sampling, no
shared base class and no wrapper. Neither may inherit _BaseRobot in the first
place, for opposite reasons: greedy is DEFINED by not having its contention
demotion, and bayes replaces the whole ranking with a posterior over
allocations. bayes does not even need BETA: it already has a genuine posterior
over sub-tasks (its belief), so its draw samples that directly rather than
lifting a value function into one.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from overcooked_ai_py.mdp.actions import Action                          # noqa: E402
from common import geometry as geo                                       # noqa: E402
from common.tasks import legal_subtasks, TIER_NAME, T_STASH, T_EXPLORE   # noqa: E402
from common.views import TruthView                                       # noqa: E402
# Imported, never redefined: the calibration argument for these two numbers is
# in bayesian_delegation.py and there must be exactly one copy of it.
from robot.nominal_policy.bayesian_delegation import GAMMA, TIER_GAIN    # noqa: E402

BETA = 8.0        # sharpness of pi(tau) ~ value(tau) ** beta
RHO = 0.95        # hold ~20 ticks between spontaneous re-draws

# WHY beta = 8 AND NOT 1. Two sub-tasks on the same tier five tiles apart differ
# in value by GAMMA**5 = 0.77, so beta = 1 -- bayes's own prior -- puts them at
# 56/44. That is too flat to be a policy: the robot would barely prefer the
# nearer of two identical jobs. beta = 8 puts that pair at roughly 90/10, while a
# one-tier gap sits at 3**8, so the ladder stays effectively deterministic and a
# possible DELIVER still preempts absolutely. It is a sharpness knob on a fixed
# distribution, not a second ranking rule.
#
# THE STICKY KERNEL. Drawing a fresh target every tick makes the robot
# oscillate: two steps toward the sink, re-draw, turn around. Holding until a
# sub-task completes removes the randomness. The way out is a sticky kernel --
# hold with probability rho, otherwise re-draw from pi:
#
#     K(tau -> tau') = rho * delta(tau' = tau) + (1 - rho) * pi(tau')
#
# which has pi as its EXACT stationary distribution:
#
#     sum_tau pi(tau) K(tau -> tau')
#         = rho * pi(tau') + (1 - rho) * pi(tau') * sum_tau pi(tau)
#         = pi(tau')
#
# That is the whole point: the occupancy distribution over ticks EQUALS the
# draw distribution, so stickiness costs nothing in fidelity. It holds only
# under two conditions, which are requirements on this code rather than
# preferences, and test_baselines.py asserts both by showing the check FAILS
# when either is broken:
#
#   * rho is UNIFORM across sub-tasks. Any per-sub-task hold probability --
#     "hold longer for distant targets" is the tempting one -- reintroduces
#     dwell-time bias and pi stops being stationary.
#   * a re-draw samples the FULL pi, INCLUDING the currently held sub-task.
#     Excluding it and renormalising is the standard way to get this wrong; it
#     makes the chain actively avoid its own mode.
#
# Forced re-draws (the held sub-task finished or became illegal, or a strictly
# more urgent tier appeared) are driven by the world rather than by the
# sampler, so they are not pi-preserving in general -- but on those ticks the
# candidate set changed, so pi is a different distribution anyway. The claim
# this file makes, and the one the tests check, is per-tick: the draw is
# always from the CURRENT true pi, and between forced events holding is
# pi-preserving.


def short_subtask(sub):
    """(tier, verb, cell) -> 'COMP plate_steak@12,5'. For one line of HUD.

    Three of these plus a prefix has to fit inside play.py's MIN_W of 980 px at
    couriernew 13, so 956 px of usable width. The full form -- tier spelled out,
    `(12, 5)` with its space -- measures 1160 px for three entries and silently
    runs off the right edge of the window, which is how a HUD line stops being
    read rather than stops being true. This form measures 776 px on the widest
    real line seen (three stash targets, which have the longest cells and a
    redraw tag). Tier is cut to four characters because COMPLETE/COLLECT and
    START/STASH are the pairs that nearly collide; all nine stay distinct, which
    is asserted rather than assumed.

    Lives here rather than in either harness so both print the same string.
    """
    tier, verb, cell = sub
    return "%-4s %s@%d,%d" % (TIER_NAME[tier][:4], verb, cell[0], cell[1])


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
    """The term _BaseRobot sorts on BETWEEN tier and distance, or 0.

    _BaseRobot ranks on (tier, _bias + contested, distance): solo demotes a
    station the human would reach first, handoff additionally promotes stashing
    something shareable. Without this term the value function has no way to
    express that middle key, so argmax(pi) is not the policy's own rank_subtasks
    pick -- measured at 2.4% of ticks for solo and 1.2% for handoff before this
    term existed. A distribution whose mode is not the ranking's head is not the
    distribution that ranking induces, so this is a correctness requirement
    rather than a refinement.

    One unit of penalty is worth one tier here. That is the closest this scale
    allows and it is not exact -- see HOW FAITHFUL, below.

    HOW FAITHFUL, MEASURED. Over 4 layouts x fov{30,360} x 400 ticks,
    argmax(pi) agreed with rank_subtasks()[0] on:

        solo     99.5%    (was 97.6% before this term existed)
        handoff  99.7%    (was 98.8%)

    The residue is NOT fixable on this scale, and the reason is worth stating
    rather than tuning against. rank_subtasks is LEXICOGRAPHIC: tier wins
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
    -- that pi at beta=1 IS bayes's prior, which test_baselines.py checks to
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


#---- NEW: verb -> (the mdp's own "is it ready" query, the mdp's own tick
#     cap). The only two verbs whose job carries a timer that advances on
#     repeated INTERACTs rather than being finished by one -- see
#     remaining() below, which is the only reader of this.
#---- CHANGED: was keyed by the CELL'S TERRAIN (POT/BOARD/SINK), which was
#     wrong -- a board hosts load_board (one press) and later chop (the
#     timer) at different times, and keying on terrain routed load_board
#     through the timer math too. Once the onion landed the board became
#     "in progress" on the CHOP timer, and remaining() charged all 5 of
#     those ticks to load_board, which a single INTERACT had already
#     finished -- so INTERACT scored worse than STAY and the search never
#     recovered. Keying on the verb itself means remaining() only reads the
#     timer when the subtask actually IS chop/wash.
#----------------------------------------------------------------------
_TIMED = {"chop": ("garnish_ready_at_location", "chopping_time"),
          "wash": ("plate_washed_at_location", "wash_time")}


#---- NEW: was a _BaseRobot method. Moved to module level, taking `baseline`
#     explicitly, because GreedyRobot and BayesianDelegationRobot are
#     deliberately standalone (see the module docstring on why neither may
#     inherit _BaseRobot) but still need to answer this -- robot/filter/
#     core/my_fov_filter.py calls self.baseline.remaining(...) on whatever
#     baseline it's wrapping, greedy/bayes included. The logic itself was
#     never _BaseRobot-specific: it only ever touches `baseline.mdp`,
#     `baseline.agent_index` and `baseline.rank_subtasks`, which all three
#     robot classes have. Each class below exposes it as a one-line
#     `remaining()` method that just calls this.
#----------------------------------------------------------------------
def _remaining(baseline, state, subtask):
    """Ticks still owed to finish `subtask`=(tier, verb, cell) from
    `state`: the walk to `cell` plus whatever work is still owed once
    there. One call answers every verb -- fetch/stash/deliver owe the
    walk plus their one press, chop/wash additionally owe whatever is
    left on the station's own timer -- so a caller never has to know
    which kind of job it is asking about.
    """
    _, verb, cell = subtask
    me = state.players[baseline.agent_index]
    pos = tuple(me.position)
    walk = TruthView(baseline.mdp, state).walkable | {pos}
    d = geo.path_len_in(geo.dist_field(walk, pos), walk, cell)
    if d is None:
        return None

    #---- once actually adjacent, a press can only land THIS tick if the
    #     robot is already facing `cell` -- a motion action into a
    #     non-walkable cell only reorients (see overcooked_mdp.py's
    #     _move_if_direction), but that reorientation still costs a real
    #     tick, same as the press itself, so it has to be counted
    #     separately rather than assumed free. Only checked once d == 0;
    #     while still walking, the final approach step usually leaves the
    #     robot facing the right way anyway, so that case stays folded
    #     into the flat "+1 for the press" below.
    #----------------------------------------------------------------------
    turn = 0
    if d == 0:
        orient = tuple(me.orientation)
        facing = (pos[0] + orient[0], pos[1] + orient[1]) == cell
        turn = 0 if facing else 1

    #---- chop/wash specifically: read the timer straight off `state`,
    #     which already reflects whatever this tick's action did -- a
    #     real INTERACT there just advanced it, a STAY left it flat.
    #     Keyed on the VERB, not the cell's terrain -- see _TIMED's
    #     comment for why that distinction matters (load_board vs chop
    #     on the same board). Gated on "not ready yet" because a
    #     finished chop/wash is no longer this subtask's job --
    #     rank_subtasks would already be offering collect/plate
    #     instead, a one-press verb like any other, so it falls through
    #     to the same check below rather than reading d here and
    #     skipping the press that verb still needs.
    #----------------------------------------------------------------------
    timed = _TIMED.get(verb)
    if timed is not None:
        ready_fn, cook_attr = timed
        if not getattr(baseline.mdp, ready_fn)(state, cell):
            obj = state.objects.get(cell)
            if obj is not None:
                t = obj.state[-1] if isinstance(obj.state, (tuple, list)) else obj.state
                return d + turn + max(0, getattr(baseline.mdp, cook_attr) - int(t))
            # nothing loaded yet (an empty board/sink/pot) -- falls
            # through, it is one press away exactly like any other verb

    #---- one press finishes it -- an untimed verb, an empty station
    #     waiting to be loaded, or a now-ready one waiting to be
    #     collected. Not worth asking until we're actually standing
    #     there -- rank_subtasks can only say "done" by the job's
    #     absence, never "how close", so d>0 always means "still needs
    #     the walk, plus the press".
    #----------------------------------------------------------------------
    if d > 0:
        return d + 1
    done = not any((v, c) == (verb, cell) for _, v, c in baseline.rank_subtasks(state))
    return 0 if done else turn + 1


class _BaseRobot:
    """Shared machinery. Subclasses only change how subtasks are ranked."""

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
        """[(tier, verb, cell)] best first. The hook the FoV filter reads.

        Sorted by this policy's OWN key, so it IS the preference order -- but it
        is NOT what action() does this tick, because action() draws from a
        distribution over this ranking rather than taking its head. See the
        module docstring.
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

    #---- CHANGED: remaining() is now a one-line call into the module-level
    #     _remaining() below -- see there for why this had to stop being
    #     _BaseRobot-only.
    #----------------------------------------------------------------------
    def remaining(self, state, subtask):
        """Ticks still owed to finish `subtask`=(tier, verb, cell) from
        `state`. See module-level _remaining() for the logic."""
        return _remaining(self, state, subtask)

    def _bias(self, view, state, verb, cell):
        """Within-tier ordering hook. 0 for everything unless overridden."""
        return 0

    # -- the distribution -----------------------------------------------------
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

    @staticmethod
    def top3(pi, held):
        """The three most likely sub-tasks, and which one was actually taken.

        `held` is the realised draw, not argmax(pi), and it is genuinely often
        not the top row: that is the whole point of drawing, and seeing it on
        screen is the fastest way to tell a sampled policy from a greedy one.
        """
        out = []
        for s, p in sorted(pi.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
            out.append((short_subtask(s), float(p), s == held))
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
        """One env action: sample a sub-task (sticky, see module docstring),
        then take one step of the walk towards it, or INTERACT on arrival.

        There is nothing after the step. No right-of-way clause, no unstuck
        counter, no sidestep - see the module docstring for why none of it can
        fire on these layouts.
        """
        ranked = self.rank_subtasks(state)
        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = TruthView(self.mdp, state).walkable | {pos}

        if not ranked:
            self.committed = self.last_subtask = None
            self.last_pi = {}
            self.t += 1
            return Action.STAY, self._info(None, {}, None, "none")

        pen = {s: within_tier_penalty(self, state, s, walk, pos) for s in ranked}
        pi = subtask_pi(ranked, pos, orient, walk, self.beta, pen)
        self.last_pi = pi
        if not pi:
            self.committed = self.last_subtask = None
            self.t += 1
            return Action.STAY, self._info(None, {}, None, "none")

        held = self._find(ranked, self.committed)
        # Forced when the job is finished or illegal, or when a strictly more
        # urgent tier has appeared -- never sticky across tiers, because a
        # possible DELIVER has to preempt on the tick it becomes possible.
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

        self.t += 1
        self.log.append(self.last_subtask)
        return act, self._info(held, pi, ranked, why)

    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i
        self.committed = None

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
# a control is a cone, and neither has one. Both also draw their own action() the
# same sticky way as _BaseRobot, but standalone -- see their own files. Imported
# here so BASELINES stays the single registry the harnesses read.
from robot.nominal_policy.greedy import GreedyRobot                    # noqa: E402
from robot.nominal_policy.bayesian_delegation import \
    BayesianDelegationRobot                                            # noqa: E402

BASELINES = {"solo": SoloRobot, "handoff": HandoffRobot,
             "greedy": GreedyRobot, "bayes": BayesianDelegationRobot}


def true_ranking(bot, state):
    """The baseline's OWN ordering of its legal sub-tasks. Exact, not modelled.

    `rank_subtasks` already returns them sorted by the policy's own key --
    (tier, within-tier penalty, distance, cell, verb) -- so this IS the
    preference order, with no Boltzmann in between. Anything that needs to know
    what a baseline prefers should read this rather than reconstruct it: pi is a
    DISTRIBUTION over that order, and a distribution is the wrong object when
    the question is "what does it actually rank first".
    """
    return bot.rank_subtasks(state)


def true_subtask(bot, state):
    """The sub-task the baseline is ACTUALLY doing this tick. (tier_name, verb, cell).

    NOT argmax of anything and NOT rank_subtasks()[0] either -- action() draws
    from a distribution over the ranking, so the ranking's head and the realised
    choice differ whenever a re-draw doesn't land on top. Only `action()` knows,
    because only `action()` draws.

    SIDE-EFFECTING, and unavoidably so: action() advances `t`, appends to `log`,
    updates `last_subtask`, and for BayesianDelegationRobot sets `last_action`,
    which its very next update() reads back as evidence about its own sub-task.
    Call it ONCE per tick and reuse the answer -- calling it twice makes the
    policy believe it acted twice.
    """
    _, info = bot.action(state)
    return info.get("subtask")


def true_pi(bot, state, ranked=None, pos=None, orient=None, walk=None):
    """The baseline's own distribution over sub-tasks, where it HAS one.

    Every baseline keeps `last_pi`, the distribution its most recent action()
    actually drew from -- read it rather than rebuild it, which is what keeps a
    caller's picture of a baseline from drifting away from what the baseline
    does. Falls back to modelling one only for an object with no such state at
    all (never true of anything in robot/methods.py, but true of a bare
    BayesianDelegationRobot on tick 0 before its first action()).

    Returns (pi, source) so a caller can tell whether it got the real thing or a
    model of it.
    """
    if getattr(bot, "last_pi", None):
        return dict(bot.last_pi), "drawn"
    if hasattr(bot, "_marginal"):
        m = {k: v for k, v in bot._marginal(0).items() if k is not None}
        if m:
            z = sum(m.values()) or 1.0
            return {k: v / z for k, v in m.items()}, "posterior"
    if ranked is None:
        ranked = bot.rank_subtasks(state)
    return subtask_pi(ranked, pos, orient, walk), "modelled"
