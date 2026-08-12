"""QMDP over a MASKED DEVIATION SEARCH: no pairs, no cells, no budgets.

The layer this sits beside (the superseded the superseded enumerating filter) enumerates candidates as
triples `(v, g, a)` -- job, target cell, first action -- and pays for every one of
them. Two truncations follow from that and both were doing damage:

    cell_k = 6    cells offered per job, taken in the baseline's own order
    pair_k = 6    (v, g) pairs actually rolled out

For a STASH the baseline's own order is DISTANCE FROM THE ROBOT, so `cell_k` kept
the six counters nearest the robot and dropped the rest. Measured at fov 90 over
6 layouts x 3 seeds, on ticks where the committed job was a stash:

    layout        counters offered   scored   visible to human: offered / scored
    back_bar              22.8          6.0            2.20 / 0.18
    banquet_pass          33.3          6.0            2.00 / 0.00
    butchery              26.1          6.0            2.40 / 0.47
    chefs_table           30.1          6.0            4.78 / 1.11
    divide                51.2          6.0           12.13 / 1.23
    pantry                33.4          6.0            3.65 / 1.10

On back_bar the shortlist held NO counter the human could see on 55% of stash
ticks, on banquet_pass 39%. For a layer whose entire purpose is choosing a counter
the human can see, the shortlist was excluding the answer -- and `_jobs`' own
docstring says aggregating by job "hands each job its whole G(v)", which `cell_k`
then took back.

THE FIX IS TO STOP ENUMERATING CELLS AT ALL. Legality of INTERACT is a PREDICATE
on (position, orientation): "is the cell I am facing a legal target of any of the
top-m jobs?". A predicate costs the same whether the job has three legal cells or
fifty-one, so every stash counter is in scope at zero candidate cost. What is
searched instead is ACTIONS:

    every action sequence the mask allows, until the FIRST INTERACT,
    then the tail estimates everything after it

`m` is the only bound left. `cell_k`, `pair_k` and `eps` are gone -- `eps` guarded
the `(v, g)` switch and there is no `(v, g)` any more.

WHY THIS IS NOT 5^25. Searching action SEQUENCES to the head cap would be
6^25 ~ 10^19, and no latency budget makes that finite. It is not necessary. The
rollout is deterministic in every part -- the env transition given the joint
action, the ladder given theta, A* tie-broken by cell coordinate (DESIGN.md 3.4) --
so two action sequences that arrive at an IDENTICAL simulation state have
identical futures and one of them can be dropped. That makes this a search over
reachable STATES, not sequences. The dedupe is on (position, orientation,
tick), which is exact only because the human is forecast once and shared -- see
_forecast, and section 3.2c of DESIGN.md for the 0.8% that costs.

The key has to be the whole state, not `(pos, orient, t)`. Two robot paths meeting
at the same tile at the same tick have swept different cells on the way, and which
cells the robot's body occupied gates the human's `view.robot` refresh and through
it the contention and redundancy demotions (this file's own history). So the
human can be in a different internal state at the same tile, and collapsing on
position alone would silently merge two different futures. `_key` therefore
digests the env state AND the shadow's own mutable state -- the same fields
`clone_human` copies, which is the list of everything that makes one shadow differ
from another.

ONE SHADOW, NOT A MIXTURE. `Q = C_MAP`, the MAP cone only, and no search at all
when `p(MAP) < certainty`. That is not an approximation of the mixture, it is what
the mixture already was: measured over 18,000 ticks (6 layouts x 5 true cones x 3
seeds, posterior driven by handoff), mean `p(MAP) = 0.99` and `p(MAP) >= 0.9` on
98.4% of ticks, so `min_p = 0.05` was pruning the live set to a singleton anyway.
The gate is what the sharpness buys: the MAP cone is right on 98.9% of ticks
overall and on 100.0% of the ticks where `p(MAP) >= 0.9`, so every cone error the
posterior makes sits inside the 1.6% the gate refuses to act on.

    Caveat, and it is not small: the shadows run the SAME ladder as the human, so
    that is inference against a perfectly specified model and 98.9% is a ceiling
    rather than a performance estimate. `alpha = 0.9` is the only slack absorbing
    model error, and against a human whose ladder differs this number falls.

WHAT IS KNOWINGLY LEFT OUT. There is no commitment and therefore no hysteresis.
the superseded enumerating filter holds a committed `(v, g)` and forces it back into the candidate
list whenever it drops out of the top-m, and its own comment records what happened
without that: a third of all approaches abandoned mid-walk, chefs_table
completions 47 -> 38, arrivals halved. The mask here is rebuilt every tick from a
fresh top-m, so a cell being walked towards can leave the mask and the approach
dies with it. `hold_target` re-admits the previous winning cell while it is still
legal ANYWHERE in the full ranking; it defaults to False because it is not part of
the design being tested, and it is the first thing to turn on if this thrashes.
"""
import collections
import copy
import math
import os
import sys

_NL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(_NL)))
sys.path.insert(0, _NL)

import random                                                      # noqa: E402

from overcooked_ai_py.mdp.actions import Action                    # noqa: E402
from common import geometry as geo                                 # noqa: E402
from common.tasks import TIER_NAME                                 # noqa: E402
from common.views import BeliefView, TruthView                     # noqa: E402
from human.limited_vision_human import LimitedVisionHuman          # noqa: E402
from robot.filter.core.value_tail import tail_ticks                # noqa: E402
from robot.nominal_policy.subtask_dist import true_pi              # noqa: E402

MOVES = [a for a in Action.MOTION_ACTIONS if a != Action.STAY]


def clone_human(h):
    """A faithful copy of a shadow that SHARES the mdp.

    copy.deepcopy(shadow) would deep-copy the gridworld with it, which is both
    enormous and pointless. Everything that makes one shadow differ from another
    is copied here and the mdp is aliased.

    Copying only `view`, as the superseded filter did, is NOT faithful. It holds
    a subtask across ticks (decide() is sticky within a tier) and a fresh human
    has last_subtask=None, so its first move can differ from what that shadow
    would really do next -- a systematic first-step error in every rollout. `t`
    matters just as much: BeliefView._fresh tests `self.t - rec[2]`, so a human
    restarted at t=0 against a view holding timestamps from tick T computes a
    NEGATIVE age and nothing it inherited can ever decay.
    """
    c = LimitedVisionHuman.__new__(LimitedVisionHuman)
    c.seed = h.seed
    c.mdp = h.mdp                                   # shared on purpose
    c.fov = h.fov
    c.agent_index = h.agent_index
    c.other_index = h.other_index
    c.patience = h.patience
    c.react_to_robot = h.react_to_robot
    c.forget_horizon = h.forget_horizon
    c._rng = copy.deepcopy(h._rng)
    c.view = copy.deepcopy(h.view)
    c.t = h.t
    c.explore_ticks = h.explore_ticks
    c.abandoned = set(h.abandoned)
    c.last_subtask = h.last_subtask
    c._recent = list(h._recent)
    c._goal = h._goal
    c.log = []
    return c


def norm_entropy(p):
    """H(p)/log|cones| in [0, 1]. Logged, never scored against."""
    import math
    vals = [q for q in p.values() if q > 0]
    if len(vals) < 2:
        return 0.0
    h = -sum(q * math.log(q) for q in vals)
    return h / math.log(len(p))


def fast_clone_human(h):
    """clone_human without copy.deepcopy. Same result, ~7x cheaper.

    `clone_human` is correct and stays the reference; it is also 68% of
    this search's runtime, because `copy.deepcopy(h.view)` walks every entry of
    every store one object at a time -- 6.4 million deepcopy calls for 2,774
    clones, measured. Where the enumerating filter cloned once per candidate this
    clones once per search NODE, so the constant matters here in a way it never
    did there.

    Nothing needs deep copying, and the reason is a property of the stores rather
    than a hope: every value in `known_terrain`, `contents`, `seen_count` is a
    str, an int or a tuple, all immutable, so a shallow dict copy cannot be
    aliased into. `_stations` maps a char to a mutable SET and is the one store
    that needs its values rebuilt. `robot` is a tuple or None.

    The rng is rebuilt from `getstate()` rather than deep-copied: the state is an
    immutable tuple of 625 ints, and deep-copying the Random object was most of
    the `_deepcopy_tuple` traffic in the profile.

    Equivalence against clone_human is asserted in
    robot/filter/tests/test_qmdp.py over full rollouts, not argued here.
    """
    c = LimitedVisionHuman.__new__(LimitedVisionHuman)
    c.seed = h.seed
    c.mdp = h.mdp                                   # shared on purpose
    c.fov = h.fov
    c.agent_index = h.agent_index
    c.other_index = h.other_index
    c.patience = h.patience
    c.react_to_robot = h.react_to_robot
    c.forget_horizon = h.forget_horizon
    r = random.Random()
    r.setstate(h._rng.getstate())
    c._rng = r

    v = h.view
    w = BeliefView.__new__(BeliefView)
    w.forget_horizon = v.forget_horizon
    w.known_terrain = dict(v.known_terrain)
    w._stations = {k: set(s) for k, s in v._stations.items()}
    w.contents = dict(v.contents)
    w.seen_count = dict(v.seen_count)
    w.optimistic = v.optimistic
    w.robot = v.robot
    w.t = v.t
    c.view = w

    c.t = h.t
    c.explore_ticks = h.explore_ticks
    c.abandoned = set(h.abandoned)
    c.last_subtask = h.last_subtask
    c._recent = list(h._recent)
    c._goal = h._goal
    c.log = []
    return c


class QMDPFilter:
    """Search every masked action sequence to its first INTERACT; act on the best.

    C(seq) = t_end + tail(state at t_end), ticks, lower better -- the same value
    function as the superseded enumerating filter, evaluated at the terminals of an action search
    instead of at the head of an enumerated (v, g, a) triple.

    Q(a) = min over every terminal whose FIRST action was `a`. Min for the same
    reason the pair collapse was a min: after taking `a` the robot is free to
    continue however is best from where it lands, so its own continuation is a
    decision and not epistemic uncertainty.
    """

    def __init__(self, mdp, baseline, posterior, agent_index=0,
                 m=3, depth=30, waits=2, eps=2.0, eps_a=2.0, blind=8.0,
                 certainty=0.9, no_handoff=200.0, frozen_fov=None,
                 allow_stay=True,
                 hold_target=False, only_base=False, max_states=200000,
                 temperature=4.0, debug=False):
        self.mdp, self.baseline, self.post = mdp, baseline, posterior
        self.agent_index, self.other_index = agent_index, 1 - agent_index
        # `depth` replaces the superseded enumerating filter's `T`, and it is a HARDER bound than `T`
        # ever was. There the head was one beeline per candidate, so 25 ticks cost
        # nothing; here the frontier branches and the exact state dedupe barely
        # merges, because two robot paths put the robot in different cells and the
        # human's `view.robot` and `seen_count` diverge with them. Measured state
        # counts on divide: 55 at depth 4, 355 at 6, 2281 at 8 -- about 2.5x per
        # level, so this is exponential in practice and no latency budget changes
        # that. 6 is where the cost sits at a few hundred ms a tick.
        #
        # DO NOT RAISE THIS TO REACH THE FAR COUNTERS. It is the obvious thing to
        # want -- the farthest legal target measured over 655 ticks is 28 steps
        # away on banquet_pass, so a `depth` that could certify it would have to be
        # about 30 -- and it is arithmetically impossible here, which is the single
        # most important fact about this file. Extrapolating the measured growth:
        #
        #     depth 29  ~  2281 * 2.5**21  ~  1.6e12 states
        #
        # each needing its own human step and clone. A BRANCHING search cannot
        # reach a far cell at all. Enumerating (cell, arrival) instead reaches the
        # same cell in 30 ticks of simulation for that ONE candidate, because it
        # walks one path per cell rather than every path -- and
        # dev/route_channel.py measures what the extra paths were worth: the
        # human's behaviour changes with the robot's route on 0.8% of ticks, and
        # 0.0% on divide and pantry.
        #
        # So the trade is: branching buys route variation worth 0.8% and gives up
        # far cells entirely; enumeration gives up the 0.8% and reaches everything.
        # For THIS suite, where every dish crosses a counter chosen for visibility,
        # enumeration is simply the better object and this file is the control that
        # shows what the exhaustive version would have added.
        #
        # Reaching no INTERACT within `depth` is therefore expected, not an error:
        # roughly a third of ticks at depth 6. `no_reach` counts it so the bound is
        # never silent.
        self.m, self.depth = m, depth
        # Arrival times scored per cell: `waits + 1` earliest. This is what
        # makes "same counter, two ticks later" a distinct plan instead of a
        # re-plan artifact. Bounded because a cell reached much later is
        # dominated in t_end and the tail cannot overturn it.
        self.waits = waits
        # GUARD (ii), the commitment margin. Without it the layer re-decides from
        # scratch every tick and, when its own plan fails the eps_a test by a hair,
        # emits the BASELINE's action instead -- which walks toward a different
        # counter. Traced on back_bar/qmdp-greedy: the robot alternated (13,3) and
        # (13,2) forever, deferring to the baseline at gain +2.0 and overriding it
        # at +17.0 on alternate ticks, never pressing anything. A committed target
        # is held until something beats it by eps TICKS.
        self.eps = eps
        # Only the ACTION margin survives. There is no (v, g) to switch between,
        # so `eps` has nothing to guard; 2.0 is kept because one wasted step
        # costs ~2 ticks, so an alternative still has to beat a wasted step.
        self.eps_a = eps_a
        self.blind = blind
        self.certainty = certainty
        # WHAT A COUNTER THE PARTNER CANNOT REACH COSTS. Not infinity, on purpose:
        # on these layouts the robot often has no reachable-by-both counter at all
        # for a given item, and an infinite cost would make every candidate equal
        # and the ordering arbitrary. A large finite number keeps a failed handoff
        # strictly worse than every real one, while still ranking the failures among
        # themselves by t_end.
        # WHAT A FAILED HANDOFF COSTS. ONE number for both ways it fails -- the
        # partner never looks at the counter after the item lands there, or they
        # cannot stand beside it at all -- because within the horizon we have no
        # evidence that either is better than the other. Both mean the same thing
        # for this dish: the partner does not get this item. Ranking them against
        # each other would be ranking two guesses, so they are charged alike and
        # what separates counters INSIDE the failed class is just t_end.
        #
        # IT MUST EXCEED THE LARGEST REAL PICKUP or the layer prefers a guess to a
        # measurement -- which it did at first, when `blind=8` made "nobody ever
        # looks here" the cheapest option on the board and the robot stashed onto a
        # counter its partner had already walked past. In-horizon pickups are bounded
        # by depth plus the map diameter, comfortably under 200.
        self.no_handoff = no_handoff
        self.frozen_fov = frozen_fov
        self.allow_stay = allow_stay
        self.hold_target = hold_target
        # PARITY CONTROL. The mask collapses to the baseline's own realised
        # cell, so the only plan available is the baseline's own and the filter
        # must play identically to it. With waits=0 alongside it there is not
        # even an arrival choice left. This is the check that caught the filter
        # anchoring its fallback on argmax pi rather than the realised draw.
        self.only_base = only_base
        self.max_states = max_states
        self.temperature = temperature
        self.debug = debug
        self.last_subtask = None
        self.last_Q = {}
        self.last_target = None
        self.committed = None
        self._walk = None

    # -- plumbing -----------------------------------------------------------
    def set_agent_index(self, i):
        self.agent_index, self.other_index = i, 1 - i
        if hasattr(self.baseline, "set_agent_index"):
            self.baseline.set_agent_index(i)

    def set_mdp(self, mdp):
        self.mdp, self._walk = mdp, None

    def _joint(self, r, h):
        return (r, h) if self.agent_index == 0 else (h, r)

    def walkable(self, state):
        if self._walk is None:
            self._walk = set(TruthView(self.mdp, state).walkable)
        return self._walk

    # -- the cone -----------------------------------------------------------
    def _map_cone(self):
        """(fov, p) for the MAP cone, or the frozen one at probability 1."""
        if self.frozen_fov is not None:
            return self.frozen_fov, 1.0
        p = self.post.p
        if not p:
            return None, 0.0
        f = max(p, key=p.get)
        return f, p[f]

    # -- the mask -----------------------------------------------------------
    def _mask(self, state, ranked, pos, orient, walk):
        """(mask_cells, jobs, cell_owner). Every legal target of the top-m jobs.

        Jobs are aggregated at the (tier, verb) level and ranked by pi mass, with
        the baseline's REALISED pick forced to the front -- the same anchor
        `the superseded enumerating filter._jobs` uses, and for the same reason: pi is a distribution
        and the pick is a draw from it, so `argmax pi` can name a job the baseline
        is not doing, and everything downstream reads job 0 as "what the nominal
        policy would have done".

        No cell is dropped. That is the entire point of the rewrite: for a stash
        this returns all 23-51 reachable free counters rather than the six nearest
        the robot.
        """
        pi, _src = true_pi(self.baseline, state, ranked, pos, orient, walk)
        mass, cells, order = {}, {}, []
        for tier, verb, cell in ranked:
            v = (tier, verb)
            if v not in cells:
                cells[v], mass[v] = [], 0.0
                order.append(v)
            cells[v].append(cell)
            mass[v] += pi.get((tier, verb, cell), 0.0)
        jobs = sorted(order, key=lambda v: (-mass[v], order.index(v)))[:self.m]

        pick = getattr(self.baseline, "last_subtask", None)
        if pick is not None:
            hit = next((s for s in ranked
                        if (TIER_NAME[s[0]],) + tuple(s[1:]) == tuple(pick)), None)
            if hit is not None:
                pv = (hit[0], hit[1])
                jobs = [pv] + [v for v in jobs if v != pv]
                jobs = jobs[:max(1, self.m)]

        owner, mask = {}, set()
        if self.only_base and pick is not None:
            hit = next((s for s in ranked
                        if (TIER_NAME[s[0]],) + tuple(s[1:]) == tuple(pick)), None)
            if hit is not None:
                return {hit[2]}, [(hit[0], hit[1])], {hit[2]: (hit[0], hit[1])}
            return set(), [], {}
        for v in jobs:
            for c in cells[v]:
                mask.add(c)
                owner.setdefault(c, v)       # first job in rank order owns it
        # Optional anti-abandonment re-admission -- off by default, see module doc.
        if self.hold_target and self.last_target is not None:
            legal = {c for _t, _v, c in ranked}
            if self.last_target in legal and self.last_target not in mask:
                mask.add(self.last_target)
                owner.setdefault(self.last_target,
                                 next((( t, vb) for t, vb, c in ranked
                                       if c == self.last_target), jobs[0]))
        return mask, jobs, owner

    def _step(self, pos, orient, a):
        """(pos, orient) after action `a`, using the mdp's OWN movement rule.

        Delegates to `_move_if_direction` rather than reimplementing it. Rewriting
        it here would be a second copy of the wall/orientation rule -- a move into
        a wall turns the robot without moving it, which is a real and easily
        mis-copied detail, and orientation is what the INTERACT mask reads.
        """
        return self.mdp._move_if_direction(pos, orient, a)

    def _faces(self, pos, orient, mask):
        """The mask cell this (pos, orient) is pressing, or None.

        INTERACT is admitted exactly when the cell directly in front is a legal
        target. That is the whole action gate, and it is O(1) in the number of
        candidate cells -- which is what lets every stash counter be in scope.
        """
        c = (pos[0] + orient[0], pos[1] + orient[1])
        return c if c in mask else None

    # -- the search ---------------------------------------------------------
    def _hkey(self, human):
        """Digest of everything that makes one shadow behave differently.

        Exactly the fields `clone_human` copies, because that list IS the
        definition of shadow state. Computed ONCE PER NODE rather than once per
        successor: all six successors of a node share one post-decision shadow, so
        the human half of their keys is identical by construction and recomputing
        it six times was pure waste.
        """
        v = human.view
        return (human.t, human.explore_ticks, human.last_subtask, human._goal,
                tuple(human._recent), tuple(sorted(human.abandoned)),
                v.robot, v.t,
                tuple(sorted(v.contents.items())),
                tuple(sorted(v.seen_count.items())),
                tuple(sorted(v.known_terrain.items())),
                tuple(sorted((k, tuple(sorted(vs))) for k, vs in v._stations.items())),
                human._rng.getstate())

    def _key(self, sim, human):
        """Exact dedupe key: the env state plus the shadow's mutable state.

        `hash(sim)` is value-based -- OvercookedState defines both __eq__ and
        __hash__ -- and orientation-sensitive, so two paths ending on the same tile
        facing different ways stay distinct, which they must: NORTH-then-SOUTH and
        SOUTH-then-NORTH both return to the start square facing opposite ways and
        are genuinely different plans.
        """
        return (hash(sim), self._hkey(human))

    def _forecast(self, state, fov, horizon, walk, ref):
        """The MAP shadow's action sequence for the next `horizon` ticks, once.

        THIS IS THE ONE APPROXIMATION IN THIS FILE AND ITS SIZE IS MEASURED. The
        human's actions depend on the world, which depends on where the robot
        walked, so strictly every branch of the search deserves its own human.
        Doing that is what made the exact version exponential -- the state key had
        to include the shadow, nothing merged, and 2.5x per level capped the search
        at depth 6, which cannot even reach a 28-step counter.

        dev/route_channel.py measures what that buys: over 720 ticks on all six
        layouts, changing the robot's first action changes the human's next six
        actions on 0.8% of ticks, and 0.0% on divide and pantry. The human sees the
        robot 70.7% of the time, but seeing it almost never changes what it does --
        the only path from robot position to human behaviour is the contention
        demotion, and that fires only at stations inside the dividing wall.

        So the human is forecast ONCE against the baseline's own trajectory and
        that forecast is shared by every branch. The robot's search then becomes a
        shortest-path problem over (position, orientation, tick), which is
        ~200 x 4 x 30 = 24,000 states and runs at depth 30 -- far enough to reach
        every stash counter on every layout in the suite, which is the requirement
        the exact version could not meet at any price.
        """
        sim = state.deepcopy()
        human = fast_clone_human(self.post.shadows[fov])
        out = []
        for _ in range(horizon):
            if self.mdp.is_terminal(sim):
                break
            ah, _ = human.action(sim)
            # THE REFERENCE ROBOT TRAJECTORY IS A PURE BEELINE, NOT baseline.action().
            # Calling the baseline here would be a correctness bug, not a slow path:
            # StochasticSubtask.action mutates `committed`, `last_subtask`,
            # `inner.t`, `inner.log`, draws from its own rng, and sets
            # `inner.last_action`, which bayes reads back as inverse-planning
            # evidence on its very next update. Thirty forecast steps per real tick
            # would advance the baseline's clock thirtyfold and overwrite the
            # realised pick that `_mask` anchors on -- `true_subtask`'s docstring
            # says call it ONCE per tick and means it.
            #
            # geo.step_towards is pure and reproduces what the baseline does
            # exactly: every baseline's action() is step_towards to its realised
            # cell, INTERACT on arrival. So this is the baseline's own trajectory
            # for as long as it holds that sub-task, which is the same assumption
            # the forecast already makes.
            if ref is not None:
                mv, arrived = geo.step_towards(walk, tuple(sim.players[self.agent_index].position),
                                               tuple(sim.players[self.agent_index].orientation), ref)
                ar = Action.INTERACT if arrived else (mv or Action.STAY)
            else:
                ar = Action.STAY
            hp = sim.players[self.other_index]
            out.append((ah, fast_clone_human(human),
                        tuple(hp.position), tuple(hp.orientation)))
            sim, _, _, _ = self.mdp.get_state_transition(sim, self._joint(ar, ah))
        return out

    def _sight(self, fc, mask, fov):
        """{cell: [every forecast tick the human can SEE it]}.

        EVERY tick, not just the first, because a glance only tells them
        something if the item is ALREADY THERE. A counter they swept at t+1 and
        never looked at again is useless for a stash landing at t+8 -- the belief
        that matters is the one formed by the first look AT OR AFTER the press.
        `_collect` does that lookup.

        THIS IS WHAT REPLACES THE FLAT `blind` PENALTY, and it is the whole reason
        the layer simulates the human at all. `blind = 8.0` charges the same eight
        ticks whether the partner looks at that counter three ticks from now or
        never looks at it at all -- so two counters, one about to be swept by their
        cone and one behind them, priced identically. The forecast already knows:
        it carries the human's position and orientation for every tick of the
        horizon, so the tick their cone first covers a cell is a lookup, not a
        guess.

        No cone is hardcoded anywhere here. `fov` is the MAP hypothesis from the
        posterior, and geo.visible_cells is the same cone-plus-line-of-sight test
        the human itself perceives through, so a cell counts as seen exactly when
        the human would have seen it.
        """
        out = {c: [] for c in mask}
        terrain = self.mdp.terrain_mtx
        for k, (_ah, _h, hpos, horient) in enumerate(fc):
            vis = geo.visible_cells(terrain, hpos, horient, fov)
            for c in mask:
                if c in vis:
                    out[c].append(k)
        return out

    def _fields(self, state):
        """Cache of {partner position: distance field}, built lazily.

        The partner stands in at most `depth` distinct places over a forecast, and
        each field is one BFS over the walkable floor, so this is cheap and is
        rebuilt per tick because the floor's occupancy changes.
        """
        if getattr(self, "_fcache_t", None) is not self.post:
            self._fcache = {}
        return self._fcache

    def _collect(self, state, fc, sight, cell, t_end):
        """(see tick, in-hand tick) for an item placed at `cell` at `t_end`.

        TWO EVENTS, IN ORDER, AND BOTH MATTER -- which is why this returns both and
        the cost uses the second while the readouts show the first:

          SEE   the first forecast tick AT OR AFTER `t_end` at which their cone
                covers the counter. A glance BEFORE the press is worthless: they
                looked at an empty counter and learned that it was empty. Scoring
                the earlier glance was a bug -- it rewarded stashing on a counter
                they had already walked past.
          HOLD  that tick plus their walk from wherever the forecast puts them then
                to a standing cell beside the counter, plus one for the press.

        Returns (None, None) when they never look at it again inside the horizon,
        and (see, None) when they would see it but cannot reach it -- the stranded
        case, which is a different failure and gets a different penalty.
        """
        ks = sight.get(cell) or []
        k = next((x for x in ks if x >= t_end), None)
        if k is None:
            return None, None
        walk = self.walkable(state)
        hp = fc[k][2] if k < len(fc) else fc[-1][2]
        fields = self._fields(state)
        if hp not in fields:
            fields[hp] = geo.dist_field(walk | {hp}, hp)
        d = geo.path_len_in(fields[hp], walk | {hp}, cell)
        return k, (None if d is None else k + d + 1)

    def _search(self, state, fov, mask, ref, owner=None):
        """{action_index: (best C, target cell, t_end)} over masked sequences.

        A shortest-path search over (position, orientation, tick) rather than over
        action sequences. Every masked action sequence is still explored -- the
        dedupe is what keeps "every sequence" finite, and with the human forecast
        fixed two sequences reaching the same (pos, orient, t) are interchangeable
        by construction.

        The mask is a UNION over the top-m jobs and is live at EVERY step, not only
        on arrival at one committed cell. That is the difference from
        the superseded enumerating filter, which offers INTERACT only when `arrived(g)` for the g of
        the candidate being rolled: it can never press a counter it passes on the
        way to another. Here it can, which is the whole point.

        Terminals are (cell, arrival tick). Per cell only the `waits + 1` earliest
        arrivals are scored, because a cell reached later than that is dominated in
        the head and the tail would have to overturn a t_end it cannot see. That is
        a real bound and `n_tails` reports it.
        """
        walk = self.walkable(state) | {tuple(state.players[self.agent_index].position)}
        fc = self._forecast(state, fov, self.depth + 1, walk, ref)
        if not fc:
            return {}, 0, 0, False, []
        sight = self._sight(fc, mask, fov)
        self._fcache = {}

        me = state.players[self.agent_index]
        root = (tuple(me.position), tuple(me.orientation))
        # (pos, orient, t) -> first action index that reached it soonest
        # DEDUPE WITHIN A FIRST-ACTION BRANCH, NOT ACROSS BRANCHES. Keying `seen` on
        # (pos, orient, t) alone credits each state to whichever action reached it
        # first, so any action that is merely SLOWER to everything gets no terminal
        # and no Q at all -- including, routinely, the baseline's own action, which
        # then made `base_c` fall back to the global min and the commitment get
        # dropped. Traced on back_bar: the robot walked back west for three ticks
        # with no plan at all. `first` in the key costs 5x the states, and states
        # are cheap here because the human is forecast once rather than per node.
        seen = {(None, root[0], root[1], 0)}
        # key -> (parent key, action taken). The searched path is REPLAYED EXACTLY
        # rather than re-derived by beelining: a beeline is a different trajectory
        # from the one the search scored, it can fail to be adjacent-and-facing at
        # t_end, and INTERACT then fires off-target and certifies nothing. That is
        # what left the robot frozen on STAY with a flat C.
        via = {(None, root[0], root[1], 0): None}
        frontier = collections.deque([(root[0], root[1], 0, None)])
        # (first action, cell, press tick) -> the plan. The FIRST ACTION IS PART OF
        # THE KEY, not a set hanging off it. Keying only on (cell, tick) and
        # replaying once was a real bug: one C got assigned to every action that
        # could reach that terminal, so Q(a) did not depend on a and the emitted
        # action fell out of the tie-break instead of out of the value. On back_bar
        # it made the robot alternate between two adjacent counters forever, Q
        # swinging 86 <-> 121 for one sideways step.
        presses = {}
        n_states = 0

        acts = list(MOVES) + ([Action.STAY] if self.allow_stay else [])
        while frontier:
            pos, orient, t, first = frontier.popleft()
            n_states += 1
            c = self._faces(pos, orient, mask)
            if c is not None:
                ia = (Action.ACTION_TO_INDEX[Action.INTERACT] if first is None
                      else Action.ACTION_TO_INDEX[first])
                presses[(ia, c, t + 1)] = (first, pos, orient, t)
            if t + 1 > self.depth or t >= len(fc):
                continue
            for a in acts:
                np_, no_ = self._step(pos, orient, a)
                nf = first if first is not None else a
                k = (nf, np_, no_, t + 1)
                if k in seen:
                    continue
                seen.add(k)
                via[k] = ((first, pos, orient, t), a)
                frontier.append((np_, no_, t + 1, first if first is not None else a))

        # Score the terminals. Only here does the simulator run, and only once per
        # scored terminal rather than once per node -- which is what makes depth 30
        # affordable at all.
        best = {}
        per = collections.Counter()
        n_tails = 0
        scored = []
        for (ia, cell, t_end) in sorted(presses, key=lambda x: (x[0], x[2], x[1])):
            # waits+1 earliest arrivals per (action, cell): the wait is priced, and
            # a cell reached much later is dominated in t_end anyway.
            if per[(ia, cell)] > self.waits:
                continue
            per[(ia, cell)] += 1
            seq, node = [], presses[(ia, cell, t_end)]
            while via.get(node) is not None:
                pk, a = via[node]
                seq.append(a)
                node = pk
            seq.reverse()
            sim, human = self._replay(state, fov, fc, seq)
            if sim is None:
                continue
            n_tails += 1
            # A STASH is not done when the robot lets go of it -- it is done when
            # the partner can act on it. So the head ends at whichever is later:
            # the press, or the tick their cone first covers that counter. A counter
            # the forecast never sweeps falls back to `blind`, which is now only the
            # I-have-no-idea case rather than the answer for every counter.
            avail, see_k, got_k = t_end, None, None
            if owner.get(cell, (None, None))[1] == "stash":
                # A stash is finished when the PARTNER HAS THE ITEM. Three outcomes,
                # three costs, and collapsing any two of them loses the distinction
                # the whole layer exists to make:
                #
                #   collected   they look after the press AND can walk to it ->
                #               charge the tick it is in their hand
                #   not collected  either they never look at it again inside the
                #               horizon, or no standing cell of theirs is adjacent to
                #               it. Same cost: `no_handoff`. On these layouts most
                #               counters are on the robot's side only, so this is the
                #               common case, and it is not a slow handoff -- it is
                #               not a handoff at all.
                see_k, got_k = self._collect(state, fc, sight, cell, t_end)
                if got_k is not None:
                    avail = got_k
                else:
                    # Never looked at it, or cannot reach it: same cost, because
                    # both mean the partner does not get this item and we have no
                    # basis inside the horizon for calling one better. `see_k` still
                    # says WHICH failure it was, for the readouts.
                    avail = t_end + self.no_handoff
            tl = tail_ticks(self.mdp, sim, human_view=human.view,
                            human_index=self.other_index, blind=self.blind)
            c = avail + tl
            cur = best.get(ia)
            if cur is None or c < cur[0]:
                best[ia] = (c, cell, t_end)
            # Every scored candidate, kept for the HUD and the tracer. This is the
            # table you want when the question is "why THAT counter" -- the score
            # per counter, the tick it would be pressed, and when the human's cone
            # first covers it.
            # avail and tail kept APART in the readout. C alone cannot tell you
            # whether a candidate won on being quick to reach or on leaving the
            # kitchen in a better shape, and for every verb except stash the entire
            # difference between jobs is the tail -- `avail` is just t_end there.
            scored.append((owner.get(cell, (None, "?"))[1], cell, t_end,
                           round(c, 1), see_k, got_k, round(avail, 1), round(tl, 1)))
        return best, len(seen), n_tails, False, scored

    def _replay(self, state, fov, fc, seq):
        """Execute `seq` then INTERACT, against the forecast human.

        One simulated trajectory per scored terminal, and it is the EXACT sequence
        the sweep found. The INTERACT lands adjacent-and-facing by construction,
        because being adjacent-and-facing is the condition under which the sweep
        recorded the press in the first place.
        """
        sim = state.deepcopy()
        human = None
        for k, a in enumerate(list(seq) + [Action.INTERACT]):
            if k >= len(fc):
                return None, None
            ah, human = fc[k][0], fc[k][1]
            sim, _, _, _ = self.mdp.get_state_transition(sim, self._joint(a, ah))
            if self.mdp.is_terminal(sim):
                break
        return sim, human

    def _top3(self, jobs, owner, scored, chosen_cell):
        """[(label, best C, is the one taken)] for the HUD's `robot weighs` line.

        The same shape every method in the registry emits, so that line reads the
        same way across the whole table -- but the NUMBER here is ticks-to-finish,
        lower better, where a baseline puts a probability. `top3_kind` says which.

        One entry per JOB the baseline handed over, scored by the best plan any of
        that job's cells achieved. A job with no reachable plan still gets a row, at
        inf, because "the baseline offered it and nothing could be scored for it" is
        exactly what you want to see when the robot ignores an obvious option.
        """
        best_of = {}
        for row in scored:
            cell, c = row[1], row[3]
            v = owner.get(cell)
            if v is None:
                continue
            if v not in best_of or c < best_of[v]:
                best_of[v] = c
        taken = owner.get(chosen_cell)
        return [("%s/%s" % (TIER_NAME[v[0]], v[1]),
                 best_of.get(v, float("inf")), v == taken) for v in jobs]

    def action(self, state):
        base_act, base_info = self.baseline.action(state)
        ranked = self.baseline.rank_subtasks(state)
        fov, pmap = self._map_cone()
        info = {"subtask": base_info.get("subtask"),
                "base_subtask": base_info.get("subtask"),
                "fov_post": dict(self.post.p),
                "theta_entropy": norm_entropy(self.post.p),
                "map_fov": fov, "p_map": pmap,
                "deviated": False, "gated": False,
                "base_action": Action.ACTION_TO_INDEX[base_act]}

        # THE CERTAINTY GATE. Below it the cone is a guess and every C would be a
        # claim about a human we have not identified, so the baseline acts. The
        # gate is cheap because the posterior is sharp: it refuses ~1.6% of ticks,
        # and those are where 100% of the MAP errors live.
        if fov is None or pmap < self.certainty:
            info["gated"] = True
            self.last_target = None
            return base_act, info
        if not ranked:
            return base_act, info

        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = self.walkable(state) | {pos}
        mask, jobs, owner = self._mask(state, ranked, pos, orient, walk)
        if not mask:
            return base_act, info

        best, n_states, n_nodes, capped, scored = self._search(
            state, fov, mask, base_info.get("subtask", (None, None, None))[2]
            if base_info.get("subtask") else None, owner)
        if not best:
            # No masked sequence reached an INTERACT inside `depth`. Reported, not
            # swallowed: a high no_reach rate means `depth` is too small for the
            # layout and the layer is silently degrading to its baseline.
            info.update({"no_reach": True, "n_mask": len(mask),
                         "n_states": n_states, "search_capped": capped,
                         "cands": [], "jobs": [(TIER_NAME[t], v) for t, v in jobs]})
            self.last_target = None
            return base_act, info

        qa = {ia: c for ia, (c, _cell, _te) in best.items()}
        ib = Action.ACTION_TO_INDEX[base_act]
        base_c = qa.get(ib, min(qa.values()))
        lo = min(qa.values())

        # GUARD (ii). If we are already walking to a cell and a plan for that cell
        # is still on offer, keep executing it unless another cell beats it by more
        # than eps. Executing means emitting THAT plan's first action -- not falling
        # back to base_act, which is what produced the ping-pong.
        held = None
        if self.committed is not None:
            same = [ia for ia, (_c, cell, _t) in best.items()
                    if cell == self.committed]
            if same:
                held = min(same, key=lambda ia: best[ia][0])
        if held is not None and qa[held] <= lo + self.eps:
            act = Action.INDEX_TO_ACTION[held]
            info["deviated"] = (act != base_act)
            info["held_plan"] = True
            chosen = held
            c, cell, t_end = best[chosen]
            self.committed = cell
            v = owner.get(cell)
            self.last_subtask = ((TIER_NAME[v[0]], v[1], cell) if v is not None
                                 else base_info.get("subtask"))
            self.last_Q = dict(qa)
            self.last_target = cell
            full = {i: qa.get(i, float("inf")) for i in range(len(Action.ALL_ACTIONS))}
            w = {i: math.exp(-(qa[i] - lo) / max(self.temperature, 1e-9)) for i in qa}
            z = sum(w.values()) or 1.0
            info.update({"subtask": self.last_subtask, "Q_action": full,
                         "action_dist": {i: (w.get(i, 0.0) / z)
                                         for i in range(len(Action.ALL_ACTIONS))},
                         "Q": qa[chosen], "gain": base_c - qa[chosen],
                         "n_mask": len(mask), "n_jobs": len(jobs),
                         "n_states": n_states, "n_nodes": n_nodes,
                         "search_capped": capped,
                         "plan": (v[1], cell) if v is not None else None,
                         "plan_t": t_end,
                         "jobs": [(TIER_NAME[t], vb) for t, vb in jobs],
                         "cands": sorted(scored, key=lambda x: x[3]),
                         "top3": self._top3(jobs, owner, scored, cell),
                         "top3_kind": "t"})
            return act, info
        # Tie-break on the baseline's own action, so a flat Q reproduces the
        # nominal policy rather than merely resembling it -- guard (i), which
        # survives the rewrite because base_act is still available even though
        # pair index 0 is not.
        win = min(qa, key=lambda ia: (qa[ia], ia != ib, ia))
        act = base_act
        if qa[win] < base_c - self.eps_a:
            act = Action.INDEX_TO_ACTION[win]
            info["deviated"] = True

        chosen = Action.ACTION_TO_INDEX[act]
        cell = best.get(chosen, (None, None, None))[1]
        self.last_target = cell
        # Released at the press: the first INTERACT certifies the subtask, so
        # that is exactly where the commitment stops binding (DESIGN.md 3.2).
        self.committed = None if act == Action.INTERACT else cell
        v = owner.get(cell)
        self.last_subtask = ((TIER_NAME[v[0]], v[1], cell) if v is not None
                             else base_info.get("subtask"))
        self.last_Q = dict(qa)

        full = {i: qa.get(i, float("inf")) for i in range(len(Action.ALL_ACTIONS))}
        lo = min(qa.values())
        w = {i: math.exp(-(qa[i] - lo) / max(self.temperature, 1e-9)) for i in qa}
        z = sum(w.values()) or 1.0
        info.update({"subtask": self.last_subtask, "Q_action": full,
                     "action_dist": {i: (w.get(i, 0.0) / z)
                                     for i in range(len(Action.ALL_ACTIONS))},
                     "Q": qa[win], "gain": base_c - qa[win],
                     "n_mask": len(mask), "n_jobs": len(jobs),
                     "n_states": n_states, "n_nodes": n_nodes,
                     "search_capped": capped,
                     "plan": (v[1], cell) if v is not None else None,
                     "plan_t": best.get(chosen, (None, None, None))[2],
                     "jobs": [(TIER_NAME[t], vb) for t, vb in jobs],
                     "cands": sorted(scored, key=lambda x: x[3]),
                     "top3": self._top3(jobs, owner, scored, cell),
                     "top3_kind": "t"})
        return act, info
