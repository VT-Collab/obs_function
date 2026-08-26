"""A FIELD-OF-VIEW FILTER, and nothing else, laid over the nominal policy.

qmdp.py is a second scheduler. It re-derives what the kitchen needs -- the recipe
leg, the orders left, the fetch count, a stash's handoff -- and then argues with
the baseline about it, which means every number it emits mixes "the human should
see me" with "this is the better job". You cannot read a cap off that, and you
cannot say what the cone bought, because the cone was never the only thing that
moved.

THIS FILE KEEPS EXACTLY TWO THINGS FROM qmdp.py AND DELETES EVERYTHING BETWEEN
THEM: the certainty gate on the cone posterior at one end, and the baseline's own
sub-task distribution at the other. In place of the middle there is one question,

    HOW OFTEN AND HOW SOON, OVER THE TICKS OF DOING THIS SUB-TASK, WOULD THE
    HUMAN SEE ME -- AND IS THE ANSWER WORTH WHAT IT COSTS THE BASELINE?

and one cost,

    C(plan) = R(cell) + D(action, cell) - bonus(plan)

    bonus(plan) = cap * SUM_{j=1..n} decay**(j-1) * gamma**(k_j - 1)

with R the sub-task regret in TICKS against the baseline's realised draw, D the
detour in TICKS this first action adds over the shortest route to that cell, and
k_1 < ... < k_n the ticks of the sub-task's own trajectory on which the human's
cone covers the ROBOT. The nominal policy owns task success; this layer owns only
being seen.

THE BONUS IS A NUMBER OF TICKS, NOT A FRACTION, and that is the one thing to
carry over from the previous shape of this file. There used to be an s in [0, 1]
-- discounted sightings over discounted trajectory length -- and `cap` was the
TOTAL budget that s spent a fraction of. Now:

    cap     the MOST this layer will pay, in ticks of baseline cost, to be seen
            ONCE. A plan seen once on tick 1 is worth exactly `cap`, because
            decay**0 * gamma**0 == 1. That identity is the definition of the knob
            and there is a test that pins it.
    decay   how much less each SUBSEQUENT sighting is worth than the one before,
            0 < decay < 1, asserted at construction. The j-th sighting is worth
            cap * decay**(j-1) * gamma**(k_j-1): it decays because it is LATER
            (gamma, unchanged, the open-loop forecast's own distrust of its far
            end) and because it is the j-th (decay, the human's belief being
            refreshed by seeing the robot at all, so the second look tells them
            less than the first did).
    bound   cap / (1 - decay), exposed as `max_influence` and in the info dict.
            THIS, not `cap`, is what the theorem below bounds C by, what the tier
            lock compares against a rung, and what anybody quoting "this layer can
            spend at most N ticks" has to quote. At the defaults -- cap 4,
            decay 0.5 -- it is 8 ticks.

WHY A PER-SIGHTING PRICE AND NOT A CAPPED FRACTION. Under s the tenth tick of
being seen was worth as much as the first, so "seen once" and "not seen at all"
were nearly the same number and the filter would happily spend its whole budget
turning six sightings into nine. What the cone is FOR says the opposite: the
human's belief is refreshed by seeing the robot at all. So the price is set by
what the FIRST sighting is worth and everything after it is a geometric tail.

THE BONUS IS A SUM OVER A TRAJECTORY, NOT A VERDICT ON A MOMENT, and that is the
whole reason this file exists rather than being a knob on qmdp.py. "Will this
drop be witnessed" is a yes/no about one tick and it is the HANDOFF question --
task knowledge, priced by a recipe, which this layer is not allowed to have.
"How often and how soon, over the ticks I spend doing this, am I in view" is a
statement about presence that is true whatever the item is and whatever the
counter is for. The tick the robot presses on is one ordinary member of that sum,
weighed by exactly the test every other tick gets: being seen as you put
something down is one more sighting, no more and no less. What decay changes is
only how much a REPEAT sighting is worth, and it does not care which tick of the
trajectory the repeat lands on.

WHY VISIBILITY OF THE ROBOT'S OWN POSITION, NOT OF A COUNTER. This is not an
analogy for the mechanism, it IS the mechanism. common/views.py:186-188 writes
`view.robot = (pos, orient, held, t)` if and only if the robot's cell is in the
human's `seen_cells` on that tick. So a discounted sum over the ticks the robot's
position is covered is the human's belief-refresh signal, literally, and it is
computed here with the same geo.visible_cells the human perceives through -- a
cell counts as seen exactly when the human would have seen it. qmdp's `_sight`,
which mapped COUNTER cells to the ticks a cone swept them, answers a different
question and is gone.

WHY A SUM AND NOT FIRST-SEEN LATENCY. The bonus subsumes both, and now it does so
along two axes rather than one. gamma alone traded coverage against soonness:
gamma = 1 is a flat count of the ticks I am seen on, gamma small collapses onto
"how soon am I seen". decay trades coverage against NOVELTY: decay -> 1 pays the
same for the tenth sighting as the first, decay -> 0 is all-or-nothing, "was I
seen at all". Both are knobs and neither is a second objective smuggled in.
gamma = 0.9 is not aesthetic: the forecast human is OPEN-LOOP in the robot (see
_forecast), so a cone predicted 12 ticks out is an extrapolation and later ticks
deserve less trust. decay = 0.5 is likewise a claim about the HUMAN -- a second
look refreshes a belief that is already fresh -- and it is the knob that decides
what fraction of the budget is spent getting seen once.

THE WINDOW IS THE SUB-TASK'S OWN TRAJECTORY: ticks 1..A, where A is the arrival.
The ticks after A are NOT scored, and getting that wrong was the one substantive
error this file has had. _rollout walks the robot down the distance field and then
FREEZES on the standing cell, so scoring the full depth meant a plan arriving at
tick 4 spent eight of its twelve samples parked on the very tile it presses from.
Measured before the cut: 36% of all sampled ticks were frozen ones and they
carried 22% of the entire score. That is not a rounding error -- it is the stash
spot counted eight times over, which is qmdp's "will they be looking at this
counter" reappearing as a sampling artefact in a filter built to be rid of it.

THERE IS NO DENOMINATOR ANY MORE, and that retires the awkward corner the old
shape had. Under s the divisor was the plan's own trajectory length, so a 2-tick
trajectory seen twice scored 1.0 against a 12-tick one seen eight times -- true to
"how much of my time am I in view", but it meant the same physical sighting was
worth different amounts depending on how long the job around it happened to be,
and length is a task property. A per-sighting price has no such handle: a sighting
on tick 3 is worth cap * decay**(j-1) * gamma**2 whether the plan ends on tick 4
or tick 12. A long plan can accumulate more sightings, which is the honest way for
length to matter, and the geometric decay is what stops that from being a reason
to prefer long jobs.

`depth` still bounds the window: a trajectory longer than 12 ticks is truncated
there and scored on what was actually modelled, which is the fixed-horizon promise
and the reason the forecast runs for depth + 1.

THE CANDIDATE SET IS (first action, target cell), NOT a branching action search.
qmdp's frontier grows ~2.5x per level and provably cannot reach the 28-step
counter on banquet_pass at any affordable depth (qmdp.py:262-289 has that
arithmetic). A beeline rollout per (action, cell) reaches it in a list
comprehension. The depth bound then costs nothing in REACH -- only in forecast
horizon, which is a different and much cheaper thing to be short of. The whole
search collapses to at most 6 x |mask| ~ 300 pure-geometry rollouts backed by one
multi-source BFS per candidate cell: no state.deepcopy(), no get_state_transition,
no human clone per terminal. The only simulation left in the file is the single
shared 13-step forecast.

THE COST IS WRITTEN IN TICKS, AND THAT IS THE MOST IMPORTANT CHOICE HERE.
Dividing the baseline's own log-probability ratio by

    kappa = beta * ln(1 / GAMMA) = 8 * ln(1/0.95) = 0.4103 nats per tick

converts nats into ticks-of-walking using the calibration the value function
already carries: bayesian_delegation.py sets TIER_GAIN**1 == GAMMA**-21, so one
rung of the ladder is 21.4 ticks of walking. That makes `cap` denominated in the
one unit anybody can reason about -- the number of extra ticks of baseline cost
one sighting is allowed to buy.

    kappa  = beta * ln(1/GAMMA) = 0.4103 nats per tick
    R(c)   = max(0, ln pi(ref) - ln pi(c)) / kappa      TICKS,  R(ref) = 0
             For two cells of the SAME (tier, verb) this is exactly the
             difference in steps_to_finish. Across one tier rung it is
             ln(3) / ln(1/0.95) = 21.4 ticks.
    D      = t_end(a, c) - min_a' t_end(a', c)          TICKS,  >= 0
             Measured against the BEST first action for that cell, not against
             steps_to_finish -- see the comment at the plan loop for the
             off-by-one that makes the difference load-bearing.
    bonus  TICKS, >= 0. cap * SUM_j decay**(j-1) * gamma**(k_j-1) over the
             sightings on this sub-task's own trajectory (1..A, arrival included,
             parked tail excluded). No denominator: it is a price, not a share.
    cap    TICKS. What ONE sighting on tick 1 is worth, and the most the layer
             will pay for it.
    decay  in (0, 1). What the NEXT sighting is worth relative to the last.
    max_influence = cap / (1 - decay)   TICKS. The total the layer can spend.

THEOREM (bounded influence). R + D is zero for the baseline's own action and
non-negative everywhere else, and 0 <= bonus <= cap * SUM_{j>=1} decay**(j-1) =
cap/(1-decay), because gamma**(k-1) <= 1 for every term. So C(p) <= C(b) requires

    R_p + D_p <= bonus_p - bonus_b <= cap / (1 - decay)

-- the robot NEVER emits a plan whose baseline cost exceeds cap/(1-decay) ticks
above the baseline's own, for any cone, any posterior, any layout. THAT NUMBER,
NOT `cap`, IS THE BOUND, and it is why decay = 1 is refused at construction: there
the geometric series diverges and this theorem states nothing. It is exposed as
`max_influence` and in info["max_influence"] so that no reader of a log has to
re-derive it, because re-deriving it as `cap` is off by 1/(1-decay) and looks
right. On a tick where guard (ii) holds a committed cell the bound is
max_influence + eps, because that guard admits a plan within `eps` of the best --
this is stated rather than hidden, and the tests assert both forms separately.

THE ALGEBRA IS SOUND AND R IS THE HOLE. The theorem bounds C in units of R, and
says nothing at all on a tick where R is zero for EVERY candidate: there the bonus
is compared against a field of zeros and restrains nothing. Measured, five
layouts x two seeds x 200 ticks: R collapsed to 0.0 on every candidate cell on
12.3% of scoring ticks on the bayes floor (194/1581) and 5.9% on handoff
(93/1574). Two mechanisms, both real, and only one of them is a bayes problem:

  * pi NEVER NAMES THE REALISED DRAW. `bayes` with source="posterior" fills
    last_pi from BayesianDelegationRobot._marginal(0), which reads the belief as
    it stood LAST tick, while rank_subtasks re-projects onto THIS tick's
    allocation set. The two key sets are frequently disjoint -- 144 of 1946 ticks
    -- so every candidate is floored to PI_FLOOR together, p_ref is floored with
    them, and R == 0 everywhere at once.
  * THE REALISED DRAW IS THE LEAST LIKELY CANDIDATE. R = max(0, ln pi(ref) -
    ln pi(c)) / kappa prices a cell against the baseline's OWN DRAW, not against
    the best cell, so it is 0 for every cell the baseline likes at least as much
    as the one it drew. RHO = 0.95 means the draw is often not the mode, so this
    fires on the drawn baselines too and is not a bug in anybody's posterior.

On those ticks the filter re-targeted ACROSS A TIER RUNG at 4 and at 8 ticks of
budget -- the thing the 2-8 line below promised was impossible. The bound is
therefore carried by two further mechanisms, neither of which asks pi to
discriminate:

GUARD (0), THE UNPRICED-REFERENCE REFUSAL. If pi gives the baseline's realised
cell no mass at all, this layer has no reference to price a deviation against, so
it emits base_act and clears any commitment, as the certainty gate does. That is
the conservative reading of its own charter -- the nominal policy owns task
success and this layer owns only being seen -- and it replaces a fallback that
re-anchored p_ref on the MODE and then pinned R(ref) = 0 on top, which made the
baseline's own cell free while every other cell was priced against an anchor the
baseline had not drawn. Modelling pi with subtask_pi on those ticks was the other
option and was rejected: true_pi's docstring is explicit that rebuilding what can
be read exactly is how a wrapper's picture of its baseline drifts, and a filter
that walks off on a preference its baseline never held is worse than one that
does nothing for a tick. Guards (0) and (0a) together cost 126 of 1946 ungated
ticks of authority on fov-bayes -- 144 of 1946 on the pre-fix trajectory they
replaced -- and 0 on every drawn baseline, where last_pi IS this tick's pi over
this tick's ranking and always names its own draw.

GUARD (0a), THE STALE-PICK REFUSAL. The same disposal for the same reason one
step earlier: if the sub-task the baseline says it drew is not a ROW of this
tick's ranking, there is no reference job, only a reference cell that some other
job now owns. bayes reaches that state by sampling from _marginal over last
tick's allocation set; the cell survives, its job does not, and pricing R against
the mass of whatever took the cell over is not measuring the baseline's regret at
all. It fires on 14 ticks per 1000 on the bayes floor and never on a drawn one.
Every one of the pre-fix cross-rung ticks was one of these -- a tier readout
contradicting the baseline while the robot stood on the baseline's own cell.

THE ABSOLUTE TIER GUARD. At max_influence <= one rung the candidate set is
filtered, BEFORE anything is scored, to the cells whose owner shares the TIER of
the baseline's realised job. It is max_influence and not `cap` that is compared,
because what the layer can actually spend on a plan is cap/(1-decay) and arming on
`cap` would leave a filter with twice a rung of budget believing itself inside
one. This is not a second cost term and nothing can be traded against it: below a
rung of budget a cross-rung cell is not a plan, whatever R says about it. That is
the only form of the safety property worth stating, because the R-derived form
held on the handoff floor and failed on the bayes floor on the same code at the
same budget -- a guarantee that depends on which baseline you wrapped is not a
guarantee. The guard lifts strictly ABOVE one rung, so fov-free still crosses
rungs and the test proving the safety check is capable of failing still fails it.

THE INVARIANT, and it is now structural rather than measured: for max_influence <=
one rung the TIER of the sub-task this filter emits is the tier of the baseline's realised
draw, on every baseline, every layout and every tick. Either the tick returns
early -- gated, no_reach, guard (0), guard (0a) -- and info["subtask"] is the
baseline's own, or ref_job exists, every scored cell is owned by a job on
ref_job's tier, and the emitted plan's cell is one of those. There is no third
path through action(). None of the three is redundant, and that is ablated rather
than asserted -- five layouts x 200 ticks on the bayes floor, cross-rung ticks:

    pre-fix, none of the three          11/919 at cap 4,  10/957 at cap 8
    tier guard alone                    11/919 at cap 4,  10/957 at cap 8
    refusals alone                       0/845 at cap 4,   1/845 at cap 8
    all three                            0/845 at cap 4,   0/845 at cap 8

The tier guard alone changes NOTHING, which is the whole argument for guards (0)
and (0a) in one line: every pre-fix crossing was anchored on a reference the
baseline could not price, so the guard was reading its reference rung off the
wrong job and locking the candidate set onto it. And the refusals alone leave one
crossing at cap = 8, which is the argument for the tier guard: R had collapsed on
a tick where the reference was perfectly well priced.

THE LADDER IS IN max_influence = cap/(1-decay), NOT IN cap, and that is the whole
of what the per-sighting price changed about this table. Every row below names the
BUDGET first and the cap that produces it at the default decay = 0.5 second, in
brackets. Reading a row as a cap is a factor of 1/(1-decay) -- at the default, a
factor of two -- and it is the error the `max_influence` property exists to make
impossible to write down by accident.

    budget 0       [cap 0]     reproduces the baseline EXACTLY, tick for tick,
                 and does so structurally rather than by luck. cap = 0 zeroes the
                 whole weight table, so bonus == 0 for every plan whatever decay
                 is; R + D >= 0 everywhere and == 0 for (base_act, ref_cell), so
                 C == R + D and its global minimum is 0; guard (i) tie-breaks to
                 the baseline's index; the acceptance test `C(win) < base_c -
                 eps_a` reads `0 < -eps_a` and is false; guard (ii) is disarmed
                 because commitment now requires a prior deviation; guard (iii)
                 never fires.
    budget 0.41    [cap 0.21]  under one tick of detour, so the FoV term may
                 break exact ties and nothing else.
    budget 2-8     [cap 1-4]   the intended band, and the default sits at the top
                 of it: reorders cells and near-ties WITHIN a tier. Cannot cross a
                 rung, by the absolute guard, on any baseline and on any tick --
                 NOT merely on the ticks where pi discriminates, which is all the
                 R algebra ever bought and is what the bayes floor falsified.
                 What decay buys inside this band: at 8 ticks of budget the FIRST
                 sighting is worth 4 of them, so a plan that is seen at all beats
                 a plan that is not by more than any two sightings can separate
                 two plans that are both seen.
    budget 21.4    [cap 10.7]  one full tier rung: the point at which R ALONE
                 stops forbidding a crossing. R across a rung is exactly 21.4
                 ticks only for two cells the SAME distance away; real cross-tier
                 cells also differ in distance, and the winning plan additionally
                 has to win on the bonus by nearly the whole budget -- so even
                 without the guard, crossing was possible here and observed only
                 far above. The absolute guard is inclusive at exactly one rung
                 (it arms on max_influence <= rung), so this is the LAST budget
                 that cannot cross rather than the first that can, and 21.4 <
                 rung = ln(3)/ln(1/0.95) = 21.4182 either way. (That constant
                 read 21.4017 here for a long time and was simply wrong in the
                 last digits; it is printed by self.rung and can be checked.)
    budget inf     [cap 1e6]   unbounded: the guard is off, and the filter
                 maximises the bonus and ignores pi. Over butchery / chefs_table /
                 back_bar, 200 ticks each, the tier of the emitted sub-task
                 differs from the baseline's on 37 of 537 ticks here and on 0
                 anywhere below a rung. Both halves are asserted: a safety check
                 that cannot fail is indistinguishable from a term that does
                 nothing.

A BUDGET ALONE DOES NOT STOP THE ROBOT LOITERING IN VIEW. A receding-horizon
controller forgets its own accumulated delay every tick, so "one more tick of
detour, cheap" is available forever -- and note that the decay does NOT fix this
either, because it decays sightings WITHIN one plan and every tick starts a fresh
plan whose first sighting is worth the full cap again. Guard (iii) is the answer:
a leaky-bucket budget of `stall_max` CONSECUTIVE deviating ticks, after which
base_act is forced for one tick. That gives three bounds in three different units
-- cap/(1-decay) in ticks of baseline cost, eps_a in ticks of required margin,
stall_max in ticks of wall-clock authority -- and none of them substitutes for the
others.

TWO HONEST LIMITATIONS, stated here rather than discovered later:

  * The forecast human does NOT react to where this filter actually goes. It is
    forecast once against a beeline to the baseline's own cell and shared by every
    plan. dev/route_channel.py measures the channel at 0.8% of ticks overall and
    0.0% on divide and pantry -- but that number was measured for qmdp, where the
    robot's route was incidental. Here the entire score is about being seen, so
    this approximation is more load-bearing than it was there.
  * The beeline tail of each rollout descends the distance field with a fixed
    coordinate tie-break, so among two equal-length routes the filter scores one
    of them and may score the visible route under the invisible route's
    continuation. It discovers the visible one by re-planning greedily every tick
    (only the FIRST action is ever committed) rather than certifying it up front.
    The gamma discount front-loads the score onto the near ticks, which is what
    makes that acceptable -- it is not an optimality claim and must not be read
    as one.

WHAT IS COPIED, MODIFIED AND DROPPED
------------------------------------
COPIED UNCHANGED from qmdp.py: the sys.path bootstrap; MOVES; fast_clone_human;
norm_entropy; set_agent_index; set_mdp; _joint; walkable; _map_cone; _step;
_faces; _forecast including its beeline-reference argument; _top3; the certainty
gate; guard (i)'s `min(qa, key=lambda ia: (qa[ia], ia != ib, ia))`; and the
Q_action / action_dist Boltzmann emission.

MODIFIED:
  _mask            also returns pi_cell and the pi source, and OWNERSHIP OF A
                   CELL IS BY ARGMAX pi rather than by rank order -- R is defined
                   per CELL, so a cell must be attributed to the sub-task that
                   actually carries its mass. The ONE exception is the realised
                   cell, which is forced back onto the realised job; the tier
                   guard reads its tier and the readout reports its name, and
                   both were wrong when a heavier job on the same cell renamed it.
  guard (ii)       same form, but `self.committed` is set ONLY on a deviating
                   tick. Commitment is a consequence of deviating, not a
                   permanent state. In qmdp it could emit a stale cell's first
                   action on a tick where the filter was otherwise inert, which
                   is exactly the leak that would break the cap = 0 claim.
  best[ia]         tie-break is (C, cell != ref_cell, t_end, cell). At cap = 0
                   every plan for the realised cell ties at C = 0 with plans for
                   equally likely cells, and without the second key the committed
                   cell drifts off the baseline's own.
  eps_a            2.0 -> 0.5 ticks. qmdp's 2.0 meant "one wasted step"; here it
                   would silently null any small cap.
  cands rows       same 8 fields and the same lower-is-better slot 3, with slots
                   4/5/6/7 re-tenanted as sightings / window length / (R + D) /
                   the FoV bonus as a NEGATIVE cost contribution. Slot 7 was
                   -cap*s while the FoV term was a capped fraction and is -bonus
                   now; same sign, same meaning, same decomposition C = slot6 +
                   slot7, so watch.py never learns that s stopped existing.
  the DP           carries the SIGHTING COUNT in its state, (pos, orient, j), with
                   an exact Pareto prune over (j, accumulated bonus) and a clamp at
                   j_max. Under the old score a tick was worth gamma**k regardless
                   of history and merging on the pose alone was exact; with a
                   per-sighting price it is not, and two routes at the same pose
                   with different sighting counts have different futures. See the
                   comment at the plan loop.

DROPPED ENTIRELY, and this file must not import or reimplement any of them:
value_tail.py and its tail_ticks, progress.py, orders_remaining, the recipe-leg
A* makespan, the fetch count, the `blind` constant, `no_handoff` and the
stash-only got_k/avail branch, _sight, _collect, _fields, _replay, _hkey, _key,
the BFS frontier / seen / via / presses machinery of _search, `waits`,
`max_states` and search_capped as a live concept, clone_human, and every call to
state.deepcopy() or mdp.get_state_transition() outside _forecast.
"""
import collections
import math
import os
import sys

_NL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(_NL)))
sys.path.insert(0, _NL)

from overcooked_ai_py.mdp.actions import Action                    # noqa: E402
from common import geometry as geo                                 # noqa: E402
from common.tasks import TIER_NAME                                 # noqa: E402
from common.views import TruthView                                 # noqa: E402
from robot.nominal_policy.baselines import true_pi, BETA           # noqa: E402
from robot.nominal_policy.bayesian_delegation import (             # noqa: E402
    GAMMA, TIER_GAIN)

MOVES = [a for a in Action.MOTION_ACTIONS if a != Action.STAY]

# A cell pi never names still has to be comparable, and ln(0) is not a number.
# The floor puts such a cell at R = ln(1/1e-6) / kappa = 33.7 ticks: finite, so
# nothing crashes, and larger than every sane cap, so the FoV term cannot reach
# it. That is the right shape -- "pi says no" should be a wall, not an error.
PI_FLOOR = 1e-6
KAPPA_MIN = 1e-6
_INF = 1 << 30
# How much of the geometric tail of sightings the DP is allowed to throw away, in
# units of `cap`. See _weights: the clamp under-counts by at most J_TOL * cap
# ticks and never over-counts, so the bound in the theorem survives it exactly.
J_TOL = 1e-3


def norm_entropy(p):
    """H(p)/log|cones| in [0, 1]. Logged, never scored against."""
    vals = [q for q in p.values() if q > 0]
    if len(vals) < 2:
        return 0.0
    h = -sum(q * math.log(q) for q in vals)
    return h / math.log(len(p))


def fast_clone_human(h):
    """A faithful, cheap copy of a shadow. Delegates to the human's own
    clone() (human/limited_vision_human.py) rather than reimplementing it
    here.

    THIS USED TO BE A HAND-WRITTEN FIELD LIST, copied verbatim from qmdp.py,
    and it drifted from correct: its generic "anything not named above" sweep
    dispatched only on dict/set/list, so `_rank_memo`/`_field_memo` -- tuples
    wrapping a mutable list/dict -- fell into the bare-reference-assignment
    default and were ALIASED between clone and original. h.clone() does not
    have this gap: it names every field by hand and explicitly resets those
    two memos to None, because (its own docstring) "a shadow about to be
    stepped from a different state shares our belief version and position, so
    it would hit a key that matches and is answering a different question."
    Delegating also means a field added to the human later only has to be
    taught to clone() -- the one place already colocated with the fields --
    instead of to every hand-rolled copy of it scattered through
    robot/filter/core/.

    THE CLASS COMES FROM h, NOT FROM THIS FILE, same as before: `h.clone()`
    dispatches on whatever type h actually is, so this still cannot rebuild
    the wrong agent the way naming LimitedVisionHuman here once did.
    """
    return h.clone()


class FOVFilter:
    """Pick the first action that keeps the robot in the human's cone, on a budget.

    Q(a) = min over the plans whose FIRST action was `a`, for the same reason
    qmdp's collapse was a min: after taking `a` the robot is free to continue
    however is best from where it lands, so its own continuation is a decision and
    not epistemic uncertainty.
    """

    def __init__(self, mdp, baseline, posterior, agent_index=0,
                 # cap IS WHAT ONE SIGHTING IS WORTH, AND cap/(1 - fov_decay) IS
                 # WHAT THE WHOLE PLAN IS WORTH -- the second number is the one
                 # that has to stay under a rung (21.4), not the first. At the
                 # defaults that is 4 / 0.5 = 8 ticks. cap was briefly 28 back when
                 # it WAS the total -- seven rungs -- which is above the point where
                 # R alone stops forbidding a re-target, so the FoV term dominated
                 # the baseline and the filter walked past the counter in front of
                 # the human to reach one nine tiles away. The tier guard still
                 # held, so nothing crashed and no rung was crossed; it just quietly
                 # stopped being a filter and started being a scheduler. Read
                 # `max_influence`, never `cap`, when you want that number.
                 m=3, depth=12, gamma=0.9, cap=4.0, fov_decay=0.5,
                 count_parked=False,
                 eps=2.0, eps_a=0.5, stall_max=6,
                 certainty=0.9, frozen_fov=None, allow_stay=True,
                 only_base=False, hold_target=False,
                 temperature=4.0, debug=False):
        self.mdp, self.baseline, self.post = mdp, baseline, posterior
        self.agent_index, self.other_index = agent_index, 1 - agent_index
        self.m = m
        # `depth` IS THE FORECAST HORIZON AND NOTHING ELSE. It is not a reach
        # bound the way qmdp's was: a plan whose t_end exceeds the window is still
        # scored, it just never presses inside it, and its detour term is still
        # exact. That is the whole reason this file rolls beelines instead of
        # branching -- reach is free, so the bound can be set by what the forecast
        # is worth believing rather than by what the geometry needs. Twelve ticks
        # is where the open-loop human is still roughly right; twenty is where the
        # forecast is fiction and the score would be measuring the fiction.
        assert 1 <= depth < 20, "depth is a forecast horizon, not a reach bound"
        self.depth = depth
        self.gamma = gamma
        self.cap = cap
        # DIMINISHING RETURNS ON BEING SEEN AGAIN, and the reason the bound is
        # cap/(1 - fov_decay) rather than cap. The j-th sighting of a plan is worth
        # cap * fov_decay**(j-1) ticks before the time discount, so the first one
        # is worth the whole cap and every one after it is worth `fov_decay` of the
        # one before.
        #
        # 0 < fov_decay < 1, ASSERTED. At exactly 1 every sighting is worth `cap`
        # and sum_j 1 diverges: the bounded-influence theorem would have no bound
        # at all and the tier lock below would be reading a number that means
        # nothing. That is not a degenerate-but-harmless setting the way beta = 0
        # is -- it is the whole safety property gone -- so it is refused at
        # construction rather than clamped silently. At exactly 0 the second
        # sighting is worth nothing, which is a legitimate limit but makes
        # cap/(1-0) == cap and every j >= 1 state identical; excluded too, so that
        # `max_influence` is always the tight bound and never an accident.
        assert 0.0 < fov_decay < 1.0, (
            "fov_decay must be in (0, 1): at 1 every sighting is worth `cap` and "
            "cap/(1-decay) -- the bound the whole file rests on -- is infinite")
        self.fov_decay = fov_decay
        # The value of a sighting, TABULATED, and there is exactly one of these in
        # the file. _bonus and the DP both read it, which is the structural fix for
        # the failure recorded at the `raw` loop: the DP used to compute the score
        # inline while _seen_score used _shape, so `fov_decay` moved one of them and
        # not the other and a decay sweep printed three identical rows. Keyed on
        # (cap, decay, gamma, depth) and rebuilt when any of them moves, because dev
        # drivers do set bot.cap after construction and a stale table there is the
        # same dead knob wearing a different hat.
        self._wt = self._wt_key = None
        self.count_parked = count_parked
        self.eps = eps
        self.eps_a = eps_a
        self.stall_max = stall_max
        self.certainty = certainty
        self.frozen_fov = frozen_fov
        self.allow_stay = allow_stay
        self.only_base = only_base
        self.hold_target = hold_target
        self.temperature = temperature
        self.debug = debug

        # THE NATS -> TICKS CONVERSION, derived once. `beta` is the baseline's own
        # sharpness knob, so a run at an unusual beta keeps a cap that means the
        # same thing. Degenerate betas are real -- subtask_pi supports 0 (uniform)
        # and inf (argmax) -- and both make the conversion meaningless rather than
        # merely extreme, so they fall back to the default and the cap's tick
        # reading becomes nominal. Said out loud because a silent fallback here
        # would quietly change what every reported cap meant.
        b = getattr(baseline, "beta", BETA)
        if b is None or b == 0.0 or b == float("inf"):
            b = BETA
        self.kappa = max(b * math.log(1.0 / GAMMA), KAPPA_MIN)
        # BETA=8, GAMMA=0.95 -> 0.4103 nats/tick; one rung = 8*ln(3) = 8.789 nats
        # = 21.4 ticks, which is TIER_GAIN**1 == GAMMA**-21 restated in this unit.
        self.rung = math.log(TIER_GAIN) * b / self.kappa

        self.last_subtask = None
        self.last_Q = {}
        self.last_target = None
        # Set ONLY on a deviating tick. See guard (ii): commitment is a
        # consequence of having deviated, and making it anything else is what
        # would let a stale cell's first action escape on an otherwise inert tick.
        self.committed = None
        self.debt = 0
        self._walk = None

    # -- what a sighting is worth -------------------------------------------
    @property
    def max_influence(self):
        """THE BOUND, in ticks of baseline cost. cap / (1 - fov_decay).

        `cap` is what ONE sighting is worth and this is what ALL of them are
        worth, so this -- not `cap` -- is the number in the bounded-influence
        theorem, the number the tier lock compares against a rung, and the number
        anyone quoting "the filter can spend at most N ticks" must quote. It is a
        property and it is in the info dict because the first thing that goes
        wrong with a two-knob budget is somebody re-deriving it in a plotting
        script and getting it wrong by a factor of 1/(1-decay).

        Tight rather than conservative: a plan seen on every tick of a long
        trajectory approaches it from below, and a plan seen ONCE ON TICK 1 is
        worth exactly `cap`, which is what makes the knob's name true.
        """
        return self.cap / (1.0 - self.fov_decay)

    def _weights(self):
        """(wt, j_max). wt[j][k] = what the (j+1)-th sighting is worth at tick k+1.

            wt[j][k] = cap * fov_decay**j * gamma**k          j < j_max
                     = 0                                      j >= j_max

        THE TWO DECAYS ARE DIFFERENT THINGS AND BOTH APPLY. gamma**k is the time
        discount and is unchanged: a sighting forecast twelve ticks out is an
        extrapolation of an OPEN-LOOP human and deserves less trust than one two
        ticks out. fov_decay**j is the novelty discount: the human's belief is
        refreshed by seeing the robot AT ALL, so the second look tells them much
        less than the first did, whenever it happens.

        THE j_max CLAMP AND EXACTLY WHAT IT COSTS. The DP below carries the
        sighting count in its state, so an unbounded count would be an unbounded
        state space. Sightings past j_max are worth ZERO here rather than
        decay**(j_max-1) each, and the direction of that error is the point: it
        can only UNDER-count, so bonus <= cap/(1-decay) survives as a hard bound
        instead of being exceeded by the clamp's own tail. What it costs is a plan
        seen more than j_max times being undervalued by at most

            cap * decay**j_max / (1 - decay)   ticks,

        which is what J_TOL pins at 1e-3 caps: at decay 0.5 that is j_max = 11 and
        0.00098 * cap = 0.004 ticks at the default cap, a hundred times smaller
        than eps_a and below the 0.1-tick resolution of the readout. It is also
        capped at `depth`, and there it is not an approximation at all: the window
        is at most `depth` ticks long, so a plan cannot be seen more than `depth`
        times and the clamp is never reached. At depth 12 that branch takes over at
        any decay above ~0.51 (0.2 -> j_max 5, 0.4 -> 9, 0.5 -> 11, 0.51 and up ->
        12), so the whole high-decay end of a sweep -- the end where the tail is
        fattest and an approximation would hurt most -- is EXACT.
        """
        key = (self.cap, self.fov_decay, self.gamma, self.depth)
        if self._wt_key == key:
            return self._wt
        d = self.fov_decay
        j_max = 1
        while j_max < self.depth and (d ** j_max) / (1.0 - d) > J_TOL:
            j_max += 1
        wt = tuple(tuple(self.cap * (d ** j) * (self.gamma ** k)
                         for k in range(self.depth + 1))
                   for j in range(j_max))
        self._wt, self._wt_key = (wt, j_max), key
        return self._wt

    # -- plumbing -----------------------------------------------------------
    def set_agent_index(self, i):
        self.agent_index, self.other_index = i, 1 - i
        if hasattr(self.baseline, "set_agent_index"):
            self.baseline.set_agent_index(i)

    def set_mdp(self, mdp):
        self.mdp, self._walk = mdp, None

    def reset(self):
        self.committed = self.last_target = self.last_subtask = None
        self.debt = 0
        self.last_Q = {}
        if hasattr(self.baseline, "reset"):
            self.baseline.reset()

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
    def _mask(self, state, ranked, pos, orient, walk, pick):
        """(mask, jobs, owner, pi_cell, pi_src, ref_job). Every legal target of the
        top-m jobs.

        Jobs are aggregated at the (tier, verb) level and ranked by pi mass, with
        the baseline's REALISED pick forced to the front. That anchor is not a
        detail: pi is a distribution and the pick is a draw from it, so argmax pi
        can name a job the baseline is not doing, and everything downstream reads
        job 0 as "what the nominal policy would have done".

        `pick` IS PASSED IN rather than read off `self.baseline.last_subtask`, and
        `ref_job` -- the (tier, verb) of that pick as a row of THIS tick's ranking,
        or None if it is not one -- is handed back rather than recomputed by the
        caller. Both for the same reason: there is exactly one realised draw per
        tick and it must have exactly one reading. This method used to take it off
        the object while action() took it out of base_info, which are the same
        tuple until they are not, and `ref_job is None` -- the case where the
        baseline drew a sub-task this tick's ranking does not contain -- was
        invisible to the caller entirely.

        OWNERSHIP OF A CELL IS BY ARGMAX pi, which is the one change from qmdp's
        version. There a cell went to whichever job ranked first among those that
        could legally target it. Here R is defined PER CELL, so the cell has to be
        attributed to the sub-task that actually carries its probability mass or R
        is measured against the wrong thing. Ties keep the first in rank order,
        which is what the strict `>` below buys.

        No cell is dropped. For a stash this is all 23-51 reachable free counters,
        because the filter's whole job is choosing among them by visibility.
        """
        pi, pi_src = true_pi(self.baseline, state, ranked, pos, orient, walk)
        mass, cells, order = {}, {}, []
        for tier, verb, cell in ranked:
            v = (tier, verb)
            if v not in cells:
                cells[v], mass[v] = [], 0.0
                order.append(v)
            cells[v].append(cell)
            mass[v] += pi.get((tier, verb, cell), 0.0)
        jobs = sorted(order, key=lambda v: (-mass[v], order.index(v)))[:self.m]

        hit = None
        if pick is not None:
            hit = next((s for s in ranked
                        if (TIER_NAME[s[0]],) + tuple(s[1:]) == tuple(pick)), None)
            if hit is not None:
                pv = (hit[0], hit[1])
                jobs = [pv] + [v for v in jobs if v != pv]
                jobs = jobs[:max(1, self.m)]
        ref_job = (hit[0], hit[1]) if hit is not None else None

        # PARITY CONTROL. The mask collapses to the baseline's own realised cell,
        # so the only cell available is the baseline's own, every plan shares
        # R = 0, and only D and the bonus can differ. base_act still attains D = 0,
        # so the parity claim survives -- this is the recipe that caught the filter
        # anchoring on argmax pi rather than the realised draw. qmdp paired this
        # with waits=0; there is no arrival choice here, so there is nothing to
        # pair it with.
        if self.only_base:
            if hit is not None:
                return ({hit[2]}, [ref_job], {hit[2]: ref_job},
                        {hit[2]: pi.get(hit, 0.0)}, pi_src, ref_job)
            return set(), [], {}, {}, pi_src, None

        owner, mask, pi_cell = {}, set(), {}
        jobset = set(jobs)
        for tier, verb, cell in ranked:
            v = (tier, verb)
            if v not in jobset:
                continue
            mask.add(cell)
            p = pi.get((tier, verb, cell), 0.0)
            if cell not in pi_cell or p > pi_cell[cell]:
                pi_cell[cell] = p
                owner[cell] = v
        # Optional anti-abandonment re-admission -- off by default. It is a weaker
        # mechanism than guard (ii) and is not part of the design being tested; it
        # is the first thing to turn on if this thrashes.
        if self.hold_target and self.last_target is not None:
            legal = {c for _t, _v, c in ranked}
            if self.last_target in legal and self.last_target not in mask:
                mask.add(self.last_target)
                row = next(((t, vb) for t, vb, c in ranked
                            if c == self.last_target), None)
                if row is not None:
                    owner.setdefault(self.last_target, row)
                    pi_cell.setdefault(self.last_target,
                                       pi.get(row + (self.last_target,), 0.0))
        # THE REALISED CELL BELONGS TO THE REALISED JOB, argmax pi or not. Two
        # jobs can legally target one cell -- a stash counter is a STASH target
        # and a COLLECT target on the same tick -- and argmax ownership then hands
        # that cell to whichever carries more mass. Everywhere ownership only
        # feeds R that is harmless, because R is pinned to 0 on this cell anyway;
        # but the tier guard below reads owner[ref_cell][0] as "the rung the
        # baseline is on" and info["subtask"] reports owner[cell] as what the
        # robot is doing. With argmax ownership both read the OTHER job: on the
        # bayes floor at cap = 0, where the filter provably emits base_act on
        # every tick, the readout still claimed a different TIER on 26 of 1946
        # ticks. Nothing had moved; the label had. And under the guard the
        # mislabel is worse than cosmetic -- it would lock the candidate set to a
        # rung the baseline is not on, or drop the reference cell outright.
        if hit is not None and hit[2] in mask:
            owner[hit[2]] = ref_job
        return mask, jobs, owner, pi_cell, pi_src, ref_job

    def _step(self, pos, orient, a):
        """(pos, orient) after action `a`, using the mdp's OWN movement rule.

        Delegates to `_move_if_direction` rather than reimplementing it. A move
        into a wall turns the robot without moving it, which is a real and easily
        mis-copied detail, and orientation is what the INTERACT gate reads.
        """
        return self.mdp._move_if_direction(pos, orient, a)

    def _faces(self, pos, orient, mask):
        """The mask cell this (pos, orient) is pressing, or None.

        INTERACT is admitted exactly when the cell directly in front is a legal
        target, which is O(1) in the number of candidate cells. Without this the
        robot could never press anything: step_towards returns (None, True) on
        arrival, so every MOVE plan would still show t_end = 1 while the emitted
        action walked away.
        """
        c = (pos[0] + orient[0], pos[1] + orient[1])
        return c if c in mask else None

    # -- geometry: one BFS per candidate cell, then pure lookups -------------
    def _field_to(self, walk, cell):
        """{walkable cell: steps from it to a STANDING cell of `cell`}.

        Multi-source BFS with every standing cell at distance 0. One sweep answers
        the distance question for every (position) this tick's rollouts will ever
        ask about that cell, which is what makes 5 actions x 50 cells affordable
        without a single A*.

        Equivalent to A* and not approximately: the grid is 4-connected with unit
        edges, so BFS order IS shortest-path order. `{}` when the cell has no
        standing cell at all, which _tf then reports as unreachable exactly as
        geo.path_len would.
        """
        field = {}
        q = collections.deque()
        for s in sorted(geo.adjacent_standing_cells(walk, cell)):
            if s not in field:
                field[s] = 0
                q.append(s)
        while q:
            c = q.popleft()
            nd = field[c] + 1
            for dx, dy in geo.DIRECTIONS:
                n = (c[0] + dx, c[1] + dy)
                if n in walk and n not in field:
                    field[n] = nd
                    q.append(n)
        return field

    def _tf(self, field, pos, orient, cell):
        """steps_to_finish read off `field` in O(1). None if unreachable.

        Byte-identical in semantics to baselines.steps_to_finish -- walk (d),
        turn to face it (1), press (1), with standing-beside-it-already-facing-it
        costing just the press. That equivalence is what licenses reading the whole
        of this tick's geometry off one BFS per cell instead of one A* per query,
        and the test asserts it cell for cell rather than arguing it here.
        """
        d = field.get(pos)
        if d is None:
            return None
        if d == 0:
            return 1 if (pos[0] + orient[0], pos[1] + orient[1]) == cell else 2
        return d + 2

    def _rollout(self, field, walk, p1, n):
        """The robot's POSITION at forecast ticks 1..n, as a list of length n.

        Descends the distance field one step a tick and then FREEZES at the
        standing cell, which is exactly what the robot does after it presses:
        d == 0 means "beside the target", and there is nothing in this layer's
        model of the future that would move it on.

        Orientation is never tracked and that is not laziness. views.py:186-188
        tests only `pos in seen_cells`, so which way the robot faces cannot affect
        whether the human sees it. Turning shows up in the cost through _tf's
        turn tick, where it belongs, and nowhere else.

        The tie-break among equal-distance neighbours is the cell coordinate, so
        this is deterministic -- and it is also the known approximation named in
        the module docstring: two equally short routes with different visibility
        are not both scored.
        """
        cur = p1
        out = []
        for _ in range(n):
            out.append(cur)
            d = field.get(cur)
            if not d:                       # None (stuck) or 0 (arrived): freeze
                continue
            nxt = None
            for dx, dy in geo.DIRECTIONS:
                c = (cur[0] + dx, cur[1] + dy)
                if c in walk and field.get(c, _INF) == d - 1:
                    if nxt is None or c < nxt:
                        nxt = c
            if nxt is not None:
                cur = nxt
        return out

    # -- the human's cone, once per tick ------------------------------------
    def _forecast(self, state, fov, horizon, walk, ref):
        """The MAP shadow's trajectory for the next `horizon` ticks, once.

        THIS IS THE ONE APPROXIMATION IN THIS FILE AND ITS SIZE IS MEASURED. The
        human's actions depend on the world, which depends on where the robot
        walked, so strictly every plan deserves its own human. dev/route_channel.py
        measures what that buys: over 720 ticks on all six layouts, changing the
        robot's first action changes the human's next six actions on 0.8% of
        ticks, and 0.0% on divide and pantry. The human sees the robot 70.7% of
        the time, but seeing it almost never changes what it DOES -- the only path
        from robot position to human behaviour is the contention demotion, and
        that fires only at stations inside the dividing wall.

        Carried over from qmdp unchanged, with one caveat that is bigger here than
        it was there: in qmdp the robot's route was incidental to the score, and
        here the score IS the route. A human that reacted to being approached
        would change these numbers and this file cannot tell you by how much.
        """
        sim = state.deepcopy()
        human = fast_clone_human(self.post.shadows[fov])
        out = []
        for _ in range(horizon):
            if self.mdp.is_terminal(sim):
                break
            ah, _ = human.action(sim)
            # THE REFERENCE ROBOT TRAJECTORY IS A PURE BEELINE, NOT baseline.action().
            # Calling the baseline here would be a correctness bug, not a slow
            # path: every baseline's own action() (baselines.py) mutates
            # `committed`, `last_subtask`, `t`, `log`, draws from its own rng and
            # sets `last_action`, which bayes reads back as inverse-planning
            # evidence on its very next update. Thirteen forecast steps per real
            # tick would advance the baseline's clock thirteenfold and overwrite
            # the realised pick that `_mask` anchors on -- true_subtask's docstring
            # says call it ONCE per tick and means it.
            #
            # geo.step_towards is pure and reproduces what the baseline does
            # exactly: every baseline's action() is step_towards to its realised
            # cell, INTERACT on arrival.
            if ref is not None:
                mv, arrived = geo.step_towards(
                    walk, tuple(sim.players[self.agent_index].position),
                    tuple(sim.players[self.agent_index].orientation), ref)
                ar = Action.INTERACT if arrived else (mv or Action.STAY)
            else:
                ar = Action.STAY
            hp = sim.players[self.other_index]
            out.append((ah, tuple(hp.position), tuple(hp.orientation)))
            sim, _, _, _ = self.mdp.get_state_transition(sim, self._joint(ar, ah))
        return out

    def _cone_ticks(self, fc, fov):
        """[frozen set of visible cells] for forecast ticks 0 .. len(fc)-1.

        The ONLY use the score makes of the forecast. No cone is hardcoded
        anywhere: `fov` is the MAP hypothesis from the posterior and
        geo.visible_cells is the same cone-plus-line-of-sight test the human
        perceives through, so a cell counts as seen exactly when the human would
        have seen it.
        """
        terrain = self.mdp.terrain_mtx
        return [geo.visible_cells(terrain, hpos, horient, fov)
                for (_ah, hpos, horient) in fc]

    def _window(self, path):
        """How many ticks of `path` are this sub-task's own trajectory, 1..A.

        The freeze is read off the PATH rather than passed in as t_end, because
        _rollout also freezes when the field says `stuck` (no descending
        neighbour) and that is not an arrival -- but it is equally not part of
        doing the sub-task, so both want the same cut.

        Lifted out of _bonus so the readout can print the LENGTH the sightings
        were counted over. A HUD that shows "seen 3" without saying 3 out of how
        many ticks is showing a count that cannot be compared between two plans of
        different lengths, and slot 5 of a cands row is that number.
        """
        stop = len(path)
        if not self.count_parked:
            while stop > 1 and path[stop - 1] == path[stop - 2]:
                stop -= 1
        return stop

    def _bonus(self, path, cones):
        """(bonus, first_k, last_k, n_seen). bonus in [0, cap/(1-decay)] TICKS.

            bonus = cap * SUM_{j=1..n} decay**(j-1) * gamma**(k_j - 1)

        where k_1 < k_2 < ... < k_n are the ticks of this sub-task's own
        trajectory on which the human's cone covers the ROBOT. It is a NUMBER OF
        TICKS OF BASELINE COST, not a fraction of anything: there is no
        denominator here any more, and looking for one is looking for the shape
        this file had before the cap became a per-sighting price.

        WHAT EACH KNOB NOW MEANS, and they no longer mean the same thing:
          cap     the MOST this layer will pay to be seen ONCE. A plan seen once
                  on tick 1 is worth exactly `cap` -- gamma**0 * decay**0 == 1 --
                  which is the whole reason the knob is allowed that name.
          decay   how much less the NEXT sighting is worth than the one before,
                  wherever in the window it lands.
          gamma   how much less a LATER sighting is worth than an earlier one,
                  whatever its number. Unchanged: the forecast human is open-loop
                  in the robot, so tick 12 is an extrapolation and tick 2 is not.

        BOUNDED, which is the property the rest of the file leans on:
            bonus <= cap * SUM_{j>=1} decay**(j-1) = cap / (1 - decay)
        because gamma <= 1 kills nothing and adds nothing. That number is
        `max_influence`, it is what the theorem in the module docstring quotes,
        and it is the number the tier lock compares against a rung.

        ALIGNMENT, and it is easy to get wrong by one: fc[k] holds the human's
        pose BEFORE its k-th action, so fc[0] is the human right now and the robot
        is at its current cell right now. Tick 0 carries no information about our
        choice and is excluded -- including it would add the same constant to
        every plan and dilute every difference. The window is k = 1..D, which is
        why the forecast is run for depth + 1 ticks.

        IT IS A SUM OVER THE WHOLE TRAJECTORY, NOT A VERDICT ON ANY ONE TICK.
        The window is 1 .. A, where A is the arrival, and the question it answers
        is HOW OFTEN AND HOW SOON AM I SEEN -- not "does this drop get witnessed",
        which is a yes/no about a single moment and is the handoff question this
        filter is not allowed to ask. Every tick in the window is scored by the
        same cone test, the arrival tick included: being seen as you put the item
        down is one more sighting, and nothing here can tell it apart from being
        seen two steps earlier.

        THAT IS THE WHOLE POINT, and the shape it replaces is worth recording.
        _rollout descends the distance field and then FREEZES on the standing cell,
        so a plan arriving at tick 4 of a 12-tick window spent eight of its twelve
        samples parked on the very tile it presses from. Scoring those asks "will
        the partner be looking at the counter I stash on, around the time I stash
        on it" -- a question about the HANDOFF, which is exactly the task knowledge
        this filter is not allowed to have, smuggled back in as a sampling
        artefact. Measured before the cut: 36% of all sampled ticks were frozen
        ones and they carried 22% of the entire score. A fifth of the objective was
        quietly qmdp's, and it was concentrated on the stash moment.

        Set count_parked=True to restore the old full-window behaviour. It is kept
        as an ablation because "does the parked tail matter" is the first question
        a reviewer will ask about that cut, and the honest answer is a number.

        THE LIVE PATH IS THE DP IN action(), NOT THIS. This is the reference
        definition -- one materialised path, no search -- and the tests pin the
        semantics on it. Both read the SAME `_weights` table, which is what stops
        the two drifting apart: they drifted once already, when the DP normalised
        inline while this method went through _shape, and `fov_decay` moved one and
        not the other so a whole decay sweep printed identical rows.
        """
        # The freeze is detected from the PATH rather than passed in as t_end,
        # because _rollout also freezes when the field says `stuck` (no descending
        # neighbour) and that is not an arrival -- but it is equally not part of
        # doing the subtask, so both want the same cut.
        stop = self._window(path)
        wt, j_max = self._weights()
        kmax = len(wt[0]) if wt else 0
        tot, n_seen, first, last = 0.0, 0, None, None
        for k, cell in enumerate(path[:stop]):
            if cell in cones[k + 1]:
                # n_seen is the count BEFORE this one, so it indexes decay
                # directly: the (n_seen+1)-th sighting is worth decay**n_seen.
                # Past j_max the sighting is worth ZERO -- see _weights for why
                # that direction of error is the safe one.
                if n_seen < j_max and k < kmax:
                    tot += wt[n_seen][k]
                n_seen += 1
                if first is None:
                    first = k + 1
                last = k + 1
        return tot, first, last, n_seen

    def _top3(self, jobs, owner, scored, chosen_cell):
        """[(label, best C, is the one taken)] for the HUD's `robot weighs` line.

        The same shape every method in the registry emits, so that line reads the
        same way across the table -- but the NUMBER here is a cost in ticks, lower
        better, where a baseline puts a probability. `top3_kind` says which. A job
        with no scorable plan still gets a row, at inf, because "the baseline
        offered it and nothing could be scored for it" is exactly what you want to
        see when the robot ignores an obvious option.
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

    # -- the tick -----------------------------------------------------------
    def action(self, state):
        # ONCE per tick. base_info["subtask"] is the REALISED draw and everything
        # in this file is anchored on it; calling the baseline twice would make it
        # believe it acted twice and would move that anchor underneath us.
        base_act, base_info = self.baseline.action(state)
        ranked = self.baseline.rank_subtasks(state)
        fov, pmap = self._map_cone()
        info = {"subtask": base_info.get("subtask"),
                "base_subtask": base_info.get("subtask"),
                "fov_post": dict(self.post.p),
                "theta_entropy": norm_entropy(self.post.p),
                "map_fov": fov, "p_map": pmap,
                "deviated": False, "gated": False,
                # THREE NUMBERS, NOT ONE, AND THEY ARE NOT INTERCHANGEABLE. `cap`
                # is what ONE sighting is worth; `decay` is how fast the next one
                # is worth less; `max_influence` = cap/(1-decay) is the total this
                # layer can ever spend and is the bound in the theorem. It is
                # emitted rather than left to be re-derived because a script that
                # quotes `cap` as the bound is wrong by 1/(1-decay) and every
                # number in it still looks sane. On every tick, gated ones
                # included, so a log can always say what budget produced it.
                "cap": self.cap, "decay": self.fov_decay,
                "max_influence": self.max_influence, "debt": self.debt,
                "base_action": Action.ACTION_TO_INDEX[base_act]}

        # THE CERTAINTY GATE, unchanged from qmdp. Below it the cone is a guess and
        # the bonus would be a claim about a human we have not identified, so the
        # baseline acts. The gate refuses ~1.6% of ticks and 100% of the posterior's MAP
        # errors live inside them. Clearing `committed` here is not tidiness: a
        # commitment surviving a gated stretch fires on the first ungated tick
        # afterwards, against a plan nobody scored in between.
        if fov is None or pmap < self.certainty:
            info["gated"] = True
            self.committed = self.last_target = None
            return base_act, info
        if not ranked:
            return base_act, info

        # THE REFERENCE IS THE BASELINE'S REALISED DRAW, NEVER argmax(pi). RHO=0.95
        # means the baseline spontaneously re-draws, so the realised pick is often
        # not the mode -- that is the point of drawing. Anchoring R on the mode is
        # exactly the bug the parity control exists to catch: it makes cap = 0
        # deviate.
        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)
        walk = self.walkable(state) | {pos}
        sub_ref = base_info.get("subtask")
        mask, jobs, owner, pi_cell, pi_src, ref_job = self._mask(
            state, ranked, pos, orient, walk, sub_ref)
        info["pi_src"] = pi_src
        if not mask:
            return base_act, info

        ref_cell = sub_ref[2] if sub_ref else None
        if ref_cell is None or ref_cell not in mask:
            # _mask forces the realised pick to the front of `jobs`, so this is the
            # nothing-ranked case rather than a masking accident.
            info["no_reach"] = True
            self.committed = self.last_target = None
            return base_act, info
        # GUARD (0a), THE STALE-PICK REFUSAL. ref_job is None when the sub-task the
        # baseline says it drew is not a row of THIS tick's ranking at all. That is
        # not hypothetical and it is not the same as the cell being unreachable:
        # bayes's own action() samples from its _marginal directly (baselines.py
        # no longer models one), whose keys are last tick's allocation set, so
        # the draw can name a job
        # rank_subtasks no longer offers. The cell usually survives anyway --
        # another job took it over -- and that is exactly what makes this
        # dangerous rather than harmless: R would be priced against the mass of a
        # DIFFERENT job on that cell, and the tier guard below would lock onto that
        # other job's rung. Traced on chefs_table t14, where the baseline drew
        # START/load_board@(11,5), the ranking held only WORK/chop@(11,5), and the
        # filter reported a WORK sub-task against a START baseline while the tier
        # lock was on and believed itself to be holding the line. 14 such ticks
        # per 1000 on the bayes floor and 0 on any drawn one -- rare, and the
        # whole of the difference between an invariant that is observed and one
        # that is structural: EVERY pre-fix cross-rung tick was one of these.
        if ref_job is None:
            info["stale_pick"] = True
            self.committed = self.last_target = None
            return base_act, info
        # GUARD (0), THE UNPRICED-REFERENCE REFUSAL, this layer's whole answer to
        # "what if the distribution I am pricing against says nothing this tick".
        # pi(ref) == 0 means the baseline's OWN realised draw carries no mass in
        # the distribution the baseline claims to have drawn it from, which is not
        # a near-miss: it happens whenever bayes's last_pi came from _marginal(0)
        # over LAST tick's belief while rank_subtasks re-projected onto this
        # tick's allocation set, and the two key sets are disjoint. Then every
        # candidate, ref included, is floored to PI_FLOOR, R is 0 everywhere, and
        # the bonus is being compared against a field of zeros -- 126 of 1946 ungated
        # ticks on fov-bayes here, 144 on the pre-fix trajectory this replaced,
        # and 0 on every drawn baseline.
        #
        # The previous line here re-anchored p_ref on the MODE and left the pin
        # R(ref) = 0 to paper over the inconsistency. That kept the arithmetic
        # finite and made the bound meaningless: the baseline's cell was free by
        # fiat while every other cell was priced against a cell the baseline had
        # not drawn. Refusing is the conservative reading of the charter in the
        # first paragraph of this file -- the nominal policy owns task success --
        # and it is the same disposal the certainty gate makes for the same
        # reason: no reference, no deviation. `committed` is cleared for the gate's
        # reason too, so a commitment cannot survive a refused stretch and fire on
        # the far side of it against a plan nobody scored.
        if pi_cell.get(ref_cell, 0.0) <= 0.0:
            info["pi_no_ref"] = True
            self.committed = self.last_target = None
            return base_act, info
        p_ref = max(pi_cell.get(ref_cell, 0.0), PI_FLOOR)

        # ONE forecast, shared by every plan. depth + 1 ticks because the window is
        # k = 1..D and fc[0] is the human as it stands right now.
        fc = self._forecast(state, fov, self.depth + 1, walk, ref_cell)
        if not fc:
            return base_act, info
        cones = self._cone_ticks(fc, fov)
        D = min(self.depth, len(cones) - 1)
        if D < 1:
            return base_act, info

        # THE ABSOLUTE TIER GUARD. Below one rung of budget a cross-rung cell is
        # not a plan at all -- it is removed from the candidate set here, before a
        # single BFS is run, rather than priced and then out-argued by s.
        #
        # It is here and not in the cost because the cost cannot carry it. C is
        # bounded in units of R, and R is 0 for every cell the baseline likes at
        # least as much as the one it drew -- which is EVERY cell whenever pi is
        # flat over the candidates, and every cell whenever the draw was the least
        # likely one, and every cell whenever pi prices nothing at all -- guard (0)
        # now turns back that third case, and the first two are still live and
        # still legitimate. On the bayes floor R collapsed to 0.0 everywhere on
        # 12.3% of scoring ticks, and there the filter re-targeted across a rung at
        # cap = 4 and cap = 8, which is exactly what `cap` was advertised to make
        # impossible. A guarantee that holds only while pi discriminates is not a
        # guarantee; this one holds on any baseline, any layout, any tick.
        #
        # IT IS max_influence THAT IS COMPARED, NOT cap, and that is the whole of
        # what the decay change does to this guard. `cap` is now the price of ONE
        # sighting; what the layer can actually spend on a plan is cap/(1-decay),
        # which at the default decay is TWICE cap. Arming on `cap` would have left
        # a filter with a budget of 2 * 21.4 = 42.8 ticks believing itself to be
        # inside a rung, and the lock is the only absolute thing in the file.
        #
        # `<=` and not `<`: at max_influence == rung the FoV term can just barely
        # pay for a crossing in R, and the registry's one-rung row is set to
        # cap = rung * (1 - decay) so that its BOUND is one rung to within rounding
        # (rung = 21.4182). So one rung of influence is the last budget that cannot
        # cross, and everything strictly above it -- fov-free at cap 1e6 -- is
        # unguarded and does cross, which is what keeps the safety test falsifiable.
        #
        # ref_job is the REALISED job -- guard (0a) has already turned back the
        # tick where there was not one, and _mask has forced owner[ref_cell] onto
        # it, so the reference cell is guaranteed to survive the filter below.
        # Reading the rung off argmax-pi ownership instead would lock the set to a
        # rung the baseline is not on, or drop the reference cell and leave the
        # layer inert while looking like it was working.
        lock = self.max_influence <= self.rung + 1e-9
        ref_tier = ref_job[0]
        info["tier_lock"] = lock
        n_locked = 0

        # ONE multi-source BFS per candidate cell, OUTSIDE the action loop. Inside
        # it this file would be slower than the branching search it replaced.
        fields, R = {}, {}
        for c in sorted(mask):
            v = owner.get(c)
            if lock and (v is None or v[0] != ref_tier):
                n_locked += 1
                continue                    # a different rung: not a plan at all
            f = self._field_to(walk, c)
            # `_tf` is the reachability test and nothing more here -- the detour is
            # measured against the best FIRST ACTION further down, not against
            # this. n0 is kept beside the field so the drop is auditable.
            n0 = self._tf(f, pos, orient, c)
            if n0 is None:
                continue                    # walled off from here: not a plan
            fields[c] = (f, n0)
            R[c] = max(0.0, (math.log(p_ref)
                             - math.log(max(pi_cell.get(c, 0.0), PI_FLOOR)))
                       / self.kappa)
        if ref_cell not in fields:
            # No reference to measure R against, so there is no cost function.
            info["no_reach"] = True
            self.committed = self.last_target = None
            return base_act, info
        # R(ref) == 0 IS A REQUIREMENT, NOT AN OBSERVATION. It now falls out of the
        # formula on every tick that gets this far, because guard (0) has already
        # turned back the one case where it did not -- p_ref anchored on the mode
        # with the realised cell sitting below it. Pinned anyway: the theorem, and
        # with it the cap = 0 parity claim, rests on the baseline's own plan
        # costing exactly zero, and that is too load-bearing to leave to the
        # associativity of two logs and a divide.
        R[ref_cell] = 0.0
        # DIAGNOSTIC, not a control. R == 0 on every candidate is still reachable
        # -- pi flat over the mask, or the realised draw the least likely of them
        # -- and on such a tick `cap` is not restraining the choice, it is only the
        # tier guard that is. Reported so the collapse rate can be measured from a
        # run's own readouts instead of re-derived from `cands`, which is how it
        # went unnoticed the first time.
        info["r_collapse"] = len(R) > 1 and max(R.values()) <= 0.0
        info["n_locked"] = n_locked

        # PLANS: A REAL ROLLOUT OVER LOW-LEVEL ACTIONS. This used to descend a
        # distance field one step a tick -- a shortest-path BEELINE -- and that was
        # the file's one substantive design error. With a beeline the trajectory is
        # FIXED once you name (first action, target cell), so the visibility term
        # could only ever choose BETWEEN shortest routes. It could never do the one
        # thing this filter exists for: walk a LONGER way because the longer way is
        # where the human is looking. The symptom was that the score came out 0 for
        # the overwhelming majority of candidates -- not because being seen was
        # impossible, but because the single route on offer happened not to be
        # visible and there was no second route to consider.
        #
        # So: dynamic programming over (pos, orient, SIGHTINGS SO FAR) per tick.
        #
        # THE THIRD COMPONENT IS NEW AND IT IS NOT OPTIONAL. Under the old score a
        # tick was worth gamma**k WHATEVER had happened before it, so two routes
        # reaching the same pose at the same tick had identical futures and merging
        # them on the better accumulated sum was exact. Now the j-th sighting is
        # worth decay**(j-1) of a sighting, so the future value of a pose depends
        # on HOW MANY TIMES the route has already been seen: a route seen five
        # times and a route seen never, standing on the same tile at the same tick,
        # price their next sighting a factor decay**5 apart. Merging them on `acc`
        # alone would have let a well-seen route inherit a never-seen route's
        # appetite -- the exact failure the count in the state prevents.
        #
        # IT IS STILL EXACT, and cheaper than the state space suggests, for two
        # reasons kept separate on purpose:
        #   * PARETO DOMINANCE, exact and doing the real work. If two states at the
        #     same (pos, orient, tick) have j1 <= j2 and acc1 >= acc2, state 1
        #     dominates: the set of futures from a pose is identical whatever j is,
        #     and every future sighting is worth decay**(j+i-1), which is
        #     non-increasing in j -- so state 1's best future is >= state 2's and
        #     its head start is >= too. Dropping state 2 cannot lose the optimum.
        #     That is prune() below, and it leaves at most an antichain per pose:
        #     acc strictly increasing in j.
        #   * THE j_max CLAMP, the only approximation, and it is bounded: see
        #     _weights. Sightings past j_max are worth nothing, so states past it
        #     merge into one absorbing count. At the default decay that costs at
        #     most 0.001 * cap ticks on a plan seen more than eleven times, and at
        #     any decay above ~0.51 it costs exactly nothing, because j_max hits
        #     `depth` and a window of `depth` ticks cannot hold more sightings than
        #     that.
        # Either way it is a frontier over POSES, not over sequences: |floor| x 4 x
        # depth x (a small antichain) states instead of 5**12 = 244M sequences.
        # MEASURED, uncontended, 60 ticks: butchery 33.4 -> 44.5 ms/tick and
        # banquet_pass 19.8 -> 21.8 against the pre-decay DP, so the count costs
        # 1.1-1.3x rather than the depth-fold the state space suggests -- that gap
        # is the prune. qmdp on the same box is 1232 ms/tick, and the repo test
        # asserting this file is the cheaper of the two has 26x of headroom.
        #
        # EXACTNESS IS CHECKED AGAINST BRUTE FORCE, not argued: at depth 5, every
        # action sequence enumerated and scored by the formula directly, the DP's
        # best plan per (first action, cell, arrival) matched to within the 0.005
        # rounding of the readout on 1223 plans over three layouts x three decays.
        #
        # THE DETOUR IS THE MECHANISM, AND IT IS PRICED. A route that takes k extra
        # ticks pays k in the cost below and can earn at most cap/(1-decay) back, so
        # it is taken exactly when being seen is worth more than the delay -- and
        # never by more than that, which is the bounded-influence theorem restated
        # in terms of routes rather than of jobs.
        #
        # Reach is unaffected. The old comment argued a beeline reaches a 28-step
        # counter that a growing frontier provably cannot; that argument was about
        # REACH and is still true, but it is not an argument for a beeline. The DP
        # is a frontier over POSES rather than over sequences, so it neither grows
        # exponentially nor stops short: every pose reachable within depth is in it.
        acts = list(MOVES) + ([Action.STAY] if self.allow_stay else [])
        # wt[j][k] IS THE WHOLE COST FUNCTION'S FoV HALF, cap included. Read it,
        # never gamma**k -- see _weights and the dead-knob failure recorded there.
        wt, j_max = self._weights()

        # One transition table per tick, built lazily. _step and _faces are pure
        # given (pos, orient) and this tick's walkable set, and the DP asks for the
        # same pose thousands of times, so memoising turns the inner loop into dict
        # lookups. This is the whole of the difference between "a few hundred ms a
        # tick" and "a few tens".
        step_tab, face_tab = {}, {}

        def moves_from(q, o):
            v = step_tab.get((q, o))
            if v is None:
                v = tuple(self._step(q, o, b) for b in acts)
                step_tab[(q, o)] = v
            return v

        def faced_from(q, o):
            if (q, o) not in face_tab:
                face_tab[(q, o)] = self._faces(q, o, mask)
            return face_tab[(q, o)]

        # (first action, cell, t_end) -> (best BONUS in ticks, n_seen)
        term, short = {}, {}

        def offer(ia, c, t_end, acc, n):
            k = (ia, c, t_end)
            cur = term.get(k)
            if cur is None or acc > cur[0]:
                term[k] = (acc, n)
            if t_end < short.get(c, _INF):
                short[c] = t_end

        def prune(cand):
            """Drop every DOMINATED (j, acc) from each pose. Exact, not a heuristic.

            j1 <= j2 and acc1 >= acc2 means state 2 can never beat state 1 from
            here: same reachable futures, every future sighting worth no less to
            state 1 (decay**(j1+i-1) >= decay**(j2+i-1)), and a head start that is
            already no smaller. So what survives per pose is an antichain -- acc
            strictly increasing in j -- which is what keeps the extra dimension
            from multiplying the state count by depth. Being seen more times is
            only ever worth carrying if it bought a strictly better sum.
            """
            for key, byj in cand.items():
                if len(byj) < 2:
                    continue
                best, keep = -1.0, {}
                for j in sorted(byj):
                    v = byj[j]
                    if v[0] > best + 1e-12:
                        keep[j] = v
                        best = v[0]
                if len(keep) < len(byj):
                    cand[key] = keep
            return cand

        # INTERACT IS A FIRST ACTION TOO, and omitting it is not a small loss:
        # without it the robot can never press anything, because on arrival
        # step_towards returns (None, True) and every MOVE plan would keep
        # reporting t_end = 1 while the emitted action walked away. Its trajectory
        # is one tick long -- the press -- scored at the cell it presses from, by
        # the same test every other tick gets. Seen on it, and it is the FIRST
        # sighting of that one-tick plan, so it is worth exactly wt[0][0] == cap.
        faced0 = faced_from(pos, orient)
        if faced0 is not None and faced0 in fields:
            h = 1 if pos in cones[1] else 0
            offer(Action.ACTION_TO_INDEX[Action.INTERACT], faced0, 1,
                  wt[0][0] if h else 0.0, h)

        for a in acts:
            ia = Action.ACTION_TO_INDEX[a]
            p1, o1 = self._step(pos, orient, a)
            h = 1 if p1 in cones[1] else 0
            # {(pos, orient): {sightings so far: (bonus in ticks, n_seen)}}.
            # n_seen rides along for the readout only -- it is the count of the
            # path that won, and past j_max it keeps counting while the bonus
            # stops, which is what "seen 14/14" beside a flat bonus means.
            dp = {(p1, o1): ({1: (wt[0][0], 1)} if h else {0: (0.0, 0)})}
            for k in range(1, D):
                nxt = {}
                for (q, o), byj in dp.items():
                    # TERMINAL: press at tick k+1 from a pose reached after k
                    # actions. The robot does not move on a press, so the press
                    # tick is scored at `q` -- one ordinary tick of the window.
                    c = faced_from(q, o)
                    if c is not None and c in fields:
                        if q in cones[k + 1]:
                            for j, (acc, n) in byj.items():
                                offer(ia, c, k + 1,
                                      acc + (wt[j][k] if j < j_max else 0.0),
                                      n + 1)
                        else:
                            for j, (acc, n) in byj.items():
                                offer(ia, c, k + 1, acc, n)
                    for (q2, o2) in moves_from(q, o):
                        cur = nxt.get((q2, o2))
                        if cur is None:
                            cur = nxt[(q2, o2)] = {}
                        if q2 in cones[k + 1]:
                            for j, (acc, n) in byj.items():
                                # The (j+1)-th sighting, discounted twice: by when
                                # it happens (k) and by how many came first (j).
                                if j < j_max:
                                    v, j2 = acc + wt[j][k], j + 1
                                else:
                                    v, j2 = acc, j      # absorbing: worth nothing
                                p = cur.get(j2)
                                if p is None or v > p[0]:
                                    cur[j2] = (v, n + 1)
                        else:
                            for j, av in byj.items():
                                p = cur.get(j)
                                if p is None or av[0] > p[0]:
                                    cur[j] = av
                dp = prune(nxt)
                if not dp:
                    break

        # THE PLAN CARRIES ITS BONUS, IN TICKS. There is no normalisation step here
        # and no `s`: the DP accumulated cap * sum_j decay**(j-1) * gamma**(k_j-1)
        # directly out of the one weight table, so what comes out is already the
        # number of ticks of baseline cost this plan's visibility is worth. The
        # window is t_end by construction, so there is no parked tail to strip and
        # nothing to detect.
        raw = [(ia, c, t_end, bonus, n, t_end)
               for (ia, c, t_end), (bonus, n) in term.items()]

        # THE DETOUR IS MEASURED AGAINST THE BEST FIRST ACTION FOR THAT CELL, not
        # against steps_to_finish from where the robot stands. That is the one
        # place this file cannot use the baseline's own distance function, and the
        # reason is an off-by-one that breaks the cap = 0 proof outright:
        # steps_to_finish charges d + 2 -- walk, TURN, press -- unconditionally,
        # but a first action that steps onto a standing cell ALREADY FACING the
        # target never pays that turn. Robot at (7,4) facing south, counter at
        # (7,6): steps_to_finish says 3, stepping south and pressing takes 2, so
        # `detour` came out at -1, R + D went NEGATIVE, and at cap = 0 the filter
        # deviated from the baseline it is supposed to reproduce exactly (traced
        # on back_bar seed 0 tick 10).
        #
        # Taking the min over this cell's own rows makes D >= 0 by construction
        # with equality attained, which is what the theorem needs. It also still
        # gives the BASELINE's own action D = 0 on the reference cell, which is
        # what parity needs, and that is a fact about step_towards rather than a
        # hope: for d >= 2 every action on a shortest route ties at d + 2; for
        # d == 1 a collinear standing cell (pos + dir with target at pos + 2*dir)
        # is the UNIQUE standing cell at distance 1, so astar cannot pick a
        # different one and miss the saved turn; and for d == 0 the baseline turns
        # straight onto the target, which is the minimum by inspection.
        # THE BASELINE'S OWN PLAN COSTS ZERO BY DEFINITION, and pinning that is what
        # makes cap = 0 an identity instead of a hope. `short[c]` is the shortest
        # t_end the DP can find for c, and the DP is a BETTER WALKER than the
        # baseline: it shares turns across a route where geo.step_towards commits to
        # a direction and pays for the turn later. So on the reference cell the
        # baseline's realised action was coming out one tick "slow", eating a detour
        # charge it never agreed to, and at cap = 0 some other action tied at zero
        # and won -- fov-base diverged from handoff on 115 of 1000 ticks with the
        # FoV term switched off entirely, which is a filter second-guessing the
        # nominal policy's WALKING. That is not this layer's job: task success is
        # the baseline's, and "you could have turned earlier" is task success.
        #
        # So the reference cell is measured against the baseline's own arrival, and
        # every detour is clamped at zero. A route that is genuinely faster is then
        # free rather than negative, which keeps R + D >= 0 -- the theorem needs
        # that -- and leaves the tie to guard (i), which prefers base_act.
        _ib = Action.ACTION_TO_INDEX[base_act]
        ref_t = min((te for (ia, c, te) in term if ia == _ib and c == ref_cell),
                    default=None)
        if ref_t is not None:
            short[ref_cell] = ref_t
        plans = []
        for (ia, c, t_end, bonus, nsn, win) in raw:
            # D is what makes waiting and orbiting cost something: R is per-cell,
            # so a filter paying only R would wander sideways to stay visible at
            # zero charge.
            plans.append((ia, c, t_end, R[c] + max(0.0, t_end - short[c]),
                          bonus, nsn, win))

        if not plans:
            info.update({"no_reach": True, "n_mask": len(mask), "n_states": 0,
                         "search_capped": False, "cands": [],
                         "jobs": [(TIER_NAME[t], v) for t, v in jobs]})
            self.committed = self.last_target = None
            return base_act, info

        # THE COST. `cap` is not here any more -- it is inside `bonus`, folded in by
        # the weight table, which is what makes this the ONE line that combines the
        # two halves and nothing else in the file a second opinion about the price.
        best, scored = {}, []
        for (ia, cell, t_end, base_cost, bonus, nsn, win) in plans:
            c = base_cost - bonus
            # THE `cell != ref_cell` KEY IS LOAD-BEARING. At cap = 0 every plan for
            # the realised cell ties at C = 0 with plans for equally likely cells,
            # and without it the committed cell drifts off the baseline's own while
            # the emitted action still looks identical -- which is parity that
            # holds until the first tick it does not.
            key = (c, cell != ref_cell, t_end, cell)
            cur = best.get(ia)
            if cur is None or key < cur[0]:
                best[ia] = (key, c, cell, t_end, base_cost, bonus, nsn, win)
            # avail and tail kept apart in the readout under their old names: slot 6
            # is the BASELINE half R + D and slot 7 the FoV half -bonus, so the
            # HUD's "C = av + tl" decomposition still adds up. Slot 7 was -cap*s
            # when the FoV term was a capped fraction; it is the same slot with the
            # same sign and the same meaning -- the ticks the cone bought -- and
            # watch.py never has to know that s stopped existing.
            # SLOTS 4/5 ARE THE COUNT AND ITS DENOMINATOR, not a first-seen and a
            # last-seen tick. qmdp put "the tick their cone reaches the counter"
            # and "the tick they hold the item" here because its score was about
            # a moment; ours is a count over a trajectory, so the honest readout
            # is "seen on n of w ticks". Printing t+k here would be describing
            # this filter in the vocabulary of the one it replaced.
            scored.append((owner.get(cell, (None, "?"))[1], cell, t_end,
                           round(c, 1), nsn, win,
                           round(base_cost, 1), round(-bonus, 1)))

        qa = {ia: v[1] for ia, v in best.items()}
        # PER FIRST ACTION, ON THE RECORD. Q_action alone says WHICH action won and
        # by how much, but not WHY -- and the question anyone actually asks of this
        # filter is "it could have stepped right and been seen four more times, why
        # did it not". That is answerable only with the losing action's own best
        # plan beside the winner's: its target, its cost split into the baseline
        # half and the FoV half, and how much of it would have been seen.
        info["per_action"] = {
            ia: {"C": round(v[1], 2), "cell": v[2], "t_end": v[3],
                 "base": round(v[4], 2), "fov": round(-v[5], 2),
                 "seen": v[6], "win": v[7]}
            for ia, v in best.items()}
        ib = Action.ACTION_TO_INDEX[base_act]
        base_c = qa.get(ib, min(qa.values()))
        lo = min(qa.values())

        # GUARD (ii), the committed cell. Armed ONLY after a deviating tick, which
        # is the difference from qmdp and the reason cap = 0 is exact rather than
        # nearly exact. Its job is unchanged: without something held, the layer
        # re-decides from scratch each tick and, when its own plan misses eps_a by
        # a hair, emits the baseline's action instead -- which walks toward a
        # different counter. Traced on back_bar: the robot alternated two cells
        # forever and pressed nothing.
        held = None
        if self.committed is not None:
            same = [ia for ia, v in best.items() if v[2] == self.committed]
            if same:
                held = min(same, key=lambda ia: (qa[ia], ia != ib, ia))
        # ...but a HELD PLAN MUST NEVER OUTRANK AN EQUALLY GOOD BASELINE ACTION.
        # The hold exists to stop the layer ping-ponging between two counters it
        # already wanted; it is not a licence to deviate for free. Letting it emit
        # its own action on a tie is what broke cap = 0 parity when the beeline
        # became a real rollout: the DP finds SEVERAL equally short routes to a
        # cell where the beeline found one, so on a flat Q the held branch started
        # picking a tied-but-different action and fov-base diverged from handoff on
        # 115 of 1000 ticks. Deferring on ties costs the guard nothing -- if the
        # held plan is genuinely better it is not a tie -- and it restores parity
        # as an identity rather than as a measurement.
        if (held is not None and qa[held] <= lo + self.eps
                and qa[held] < qa.get(ib, lo) - 1e-12):
            act = Action.INDEX_TO_ACTION[held]
            chosen = held
            info["held_plan"] = True
        else:
            # GUARD (i). Tie-break on the baseline's own action, so a flat Q
            # reproduces the nominal policy rather than merely resembling it.
            win = min(qa, key=lambda ia: (qa[ia], ia != ib, ia))
            act = base_act
            if qa[win] < base_c - self.eps_a:
                act = Action.INDEX_TO_ACTION[win]
            chosen = Action.ACTION_TO_INDEX[act]

        # GUARD (iii), the deviation budget. This is the hard answer to "you cannot
        # just try to stay in the human's cone forever": at most `stall_max`
        # CONSECUTIVE ticks of authority, then the baseline gets one back,
        # regardless of cap. A cap alone cannot do this job -- a receding-horizon
        # controller forgets its accumulated delay every tick, so each individual
        # tick of loitering is affordable while the sum of them is not.
        dev = (act != base_act)
        if dev and self.debt >= self.stall_max:
            act, chosen, dev = base_act, ib, False
            info["stalled"] = True
        # Leaky rather than a hard counter: one tick of deference pays back one
        # tick of budget, so a filter that deviates half the time keeps deviating
        # and one that deviates constantly is throttled.
        self.debt = (self.debt + 1) if dev else max(0, self.debt - 1)
        info["deviated"] = dev
        info["debt"] = self.debt

        _k, C, cell, t_end, base_cost, bonus, _nsn, _win = best.get(
            chosen, (None, float("nan"), None, None, 0.0, 0.0, 0, 0))
        self.last_target = cell
        v = owner.get(cell)
        self.last_subtask = ((TIER_NAME[v[0]], v[1], cell) if v is not None
                             else base_info.get("subtask"))
        # COMMITMENT IS A CONSEQUENCE OF DEVIATING. Released at the press, because
        # the first INTERACT certifies the sub-task and that is where holding a
        # cell stops meaning anything. With cap = 0 the filter never deviates, so
        # this is never set, so guard (ii) can never emit a non-baseline action on
        # a later tick -- which is what makes the parity claim structural.
        self.committed = cell if (dev and act != Action.INTERACT) else None

        full = {i: qa.get(i, float("inf")) for i in range(len(Action.ALL_ACTIONS))}
        w = {i: math.exp(-(qa[i] - lo) / max(self.temperature, 1e-9)) for i in qa}
        z = sum(w.values()) or 1.0
        info.update({"subtask": self.last_subtask, "Q_action": full,
                     "action_dist": {i: (w.get(i, 0.0) / z)
                                     for i in range(len(Action.ALL_ACTIONS))},
                     "Q": qa.get(chosen, float("nan")),
                     "gain": base_c - qa.get(chosen, base_c),
                     "n_mask": len(mask), "n_jobs": len(jobs),
                     "n_states": len(plans),      # plans enumerated
                     "n_nodes": len(fields),      # candidate cells
                     "search_capped": False,      # no bounded frontier here
                     "plan": (v[1], cell) if v is not None else None,
                     "plan_t": t_end,
                     "jobs": [(TIER_NAME[t], vb) for t, vb in jobs],
                     "cands": sorted(scored, key=lambda x: x[3]),
                     "top3": self._top3(jobs, owner, scored, cell),
                     "top3_kind": "t",
                     # THE CANDS ROW IS THE SAME SHAPE AS QMDP'S AND MEANS SOMETHING
                     # ELSE, which is exactly how a readout lies. qmdp's slots 4/5
                     # are "the tick the partner's cone reaches THAT COUNTER" and
                     # "the tick they have the item in hand"; ours are the count of
                     # ticks their cone is on THE ROBOT and the length it was
                     # counted over, and slot 7 is the FoV bonus in ticks rather
                     # than a tail. Holding the shape keeps watch.py's positional
                     # unpack working; this key is what stops it printing our
                     # numbers under qmdp's stash-and-handoff labels, which is a
                     # claim about task knowledge this filter deliberately does not
                     # have.
                     "cands_kind": "fov",
                     # Extra and non-breaking: the two halves of C on their own, so
                     # an analysis can ask what the cone bought without re-deriving
                     # it from the cands table. `fov_bonus` keeps its old sign --
                     # the FoV term's contribution to the COST, so C = base_cost +
                     # fov_bonus still adds up -- and it is now a number of ticks
                     # outright rather than cap times a fraction.
                     #
                     # `cap`, `decay` and `max_influence` are set on EVERY tick, up
                     # at the top of action(), gated ones included -- see there for
                     # why the bound travels with the readout.
                     "fov_bonus": -bonus,
                     "base_cost": base_cost, "kappa": self.kappa,
                     "depth": self.depth})
        self.last_Q = dict(qa)
        return act, info
