"""
ALTERNATIVE COST FUNCTION -- maximise what the partner WANTS to be doing.

Drop-in replacement for cost_function.py. Same two entry points, same
signatures, so qmdp.py can swap between them with one flag and nothing else
changes:

    tick_score(mdp, planner, shadow, state, wrong_before, explored, robot_index)
        -> (score, wrong_after)          higher score = better
    wrong_beliefs(shadow, state) -> set

===========================================================================
THE IDEA
===========================================================================
Roll the partner forward and score, every tick, HOW VALUABLE THE JOB THEY ARE
ON IS -- by their own reckoning. Prefer the robot action that leaves them
working the most valuable line.

The valuation is the human model's own PRIORITY table
(limited_vision_human.py:161), the same one its sampler draws from:

    pickup_washed_plate 5   ASSEMBLE a ready order -- finish it
    pickup_meat      4   start the steak    (critical path)
    pickup_onion     4   start the garnish  (critical path, EQUAL to meat)
    chop_onion       4   finish an in-progress chop
    heat_washed_plate   3   finish an in-progress wash
    pickup_plate     3   start a plate washing

score(tick) = PRIORITY[the partner's standing line]     0 if they have none

===========================================================================
WHAT THIS BUYS, AND WHAT IT COSTS
===========================================================================
BUYS -- it can see an UPGRADE, which cost_function.py structurally cannot.
The plate->assemble case: partner on `pickup_plate` (3) sees the robot carrying
a ready hot plate, re-samples onto `pickup_washed_plate` (5), and the score rises
by 2 per tick. The old cost function only ever detects that a plan DIED; this
one detects that a better one was found.

It also absorbs "left them with nothing to do" for free: explore, wait and
check_* are not keys in PRIORITY, so `.get(subtask, 0)` returns **0** with no
list of names to maintain. Being lost scores zero automatically.

COSTS -- **this reads task-value knowledge.** PRIORITY says assembling beats
prepping beats washing, which is the recipe's critical path written down. A real
person has no such table to read, so unlike `WASTED_IF_OCCUPIED` (a claim about
which station an errand concerns) or `visible()` (a claim about perception),
this one does NOT survive swapping the scripted human for a human being.

Treat it accordingly. It is the ORACLE-VALUE arm: "what does this look like if
the partner's preferences were known?" -- an upper bound on the learned-weights
version, not a method you can ship as domain-knowledge-free.

Everything else it refuses as before: no sparse reward, no shaped reward, no
order count, no step budget.

===========================================================================
NOT SCORED
===========================================================================
No information term and no clash term. The claim here is single: leave your
partner on the most valuable line they can be on. If the robot blindsides them,
their line becomes unavailable, their own sampler moves them somewhere worse or
to a look, and the score falls on its own -- so the consequence is measured
rather than asserted, the same way the rollout measures everything else.
"""
import baseline  # noqa: F401   sys.path shim -- import before anything below

from fov.human.agent.limited_vision_human import PRIORITY      # noqa: E402


def wrong_beliefs(shadow, state):
    """Interface compatibility only -- this cost reads no beliefs.

    qmdp.rollout() seeds `wrong` before the loop and threads it forward, so the
    symbol has to exist and return something set-like. Returning an empty set
    makes every subsequent set-difference empty too, which is exactly right:
    there is no information term here.
    """
    return set()


def tick_score(mdp, planner, shadow, state, wrong_before, explored,
               robot_index=0):
    """Value of the line the partner is on this tick, by their own PRIORITY.

    Returns (score, wrong_after) to match cost_function.tick_score. Higher is
    better; qmdp maximises.
    """
    #`_sampled` is the standing line -- the last thing they freely CHOSE. Use it
    #rather than `_current`, which gets overwritten by forced errands: a human
    #carrying meat to the pot is on `drop_meat`, which is not a PRIORITY key and
    #would score 0 for the whole carry even though they are mid-critical-path.
    #`_sampled` survives the entire pickup -> carry -> drop excursion.
    line = getattr(shadow, "_sampled", None)

    #PRIORITY's keys are exactly SAMPLING_SUBTASKS -- the six a free choice can
    #land on. Anything else (explore, wait, check_*, a forced drop) is absent, so
    #.get(..., 0) scores it zero with no name list of our own to maintain.
    score = float(PRIORITY.get(line, 0))

    #`explored` is deliberately unused. A human who is exploring has no standing
    #line, so `line` is None and the score is already 0 -- the explore gate that
    #cost_function.py needs is redundant here, and adding it would double-count.

    return score, wrong_before
