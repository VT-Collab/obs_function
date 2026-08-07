"""THE COST FUNCTION.  Two terms, both counted once per rollout tick.

    COVER    +1 for engaging a station your partner has WRONG -- doing the
             thing they cannot know needs doing. The unseen-assistance term.

    CREDIT   +1 for every fact the human GETS RIGHT that they had wrong.
             One-sided: a fact going the other way (right -> wrong) costs
             NOTHING. Informing your partner is a favour; failing to inform them
             is not a crime, because most of the time you are simply working.

    PENALTY  -1 for DOING THE EXACT SAME THING AS YOUR PARTNER WHILE THEY
             CANNOT SEE YOU. Redundant effort, spent in secret. Not for being
             near them, not for being busy -- for duplicating the specific work
             they are doing right now, unannounced.
             Skipped when they are LOST (the explore gate below).

    score(tick) = cover + credit - penalty      summed over the rollout

Kept deliberately PURE: no sparse reward, no shaped reward, no recipe, no order
count, no step budget, and no PRIORITY (the human's hand-written value ordering
over subtasks, which is the recipe wearing a disguise). Both claims are things
you would want from any teammate in any joint task, and both are things you
could ask about a real person.

===========================================================================
THE KNOWN RISK OF STAYING PURE, AND WHY IT IS SURVIVABLE HERE
===========================================================================
There is no term for "the robot got work done", so working scores 0. Any
POSITIVE term therefore outbids work automatically, because +1 > 0. Measured on
gc00 at W_CREDIT=1.0: 96% of overrides were credit-driven and 21% of them threw
away an INTERACT -- the module traded cooking for sightseeing.

That failure was ENVIRONMENTAL, not structural. On gc00 the human's beliefs were
right essentially always (gap_timed = 0, blind fraction 0.00), so credit fired on
76% of ticks about nothing -- it was pure noise with a positive sign. On these
layouts the probe measures gap_timed at 0.3-1.8 with runs persisting 8-24 ticks,
so a credit event now means the robot actually closed a blind spot that was
going to cost somebody something.

The residual risk is handled STRUCTURALLY rather than with a task term:
qmdp.PROTECT_INTERACT refuses to override the baseline when it wants to INTERACT.
INTERACT is the one primitive that CHANGES THE WORLD; the other five only
reposition. That is a distinction in the action space, not a reward, so the cost
function stays task-agnostic.

===========================================================================
BOTH TERMS ARE DOMAIN-GENERAL.  ONLY THE DETECTOR IS NOT.
===========================================================================
    CREDIT   my partner's picture of the world got more accurate
    PENALTY  my partner and I are doing the same work, and they do not know it

Neither mentions steak, stations or a recipe. What IS domain-specific is how you
DETECT "the same work" in this kitchen: it comes out as "the same station line",
because in a steakhouse the station is what makes two efforts redundant. That
detector lives in `line_of` / `robot_line` and nowhere else, so porting this cost
function to another domain means rewriting those two functions and nothing else.

    different work, they see me     +1 credit,  0 penalty   =  +1
    different work, hidden           0 credit,  0 penalty   =   0
    SAME work, they SEE me          +1 credit,  0 penalty   =  +1
    SAME work, HIDDEN                0 credit, -1 penalty   =  -1

If you end up doing the same thing they are, there are two acceptable outcomes:
LET THEM KNOW -- be where they can see you, and their own model makes them yield
-- or DROP IT. Both are fine and neither is charged. The single thing that costs
anything is duplicating them while they have no idea, because that is the one
where BOTH of you spend the effort.

===========================================================================
THE EXPLORE GATE
===========================================================================
A partner on `pickup_meat` who cannot find the meat station does not walk there;
they wander, and hold a standing line while making no progress on it. Deferring
to a plan that is going nowhere is worse than useless, so if the human explored
this tick, no penalty. One condition covering every version of the problem.

With familiar-kitchen seeding this fires far less than it used to -- the probe
measures explore at 0.0 on every layout -- but it stays, because the shadows can
still fail to find a route through terrain they have not seen.

===========================================================================
THE TRAP IN THE DETECTOR
===========================================================================
`planner.target_kind("pickup_meat")` returns the MEAT DISPENSER, which is where
they WALK. Two agents at a dispenser duplicate nothing; it is infinite and
uncontested. The station that makes `pickup_meat` redundant is the POT, because
a full pot is what makes the errand pointless. WASTED_IF_OCCUPIED is the human's
own table for exactly that, and target_kind covers the three subtasks it does not
list, where the walk-to station IS the contested one.
"""

import _paths  # noqa: F401   MUST be first

from fov.human.agent.limited_vision_human import (                      # noqa: E402
    WASTED_IF_OCCUPIED, ROBOT_HELD_FETCH, TERRAIN_KIND)

#How loudly informing your partner competes with doing your job. See the header:
#this matters more than a weight usually does, because work scores 0.
W_CREDIT = 1.0

#=========================================================================
#COVER: the UNSEEN-ASSISTANCE term, and the one the first draft was missing.
#
#    +1 for engaging a station whose state your partner currently has WRONG.
#
#Stated without a kitchen in sight: I am handling something you cannot know
#needs handling. That is the claim the whole project is about -- "robots that
#reason over the human's field of view can provide unseen and seen assistance"
#-- and neither of the other two terms expresses it:
#
#    CREDIT   makes their picture accurate      = TELLING them        (seen)
#    PENALTY  avoids duplicating them unseen    = STAYING OUT OF THE WAY
#    COVER    acts on what they cannot know     = DOING IT FOR THEM   (unseen)
#
#Measured reason it had to exist. With credit and penalty alone the module was
#worse than the baseline at every setting tried, and WORST with the penalty
#alone (+28 ticks at fov360). Both of those terms are satisfied by WITHDRAWING,
#and under capacity pressure -- where the task cannot be finished by one agent --
#withdrawing is the expensive mistake, because unused effort costs more than
#duplicated effort. COVER is the only one of the three that can ever ask the
#robot to do MORE, which is why a score built from the other two could only
#subtract.
#
#STILL TASK-AGNOSTIC. It reads `wrong_before` -- the same partner-belief gap the
#other two terms read, computed by the same function -- and nothing else. No
#sparse reward, no shaped reward, no recipe, no order count, no PRIORITY. It does
#not know a steak from a plate; it knows the partner is wrong about that station
#and the robot is the one standing at it.
#
#It is also the term that INVERTS with the cone, which is what makes it FOV
#reasoning rather than partner modelling: at theta=360 the partner is wrong about
#almost nothing, the blind-spot set is nearly empty, and COVER goes quiet on its
#own. The probe measures exactly that -- gap_timed 1.84 at fov30 falling to 0.34
#at fov360.
#=========================================================================
W_COVER = 1.0

#The three contested station kinds. A dispenser or a counter is not contested, so
#it collapses to "" and can never match anybody's line.
CONTESTED = ("pot", "board", "sink")


def line_of(planner, subtask):
    """The contested station kind for `subtask`, or "" if it has none."""
    if not subtask:
        return ""
    #`a or b` -> use the human's wasted-trip table, fall back to their planner's
    #walk-to station. target_kind returns "" for explore/wait/check_*, so a human
    #merely looking around has line "" and can never match the robot's.
    return WASTED_IF_OCCUPIED.get(subtask) or planner.target_kind(subtask)


def robot_line(mdp, planner, state, robot_index=0):
    """The contested station kind the robot is engaging: what it CARRIES takes
    precedence, else the station it is FACING. "" if neither."""
    rp = state.players[robot_index]
    held = rp.held_object.name if rp.held_object else None

    if held:
        carried = line_of(planner, ROBOT_HELD_FETCH.get(held, ""))
        if carried:
            return carried
        #Test the RESULT, not whether the hands are full. ROBOT_HELD_FETCH knows
        #only meat/onion/plate/washed_plate, so a robot holding a STEAK gives "" --
        #and a robot carrying a steak while standing at the board IS engaging the
        #board. Returning early here would score that as engaging nothing.

    try:
        faced = (rp.position[0] + rp.orientation[0],
                 rp.position[1] + rp.orientation[1])
        #GUARD NEGATIVES EXPLICITLY. `except IndexError` does NOT cover this:
        #python list indices wrap, so terrain_mtx[-1][-1] quietly hands back the
        #terrain from the OPPOSITE CORNER of the kitchen instead of raising.
        if faced[0] < 0 or faced[1] < 0:
            return ""
        #the matrix is indexed [y][x], NOT [x][y]
        kind = TERRAIN_KIND.get(mdp.terrain_mtx[faced[1]][faced[0]], "")
        return kind if kind in CONTESTED else ""
    except Exception:
        return ""


def wrong_beliefs(shadow, state):
    """The SET of facts the human currently has wrong.

    A set rather than a count is what makes the credit term possible: to say
    which facts got FIXED this tick you have to compare the two collections.
    """
    wrong = set()
    for loc, bel in shadow.beliefs.items():
        #The try goes around ONE entry, with `continue`. Wrapping the whole loop
        #and returning set() on failure is not merely lossy, it is BACKWARDS:
        #the caller computes len(wrong_before - wrong_after), so an empty
        #wrong_after reads as "every single thing they had wrong just got fixed"
        #-- the largest possible credit, handed out because something threw.
        try:
            if loc == shadow.ROBOT:
                #1 - agent_index flips 0<->1: the OTHER player.
                rp = state.players[1 - shadow.agent_index]
                truth = rp.held_object.name if rp.held_object else "none"
            else:
                truth = shadow._true_state_of(state, loc)
        except Exception:
            continue
        #UNKNOWN is a plain string that never equals a real station state, so
        #"they have forgotten this" lands in `wrong` with no special case --
        #which is right: a forgotten fact is a fact they do not currently have.
        if bel.value != truth:
            wrong.add(loc)
    return wrong


def tick_score(mdp, planner, shadow, state, wrong_before, explored,
               robot_index=0):
    """credit - penalty for a single rollout tick. Higher is better.

        shadow        the hypothesis's human, AFTER it has acted this tick
        state         the world AFTER the joint move
        wrong_before  the set from wrong_beliefs, taken BEFORE the tick
        explored      True if the human fell back to exploring this tick
    """
    wrong_after = wrong_beliefs(shadow, state)
    #ONE-SIDED BY CONSTRUCTION: `wrong_after - wrong_before` is never computed,
    #so information LOSS can never cost anything. The asymmetry is enforced by
    #which subtraction is written, not by a rule somewhere else.
    credit = W_CREDIT * len(wrong_before - wrong_after)

    #COVER: is the robot engaging a station the partner is currently WRONG about?
    #`wrong_before` is the gap as it stood at the top of the tick, which is the
    #right reference -- it is what the partner did not know when the robot
    #committed to this move.
    mine = robot_line(mdp, planner, state, robot_index)
    cover = 0.0
    if mine:
        blind = [loc for loc in shadow.stations.get(mine, [])
                 if loc in wrong_before]
        if blind:
            cover = W_COVER

    if explored:
        #nothing to defer to -- an intention they cannot act on is not a plan
        #worth protecting
        penalty = 0
    else:
        #Use `_current`, what they are DOING this tick -- not `_sampled`, what
        #they would like to be doing.
        theirs = line_of(planner, shadow._current)
        #`and` short-circuits, so the cone test is only paid for when the lines
        #actually match. shadow.visible is the HUMAN'S OWN test -- perception is
        #never re-implemented here, we ask them.
        if theirs and theirs == mine:
            rp = state.players[robot_index]
            penalty = 0 if shadow.visible(state, rp.position) else 1
        else:
            penalty = 0

    return credit + cover - penalty, wrong_after
