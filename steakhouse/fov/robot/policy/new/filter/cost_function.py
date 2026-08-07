"""
THE COST FUNCTION.  Two things, both counted once per rollout tick.

    CREDIT   +1 for every fact the human GETS RIGHT that they had wrong.
             One-sided: a fact going the other way (right -> wrong) costs
             NOTHING. Informing your partner is a favour; failing to inform them
             is not a crime, because most of the time you are simply working.

    PENALTY  -1 for DOING THE EXACT SAME THING AS YOUR PARTNER WHILE THEY
             CANNOT SEE YOU. Redundant effort, spent in secret. Not for being
             near them, not for being busy -- for duplicating the specific work
             they are doing right now, unannounced.
             Skipped when they are LOST (the explore gate below).

    score(tick) = credit - penalty          summed over the rollout

===========================================================================
BOTH TERMS ARE DOMAIN-GENERAL.  ONLY THE DETECTOR IS NOT.
===========================================================================
Stated without a kitchen in sight, the two claims are:

    CREDIT   my partner's picture of the world got more accurate
    PENALTY  my partner and I are doing the same work, and they do not know it

Neither mentions steak, stations, or a recipe. Both are things you would want
from a teammate in any joint task, and both are things you could ask about a
real person.

What IS domain-specific is how you DETECT "the same work" in this kitchen: here
it comes out as "we are both engaged on the same station line", because in a
steakhouse the station is what makes two efforts redundant. In another domain
the detector changes and the claim does not. Keep that boundary clean -- the
detector lives in `line_of` / `robot_line` below and nowhere else, so porting
this cost function means rewriting those two functions and nothing else.

===========================================================================
TWO GOOD WAYS TO DUPLICATE, AND ONE BAD ONE
===========================================================================
    different work, they see me     +1 credit,  0 penalty   =  +1
    different work, hidden           0 credit,  0 penalty   =   0
    SAME work, they SEE me          +1 credit,  0 penalty   =  +1
    SAME work, HIDDEN                0 credit, -1 penalty   =  -1

If you end up doing the same thing they are, there are two acceptable outcomes:
LET THEM KNOW -- be where they can see you, and their own model makes them yield
(`_robot_redundant` fires and they re-sample onto something else) -- or DROP IT
and go do something else. Both are fine and neither is charged.

The single thing that costs anything is duplicating them while they have no
idea, because that is the one where BOTH of you spend the effort.

So the penalty carries a `not seen` condition; it is not cancelled by the credit
happening to arrive in the same tick. Those are different claims and only the
first one gives the right ordering.

===========================================================================
THE EXPLORE GATE, AND WHY IT MATTERS MORE THAN IT LOOKS
===========================================================================
A blind partner on `pickup_meat` who has never FOUND the meat station does not
walk there. `execute()` finds no known target, falls back to `_explore_action`,
and `action()` bumps `n_explore`. They hold a standing line and make no progress
on it.

Deferring to a plan that is going nowhere is worse than useless, so: if the
human explored this tick, no penalty. One condition, and it covers every version
of the problem -- no line at all, a line whose station they have never seen, a
line with no known route.

The payoff: against a blind partner almost every tick is an explore tick, so
every candidate action scores the same, they tie, and the tie breaks to the
baseline. The robot stops deferring and just cooks. The old package needed a
hand-built `defer` weight to get that; here it is one `if`.

===========================================================================
THE DETECTOR: WHAT COUNTS AS "THE SAME WORK" HERE -- AND THE TRAP IN IT
===========================================================================
Two efforts are redundant in this kitchen when they are aimed at the same
STATION -- if we are both filling the pot, one of us wasted the trip. So "same
work" is implemented as "same station line".

The obvious way to find that station is wrong. `planner.target_kind("pickup_meat")`
returns "meat" -- the MEAT DISPENSER, which is where they WALK. But two agents
at a dispenser are not duplicating anything; it is infinite and uncontested. The
station that makes `pickup_meat` redundant is the POT, because a full pot is what
makes the errand pointless.

The human model already has that mapping, in `WASTED_IF_OCCUPIED`:
    pickup_meat -> pot     pickup_onion -> board     pickup_plate -> sink
    drop_meat   -> pot     drop_onion   -> board     drop_plate   -> sink
and for the three subtasks it does not list (chop_onion, heat_washed_plate,
pickup_washed_plate) the walk-to station IS the contested one, so `target_kind`
covers them. Two of the human's own tables, one fallback between them, none of
our own.

===========================================================================
WHAT THIS READS, AND WHAT IT REFUSES TO
===========================================================================
READS   the partner's current subtask (inferred, never told), their beliefs,
        their own two station tables, and the true world.
REFUSES sparse reward, shaped reward, the recipe, the order count, the step
        budget -- and `PRIORITY`, the human's hand-written value ordering over
        subtasks, which is the recipe wearing a disguise.

The price of refusing PRIORITY, stated so it is not a surprise later: this cost
can tell that the partner's job got CONTESTED, and cannot tell that the partner
UPGRADED to a better job. Catching the upgrade needs values; values come from
preferences; that is the learned-weights version, later.
"""

# =========================================================================
# WHAT TO WRITE, ONE STEP PER LINE OF CODE.
#
# Same convention as baseline.py: each comment is one line of code, and the
# indentation of the comment is the indentation your line needs. Python uses
# indenting instead of { } to decide what sits inside what, so it is syntax,
# not style. 4 spaces per level, never tabs.
# =========================================================================

# ---- STEP 0: THE IMPORTS ------------------------------------------------
#`import X` loads a module and gives you the name X. `from X import Y` reaches
#inside and pulls out just Y, so you write Y instead of X.Y. Use the second when
#you only need one or two names.
#
#write these, in this order:
import baseline   # noqa: F401
#      not for anything inside it -- importing it RUNS the sys.path shim at the
#      top of that file, which is what makes every import below resolvable. It
#      must come first. The `# noqa: F401` tells linters "yes, unused, on
#      purpose"; without it they flag it and someone eventually deletes it.
#
from fov.human.agent.limited_vision_human import (
       WASTED_IF_OCCUPIED, ROBOT_HELD_FETCH, TERRAIN_KIND)

#=========================================================================
#WEIGHT ON THE CREDIT TERM.  This matters far more than a weight usually does.
#
#The score has NO term for "the robot got work done" -- by design, we read no
#reward. So working scores 0. Any POSITIVE term therefore outbids work
#automatically, because +1 > 0, and the module will walk away from an INTERACT
#to collect it.
#
#Measured on gc00 at W_CREDIT=1.0: 96% of all overrides were credit-driven, and
#21% of them threw away an INTERACT. Turning this down is what stops the module
#trading cooking for sightseeing. At small values the penalty dominates and
#credit survives only as a near-tiebreaker among actions the penalty cannot
#separate.
#=========================================================================
W_CREDIT = 1.0
#      ALL THREE, and only these three -- every name the steps below use.
#      Parentheses let one import span several lines.
#        WASTED_IF_OCCUPIED  subtask -> the station whose being non-empty
#                            wastes it        (step 1)
#        ROBOT_HELD_FETCH    held item -> the human fetch it makes redundant
#                                         (step 2)
#        TERRAIN_KIND        terrain letter -> station kind   (step 2)
#      NOT `UNKNOWN`: the prose mentions it, but no line of code below ever
#      names it -- step 3 compares belief to truth directly and UNKNOWN falls
#      out as "not equal" on its own. Importing a name you never use is how
#      files rot.


# ---- STEP 1: WHICH STATION LINE IS A SUBTASK ON -------------------------=
#just return the station the human wants to be on
def line_of(planner, subtask):
    """the contested station kind for `subtask`, or "" if it has none."""
    if not subtask:
        return ""
    
    #  * otherwise return
#        WASTED_IF_OCCUPIED.get(subtask) or planner.target_kind(subtask)
#      read it left to right:
#      - d.get(k) looks up k in dict d and hands back None if it is missing,
#        instead of crashing the way d[k] would.
#      - `a or b` evaluates to a when a is truthy, otherwise b. So this means
#        "use the human's wasted-trip table, and fall back to their planner's
#        walk-to station when that table has no entry".
#      - target_kind returns "" for anything it does not know (explore, wait,
#        check_*), so a human who is merely looking around has line "" and can
#        never match the robot's line. That is exactly what we want.

#WASTED_IF_OCCUPIED is a static dict of strings — subtask name → station kind. No state, no beliefs, no observations in it:
#{"pickup_meat": "pot", "drop_meat": "pot", "pickup_onion": "board", ...}
    return WASTED_IF_OCCUPIED.get(subtask) or planner.target_kind(subtask)



# ---- STEP 2: WHICH STATION LINE IS THE *ROBOT* ON -----------------------
def robot_line(mdp, planner, state, robot_index=0):
    
#docstring: "the contested station kind the robot is engaging: what it is
#carrying takes precedence, else the station it is facing. "" if neither."

#body:
#
#  * get the robot's player object:
#        rp = state.players[robot_index]
#    `state.players` is a list; [0] is the robot, [1] is the human. Square
#    brackets index a list. Using the named constant instead of a bare 0 means
#    the day the indices swap you edit one line.
    rp = state.players[robot_index]
    
#  * work out what it is holding, as a plain string or None:
#        held = rp.held_object.name if rp.held_object else None
#    `A if test else B` is a conditional EXPRESSION -- it evaluates to A or B,
#    so it can sit on the right of an `=`. It is not the same as an `if`
#    statement. Here it guards against rp.held_object being None: reading
#    `.name` on None would crash.
    held = rp.held_object.name if rp.held_object else None
    
    
#  * if what it is holding maps to a line, THAT is the line. The human's own
#    ROBOT_HELD_FETCH maps a held item to the human FETCH it makes redundant,
#    and line_of() then maps that fetch to its contested station:
#        carried = line_of(planner, ROBOT_HELD_FETCH.get(held, ""))
#        if carried:
#            return carried
#    carrying meat -> "pickup_meat" -> "pot". Two of their tables composed;
#    still nothing of ours.
    if held:
        carried = line_of(planner, ROBOT_HELD_FETCH.get(held, ""))
        if carried:
            return carried
        
#    ROBOT_HELD_FETCH only knows four items -- meat, onion, plate, washed_plate.
#    A robot holding a STEAK, a GARNISH or a DISH is not in it, so the lookup
#    gives "" and the one-line version would return "" and never reach the
#    faced-cell check below. A robot carrying a steak while standing at the
#    board IS engaging the board, and that version would score it as engaging
#    nothing. Test the RESULT, not whether the hands are full.
    try:
    #  * otherwise fall back to the cell it is FACING. A player has a position and
    #    an orientation, both (x, y) pairs; the cell in front is their sum:
        #faced x and y coords
        faced = (rp.position[0] + rp.orientation[0],
                rp.position[1] + rp.orientation[1])

        #GUARD THE NEGATIVES EXPLICITLY. `except IndexError` does NOT cover
        #this: python list indices wrap around, so terrain_mtx[-1][-1] does not
        #raise -- it quietly hands back the terrain from the OPPOSITE CORNER of
        #the kitchen. A robot facing off the top or left edge would be scored as
        #engaging whatever station happens to sit at the far end of the map, and
        #nothing would ever tell you. Test before you index.
        if faced[0] < 0 or faced[1] < 0:
            return ""

    #  * turn that cell into a station kind and return it. The terrain matrix is
    #    indexed [y][x] -- NOT [x][y]. That is the single most common bug in this
    #    codebase; the human model has a comment about it too:
        terrain = mdp.terrain_mtx[faced[1]][faced[0]]

    #    then map the terrain letter to a kind. TERRAIN_KIND in the human model
    #    already does it ('P'->pot, 'B'->board, 'W'->sink, ...), so import it and:
        kind = TERRAIN_KIND.get(terrain, "")

    #    finally return kind if it is one of the three contested kinds, else "":
    #        return kind if kind in ("pot", "board", "sink") else ""
        return kind if kind in ("pot", "board", "sink") else ""
    #    `x in (a, b, c)` tests membership in a tuple. Dispensers and counters are
    #    not contested, so they collapse to "" and never match anybody's line.
    #  * `except Exception`, not `except IndexError`: off the right/bottom edge
    #    raises IndexError, but a malformed coordinate raises TypeError and the
    #    whole rollout should not die for a cosmetic lookup either way.
    except Exception:
        return ""

# ---- STEP 3: THE KNOWLEDGE-BASE GAP -------------------------------------
def wrong_beliefs(shadow, state):
#Returning a SET rather than a count is what makes the credit term possible: to
#say "which facts got FIXED this tick" you need to compare the two collections,
#not two numbers.
#
#body:
#
#  * start an empty set:
    wrong = set()

#  * loop over every belief the human holds:
#        for loc, bel in shadow.beliefs.items():
#    `.items()` on a dict yields (key, value) pairs, and naming two variables in
#    the for-line unpacks each pair automatically. 
# `loc` is a cell coordinate or
#    the special ROBOT key; 
# `bel` in and of itself is a Belief(value, seen_at) namedtuple, so
#    `bel.value` is what they think is there.
    #`beliefs`, PLURAL -- limited_vision_human.py:447 is `self.beliefs = {}`.
    #`shadow.belief` (singular) is an AttributeError.
    for loc, bel in shadow.beliefs.items():
        #THE try GOES HERE, AROUND ONE ENTRY, WITH `continue`.
        #Wrapping the whole loop and returning set() on failure is not merely
        #lossy, it is BACKWARDS: the caller computes
        #     credit = len(wrong_before - wrong_after)
        #so an empty wrong_after reads as "every single thing they had wrong
        #just got fixed" -- the largest possible credit, handed out because
        #something threw. One bad entry would become a jackpot.
        try:
    #      * the teammate entry is stored under a special key and needs the truth
    #        looked up differently from a station. Handle it first:
    #        `1 - agent_index` flips 0<->1: the OTHER player. Cute, and it means the
    #        line is correct whichever index the human was built with.
            if loc == shadow.ROBOT:
                rp = state.players[1 - shadow.agent_index]
                truth = rp.held_object.name if rp.held_object else "none"

    #      * otherwise it is a station cell, and the human's own method reads it:
    #        `_true_state_of` is a pure function of (mdp, state, loc) -- the shadow
    #        only supplies the mdp and the station timings, so which shadow you ask
    #        does not matter.
            else:
                #LEADING UNDERSCORE. The method is `_true_state_of`
                #(limited_vision_human.py:527); `true_state_of` does not exist.
                truth = shadow._true_state_of(state, loc)
        except Exception:
            #skip this one entry and keep the rest of the set intact
            continue
    #
    #      * compare, and record the disagreements:
    #        `!=` is "not equal". `.add` puts one item in a set. Note UNKNOWN is a
    #        plain string that never equals a real station state, so "they have
    #        forgotten this" lands in `wrong` with no special case -- which is right:
    #        a forgotten fact is a fact they do not currently have.
        if bel.value != truth:
            wrong.add(loc)

    #NOTE the indentation: this is OUTSIDE the for, at the function's level. One
    #extra indent and you would return after the first belief.
    return wrong


# ---- STEP 4: THE TWO NUMBERS FOR ONE TICK -------------------------------
def tick_score(mdp, planner, shadow, state, wrong_before, explored,
               robot_index=0):
    
#docstring: "credit - penalty for a single rollout tick. Higher is better."

#the arguments, and why each is needed:
#    shadow        the hypothesis's human, AFTER it has acted this tick
#    state         the world AFTER the joint move
#    wrong_before  the set from step 3, taken BEFORE the tick
#    explored      True if the human fell back to exploring this tick

#body:
#
#  * CREDIT. Work out what they now have wrong, and count what got fixed:
#    `a - b` on two sets is SET DIFFERENCE: everything in a that is not in b.
#    So this is exactly "facts they had wrong and now have right". `len()` counts
#    them.
#    ONE-SIDED BY CONSTRUCTION: `wrong_after - wrong_before` (things that went
#    the other way) is never computed, so information LOSS can never cost
#    anything. That is the asymmetry we agreed on, and it is enforced by which
#    subtraction you write rather than by a rule somewhere else.
    wrong_after = wrong_beliefs(shadow, state)
    credit = W_CREDIT * len(wrong_before - wrong_after)
    #W_CREDIT: how loudly informing your partner competes with DOING YOUR JOB.
    #This matters more than it looks. The score has no term for "the robot got
    #work done" -- by design, we read no reward -- so work scores 0. Any
    #POSITIVE term therefore outbids work automatically, because +1 > 0. At
    #W_CREDIT=1.0, measured on gc00: 96% of all overrides were credit-driven and
    #21% of them threw away an INTERACT. Turning this down is how you stop the
    #module trading cooking for sightseeing.
    
#  * PENALTY. The claim being tested is "I am doing the EXACT SAME THING as my
#    partner and they cannot see me." THREE conditions, all must hold. Doing the
#    same thing is not enough on its own -- there are two fine ways to duplicate
#    someone and only one bad one:
#        they can see you    -> they yield on their own (_robot_redundant fires
#                               and they re-sample). You told them. Fine.
#        you drop it         -> also fine.
#        they cannot see you -> THE ONLY BAD CASE. Both of you spend the effort
#                               and one of the two trips was for nothing.
#
#    so, in order:
#    if they are lost, there is nothing to defer to:
    if explored:
        penalty = 0
#    otherwise compare the two lines AND check whether they can see you:
    else:
        #    `theirs and theirs == mine and not seen` is three tests chained with `and`,
        #    which is True only w`hen ALL of them are. `and` also SHORT-CIRCUITS -- it
        #    stops at the first False, so `shadow.visible(...)` is only paid for when
        #    the lines actually match. Free speed, and the reason to put the cheap test
        #    first.

        #    `shadow.visible(state, loc)` is the HUMAN'S OWN cone test, called on the
        #    robot's cell. We do not re-implement perception anywhere; we ask them.
        #
        #    Use `_current`, what they are DOING this tick -- not `_sampled`, what they
        #    would like to be doing. Your question about the blind partner is why: an
        #    intention they cannot act on is not a plan worth protecting.

        #NOTE the DOT. `shadow,_current` (comma) builds a two-item TUPLE out of
        #the shadow and a name `_current` that does not exist here -- NameError.
        #A comma where a dot belongs is legal python, which is why it stings.
        theirs = line_of(planner, shadow._current)
        mine = robot_line(mdp, planner, state, robot_index)
        rp = state.players[robot_index]
        seen = shadow.visible(state, rp.position)
        penalty = 1 if (theirs and theirs == mine and not seen) else 0
        


#  * return both, and the new set, so the caller can carry it to the next tick:
#        return credit - penalty, wrong_after
#    returning two values makes a tuple; the caller writes
#        score, wrong_before = tick_score(...)
#    and the loop threads `wrong_before` forward one tick at a time.
    return credit - penalty, wrong_after