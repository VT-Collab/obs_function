"""The limited-vision human. Everything it knows, it had to walk over and look at.

ONE SENTENCE. Every tick it observes through a cone, asks common/tasks.py what
its BELIEFS say is legal and reachable, takes the best of those on the ladder,
and walks one step of an A* laid over ground it has actually seen. Nothing is
planned ahead and nothing is carried over except a small amount of stickiness,
so "sees something better and switches" is the default path rather than a
special case. On top of that ladder it will also go and LOOK for whatever it
last saw the robot carrying -- see THE LOOK-FOR below, which is half this file.

--------------------------------------------------------------------------------
ONE FILE, ONE CLASS, AND WHY THAT IS A CORRECTNESS PROPERTY
--------------------------------------------------------------------------------
The look-for used to be a subclass in human/look_for_human.py. In the user's
words, the reason it is not any more:

    "change the human class to merge into one human class that includes the look
    for (one file, one class), don't make it separate, so the downstream forecast
    that calls it automatically includes everything."

That is not a tidiness preference, it is the fix for a bug that two classes make
possible and one class makes UNREPRESENTABLE. The robot's filter predicts the
human by building SHADOW humans, one per hypothesised cone, and rolling a clone
of one of them forward. While there were two classes, every one of those
construction sites had to be TOLD which class the seat actually held -- and the
default was the base class. So the seat held the look-for human, the shadows were
plain ladder humans, and the posterior scored hypotheses against actions the real
human would never take while the forecast predicted a cone belonging to nobody.
The `human_cls` parameter that used to be threaded through robot/methods.py and
FOVPosterior existed for no other reason and is gone with the subclass. `human_kw`
stays, because the shadows still have to match the real human's forget_horizon
and its enable_look_for setting -- otherwise --no-look-for would be predicted by
a look-for shadow, which is the same bug wearing a flag instead of a class.

enable_look_for=False is the A/B control and reproduces the plain ladder agent
ACTION FOR ACTION -- the agent as it stood before any of the look-for existed.
It is what the harnesses expose as --no-look-for. That equality was checked at
the merge, over 60 md5 action-trace hashes ({on, off} x 5 layouts x 3 cones x 2
seeds x 150 ticks) recorded against the two-class code and re-run against this
one: all 60 identical, the 30 "off" ones being exactly this claim and the 30
"on" ones saying the merged class is the old subclass to the action. The first
section of human/tests/test_look_for_human.py keeps the half of that which stays
testable once the second class is gone -- that with the feature off, none of the
machinery below can reach the trace.

WHAT THE MERGE DID NOT FIX BY ITSELF, and had to be chased: the clone helpers in
robot/filter/core/qmdp.py (since deleted -- see robot/methods.py's module
docstring) named the base class and copied its fields one by one, so they were
the site of the very downgrade described above. With one class the downgrade
stopped being silent and became an AttributeError on the shadow's first
observe(), which is how it was found. Both helpers now take their class
from type(h) and copy anything they do not name. (Historical: the now-deleted
QMDPFilter._hkey, which digested "everything that makes one shadow behave
differently", grew its sighting book for the same reason -- without it that
filter's search would dedupe two futures that differ only in what the human
remembers seeing. my_fov_filter.py has no equivalent dedupe step.)

WHY IT EXISTS AT ALL. overcooked_ai_py's SteakLimitVisionHumanModel cannot
express FOV-driven behaviour, for three independent reasons: update()
(agent.py:1231) only ever ADDS observations, so a 20-degree human that glimpsed
the pot once knows it forever; init_knowledge_base (agent.py:1075) copies the
whole start state in unfiltered; and in_bound (agent.py:882) returns True for
any tile beside the player, so the station you are standing at is legible
however narrow the cone. Measured across 768 random layouts: 642 showed exactly
zero FOV effect on subtask choice, the mean was 0.1, and what little appeared
was timing jitter. Everything below exists to put an OBSERVATION COST back in --
what the agent knows, it spent steps looking at, and how many steps that takes
is set by the cone.

The two properties this is built to have:

  (A) THE CONE CHANGES THE SUBTASK SEQUENCE, substantially and attributably.
      That is the signal a downstream FOV-aware robot has to infer.
  (B) THE HUMAN IS A FUNCTIONAL TEAMMATE AT EVERY CONE. It never deadlocks and
      never starves; a narrow cone pays in walking, not in standing still.

Both come from the same three channels, in measured order of contribution:

  DISCOVERY   it starts knowing nothing, not even where the stations ARE. A
              station never seen is not in view.stations(), so no subtask can
              name it. Strongest channel, and it degrades smoothly with the cone.
  ROUTING     A* runs over view.walkable, which is only floor that has been
              seen. A corridor never looked at cannot be planned through, so a
              narrow cone knows fewer routes and its paths are longer or do not
              exist yet.
  STALENESS   contents beliefs expire after FORGET_HORIZON. A wide cone keeps
              stations current incidentally; a narrow one goes stale and pays a
              round trip to find out it guessed wrong. See the OPTIMISM note in
              common/views.py: pessimism looked like a different bug entirely.

--------------------------------------------------------------------------------
PERCEPTION -- what gets written into the belief, and what deliberately does not
--------------------------------------------------------------------------------
visible(cell) = in_cone(cell) AND los_clear(self, cell), with '#' the ONLY thing
that blocks sight. Counters, pots, sinks, boards and dispensers are all
see-through, so occlusion is a property of the LAYOUT rather than of where the
furniture sits -- that is the knob the layout suite turns. There is NO radius:
the cone runs to the edge of the map, and at fov 360 the cone test is vacuous
but line of sight is not. Omnidirectional is never omniscient.

FELT TERRAIN is the one exception, and it is a narrow one. The four tiles the
agent is touching are known as TERRAIN ONLY -- floor or furniture, nothing about
what is on them and NOT onto the station list, so a pot brushed past but never
looked at stays undiscovered and no subtask can target it. Without this a 30
cone could not plan a single step sideways, because unseen ground is not
walkable, and the agent span on the spot instead of exploring. The line this must
not cross is "adjacency is free", which is exactly what made the stock model
FOV-blind: feeling the floor buys mobility and none of that.

--------------------------------------------------------------------------------
MEMORY -- BeliefView, and why the robot belief has two clocks
--------------------------------------------------------------------------------
Terrain and station locations are PERMANENT: walls do not move and a station once
found is never un-found. CONTENTS decay to unknown after FORGET_HORIZON, counters
included -- a counter not looked at recently is genuinely unknown, not assumed
empty. Counters are where a handoff lands, so "what is on the worktops" is
perishable, decision-relevant knowledge that a wide cone keeps for free.

The belief about the robot is SPLIT ACROSS TWO CLOCKS because its two halves are
falsified by different things. Looking at the cell they were in and finding it
empty refutes their POSITION on the spot -- and must, or a confidently wrong
position sits there until FORGET_HORIZON and misroutes the contention check. It
says nothing whatever about their HANDS, which only ever expire with time. So
the human can be in the perfectly reasonable state of knowing the robot is
carrying meat while having no idea where it went.

--------------------------------------------------------------------------------
THE LADDER -- flat, one level, and now reachability-aware
--------------------------------------------------------------------------------
A subtask is (verb, target cell). No hierarchy, no forced/chosen split, one
choice per tick. Every subtask has the same shape -- A* to the target, one
INTERACT on arrival -- and tiers rank them by WHAT THAT ARRIVAL INTERACT
PRODUCES, never by how far the walk is. Distance is only the tie-break inside a
tier. The tiers live in common/tasks.py; briefly:

  T1 DELIVER   T2 COMPLETE (yields a dish)   T3 HALF (yields steak_dish or
  garnish_dish)   T4 COLLECT   T5 WORK (chop/wash)   T6 START (load a station)
  T7 FETCH   T8 STASH   T9 EXPLORE

Cross-tier preemption is instant and unconditional: see a finished dish, know the
hatch, and a half-washed plate is abandoned on the spot. That needs no special
case, it is just "T1 outranks T5".

REACHABILITY IS PART OF LEGALITY, not a filter applied afterwards.
legal_subtasks() takes an `ok(cell)` predicate and rank() answers it from the
agent's own belief map -- "unreachable" means "I know of no route", which is the
honest thing for someone who has not yet seen the wall. Judging legality without
it left a human holding a plate, with the only sink walled off in the other room,
with a permanently "legal" load_sink: it never fell through to stash and walked
in circles for the rest of the episode. Filtering at the source instead means the
plate goes down on the pass and the robot picks it up -- the handoff the layouts
exist to force, happening by itself.

THE STASH LOOP is the one real hazard the ladder has to be built around. Hold a
steak_dish with no garnish believed anywhere; T2 is empty, so stash it; now
empty-handed, and "pick up a steak_dish" is legal, so pick it straight back up,
forever. actionable() kills it: a pickup is only offered if the agent believes in
a way to ADVANCE that item. Same rule kills every symmetric variant, and it is
what makes the ladder terminating rather than merely ordered.

--------------------------------------------------------------------------------
THE DECISION -- two passes, then stash, then stickiness
--------------------------------------------------------------------------------
rank() asks for the ladder up to three times, and the order matters:

  1. STRICT. optimistic off: only stations whose contents are actually known.
  2. OPTIMISTIC, only if strict came back empty. A station found but whose status
     has expired is offered as available. A guess must never outrank knowledge --
     on equal footing a stale sink reads as ready, collect_plate (T4) outranks
     get_plate (T7), and the human shuttles between them forever. Seen on
     banquet_pass at fov 30 as a pick-plate / collect-plate shuffle in which no
     plate was ever picked up.
  3. STASH, only if both came back empty. Offering it earlier makes the strict
     pass non-empty, so the optimistic pass never runs, and a human holding a
     garnish two tiles from a stale-but-ready sink put it down and picked it up
     for 136 of 300 ticks on banquet_pass.

STICKY WITHIN A TIER, and only within a tier. Walking changes which of two
equally good targets is nearer, so a pure argmax swaps between them every few
steps and the agent paces on the spot -- measured at about 30 target flips per
100 ticks before this. Preferring what we were already doing among EQUALS costs
nothing and is not a switch margin: cross-tier preemption is untouched.

--------------------------------------------------------------------------------
HOW THE ROBOT INFLUENCES THE HUMAN -- two soft channels, both FOV-gated
--------------------------------------------------------------------------------
CHANNEL A, SEEING ITS HANDS. If the robot is currently BELIEVED to be holding
meat/onion/plate, the matching fetch is demoted WITHIN its tier -- last, never
gone. Softness is the whole fix, not a nicety: the old agent tried a hard
redirect (pot empty + robot seen carrying meat -> chop instead) and it INVERTED
the gradient. Both agents deferred the meat to each other, neither cooked, and
the better-sighted human starved; fov60 and fov180 delivered ZERO on
steak_island from this alone. Reacting to a fait accompli in the robot's hands is
safe; predicting what it is ABOUT to pick up is what broke.

CHANNEL B, SEEING A DROP. The robot puts something on a counter inside the cone,
the contents belief updates that tick, a higher tier becomes legal, and the
ladder switches. No glancing, no scanning, no new machinery -- peripheral vision
plus instant re-decision is the entire mechanism. This is where cone width
becomes leverage the robot can USE: a 360 human notices a drop on any counter it
has line of sight to, a 30 human only on what it is pointed at, so the robot's
CHOICE OF COUNTER is a decision that depends on theta. A cone-blind robot drops
into the blind spot and the handoff rots until FORGET_HORIZON.

CONTENTION is the third reaction and it is NOT collision handling. If the robot
is SEEN to be closer to a station than we are, that subtask is demoted within its
tier so the two of us spread out instead of queueing. It can only ever fire on a
station reachable from BOTH rooms -- one embedded in the dividing wall, worked
from opposite sides. FOV-gated on purpose: a narrow cone does not know where the
robot is, contends more, and coordinates worse. That is the effect, not a bug.

The human never deliberately watches, tracks or positions itself relative to the
robot. Influence is entirely incidental: the robot walks into view, or it does
not.

--------------------------------------------------------------------------------
WHAT IS DELIBERATELY NOT HERE: ANY COLLISION HANDLING
--------------------------------------------------------------------------------
No yielding, no right-of-way, no sidestep, no unstuck counter, and the partner is
not an obstacle -- step_towards() plans straight through their tile. Every layout
in layout/layouts is two rooms joined only by pass-through counters, so the two
agents have NO SHARED FLOOR: they were measured adjacent on 0 of 4800 ticks. All
of that machinery fired zero times and was deleted rather than left as
decoration. The one loop-breaker that remains, in _explore(), breaks a loop with
OURSELVES; there is nobody else in the room to collide with.

Also gone: the softmax over subtasks, the temperature, and the closed-form
subtask_distribution() the old agent exposed. The ladder is an argmax with
tie-breaks, so the "distribution" a robot could read off it was a delta anyway.

================================================================================
THE LOOK-FOR -- going and finding what was last seen in the robot's hands
================================================================================
ONE SENTENCE. When the robot walks through the cone carrying something, this
human remembers WHERE it was and WHAT it held, and if that thing is worth more
than whatever the ladder is currently offering, it walks over to that area and
sweeps the counters there until it finds it, puts it down as somebody else's
problem, or the sighting goes stale.

WHY THIS IS NOT A NEW TIER. The obvious implementation gives "look for X" its
own rung and then spends the rest of the file arguing about where the rung goes.
It does not need one. The ladder ALREADY prices every object: the tier of
take_<X> off a counter is exactly the answer to "how much is having an X worth
to me right now", because that is what that number means. So a look-for borrows
it. A look-for-meat is worth what a meat is worth (T7 FETCH), a look-for-dish is
worth what a dish is worth (T2 COMPLETE), and the question "should I abandon
what I am doing to go and find it" becomes a comparison of two numbers that were
already there. No new ordering to tune, no new failure mode where the new rung
is one notch too high and eats the whole episode.

PREEMPTION IS STRICT -- the look-for must be STRICTLY better, not merely equal.
That single choice is what makes the first thing the design asks for come out
right with no rule of its own: a look-for-meat (T7) can never displace an
ordinary get_meat from a known dispenser (also T7), so the human only ever goes
HUNTING for a meat when it cannot simply go and FETCH one. Equally, while the
human is working the onion chain (T4 collect_garnish / T5 chop / T6 load_board /
T7 get_onion) a T7 look-for never fires; the moment that chain is exhausted and
the ladder falls to stash (T8) or explore (T9), it does. "Finish the onions,
then go and look for the meat" is not implemented anywhere in this file. It is
what strict comparison of borrowed tiers already says.

--------------------------------------------------------------------------------
THE MEMORY, AND WHY IT CANNOT LIVE IN THE BELIEF
--------------------------------------------------------------------------------
BeliefView.robot is (pos, orient, held, t) on TWO CLOCKS, and the fast half is
destroyed by precisely the event this feature is about. views.py:190-200: the
moment the human looks at the cell it last saw the robot in and the robot is not
there, POSITION goes to None on the spot. That is correct for what it was built
for -- a confidently wrong position misroutes the contention check -- but it
deletes the only copy of "the robot was over THERE with a meat", which is the
whole datum a look-for needs. The hands survive, so the belief can tell you WHAT
but never again WHERE.

So the agent keeps its own record, `sightings: item -> (robot_pos, tick)`,
written in observe() at the instant of sighting and expiring on its OWN clock.
LOOK_HORIZON is deliberately 2x FORGET_HORIZON: the hands belief dies at 30
(views.py:201) and this memory has to outlive it or the feature only ever works
while the belief that made it redundant is still alive.

--------------------------------------------------------------------------------
ARRIVAL IS NOT AN INTERACT, AND THAT IS THE POINT
--------------------------------------------------------------------------------
A look-for NEVER returns INTERACT. Its entire payoff is geometric: step_towards
turns the agent to face the counter, the next observe() folds that counter's
contents into the belief, and then ONE OF TWO THINGS happens, both of which are
existing machinery:

  FOUND     counters_holding(X) goes non-empty, suppression S2 withdraws the
            look-for, and the ORDINARY ladder offers take_X at the very tier the
            look-for borrowed. The human walks on to it without a visible switch.
            This is Channel B from the perception half above, reused rather than
            reinvented.
  NOT THERE the counter joins a per-sighting CLEARED set and the sweep moves to
            the next one.

There is therefore no new arrival semantics anywhere in this file, and no way to
get one wrong.

--------------------------------------------------------------------------------
A COUNTER IS ONLY CLEARED ONCE THE ROBOT HAS LET GO. THIS IS THE WHOLE FEATURE
--------------------------------------------------------------------------------
The sought item is, BY DEFINITION AT THE MOMENT OF SIGHTING, in the robot's
hands. So no counter holds it, and every counter the cone can currently see
truthfully reports "not this". The first version of RESOLVE below added all of
them to the cleared set on the spot, the candidate set came back empty on the
very tick the sighting was written, and S5 deleted the sighting -- the better the
human could see, the more certainly the sighting died before it was ever used.
Measured over 50 episodes of the rollout sweep: 264 sightings written, 140
destroyed by S5, 85 of those on their BIRTH tick, 117 of them while the robot was
still believed to be holding the very item, and on average 7.5 of the 11.2 known
counters were already in the cleared set at the instant of birth. The feature was
a tick cost with an 18.9% bout payoff rate, most of which was coincidence.

Two conditions now gate a clearing, and each closes one half of that:

  AT OR AFTER THE SIGHTING. The counter's contents record must be timestamped at
  or after the tick the sighting was armed. A counter looked at ten ticks BEFORE
  the robot walked past with a meat says nothing about where the meat went, and
  the record can easily still be fresh (FORGET_HORIZON is 30) when it is asked.

  AND THE ROBOT HAS LET GO. While view.robot[2] is still the sought item, looking
  at an empty counter beside the robot proves only that the robot has not put it
  down yet. So nothing clears, the candidate set stays full, and the human walks
  to the counter nearest the remembered position and WAITS THERE -- which is the
  behaviour that was asked for in the first place. It is not a timer: the hands
  half of the robot belief expires on views.py:201's own 30-tick clock, and the
  instant it does the sweep resumes with 30 ticks of LOOK_HORIZON left to run.

RE-ARMING is the other half. Seeing the robot pick the thing up again, or carry
it somewhere else, invalidates every conclusion the sweep has reached, so the
cleared set is emptied and the horizon restarts. Re-arming is deliberately NOT
"the robot is in the cone holding it again this tick": a robot that simply
lingers at the same cell must not renew the horizon every tick, or LOOK_HORIZON
stops being a bound at all and the human can stand at the pass for the whole
episode. Armed on a CHANGE -- new item, new cell, or hands we had stopped
believing in -- and expiring from the arming tick, so LOOK_HORIZON is a hard
budget for one sighting rather than a sliding window.

TERMINATION IS STRUCTURAL, not hoped for. Three independent retirements, all of
them in observe() where the evidence arrives: the item turns up in our own hands
(the success case), LOOK_HORIZON runs out from the arming tick, or the sweep runs
out of unswept, in-radius, reachable counters AFTER the robot has let go. The
cleared set IGNORES belief decay -- it is a record of where the sweep has been,
not a belief about what is on those counters -- so a candidate the ladder
un-knows does not come back and the set only ever grows. A sighting with nothing
left to look at is now SKIPPED rather than deleted on the spot, which the old
code could not afford: with the set no longer poisoned at birth, the only way a
new candidate appears is the human DISCOVERING a counter it had never seen, and
that is real new information worth sweeping.

--------------------------------------------------------------------------------
THE DROP, AND WHY IT IS NOT A TWO-PHASE PLAN
--------------------------------------------------------------------------------
This agent has no plan and one action per tick, so "put down what you are
holding, then go and look" is not scripted. It is the same derivation re-run
every tick: while the hand is full the look-for resolves to a synthesized T_STASH
at the nearest free counter; the tick after that INTERACT lands, held is None,
the identical derivation resolves to the walk. Nothing crosses that boundary
except two commitments, and a T1/T2 subtask appearing mid-drop preempts instantly
because base_tier is recomputed from scratch every tick -- the human simply
abandons the drop, still holding the item, and no un-commit logic is needed.

THE STASH-AND-TAKE-IT-BACK LOOP CANNOT COME BACK, and the reason is a property of
the ladder rather than a guard here. For all eight items, actionable(X) is
exactly the disjunction of the gates on X's held-branch verbs, and the
empty-handed take tier is uniformly one or more tiers WORSE than the in-hand tier
for the same item (dish 1->2, steak_dish 2->3, garnish_dish 2->3, washed_plate
3->4, garnish 2or3->4, raw 6->7 -- derived by executing legal_subtasks, see the
tests). So if the look-for beat the in-hand tier it also beats the after-drop
take tier; and if the in-hand verb was illegal, the after-drop take is illegal
too, by the identical gate. Putting something down can never make the ladder
offer it straight back at a tier that beats the look-for that just outranked it.

THE DROP MUST STILL BE WORTH MAKING, though, and that is a SEPARATE test which
the argument above does not give you. It is about the item in our hands; it says
nothing about the rest of the empty-handed ladder, which offers whole verbs the
in-hand branch cannot -- every dispenser and every counter pickup in the kitchen.
Beating the in-hand tier therefore only earns the right to put the thing down,
and the human walked to a counter, stashed, and was immediately outranked by a
get_<raw> at the very tier the look-for had: 30 of 45 drop legs ended in no
look-for at all, a wasted round trip each. So a look-for that needs a drop must
ALSO strictly beat the tier the ladder will offer once the hand is free
(_after_drop_tier). Same strictness, same reason, and it took the drop legs from
45 to 5 with none of them wasted.

--------------------------------------------------------------------------------
THE ONE PLACE THE DERIVATION NEEDS HELP: THE LOOK-AHEAD GATE
--------------------------------------------------------------------------------
"Robot is carrying a garnish_dish, drop the onion and go and find it" is right
when a steak is COMING, and actionable("garnish_dish") is some(ready(POT)) --
which is False for a pot that is merely COOKING (verified by execution). But a
look-for is a multi-tick walk, so the honest question is not "is this usable this
instant" but "will it be usable when I arrive", and a pot mid-timer will have
finished. LookaheadView relaxes ready(ch) to ready + in_progress for the duration
of the gate test and nothing else. That single line is what carries that example;
with lookahead_gate=False it does not fire, which the tests pin down so nobody
deletes the relaxation later thinking it is decoration.

THE HOLE, STATED PLAINLY: pot empty and nothing cooking, and there is no
look-for for a garnish_dish. That is the derivation being right rather than a
bug -- a garnish_dish with no steak on the way is unusable and the correct move
is to go and get meat, which is what the ladder already does. It is also exactly
what makes "robot has a garnish_dish but I wanted to do meat -> do meat" come out
right for free in a single-pot layout: wanting to do meat MEANS believing a pot
is empty, an empty pot is neither ready nor in progress, so the gate fails and
the tier is never even computed. gate_composites=False exempts the three
composites from the gate if you want the literal behaviour in that world too; it
is loop-safe (the pickup is still gated by actionable(), so the human walks over,
sees it, and cannot take it) but it buys a wasted trip.

OPTIMISM PARITY. actionable() reads ready/empty, which BeliefView answers through
_maybe() under `optimistic`. rank() leaves optimistic=True on exit whichever pass
actually won, so evaluating the gate naively lets a stale pot read as BOTH empty
and ready -- a guess outranking knowledge, which is the failure views.py:239-254
already documents. The gate is therefore evaluated under the SAME setting that
produced the winning ordinary ranking, recovered with one extra strict
legal_subtasks call rather than by duplicating rank()'s body.

--------------------------------------------------------------------------------
TARGETING IS L1 FROM THE REMEMBERED POSITION, NOT A PATH
--------------------------------------------------------------------------------
"That area" is scored by L1 distance from the remembered robot position, and this
is not laziness. Every layout in this suite is two rooms joined only by
pass-through counters, so the robot's room is NOT reachable over the human's
believed floor: a BFS from the remembered position returns None for every counter
and the metric would be vacuous. L1 crosses the dividing wall, which is exactly
the geometry the pass counters sit in. Only the COUNTERS have to be reachable by
us, and that is tested separately.

Both the look target and the drop target are COMMITTED once chosen. L1 to a fixed
remembered position does not move as we walk, but the walk-distance tie-break
does, and re-argmin-ing a moving tie-break is the documented pacing bug -- ~30
target flips per 100 ticks, see THE DECISION above.

--------------------------------------------------------------------------------
rank() AND decide() ARE PURE QUERIES OF THE CURRENT VIEW, AND MUST STAY THAT WAY
--------------------------------------------------------------------------------
Both are questions you may ask at any time -- the filter's shadow rollouts, its
likelihood terms and every probe in the test suite ask "what would this human do"
without stepping it. Two things broke that contract when the look-for arrived and
both are fixed rather than worked around.

THE MEMO NEEDED A BELIEF VERSION. rank() is memoised on (t, pos, held), and
observe() -- the only thing that ever changes the belief -- did not invalidate
it. self.t is incremented by action(), not by observe(), so a caller that asked
decide() BEFORE action() in the same tick filled the memo from LAST tick's
belief and action()'s own decide() then hit it: the agent acted on a belief one
tick stale, silently, only when someone else looked at it. Measured with
scratchpad/repro_memo.py on banquet_pass/fov90: without the memo the answer
changes across observe() on 18 of 200 ticks, with the t-keyed memo it changed on
1. The memo is a real speedup and stays; every memo key now carries
`_belief_version`, a counter bumped by observe() and by nothing else, so a stale
entry cannot be reached.

DECIDE() DOES NOT EDIT THE SIGHTING BOOK. Retiring a swept-out sighting used to
happen inside _scan, i.e. as a side effect of being ASKED what to do -- so a
shadow rollout that queried the human deleted the real human's memory. All three
retirements live in observe(). The committed look and drop targets are still
chosen by the same derivation, but _choose() RETURNS them and action() -- the one
method that actually takes the tick -- writes them, exactly as last_subtask is
written there and nowhere else.
"""
# =============================================================================
# IMPLEMENTATION
# =============================================================================
import os
import random
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overcooked_ai_py.mdp.actions import Action              # noqa: E402
from common import geometry as geo                            # noqa: E402
from common.tasks import (actionable, legal_subtasks, COUNTER,  # noqa: E402
                          FETCH_OF, TIER_NAME, T_EXPLORE, T_STASH,
                          _BECOMES, _CHAIN_VERBS)
from common.views import BeliefView                            # noqa: E402

# Ticks before what is ON a station decays back to unknown. This is the whole of
# the staleness channel: shorten it and even a 360 cone has to keep re-checking,
# lengthen it and a 30 cone stops paying for its blind spots.
FORGET_HORIZON = 40
# Ticks of UNBROKEN exploring before the tile we are stuck at is blacklisted as a
# frontier target and the committed goal is dropped. It is the only timer in the
# agent, and it is a way out of "the only unknown left is behind a wall I cannot
# see through", not a general-purpose give-up. Reset the instant anything on the
# ladder becomes legal again.
PATIENCE = 30

# Ticks a sighting stays live. TWICE the belief's forget_horizon on purpose:
# view.robot's hands belief dies at 30 (views.py:201) and this memory has to
# outlive it, or the look-for is only ever available in the window where the
# belief already told you what the robot was carrying anyway.
LOOK_HORIZON = 60
# Max L1 distance from the remembered robot position to a counter worth checking.
# This is the entire definition of "that area", and it is the main tuning knob:
# too small and the sweep gives up before it reaches the pass, too large and a
# single sighting sends the human across the whole kitchen.
LOOK_RADIUS = 6

# THE DROP TIER: what an empty-handed take_<item> off a counter is worth. Read
# straight off tasks.py:375-378 and VERIFIED by executing legal_subtasks against
# a synthetic view in which every station is found and reachable and one counter
# holds the item -- see human/tests/test_look_for_human.py, which re-derives this
# table rather than restating it, so a change to the ladder breaks the test
# instead of silently mispricing every look-for.
LOOK_TIER = {"dish": 2, "steak_dish": 3, "garnish_dish": 3,
             "washed_plate": 4, "garnish": 4,
             "meat": 7, "onion": 7, "plate": 7}

# THE KEEP TIER: (what I am holding, what I am looking for) -> tier, for the four
# held-branch verbs in tasks.py that TARGET A COUNTER rather than a station. For
# these four the human does not have to put anything down at all -- walking over
# holding the garnish IS the subtask -- and the tier is better than the drop tier
# in every one of them, which is why "in KEEP_TIER" and "do not drop" are the
# same test below.
#
# UNGATED, and that is not an oversight. The held branch of legal_subtasks never
# calls actionable(); it simply offers combine at counters_holding(...). This
# mirrors it. Gating it would price a look-for lower than the very subtask it is
# a search for.
KEEP_TIER = {("steak_dish", "garnish"): 2, ("garnish", "steak_dish"): 2,
             ("garnish", "washed_plate"): 3, ("washed_plate", "garnish"): 3}

# The carve-out: a human carrying a finished order never diverts to go looking
# for anything, however valuable. Expressed as a row of the table (no held-branch
# counter verb exists for a dish either, so look_tier is simply undefined for
# every column) rather than as a special case in decide(). Unconditional on
# purpose: it does not ask whether deliver is currently legal, because "I am
# holding the thing everyone is waiting for" is the reason, not "I know where the
# hatch is".
NEVER_DROP = {"dish"}

# the three things that are made of other things. gate_composites=False exempts
# exactly these from the actionable() gate; see the module docstring.
_COMPOSITES = ("dish", "steak_dish", "garnish_dish")

# verb -> the RAW item whose chain it belongs to, inverted out of
# tasks._CHAIN_VERBS so this cannot drift from the ladder's own idea of which
# verbs make what. Read only by the optional chain_guard.
_PRODUCT_RAW = {"washed_plate": "plate", "garnish": "onion",
                "steak_dish": "meat"}
_VERB_CHAIN = {v: _PRODUCT_RAW[p] for p, vs in _CHAIN_VERBS.items() for v in vs}

_PREFIX = "look_for_"


class LookaheadView:
    """A read-only proxy that answers ready() as ready-OR-IN-PROGRESS.

    Wrapped around the human's own BeliefView for the DURATION OF ONE GATE TEST
    and nothing else. Every other query -- stations, empty, in_progress,
    counters_holding, free_counters, walkable, forget_horizon -- delegates
    untouched, so the optimism setting and the decay of the underlying belief
    still apply exactly as they would have.

    WHY ONLY ready(). actionable() asks "can I use this NOW", which is the right
    question for something already in your hand and the wrong one for something
    you are deciding whether to spend twenty ticks walking towards. A pot that is
    mid-timer now is a pot that is ready when you arrive. Relaxing anything else
    -- empty(), or counters_holding() -- would be hallucination rather than
    look-ahead: those are claims about what IS somewhere, and no amount of
    walking makes them true.

    Deliberately NOT a subclass of BeliefView. tasks._counts_composites duck-types
    on `forget_horizon`, which __getattr__ forwards, so this reads as a belief
    view to the ladder without inheriting a second copy of the stores that could
    drift from the real one.

    THE ONE OTHER CLASS IN THIS FILE, and it is not a second human. "One file,
    one class" is a rule about the AGENT, because an agent that can be built two
    ways is an agent the forecast can build the wrong way. A stateless read-only
    adapter cannot be put in the human seat and so cannot reproduce that failure.
    """

    def __init__(self, view):
        # object.__setattr__ is not needed -- __getattr__ only fires for
        # attributes that are NOT found normally, and `_view` is.
        self._view = view

    def ready(self, ch):
        v = self._view
        return sorted(set(v.ready(ch)) | set(v.in_progress(ch)))

    def __getattr__(self, name):
        return getattr(self._view, name)


class LimitedVisionHuman:
    """Ladder-driven, FOV-gated teammate. See the module docstring for the design.

    Every tick: observe -> ask the ladder what BELIEF says is legal and reachable
    -> take the best of those -> walk one step or INTERACT. No plan, no
    hysteresis, no scores. The only state that survives a tick is the belief
    itself, the last subtask (for within-tier stickiness), the explore goal and
    the sighting book the look-for runs on.

    THE ONLY HUMAN CLASS THERE IS. The look-for was a subclass and is not any
    more -- see ONE FILE, ONE CLASS in the module docstring for the forecast bug
    that split made possible. Everything the subclass carried is here:

      observe()  additionally snapshots (item -> robot_pos, armed_tick) at the
                 moment of sighting, folds this tick's cone into the per-sighting
                 cleared sets, and RETIRES sightings. It is the only method that
                 writes the sighting book, which is what makes decide() safe to
                 ask speculatively.
      rank()     memoised on the belief version, so decide() can consult the
                 ordinary ladder without paying for a second BFS sweep and two
                 more legal_subtasks passes. NO behaviour change.
      decide()   a PURE QUERY: the ordinary answer from _ladder_decide(), unless
                 a look-for -- or the stash that has to precede it -- is STRICTLY
                 better.
      action()   one arm more in the dispatch: a look_for verb walks and turns
                 but never INTERACTs. Also writes the committed look and drop
                 targets that decide() only computes.
      clone()    a faithful copy that shares the mdp, for the filter's shadows.

    enable_look_for=False is the A/B control and is action-for-action identical
    to the agent before any of that existed.
    """


    def __init__(self, mdp, fov, agent_index=1, forget_horizon=FORGET_HORIZON,
                 patience=PATIENCE, react_to_robot=True, seed=0,
                 look_horizon=LOOK_HORIZON, look_radius=LOOK_RADIUS,
                 lookahead_gate=True, gate_composites=True, chain_guard=False,
                 suppress_unreachable=True, enable_look_for=True):
        self.seed = seed
        self.mdp = mdp
        self.fov = fov
        self.agent_index = agent_index
        self.other_index = 1 - agent_index
        self.patience = patience
        self.react_to_robot = react_to_robot
        self.forget_horizon = forget_horizon
        self.look_horizon = look_horizon
        self.look_radius = look_radius
        # relax ready() to ready-or-cooking while testing the gate. ON: without
        # it a robot carrying a garnish_dish past a pot that is merely COOKING
        # buys no reaction at all.
        self.lookahead_gate = lookahead_gate
        # OFF exempts dish/steak_dish/garnish_dish from the gate entirely. Buys
        # the literal "always go and look for a composite" behaviour at the cost
        # of wasted trips when the thing genuinely has no future.
        self.gate_composites = gate_composites
        # OFF by default. When ON, a look-for is only allowed if the sought item
        # is downstream of the chain the ordinary winner belongs to. Makes "robot
        # has a garnish_dish but I wanted to do meat -> do meat" literal even in
        # a multi-pot layout, at the cost of blocking legitimate preemptions
        # (R=meat, Y=washed_plate is a real one it would refuse).
        self.chain_guard = chain_guard
        # ON: knowing where the item is suppresses the look-for even when we
        # cannot reach it, because "I know where it is and cannot get to it"
        # already answers the question the look-for exists to ask. OFF falls back
        # to the literal rule and keeps looking.
        self.suppress_unreachable = suppress_unreachable
        # THE A/B CONTROL, and the reason it is a flag rather than a second
        # class. --no-look-for is a knob on one agent, so the filter's shadows
        # are built by copying the seat's kwargs (human_kw) instead of by naming
        # a class -- and a shadow that is a different agent from the human is
        # no longer something a caller can express. See the module docstring.
        self.enable_look_for = enable_look_for
        self.reset()

    def reset(self):
        self._rng = random.Random(self.seed * 2 + 1)
        self.view = BeliefView(self.forget_horizon)
        self.t = 0
        self.explore_ticks = 0
        # tiles PATIENCE gave up standing at, dropped from the frontier until the
        # cone finds new ground -- see observe(). Not a permanent blacklist: a
        # frontier that was hopeless from here may be trivial from ten tiles on.
        self.abandoned = set()
        self.last_subtask = None
        self._recent = []          # last 12 positions, so _explore can spot a loop
        self._goal = None          # the explore target we are committed to
        self.log = []
        # item -> (robot_pos, armed_tick). ONE entry per item, newest wins.
        # Immune to views.py:200 nulling the position half of the robot belief,
        # which is the entire reason it exists rather than being read out of the
        # view. armed_tick is the tick this sighting was ARMED, not the last tick
        # the robot was seen with it: a robot lingering at one cell must not
        # renew LOOK_HORIZON every tick, or the horizon stops bounding anything.
        self.sightings = {}
        # item -> {counter cells already resolved SINCE this sighting was armed}.
        # Ignores belief decay on purpose: it is a record of where the sweep has
        # been, not a belief about what is there, and that is what makes the
        # sweep terminate instead of restarting every FORGET_HORIZON ticks.
        # Emptied on re-arm, because seeing the item back in the robot's hands
        # falsifies every conclusion the sweep had reached.
        self._cleared = {}
        self._look_target = None    # (item, counter cell) committed goal
        self._drop_target = None    # counter cell committed for the drop leg
        # BUMPED BY observe() AND BY NOTHING ELSE, and it is what makes the two
        # memos below safe. Keying them on self.t instead was wrong in a way that
        # only showed up under an external caller: action() increments t at the
        # END of the tick, so a decide() asked BEFORE action() filled the memo
        # from last tick's belief under this tick's key, and action()'s own
        # decide() then hit it. See the module docstring and repro_memo.py.
        self._belief_version = 0
        self._rank_memo = None      # ((version, pos, held), ranked)
        self._field_memo = None     # ((version, pos), walk, dist_field)

    # -- perception ---------------------------------------------------------
    def observe(self, state):
        """Fold this tick's cone into the belief, then run the sighting book.

        Seeing new ground clears `abandoned`: a goal PATIENCE gave up on was only
        hopeless given what we knew then, and the map having grown is exactly the
        evidence that it might not be any more. Without this the frontier only
        ever shrinks and a long episode ends up with nothing left to aim at.

        `self.t` is still THIS tick throughout -- action() increments it only at
        the very end -- so a sighting recorded here is co-timed exactly with the
        view.robot record written a few lines above it, and the two expire on
        schedules that can be compared directly.

        THE ONLY WRITER of sightings/_cleared, which is what makes decide() a
        query anyone may ask without consequences. Retirement used to live in
        _scan, so a shadow rollout asking the real human what it would do deleted
        the real human's memory.

        Three jobs after the cone lands, in this order and it matters:

        RECORD    the robot is in the cone holding something, so remember WHERE.
                  Only a CHANGE arms a sighting -- a different item, a different
                  cell, or hands we had stopped believing in -- and arming is
                  what empties the cleared set and restarts LOOK_HORIZON. A robot
                  merely lingering at the same cell re-arms nothing: refreshing
                  the timestamp every tick it is in the cone would make
                  LOOK_HORIZON a sliding window rather than a budget, and a human
                  that can see the pass could then stand at it all episode.
        RESOLVE   a counter is CLEARED only if the cone answered for it AT OR
                  AFTER the arming tick AND we no longer believe the robot is
                  holding the thing. Both halves are load-bearing and the second
                  is the whole feature -- see the module docstring. Under those
                  two conditions peripheral vision still closes candidates for
                  free, which is what keeps the sweep short at a wide cone and
                  long at a narrow one: the FOV gradient this feature is for.
        RETIRE    three ways, all of them here because this is where the evidence
                  lands. FOUND: the item is in our own hands, which makes "do not
                  look for what you are holding" structural rather than a test in
                  the hot path -- there is nothing left to find. EXPIRED:
                  LOOK_HORIZON ticks since arming. SWEPT: no unswept, in-radius,
                  reachable counter is left AND the robot has let go, so the
                  sweep genuinely finished rather than never having started.
        """
        # captured BEFORE the view is overwritten: "did we believe, coming into
        # this tick, that the robot had this thing". That is the difference
        # between a robot picking an item back up (arm) and one still carrying
        # the item we already know about (do not).
        was_held = self.view.robot[2] if self.view.robot else None

        me = state.players[self.agent_index]
        pos = tuple(me.position)
        seen = geo.visible_cells(self.mdp.terrain_mtx, pos,
                                 tuple(me.orientation), self.fov)
        # the four tiles we are touching are FELT, not seen: terrain only, so we
        # can always step around ourselves, and never a station we can act on.
        felt = [(pos[0] + dx, pos[1] + dy) for dx, dy in geo.DIRECTIONS]
        before = len(self.view.known_terrain)
        self.view.observe(self.mdp, state, seen, self.t, self.other_index,
                          felt=felt)
        if len(self.view.known_terrain) > before:
            self.abandoned.clear()      # new ground seen: worth retrying goals

        # BEFORE the early return, so the invalidation holds in the A/B control
        # too. Nothing the ladder half of this class can read is touched by it.
        self._belief_version += 1
        if not self.enable_look_for:
            # the no-op guarantee is structural, not a promise: with the feature
            # off, nothing below this line writes any state the ladder reads.
            return seen

        p = state.players[self.other_index]
        rpos = tuple(p.position)
        if rpos in seen:
            held = p.held_object.name if p.held_object else None
            if held is not None:
                prev = self.sightings.get(held)
                if prev is None or prev[0] != rpos or was_held != held:
                    self.sightings[held] = (rpos, self.t)
                    self._cleared[held] = set()

        # the slow half of the robot belief, on views.py:201's own 30-tick clock.
        # While this IS the sought item we have no evidence about any counter:
        # the thing is in the robot's hands, so of course the worktops are bare.
        still_held = self.view.robot[2] if self.view.robot else None
        for item in sorted(self.sightings):          # sorted: determinism
            if item == still_held:
                continue
            armed = self.sightings[item][1]
            done = self._cleared.setdefault(item, set())
            for c in self.view.stations(COUNTER):    # BeliefView.stations sorts
                rec = self.view.contents.get(c)
                # freshness read off the raw record rather than through
                # _fresh(), because we want "the cone answered this question
                # recently", which is a fact about our attention and not a
                # belief that can be optimistic.
                if rec is None or self.view.t - rec[2] > self.view.forget_horizon:
                    continue
                if rec[2] < armed:
                    # looked at BEFORE the robot walked past with it. Still fresh
                    # -- FORGET_HORIZON is 30 and LOOK_HORIZON is 60 -- and still
                    # says nothing whatever about where the thing ended up.
                    continue
                if rec[0] != item:
                    done.add(c)                      # known, and not what I want

        mine = me.held_object.name if me.held_object else None
        dist = None
        for item in sorted(self.sightings):
            if item == mine:                         # FOUND
                self._forget(item)
                continue
            if self.t - self.sightings[item][1] > self.look_horizon:   # EXPIRED
                self._forget(item)
                continue
            if item == still_held:
                continue                             # not swept: not yet started
            if dist is None:
                # built at most once and only if some sighting actually reaches
                # this line -- the BFS is memoised on (belief version, pos), so
                # this tick's decide() gets it for free rather than paying twice.
                walk, field = self._dist_tools(state)

                def dist(c, _w=walk, _f=field):
                    return geo.path_len_in(_f, _w, c)
            if not self._candidates(item, dist):                       # SWEPT
                self._forget(item)
        return seen

    def _forget(self, item):
        """Retire a sighting. Both stores, always together, one place to read."""
        self.sightings.pop(item, None)
        self._cleared.pop(item, None)

    def _candidates(self, item, dist):
        """Counters still worth looking at for `item`: unswept, near, reachable.

        Shared by the retirement test in observe() and the targeting in _scan so
        the two cannot disagree about what "nothing left to look at" means -- the
        version where they did is how a sighting could be retired as swept and
        still be offered a goal, or the other way round.
        """
        rpos = self.sightings[item][0]
        done = self._cleared.get(item, ())
        return [c for c in self.view.stations(COUNTER)        # sorted()
                if c not in done
                and abs(c[0] - rpos[0]) + abs(c[1] - rpos[1]) <= self.look_radius
                and dist(c) is not None]

    def _dist_tools(self, state):
        """(walkable, dist_field) from where we stand, memoised on the belief.

        `walkable` only ever changes inside observe(), so (version, pos) keys it
        exactly -- and for the same reason self.t did not, see rank(). Shared
        between observe()'s retirement test, the candidate scan and the drop leg,
        because they are three parts of one decision and a second BFS sweep from
        the same start is pure waste.
        """
        pos = tuple(state.players[self.agent_index].position)
        key = (self._belief_version, pos)
        if self._field_memo is None or self._field_memo[0] != key:
            walk = self.view.walkable | {pos}
            self._field_memo = (key, walk, geo.dist_field(walk, pos))
        return self._field_memo[1], self._field_memo[2]

    # -- choice -------------------------------------------------------------
    def rank(self, state):
        """Everything legal right now, best first, as (tier, demotion, dist, cell, verb).

        Sorted as a plain tuple, so the ordering IS the design: tier first and
        absolutely (a finished dish beats a ready garnish however far away it is),
        then the within-tier demotions that come from having seen the robot, then
        distance, and finally the cell coordinate so two identical runs stay
        identical.

        `demotion` is 0, 1 or 2 and is the ONLY place the robot enters the
        decision. It can never move a subtask across a tier and never removes one,
        so the human stays self-sufficient: if the demoted subtask is the only
        thing in its tier it still gets done. The two terms are independent --
        `redundant` is about the robot's HANDS, `contested` about its POSITION,
        and the belief about those two runs on two separate clocks (see
        BeliefView), so it is entirely normal to have one and not the other.

        Distances come from a per-tick memo that is also the reachability
        predicate handed to legal_subtasks(). Legality and ranking used to be two
        separate judgements, which is exactly how the human ended up committed to
        a station it could see through a counter and never reach.

        THE WHOLE ANSWER IS MEMOISED ON THE BELIEF VERSION, with no behaviour
        change: decide() consults this ladder and then does its own look-for
        reachability work, and without the memo the ordinary ladder would be
        recomputed from scratch -- one dist_field sweep and up to three
        legal_subtasks passes -- every time anything asked.

        THE KEY IS NOT self.t, AND THAT WAS THE BUG. rank() is a pure function of
        (belief, position, held), and self.t is not a version of the belief: it
        is incremented by action() at the END of the tick, while the belief is
        rewritten by observe() at the START of the next one. So a caller that
        asked decide() before action() -- which this class explicitly allows, and
        which every filter shadow and likelihood does -- filled the memo under
        this tick's t with LAST tick's belief, and action()'s own decide() hit it
        and acted a tick stale. _belief_version is bumped by observe() and by
        nothing else, so the memo is now exact rather than an approximation.
        """
        me = state.players[self.agent_index]
        pos = tuple(me.position)
        held = me.held_object.name if me.held_object else None
        key = (self._belief_version, pos, held)
        if self._rank_memo is not None and self._rank_memo[0] == key:
            return self._rank_memo[1]
        walk = self.view.walkable | {pos}
        robot_held = self.view.robot[2] if self.view.robot else None

        robot_pos = self.view.robot[0] if self.view.robot else None

        # one memo, used both to decide legality and to rank it -- these were two
        # separate judgements before, which is exactly how the human ended up
        # committed to a station it could see and never reach.
        # ONE BFS sweep, not one A* per candidate. legal_subtasks hands back tens
        # of cells a tick and each used to pay for its own search; the grid is
        # unit-cost and 4-connected so a single field from `pos` answers all of
        # them with the same numbers (geometry.dist_field, checked exhaustively).
        _field = geo.dist_field(walk, pos)

        def dist(cell):
            return geo.path_len_in(_field, walk, cell)

        # THREE ASKS, IN THIS ORDER, and each one only happens because the
        # previous came back empty.
        #   1. what we KNOW. 2. what we are willing to GUESS at. 3. put it down.
        # A guess must never compete with knowledge: on equal footing a stale
        # sink reads as ready, collect_plate (T4) outranks get_plate (T7), and the
        # human shuttles between the two forever -- the pick-plate /
        # collect-plate shuffle on banquet_pass at fov 30, in which no plate was
        # ever picked up. And stash must not be offered before the optimistic
        # pass, or it makes the strict pass non-empty and that pass never runs;
        # 136 of 300 ticks on banquet_pass went on putting a garnish down and
        # picking it back up two tiles from a stale-but-ready sink.
        ok = lambda c: dist(c) is not None
        self.view.optimistic = False
        subtasks = legal_subtasks(self.view, held, ok, allow_stash=False)
        if not subtasks:                       # nothing KNOWN to do: guess
            self.view.optimistic = True
            subtasks = legal_subtasks(self.view, held, ok, allow_stash=False)
        if not subtasks:                       # nothing even guessed: put it down
            subtasks = legal_subtasks(self.view, held, ok)
        # left ON for everything downstream: the strict pass is a local
        # tightening for the duration of the decision, not the agent's default.
        self.view.optimistic = True

        # the contention test walks from the BELIEVED robot position, so it is a
        # second start and needs a second sweep -- built once here rather than
        # once per candidate inside the loop.
        _rwalk = _rfield = None
        if self.react_to_robot and robot_pos is not None:
            _rwalk = walk | {robot_pos}
            _rfield = geo.dist_field(_rwalk, robot_pos)

        scored = []
        for tier, verb, cell in subtasks:
            d = dist(cell)
            if d is None:
                continue                       # not reachable over KNOWN floor
            # Channel A: seeing the robot already carrying it demotes the
            # duplicate fetch WITHIN its tier. Never across tiers, never to
            # zero -- if it is the only option in the tier we still do it.
            redundant = int(self.react_to_robot
                            and robot_held is not None
                            and FETCH_OF.get(verb) == robot_held)
            # CONTENTION, and it is NOT collision handling. The two agents have
            # no shared floor and can never block each other; what they CAN share
            # is a station embedded in the dividing wall, worked from opposite
            # sides. If the robot is SEEN to be closer to one than we are it gets
            # there first and we would arrive to an already-done job, so demote it
            # within the tier and let the two of us spread out. The path is
            # measured from the BELIEVED robot position over BELIEVED floor, so it
            # can only ever fire on a cell reachable from both rooms -- and it is
            # FOV-gated on purpose: a narrow cone does not know where the robot
            # is, so it contends more and coordinates worse. That is the effect
            # this package exists to measure, not a bug.
            contested = 0
            if _rfield is not None:
                rd = geo.path_len_in(_rfield, _rwalk, cell)
                contested = int(rd is not None and rd < d)
            scored.append((tier, redundant + contested, d, cell, verb))
        scored.sort()
        self._rank_memo = (key, scored)
        return scored

    def _ladder_decide(self, state):
        """The ORDINARY answer: one subtask off the ladder, or EXPLORE.

        Split out from decide() rather than inlined into it because the look-for
        needs this number as the thing it has to strictly beat -- `base` in
        _scan. It is decide()'s whole body in the A/B control, and it was
        decide() itself before the look-for existed, which is why the trace with
        enable_look_for=False is identical to the agent that predates all of it.

        Everything on the ladder having come back empty is not a failure state,
        it is the normal condition of a cone that has not found the kitchen yet
        -- so EXPLORE is a real rung with a real target (None) rather than a
        fallback bolted on the side, and action() dispatches on it like any
        other.

        Reaching this line with something to do also resets the PATIENCE clock:
        that timer is meant to count UNBROKEN fruitless exploring, and a spell of
        useful work in between means the next spell of exploring deserves a full
        budget rather than the remainder of the last one.
        """
        ranked = self.rank(state)
        if not ranked:
            return (T_EXPLORE, 0, 0, None, "explore")
        self.explore_ticks = 0
        # STICKY WITHIN A TIER, and only within a tier. Walking changes which of
        # two equally good targets is nearer, so a pure argmax swaps between them
        # every few steps and the agent paces on the spot -- measured at ~30 target
        # flips per 100 ticks before this. Preferring what we were ALREADY doing
        # among equals costs nothing and is not a switch margin: the ladder still
        # preempts instantly ACROSS tiers, so seeing a finished dish still drops a
        # half-washed plate on the spot, exactly as designed.
        if self.last_subtask is not None:
            top = ranked[0][0]
            for cand in ranked:
                if cand[0] != top:
                    break
                if (TIER_NAME[cand[0]], cand[4], cand[3]) == self.last_subtask:
                    return cand
        return ranked[0]

    # -- the look-for, which borrows the ladder's own prices -----------------
    def _look_tier(self, item, held, ok):
        """What a look-for this item is worth right now, or None for "not worth it".

        min over the two ways of having it: KEEP, walking over with full hands
        because a held-branch verb already targets that counter; and DROP, the
        empty-handed take tier, available only when we are willing to put down
        what we are carrying and the item has a future at all.
        """
        keep = KEEP_TIER.get((held, item))
        drop = None
        if held not in NEVER_DROP and self._gate(item, ok):
            drop = LOOK_TIER[item]
        vals = [v for v in (keep, drop) if v is not None]
        return min(vals) if vals else None

    def _gate(self, item, ok):
        """"Would this thing have a future if I had it?" -- actionable(), relaxed.

        The relaxation is one line and it carries one example on its own; see the
        module docstring. Everything else about the test, including reachability
        and the optimism setting of the underlying view, is the ladder's own.
        """
        if not self.gate_composites and item in _COMPOSITES:
            return True
        v = LookaheadView(self.view) if self.lookahead_gate else self.view
        return actionable(item, v, ok)

    def _chain_ok(self, item, verb):
        """chain_guard: is `item` downstream of the chain `verb` belongs to?

        Off by default. Verbs that belong to no raw chain -- deliver, combine,
        add_garnish, the composite pickups, stash, explore -- are not opinions
        about which chain we are working, so they pass. VERIFIED against
        tasks._BECOMES: garnish_dish is downstream of onion and plate but NOT of
        meat, and dish is downstream of all three, which is why this one
        predicate is enough to make "I wanted to do meat, so do meat" literal.
        """
        raw = _VERB_CHAIN.get(verb)
        if raw is None:
            return True
        return item in _BECOMES.get(raw, ())

    def _look_candidate(self, state, base):
        """The best look-for that STRICTLY beats `base`, as a rank() 5-tuple.

        Returns (tier, 0, dist, goal_cell, "look_for_<item>") or None. The
        demotion slot is a hard 0: Channel A and contention are both statements
        about duplicating the ROBOT's work, and going to find what the robot is
        carrying is the opposite of duplicating it.
        """
        if not self.enable_look_for or not self.sightings:
            return None
        me = state.players[self.agent_index]
        held = me.held_object.name if me.held_object else None
        walk, field = self._dist_tools(state)

        def dist(c):
            return geo.path_len_in(field, walk, c)

        def ok(c):
            return dist(c) is not None

        # OPTIMISM PARITY. rank() leaves optimistic=True whichever of its three
        # passes actually won, so the gate below would let a stale pot read as
        # BOTH empty and ready -- a guess outranking knowledge, which is the
        # failure views.py:239-254 documents. Recover the winning pass's setting
        # with one strict ask rather than by duplicating rank()'s body: the
        # strict pass won exactly when it came back non-empty, and both later
        # passes ran with optimism on.
        self.view.optimistic = False
        strict_nonempty = bool(legal_subtasks(self.view, held, ok,
                                              allow_stash=False))
        self.view.optimistic = not strict_nonempty
        try:
            return self._scan(base, held, dist, ok)
        finally:
            # restore the agent-wide default. rank() promises the same thing on
            # exit and everything downstream of the decision assumes it.
            self.view.optimistic = True

    def _after_drop_tier(self, ok):
        """Best tier the ladder would offer us EMPTY-HANDED, or T_EXPLORE.

        The second half of S4, and it is the half that was missing. Beating the
        in-hand tier only earns the right to PUT THE THING DOWN; what happens
        next is decided by a ladder that has never been consulted, and the
        empty-handed branch offers whole verbs the in-hand branch cannot -- every
        dispenser, every counter pickup. Seen on banquet_pass/fov180: the human
        stashed a garnish for a look_for_onion (T7 against a T8 stash), came up
        empty-handed, was offered get_meat at T7, and strictness -- rightly --
        gave the tie to the dispenser. The look-for it had just paid a trip for
        never happened. 30 of 45 drop legs ended that way.

        Asked with the SAME optimism setting as the ranking that produced `base`
        -- see _look_candidate -- so this cannot be a guess outranking knowledge.
        The two-ask discipline is rank()'s own: what we KNOW, then what we are
        willing to guess, and stash excluded from both because a stash is exactly
        what we are already doing.
        """
        got = legal_subtasks(self.view, None, ok, allow_stash=False)
        if not got and not self.view.optimistic:
            self.view.optimistic = True
            try:
                got = legal_subtasks(self.view, None, ok, allow_stash=False)
            finally:
                self.view.optimistic = False
        return min((t for t, _, _ in got), default=T_EXPLORE)

    def _scan(self, base, held, dist, ok):
        best = None
        after_drop = None                            # computed at most once
        for item in sorted(self.sightings):          # sorted: determinism
            rpos = self.sightings[item][0]

            # ---- SUPPRESSION -------------------------------------------------
            # S1 "I am already holding it", S3 "the sighting expired" and S5 "the
            # area is swept" are ALL handled in observe() by retiring the entry,
            # so they are not tests here at all. That is not tidiness: doing them
            # here made decide() destructive, so asking a human what it would do
            # -- a filter shadow, a likelihood, a probe -- erased its memory.
            #
            # S2 "I already believe it is on a counter". By default this counts
            # counters we CANNOT reach as well, which is a superset of the
            # literal rule and deliberate: knowing where the thing is answers the
            # question the look-for exists to ask, and walking over to re-confirm
            # it is pure waste. suppress_unreachable=False takes the literal one.
            found = self.view.counters_holding(item)
            if found and (self.suppress_unreachable or any(ok(c) for c in found)):
                continue

            # S4 the tier, and the comparison is STRICT -- see the module
            # docstring. Equal is not better, and that is the whole of "hunt for
            # a meat only when you cannot simply fetch one".
            tier = self._look_tier(item, held, ok)
            if tier is None or tier >= base[0]:
                continue
            # ...and if getting there means putting something down first, it has
            # to beat the ladder we will be standing in front of AFTER the drop
            # as well. Same strictness, same reason. Without it the drop leg is a
            # trip paid for a look-for that the next tick refuses to take.
            if held is not None and (held, item) not in KEEP_TIER:
                if after_drop is None:
                    after_drop = self._after_drop_tier(ok)
                if tier >= after_drop:
                    continue
            if self.chain_guard and not self._chain_ok(item, base[4]):
                continue

            # ---- TARGETING ---------------------------------------------------
            cands = self._candidates(item, dist)
            if not cands:
                # Nothing left to look at in that area. SKIP -- observe() is what
                # retires it, on exactly this test plus "and the robot has let
                # go". Skipping was unsafe while the cleared set was poisoned at
                # birth, because belief decay could un-know a counter and the
                # human would sweep the same room forever; it is safe now that
                # the set only grows and only ever records counters looked at
                # after the arming tick.
                continue

            # L1 FROM THE REMEMBERED POSITION FIRST, then our own walk, then the
            # cell. L1 and not a BFS from rpos, because in every layout here the
            # robot's room is unreachable over the human's believed floor and
            # that BFS would return None for every counter -- L1 crosses the
            # dividing wall, which is exactly where the pass counters are.
            def key(c, _r=rpos):
                return (abs(c[0] - _r[0]) + abs(c[1] - _r[1]), dist(c), c)

            goal = min(cands, key=key)
            # COMMIT to the goal we already chose while it is still a candidate.
            # L1 to a fixed remembered position does not move as we walk, but the
            # walk-distance tie-break does, and re-argmin-ing a moving tie-break
            # is the pacing bug the stickiness in _ladder_decide exists to
            # prevent.
            if (self._look_target is not None and self._look_target[0] == item
                    and self._look_target[1] in cands):
                goal = self._look_target[1]
            cand = (tier, 0, dist(goal), goal, _PREFIX + item)
            if best is None or cand < best:
                best = cand
        return best

    def decide(self, state):
        """The ordinary answer, unless a look-for strictly beats it. PURE QUERY.

        Two sightings competing is not a special case: both are ordinary rank()
        5-tuples and are compared as plain tuples, so tier decides first, then
        our own walk distance, then the cell, then the verb string. Fully
        ordered, no ambiguity.

        Writes NOTHING. decide() is a question anyone may ask about the current
        view -- the filter's shadows, its likelihood terms, every probe in the
        test suite -- and it used to answer it by editing the sighting book and
        the committed targets. _choose() below does the work and hands back what
        it chose; action() is the one method that records it, exactly as it
        records last_subtask.

        With enable_look_for=False this is _ladder_decide() and nothing else,
        which is the A/B control the harnesses spell --no-look-for.
        """
        return self._choose(state)[0]

    def _choose(self, state):
        """(subtask 5-tuple, look_target or None, drop_target or None).

        The whole derivation, with the two commitments RETURNED rather than
        written, so the caller decides whether this tick is actually being taken.
        Reads self._look_target / self._drop_target -- last tick's commitments --
        for the anti-pacing stickiness, and never writes them.
        """
        base = self._ladder_decide(state)
        if not self.enable_look_for:
            return base, None, None
        cand = self._look_candidate(state, base)
        if cand is None:
            return base, None, None

        item = cand[4][len(_PREFIX):]
        look = (item, cand[3])

        me = state.players[self.agent_index]
        held = me.held_object.name if me.held_object else None
        # Empty-handed, or holding the other half of a combine: walk as we are.
        # "(held, item) in KEEP_TIER" and "no drop needed" are the same test
        # because the keep tier is better than the drop tier in all four pairs,
        # so whenever the pair exists it is what the min picked.
        if held is None or (held, item) in KEEP_TIER:
            return cand, look, None

        # ---- THE DROP LEG ------------------------------------------------------
        # Not a two-phase script -- see the module docstring. While the hand is
        # full this same derivation resolves to a stash; the tick after that
        # INTERACT lands it resolves to the walk.
        walk, field = self._dist_tools(state)
        drops = [c for c in self.view.free_counters()      # sorted(), optimistic
                 if c != cand[3]
                 and geo.path_len_in(field, walk, c) is not None]
        if not drops:
            # Nowhere to put it down, so there is no look-for to be had this
            # tick and the ordinary ladder's choice stands. The human never
            # strands itself halfway into a drop it cannot finish.
            return base, None, None
        d = min(drops, key=lambda c: (geo.path_len_in(field, walk, c), c))
        if self._drop_target in drops:
            d = self._drop_target                          # same anti-pacing commit
        # Synthesized directly, so it bypasses tasks.saturated(). That is
        # correct: saturated() vetoes PRODUCING more of something, and this is a
        # put-down rather than production.
        return (T_STASH, 0, geo.path_len_in(field, walk, d), d, "stash"), look, d

    # -- acting -------------------------------------------------------------
    def action(self, state):
        """One env action. observe -> decide -> one step of the walk, or INTERACT.

        The walk is A* over `walkable`, which is only floor this agent has SEEN,
        plus the tile it is standing on -- that last term matters on tick zero and
        after a teleporting reset, when the cone has not yet caught the ground
        under our own feet and the planner would otherwise have no start node.

        The partner's tile is NOT excluded: nothing here routes around them. See
        the module docstring -- two rooms, no shared floor, measured adjacent on 0
        of 4800 ticks.

        WHY A LOOK-FOR NEVER INTERACTS. Its payoff is that step_towards turns us
        to face the counter; the next observe() reads its contents, and then
        either the ordinary ladder offers take_<item> at the very tier the
        look-for borrowed and preempts it on the spot, or the counter joins the
        cleared set and the sweep moves on. INTERACTing at a look-for target
        would pick up whatever happened to be there.

        `arrived` IS reachable for a look-for goal, and STAY is the point rather
        than belt and braces. Standing at the target counter facing it, while the
        robot is still believed to be holding the thing, is exactly "be there when
        it puts it down" -- and it terminates, because the moment the hands belief
        expires that counter's fresh record clears it and the sweep moves on.
        Before the clearing rule was fixed this branch was dead: the counter was
        cleared out of the candidate set the instant we faced it, so the human
        never arrived anywhere.

        THE COMMITMENTS ARE WRITTEN HERE, not by decide(). This is the method
        that actually takes the tick, and it is where last_subtask is written for
        the same reason.
        """
        self.observe(state)
        choice, look, drop = self._choose(state)
        self._look_target, self._drop_target = look, drop
        tier, _, _, cell, verb = choice
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)

        walk = self.view.walkable | {pos}
        if verb == "explore":
            act = self._explore(pos, orient)
        elif verb.startswith(_PREFIX):
            move, _arrived = geo.step_towards(walk, pos, orient, cell)
            act = move or Action.STAY
        else:
            move, arrived = geo.step_towards(walk, pos, orient, cell)
            # step_towards returning no move means the target went unreachable
            # between ranking and walking. STAY costs a tick, and next tick the
            # ladder re-decides on a fresh observation -- which is cheaper than
            # any recovery rule and cannot loop, because the belief has moved on.
            act = Action.INTERACT if arrived else (move or Action.STAY)

        self.t += 1
        self.log.append(self.last_subtask)
        return act, {"subtask": self.last_subtask}

    def _explore(self, pos, orient):
        """T9. Where to go when the ladder has nothing legal to offer.

        THE OBJECTIVE CHANGES HALFWAY THROUGH AN EPISODE, and that is why this is
        three rules rather than one. While there is map left, the job is DISCOVERY
        and every unknown tile is worth the same, so the only sane tie-break is
        travel. Once the map is closed, the job is REFRESH and travel is no longer
        what separates candidates -- staleness is. A single rule that tries to do
        both is worse than either. Measured end to end: 99.8% of reachable floor
        covered, and no configuration in which it stalls.

        1. SWEEP WHEN LOOPING. Back on the same tile three times in the last
           twelve means navigating is not working, so stop navigating and look:
           take whichever of the four actions reveals the most unseen ground. It
           is not a scripted "step, then four glances" because the env will not
           allow one - a move onto walkable floor MOVES you and sets your facing,
           and only a move into furniture turns you on the spot. Scoring the four
           by what they reveal gives free glances where there is something to
           face and honest steps where there is not.

        2. WHILE THE MAP IS INCOMPLETE: NEAREST frontier, and no commitment.
           Re-picking the nearest unknown every tick costs nothing, because
           reaching any of them is progress. Biasing this phase toward the
           least-SEEN cell instead measured strictly worse - 1589 distinct tiles
           covered fell to 1258 and revisits per tile rose from 2.2 to 3.7 -
           because it walks past near unknowns to reach far ones and the counts
           shift under it on the way.

        3. ONCE THE MAP IS CLOSED: least-seen FLOOR, and COMMIT to it. seen_count
           is how many ticks each cell has spent inside the cone, so least-seen is
           very nearly most-out-of-date, which is what "go and re-check something"
           means with no frontier left. Here the commitment is essential and it is
           the same argument in reverse: walking toward a far, seldom-seen tile
           raises the counts on the way, so an argmin re-picked every tick turns
           the agent around. Without rule 3 at all the agent span on the spot the
           moment the last frontier closed.
        """
        # PATIENCE. After this many unbroken exploring ticks, blacklist the tile
        # we are standing on as a frontier target and drop the committed goal.
        # The case it is for is a frontier we can see but never close - unknown
        # ground on the far side of a wall, where the frontier tile stays a
        # frontier tile however long we stand next to it, and nearest-first keeps
        # sending us back. observe() clears the blacklist the moment new ground
        # appears, so this gives up on a place, not on exploring.
        self.explore_ticks += 1
        if self.explore_ticks > self.patience:
            self.abandoned.add(pos)
            self.explore_ticks = 0
            self._goal = None
        walk = self.view.walkable | {pos}

        # 1. sweep
        self._recent.append(pos)
        del self._recent[:-12]
        if self._recent.count(pos) >= 3:
            self._goal = None
            known = self.view.known_terrain.keys()
            best, gain_of_best = None, 0
            for d in geo.DIRECTIONS:
                n = (pos[0] + d[0], pos[1] + d[1])
                stand = n if n in walk else pos      # blocked -> we only turn
                gain = len(geo.visible_cells(self.mdp.terrain_mtx, stand, d,
                                             self.fov) - known)
                if gain > gain_of_best:
                    best, gain_of_best = d, gain
            if best is not None:
                return best
            # nothing anywhere reveals anything: take any legal step to
            # break the cycle. Not collision handling -- this breaks a loop
            # with OURSELF, and there is nobody else in the room.
            steps = [d for d in geo.DIRECTIONS
                     if (pos[0] + d[0], pos[1] + d[1]) in walk]
            return self._rng.choice(steps) if steps else Action.STAY

        # 2. WHILE THERE IS STILL MAP TO FIND: nearest frontier, no commitment.
        #    Closest-unknown is the efficient greedy choice when the goal is
        #    DISCOVERY - every frontier tile is equally unknown, so the only
        #    thing to separate them is travel, and re-picking each tick costs
        #    nothing because reaching any of them is progress. Biasing this phase
        #    toward the least-seen cell measured strictly worse (1589 distinct
        #    tiles -> 1258, revisits 2.2 -> 3.7): it walks past near unknowns to
        #    reach far ones, and the counts shift under it as it goes.
        #    Every frontier tile goes in as a goal, not just the best one, so the
        #    A* below finds the nearest of them in one search.
        frontier = [c for c in self.view.frontier() if c not in self.abandoned]
        if frontier:
            self._goal = None      # a stale rule-3 commitment must not outlive it
            goals = frontier
        else:
            # 3. MAP FINISHED, so the objective changes from DISCOVER to
            #    REFRESH, and with it the right rule. Every tile is known now, so
            #    travel is no longer the only thing separating candidates - how
            #    stale they are is. Aim at the least-seen floor, and COMMIT:
            #    walking toward a far seldom-seen tile raises the counts on the
            #    way, so an argmin re-picked every tick turns the agent around.
            #    Commitment is what makes the bias pay here, and it is only
            #    needed here.
            goal = self._goal
            if goal is not None and (goal == pos or goal not in walk
                                     or geo.astar(walk, pos, [goal]) is None):
                goal = None
            if goal is None:
                cands = [c for c in walk if c != pos]
                if cands:
                    lo = min(self.view.seen_count.get(c, 0) for c in cands)
                    path = geo.astar(walk, pos, [c for c in cands
                                     if self.view.seen_count.get(c, 0) == lo])
                    goal = path[-1] if path else None
                self._goal = goal
            goals = [goal] if goal is not None else []

        # FREE INFORMATION, NEVER A DETOUR. Where several first steps start
        # routes of the SAME length, take the one that reveals the most. The
        # route is not lengthened by a single tile, so this cannot trade progress
        # for looking - it only spends a choice that was arbitrary anyway. A* on
        # its own breaks that tie on coordinate order, which is reproducible and
        # blind.
        if goals:
            path = geo.astar(walk, pos, goals)
            if path and len(path) > 1:
                steps = self._tied_first_steps(walk, pos, goals, len(path))
                nxt = (max(steps, key=lambda c: (self._reveals(pos, c), c))
                       if len(steps) > 1 else path[1])
                return (nxt[0] - pos[0], nxt[1] - pos[1])
        # Nothing reachable to aim at: turn. One tick, a whole fresh cone, and
        # it is the only move that can still change anything when the agent is
        # boxed in by unseen ground.
        i = geo.DIRECTIONS.index(orient) if orient in geo.DIRECTIONS else 0
        return geo.DIRECTIONS[(i + 1) % 4]      # spin in place

    def _tied_first_steps(self, walk, pos, goals, best_len):
        """Neighbours that start SOME equally-short route to `goals`.

        best_len is len(path) for the best route from pos - a NODE count, so a
        neighbour is tied exactly when its own best route is one node shorter.
        Getting that off by one silently turns the caller's free look into a
        detour, which is the one thing it must not become.
        """
        out = []
        for d in geo.DIRECTIONS:
            n = (pos[0] + d[0], pos[1] + d[1])
            if n not in walk:
                continue
            p = geo.astar(walk, n, goals)
            if p is not None and len(p) == best_len - 1:
                out.append(n)
        return out

    def _reveals(self, pos, cell):
        """How much NEVER-SEEN ground we would gain by stepping from `pos` to `cell`.

        The facing is not a free variable and must not be treated as one: in this
        env a move onto floor both moves you and points you the way you went, so
        the cone from `cell` is fixed by the step that got you there. Scoring a
        step against some other orientation would rank a look the agent cannot
        actually take.

        Counted against known_terrain, not seen_count, so this asks "is any of
        that ground NEW" and not "how stale is it". Once the map is closed it is
        therefore 0 for every candidate and the caller's tie-break falls back to
        coordinate order - which is right, because in that phase the informative
        choice has already been made by picking a least-seen goal, and re-deciding
        it one step at a time is the thrash rule 3 commits against.
        """
        d = (cell[0] - pos[0], cell[1] - pos[1])
        if d == (0, 0):
            return 0
        cone = geo.visible_cells(self.mdp.terrain_mtx, cell, d, self.fov)
        return len(cone - self.view.known_terrain.keys())

    # -- plumbing so this drops into the existing harnesses ------------------
    def clone(self):
        """A faithful copy that SHARES the mdp. The filter's shadow copy, by hand.

        A FRESH HUMAN IS NOT A SUBSTITUTE FOR A CLONE, which is why this exists
        at all: _ladder_decide is sticky within a tier and `t` feeds
        BeliefView._fresh's age arithmetic, so a human restarted at t=0 against a
        view holding timestamps from tick T computes a negative age and nothing
        it inherited can ever decay.

        EVERY FIELD, INCLUDING THE SIGHTING BOOK. A copy that dropped it would be
        a human that has just forgotten what it saw the robot holding -- i.e. one
        that never look-fors, which is precisely the behaviour a forecast most
        needs to get right. That used to be a live bug rather than a hypothetical:
        the clone helpers named LimitedVisionHuman and copied its fields one by
        one while the seat held a subclass, so the shadow was silently downgraded
        partway through every rollout. With one class the downgrade cannot be
        silent -- it is an AttributeError on the shadow's first observe(), which
        is how the last two copies of it were found -- and every helper in
        robot/filter/core/ now takes its class from type(h) and copies what it
        does not name. This method is the same copy written out by hand, kept
        because a human that can hand out a faithful copy of itself is the one
        thing that stops the next such helper from being written wrong.

        Nothing needs deep copying and the reason is a property of the stores
        rather than a hope: every value in known_terrain, contents and seen_count
        is a str, an int or a tuple. _stations maps a char to a mutable SET and
        `_cleared` maps an item to a mutable SET, so those two are the ones whose
        values are rebuilt. The rng is restored from getstate() because that state
        is an immutable tuple and deep-copying the Random object was most of the
        cost this function exists to avoid.
        """
        c = type(self).__new__(type(self))
        c.seed = self.seed
        c.mdp = self.mdp                                # shared on purpose
        c.fov = self.fov
        c.agent_index = self.agent_index
        c.other_index = self.other_index
        c.patience = self.patience
        c.react_to_robot = self.react_to_robot
        c.forget_horizon = self.forget_horizon
        r = random.Random()
        r.setstate(self._rng.getstate())
        c._rng = r

        v = self.view
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

        c.t = self.t
        c.explore_ticks = self.explore_ticks
        c.abandoned = set(self.abandoned)
        c.last_subtask = self.last_subtask
        c._recent = list(self._recent)
        c._goal = self._goal
        c.log = []

        c.look_horizon = self.look_horizon
        c.look_radius = self.look_radius
        c.lookahead_gate = self.lookahead_gate
        c.gate_composites = self.gate_composites
        c.chain_guard = self.chain_guard
        c.suppress_unreachable = self.suppress_unreachable
        c.enable_look_for = self.enable_look_for
        c.sightings = dict(self.sightings)
        c._cleared = {k: set(s) for k, s in self._cleared.items()}
        c._look_target = self._look_target
        c._drop_target = self._drop_target
        # the two memos are caches of things that are cheap to rebuild and WRONG
        # to inherit: a shadow about to be stepped from a different state shares
        # our belief version and position, so it would hit a key that matches and
        # is answering a different question. Copying the version and dropping the
        # memos is the safe pairing -- the shadow's first observe() bumps it
        # again anyway, and starting it at 0 would only make a collision with the
        # original's future versions possible.
        c._belief_version = self._belief_version
        c._rank_memo = None
        c._field_memo = None
        return c

    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp
