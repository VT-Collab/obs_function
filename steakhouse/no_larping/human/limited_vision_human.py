"""

I want to remake a limited vision human in this file based on the previous limited vision human. Instructions are in the comment on top of file. first just write me a basic outline of behavior of this human in file before we go further. 

reference lightly on :
1. 
2. 

Check /Users/mishafu/Desktop/obs_function/steakhouse/no_larping/env.txt for instructions.

Clean remake of limited_vision_human:

FOV:
1. 30, 60, 90, 180, 360 angled cone
2. if vision hit a wall (#), can not see that vision line anymore (occuluded). 
Reference code below (everything but # is see through):
1.     def estimated_world_cone_vis_mask(self, fov) -> np.ndarray:
        
        from math import radians, cos, sqrt
        
        #world dimensions, initialize
        W, H = self.width, self.height
        mask = np.zeros((W, H), dtype=bool)

        #agent grid coordinate
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        
        #forward direction unit vector so liek right would be (1,0) and up would be (0,-1)
        fdx, fdy = int(self.dir_vec[0]), int(self.dir_vec[1])
        
        #get the cosine of degree ready;
        #for a vector v from agent to a cell and the forward vector f
        #cos(theta) = (v·f)/|v|
        #and inside the cone means cos(theta) >= cos(half)
        half = radians(fov / 2.0)
        cos_cut = cos(half)

        # Always see your own cell
        if 0 <= ax < W and 0 <= ay < H:
            mask[ax, ay] = True

        #Scan through entire world grid:
        for x in range(W):
            for y in range(H):
                
                #vector from agent tile ax, ay to the tile x, y
                dx = x - ax
                dy = y - ay
                
                #skip agent own tile
                if dx == 0 and dy == 0:
                    continue
                
                #check if tile is in front
                dot = dx * fdx + dy * fdy     # in-front if dot > 0
                if dot <= 0:
                    continue
                
                #check if tile is in the cone
                r = sqrt(dx * dx + dy * dy)
                if (dot / r) < cos_cut:       # angle test via cosine
                    continue
                
                # Line-Of-Sight test to stop vision half way unless see-through
                if self.see_through_walls or self._los_clear(ax, ay, x, y):
                    mask[x, y] = True

        return mask

Properties:
1. Only can know/act on things inside observation history (depending on FOV)
2. * DISCOVERY. The agent starts knowing nothing - not even where the stations
    ARE
3. BELIEF DECAY -> LOOKING. Beliefs about pot / board / sink / counter (what's on it) /robot
    state are FOV-gated and expire after FORGET_HORIZON steps.
4. EXPLORATION PATIENCE -> number of timesteps it spends looking for something before giving up 
    and continue with something else
5. Should be one level of subtask NOT 2 levels

Note:
1. human picks the most important subtask to do that completes the task, then use a A* 
to go to that subtask (path is also FOV gated)
    a. for example, if human sees garnish_dish on counter and sees that steak is ready and they can reach it
        ii. they will stop washing plate even if mid task to go do that since it higher level
    b. if they see a finished dish and knows the delivery cite, go drop current task and deliver dish.
    
2. human is influenced by robot behavior:
    a. if robot carrying meat/plate/onion, downgrade doing that 
    b. if seeing something dropped on counter mid way doing something and that is higher 
        a. swtich to that
        
3. more ideas idk brainstorm with me for max fov coervage?

"""

"""
================================================================================
OUTLINE v1 - behaviour only, no implementation. Read top to bottom.
================================================================================

ONE SENTENCE
------------
Every tick the human takes the highest-ranked action its BELIEFS say is legal,
walks there by plain A* over ground it has seen, and re-decides from scratch next
tick - so the cone decides what is on the menu, and the menu decides what it does.

No scores, no weights, no arithmetic. A strict PRIORITY LADDER of tiers; take any
legal action in the highest non-empty tier; break ties by nearest. Pure argmax.


--------------------------------------------------------------------------------
1. PERCEPTION
--------------------------------------------------------------------------------
visible(cell) = in_cone(cell) AND los_clear(self, cell)

  in_cone     dot/cosine test exactly as in the brief above. fov in
              {30,60,90,180,360}. At 360 the cone test is vacuous - but LOS is
              NOT. A 360 human is omnidirectional, never omniscient.
  los_clear   straight line self->cell, blocked ONLY by '#'. Counters, pots,
              sinks, boards and dispensers are all see-through. Walls are the
              sole occluder and exist purely to partition vision.
  NO RADIUS.  The cone runs to the edge of the map. What limits sight is angle
              and walls, nothing else.

Occlusion is therefore a property of the LAYOUT, not of the furniture. Two
kitchens with identical stations but different walls are different FOV problems.
That is the experimental knob.


--------------------------------------------------------------------------------
2. MEMORY
--------------------------------------------------------------------------------
Written ONLY from currently-visible cells. Three stores:

  known_terrain[cell] -> char        PERMANENT (walls and worktops do not move)
  station_locs[kind]  -> [cells]     PERMANENT (found by looking, never lost)
  contents[cell]      -> (what, t)   DECAYS to UNKNOWN after FORGET_HORIZON
  robot               -> (pos, orient, held, t)   DECAYS, same horizon

contents covers pot / board / sink AND COUNTERS, and counters FULLY FORGET - a
counter not looked at for FORGET_HORIZON ticks becomes UNKNOWN, not "empty". This
matters more than it sounds: counters are where the robot can hand things over,
so "what is on the worktops" is perishable, decision-relevant knowledge that a
wide cone keeps for free and a narrow cone must go and re-earn.

Start state: EMPTY. The human does not know where a single station is.

Acquisition is instant, decay is slow (FORGET_HORIZON). That asymmetry is what
keeps the agent from thrashing - a cell that flickers in and out of the cone
stays known throughout.


--------------------------------------------------------------------------------
3. SUBTASKS - flat, exactly one level
--------------------------------------------------------------------------------
A subtask is (verb, target cell). No hierarchy, no forced/chosen split, no
two-stage decomposition. One list, one ladder, one choice per tick.

Legality is judged against BELIEF, never ground truth. A station never discovered
is not on the list. A counter whose contents have decayed is not on the list.


--------------------------------------------------------------------------------
4. THE PRIORITY LADDER  (this replaces the scorer)
--------------------------------------------------------------------------------
EVERY subtask in EVERY tier has the same shape:

    A* walk to the target cell   ->   one INTERACT on arrival

The walk is implied everywhere and is usually several ticks. Tiers rank subtasks
by WHAT THAT ARRIVAL INTERACT PRODUCES, never by how long the walk is. "T2
yields a dish" means "when I get there and press INTERACT, what I am holding
becomes a dish" - not "a dish appears this tick".

Distance never enters the tier. It is only the tie-break WITHIN one.

Take ANY legal action from the highest non-empty tier. Never mix tiers.

  T1  DELIVER          holding dish, delivery site known and reachable
  T2  COMPLETE A DISH  arrival interact YIELDS a dish:
                         steak_dish   + ready board
                         steak_dish   + counter garnish
                         garnish_dish + ready pot
                         garnish      + counter steak_dish
                         empty        + counter dish          (acquire a finished one)
  T3  BUILD A HALF     arrival interact yields steak_dish or garnish_dish:
                         washed_plate + ready pot     -> steak_dish
                         washed_plate + ready board   -> garnish_dish
                         washed_plate + counter garnish
                         garnish      + ready sink    -> garnish_dish
                         garnish      + counter washed_plate
                         empty        + counter steak_dish / garnish_dish
  T4  COLLECT          empty + ready sink  -> washed_plate
                       empty + ready board -> garnish
                       empty + counter washed_plate / garnish
  T5  WORK             chop at board / wash at sink (pinned, one tick per INTERACT)
  T6  START            load pot with meat / board with onion / sink with plate
  T7  FETCH            meat / onion / plate from a dispenser or a counter
  T8  STASH            put down a held item that has no known use (see 5.2)
  T9  EXPLORE          nothing above is legal

TIE-BREAK inside a tier: shortest A* path over KNOWN FLOOR. Exact ties broken
deterministically by cell coordinate, so runs are reproducible.

NO HYSTERESIS, NO COMMITMENT, NO SWITCH MARGIN. T1 and T2 preempt instantly and
unconditionally - see a finished dish, know the serving hatch, both reachable, and
the human abandons a half-washed plate on the spot. That is the requested
behaviour and it needs no special case: it is just "T1 outranks T5".

Why no dithering falls out of this rather than needing a margin: distances to a
fixed target shrink monotonically as the human approaches, and beliefs only
change when something enters the cone (instant) or expires (slow). So the ladder
is stable between genuine observations, and every switch is a real re-plan.

4.1 WHAT TIER DOMINANCE COSTS - the honest consequence
    Because distance never crosses tiers, an empty-handed human will walk 30
    tiles for a dish on a far counter (T2) rather than take the ready garnish it
    is standing next to (T4). That IS the requested behaviour - a finished order
    outranks a component, and "switch to the dish 100% immediately" only means
    anything if distance cannot outvote it - but be clear that it is a choice,
    not an oversight.

    It bites less often than it looks, because the tiers are largely partitioned
    by what the hands are already full of:
      - holding dish        -> only T1 is legal at all. No conflict possible.
      - holding steak_dish / garnish_dish / washed_plate / garnish / raw item
                            -> lower tiers need empty hands or a different item,
                               so the only competitor is T8 stash. No real
                               conflict.
      - EMPTY-HANDED        -> this is the only case where several tiers are
                               simultaneously legal, and it is exactly the case
                               where "fetch the finished dish, not the onion" is
                               the right call.

    A mid-walk target that stops being valid (the robot took the dish) needs no
    special handling: the human re-decides from scratch next tick and re-plans.
    It only pays for the steps already walked.

    IF we later decide the long walk is wrong, the fix should stay arithmetic-
    free - e.g. a tier only counts if its target is within REACH_LIMIT steps -
    rather than reintroducing a scorer that trades value against distance.


--------------------------------------------------------------------------------
5. CHECKING THE LADDER AGAINST EVERY CASE
--------------------------------------------------------------------------------
5.1 COMPLETENESS - one row per thing the human can hold. "Best" = highest tier
    reachable at all; the fallback is what happens when it is not.

    held           advancing actions                       fallback
    ------------   -------------------------------------   ---------------------
    dish           deliver (T1)                             stash (T8)
    steak_dish     +garnish: board / counter    (T2)        stash (T8)
    garnish_dish   +steak: pot                  (T2)        stash (T8)
    washed_plate   pot / board / counter garnish (T3)       stash (T8)
    garnish        sink / counter w_plate / counter s_dish  stash (T8)
                                                (T2 or T3)
    meat           load pot if empty            (T6)        stash (T8)
    onion          load board if empty          (T6)        stash (T8)
    plate          load sink if empty           (T6)        stash (T8)
    nothing        T2/T3/T4 pickups, T5 work, T7 fetch      explore (T9)

    Every row terminates. Nothing can hold an item with no exit, because T8
    always applies when a counter is known, and T9 applies when one is not.

5.2 THE STASH LOOP - the one real hazard, and the rule that kills it.
    Naive version: human holds steak_dish, no garnish known anywhere. T2 is
    empty, so it stashes on a counter (T8). Now empty-handed - and T3 says
    "empty + counter steak_dish -> pick up". It picks it straight back up.
    Infinite loop.

    RULE: a pickup is only legal if the item is ACTIONABLE - i.e. the human
    believes in a way to advance it. steak_dish with no garnish believed
    anywhere is not actionable, so it is not offered, and the human goes and
    chops one (T5/T6/T7) instead. Same rule kills every symmetric variant.
    This single constraint is what makes the ladder terminating rather than
    merely ordered.

5.3 DEAD-END PAIRS the env allows and the ladder must not chase:
    steak_dish + garnish_dish cannot merge (env spec). Neither is actionable
    against the other, so no tier ever proposes it. Correct by construction.
    Two washed_plates likewise.

5.4 BLOCKED STATIONS: holding meat with the pot already cooking, or plate with
    the sink already washing, drops to T8 stash - then empty-handed the human
    picks up T5 WORK and advances the very station that blocked it. The blocked
    case self-heals instead of deadlocking.

5.5 NO-COUNTER LAYOUTS: if no counter is believed known, T8 is unavailable and
    the human falls to T9 explore, which is correct - it needs to find one.


--------------------------------------------------------------------------------
6. HOW THE ROBOT INFLUENCES THE HUMAN  (open - my recommendation, argue with me)
--------------------------------------------------------------------------------
The thesis: a robot that knows the human's exact FOV should beat one that does
not. That only pays off if SEEING the robot changes what the human does. Two
channels, both of which fall out of what is already above - no new machinery:

  CHANNEL A - SEEING ITS HANDS -> DO NOT DUPLICATE
    If the robot is currently believed to be holding meat/onion/plate, the
    matching T7 FETCH is deprioritised WITHIN ITS TIER - it goes last, not away.
    Not a weight and not a veto: if it is the only option in the tier the human
    still does it, so nothing starves.

    This is deliberately soft. The old agent tried a HARD redirect (pot empty +
    robot seen carrying meat -> chop instead) and it INVERTED the gradient:
    both agents deferred the meat to each other, neither cooked, and the
    better-sighted human starved. fov60 and fov180 delivered ZERO on
    steak_island from this alone. Softness is not a nicety, it is the fix.

  CHANNEL B - SEEING A DROP -> REACT TO THE HANDOFF
    The robot puts something on a counter inside the human's cone. The human's
    contents belief updates that same tick, a higher tier becomes legal, and the
    ladder switches to it immediately. No glancing, no scanning, no special
    behaviour: peripheral vision plus instant re-decision is the whole mechanism.

    This is where cone width becomes leverage the robot can USE. A 360 human
    sees a drop on any counter it has line of sight to. A 30 human sees only
    what it is pointed at. So the robot's CHOICE OF COUNTER is a decision that
    depends on the human's cone - and a robot that models the cone correctly
    picks a counter the human will actually notice, while a cone-blind robot
    drops things into the human's blind spot and the handoff silently rots
    until FORGET_HORIZON. Same action, different outcome, attributable to FOV.

  WHAT I AM NOT PROPOSING: the human never deliberately watches, tracks or
  positions itself relative to the robot. Influence is entirely incidental -
  the robot walks into view, or it does not.

  STILL OPEN: whether Channel A should also react to the robot's ORIENTATION
  (seen standing at the empty pot, about to cook) or only to its HANDS. Hands is
  a fait accompli; orientation is a prediction, and predictions are what
  inverted the gradient last time. I lean hands-only.


--------------------------------------------------------------------------------
7. MOVEMENT AND EXPLORATION
--------------------------------------------------------------------------------
A* over KNOWN FLOOR ONLY. Unknown cells are not traversable - the human cannot
plan through a corridor it has never seen. This is the second FOV channel: a
narrow cone knows fewer routes, so its paths are longer or do not exist yet.

TAKE THE SHORTEST PATH. Do not detour for information, do not prefer
high-visibility routes, do not turn to look before committing. The human sees
whatever the shortest path happens to sweep and nothing more. Information is a
by-product of walking, never a goal.

EXPLORE (T9) when nothing else is legal: head for the nearest frontier cell
(known floor bordering unknown). Turning in place is a legal explore move, since
it costs one tick and reveals a fresh cone.

PATIENCE: exploring is usually aimed at a specific missing precondition - "I hold
a plate and have never found a sink". After PATIENCE ticks without finding it,
abandon that goal, take the best action that IS legal (typically stash + do
something else), and do not re-target the same goal until something new is seen.
This is the only timer in the agent.

================================================================================
REFERENCE - the previous agent's design notes, kept for comparison
================================================================================


MISHA NEW CHANGE - a limited-vision steak human written FROM SCRATCH, because
SteakLimitVisionHumanModel structurally cannot express meaningful FOV-driven
behaviour.

WHY THE STOCK MODEL COULD NOT WORK
----------------------------------
Three properties of overcooked_ai_py/agents/agent.py's SteakLimitVisionHumanModel
each independently destroy FOV sensitivity:

1. IT NEVER FORGETS. update() (agent.py:1231) only ADDS observations. A
   20-degree human that glimpses the pot once knows its state forever, exactly
   like a 180-degree one.
2. IT STARTS OMNISCIENT. init_knowledge_base (agent.py:1075) copies every object
   in the start state in, unfiltered by FOV.
3. ADJACENCY IS FREE. in_bound (agent.py:882) returns True for tiles beside the
   player regardless of vision_bound, and the human always stands next to the
   station it is acting on.

Measured: across 768 random layouts, 642 showed exactly zero FOV effect on
subtask choice; the mean was 0.1, and what little appeared was timing jitter.

THE PROPERTIES THIS AGENT IS BUILT TO HAVE
------------------------------------------
(A) FOV CHANGES THE SUBTASK SEQUENCE SUBSTANTIALLY - the human genuinely behaves
    differently under different cones, and that difference is attributable to
    vision (see decide()). This is the property the robot must infer, and it is
    STRONG: substantial subtask divergence on every validated layout.
(B) THE HUMAN IS A FUNCTIONAL TEAMMATE AT EVERY FOV - the TEAM completes orders
    regardless of the cone (the human's prep feeding a teammate's delivery is a
    team win). More vision need not raise the HUMAN's own delivery count - a
    greedy teammate harvests shared stations either way - so success is measured
    at the team, and the FOV signal lives in (A), the behaviour.

Both come from OBSERVATION COST: everything the agent knows it had to spend
steps looking at, and how many steps that takes is set by the cone. In measured
order of contribution:

  * DISCOVERY. The agent starts knowing nothing - not even where the stations
    ARE - so it must map the kitchen by looking. This is the strongest channel
    and it degrades smoothly with the cone (audited on steak_island: a 30-degree
    human finds 87% of stations and 92% of the map, a 360-degree one 100% of
    both). Steps spent mapping are steps not spent delivering, so under a finite
    horizon this alone orders the FOVs.
  * BELIEF DECAY -> LOOKING. Beliefs about pot / board / sink state are FOV-gated
    and expire after FORGET_HORIZON steps. A wide cone keeps stations in view
    incidentally and stays current; a narrow cone goes stale and must emit an
    explicit check_* subtask - physically walking over to LOOK. Those subtasks
    appear only in narrow-FOV trajectories. That is (A): a different subtask in
    a different order, not a delay. The travel they cost feeds (B).
  * WASTED TRIPS. A stale-but-confident belief can still send the agent on the
    wrong errand - believing the pot empty when it has already been filled - and
    it pays a round trip before dumping the meat. Real, but the smaller term.

A caution learned the hard way, and the exact line the teammate-awareness below
walks: do NOT make the agent HARD-defer to its teammate on what it sees the
teammate holding. That was tried (pot empty + teammate seen carrying meat -> chop
an onion INSTEAD) and it INVERTED the gradient, because a hard redirect is itself
FOV-gated and the greedy teammate does not reliably follow through: both agents
deferred the meat to each other, neither cooked it, and the better-sighted human
starved. fov60 and fov180 delivered ZERO on steak_island purely from this.

What IS safe - and is what this agent now does (ROBOT_HELD_FETCH,
_robot_redundant) - is a SOFT reaction to the SAME signal: when the human SEES
the teammate already carrying an ingredient it DOWN-WEIGHTS (x ROBOT_SUPPRESS,
then renormalises - never zeroes) the now-redundant FETCH, and drops that fetch
if already committed. The difference from the failure is decisive. Softness keeps
the human SELF-SUFFICIENT: if the redundant fetch is the ONLY available move it
still does it (renormalise -> weight 1), so nothing starves - while a
non-redundant task, whenever one exists, is preferred. And it reacts only to a
fait-accompli in progress (an item actually in the teammate's hands, FOV-gated),
never to a prediction of what the teammate MIGHT pick up. Validated: team
delivery holds at every FOV on all layouts. This reaction is the whole point of
the finalized human - it is what lets a downstream FOV-aware robot, which knows
whether the human can currently see it, out-coordinate an FOV-unaware one: only a
human that visibly changes what it does when it sees the robot gives that robot
an advantage to exploit. The observation-cost channels above still carry the FOV
signal the robot INFERS; this channel is how that inference pays off.

The commitment is essential, but it is COMMITMENT WITH BELIEF-SENSITIVE
ABANDONMENT, exactly as a person behaves: the agent pursues its chosen errand
until it either completes OR it OBSERVES, through its own vision cone, that the
errand is pointless - e.g. it sees the teammate has already filled the pot it
set out to load. It does NOT abandon merely because a belief went stale;
forgetting is not observing, and dropping an errand on forgetting would be
dithering, not re-planning.

This is what makes the FOV effect substantial rather than marginal, and it does
so THROUGH vision: the abandonment trigger is itself FOV-gated. A wide cone
keeps the pot / board / sink incidentally in view and notices the invalidation
mid-trip, so it re-plans and wastes nothing; a narrow cone, pointed at the thing
it is fetching, never sees it, commits the whole trip, and pays for the meat it
cannot use. The cost of narrow vision is therefore measured in orders delivered,
not merely in different subtask labels. There is NO step-count "patience": a
commitment ends only on completion or on an observed contradiction, exactly as
the paper states - the agent cannot deadlock because arriving at a station ends
the errand (a no-op INTERACT still re-decides next tick) and an occupied
destination is observed and abandoned on sight.

HOW A SUBTASK IS CHOSEN - tau_t ~ P(.|o_{0:t})
----------------------------------------------
The human does NOT follow a fixed recipe order. Empty-handed, it SAMPLES among
the subtasks it currently believes are AVAILABLE and task-ADVANCING, weighted by
a sensible-order preference (softmax of PRIORITY at `temperature`), and commits
to the sample until it OBSERVES the choice is no longer helpful:

    temperature -> 0    : always the top-priority available move
    temperature ~ 0.5   : mostly-sensible, with genuine alternatives (the default)
    temperature -> inf  : uniform over available (erratic)

This is exactly the paper's model - "pick a subtask believed available that
advances the task; if several are, pick one; drop it only when it is observed no
longer useful." Because the choice among EQUALLY-preferred tasks (e.g. start the
steak vs start the garnish) is genuinely stochastic, two cones with different
beliefs produce different subtask sequences - property (A).

The distribution is KNOWN in closed form (subtask_distribution()), which is what
lets a robot do EXACT Bayesian FOV inference (fov/robot/inference/
bayes_fov_sampling.py) - it reads the probability assigned to the observed
subtask directly, with no epsilon fudge. One thing flexible ordering needs that a
fixed order gets for free: an assembly-ordering constraint (only plate up once
BOTH the garnish and the washed plate are ready, else the human can hold a steak
with no garnish and deadlock, unable to chop with full hands).

"""

# =============================================================================
# IMPLEMENTATION
# =============================================================================
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overcooked_ai_py.mdp.actions import Action, Direction   # noqa: E402
from common import geometry as geo                            # noqa: E402
from common.tasks import (legal_subtasks, FETCH_OF, T_EXPLORE,  # noqa: E402
                          TIER_NAME)
from common.views import BeliefView                            # noqa: E402

FORGET_HORIZON = 25      # ticks before a contents belief decays to UNKNOWN
PATIENCE = 20            # ticks of fruitless exploring before giving up on a goal


class LimitedVisionHuman:
    """Ladder-driven, FOV-gated teammate. See the outline above.

    Every tick: observe -> list what BELIEF says is legal -> take the best one
    on the ladder -> walk one step or INTERACT. No commitment, no hysteresis,
    no scores. Re-decided from scratch each tick, so "sees something better and
    switches" is the default path rather than a special case.
    """

    def __init__(self, mdp, fov, agent_index=1, forget_horizon=FORGET_HORIZON,
                 patience=PATIENCE, react_to_robot=True, seed=0):
        self.seed = seed
        self.mdp = mdp
        self.fov = fov
        self.agent_index = agent_index
        self.other_index = 1 - agent_index
        self.patience = patience
        self.react_to_robot = react_to_robot
        self.forget_horizon = forget_horizon
        self.reset()

    def reset(self):
        import random
        self._rng = random.Random(self.seed * 2 + 1)
        self.view = BeliefView(self.forget_horizon)
        self.t = 0
        self.explore_ticks = 0
        self.abandoned = set()      # explore goals given up on until something new
        self.last_subtask = None
        self._last_pos = None
        self._stuck = 0
        self.log = []

    # -- perception ---------------------------------------------------------
    def observe(self, state):
        me = state.players[self.agent_index]
        seen = geo.visible_cells(self.mdp.terrain_mtx, tuple(me.position),
                                 tuple(me.orientation), self.fov)
        before = len(self.view.known_terrain)
        self.view.observe(self.mdp, state, seen, self.t, self.other_index)
        if len(self.view.known_terrain) > before:
            self.abandoned.clear()      # new ground seen: worth retrying goals
        return seen

    # -- choice -------------------------------------------------------------
    def rank(self, state):
        """Legal subtasks, best first. (tier, redundant, dist, cell, verb)."""
        me = state.players[self.agent_index]
        pos = tuple(me.position)
        held = me.held_object.name if me.held_object else None
        # A subtask whose only standing cell is occupied by the teammate is not
        # reachable THIS tick. Dropping it here rather than jamming against it is
        # what stops the two agents deadlocking on the same dispenser.
        other = tuple(state.players[self.other_index].position)
        walk = (self.view.walkable | {pos}) - {other}
        robot_held = self.view.robot[2] if self.view.robot else None

        robot_pos = self.view.robot[0] if self.view.robot else None

        scored = []
        for tier, verb, cell in legal_subtasks(self.view, held):
            d = geo.path_len(walk, pos, cell)
            if d is None:
                continue                       # not reachable over KNOWN floor
            # Channel A: seeing the robot already carrying it demotes the
            # duplicate fetch WITHIN its tier. Never across tiers, never to
            # zero -- if it is the only option in the tier we still do it.
            redundant = int(self.react_to_robot
                            and robot_held is not None
                            and FETCH_OF.get(verb) == robot_held)
            # CONTENTION: if the robot is SEEN to be closer to this station than
            # we are, it will get there first and we would just queue behind it.
            # Demote within the tier so the two of us spread across stations
            # instead of both jamming one. Note this is FOV-gated on purpose --
            # a narrow cone does not know where the robot is, so it contends
            # more and coordinates worse. That is the FOV effect, not a bug.
            contested = 0
            if self.react_to_robot and robot_pos is not None:
                rd = geo.path_len(walk | {robot_pos}, robot_pos, cell)
                contested = int(rd is not None and rd < d)
            scored.append((tier, redundant + contested, d, cell, verb))
        scored.sort()
        return scored

    def decide(self, state):
        ranked = self.rank(state)
        if ranked:
            self.explore_ticks = 0
            return ranked[0]
        return (T_EXPLORE, 0, 0, None, "explore")

    # -- acting -------------------------------------------------------------
    def action(self, state):
        self.observe(state)
        tier, _, _, cell, verb = self.decide(state)
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)

        other = tuple(state.players[self.other_index].position)
        walk = self.view.walkable | {pos}
        if verb == "explore":
            act = self._explore(pos, orient, other)
        else:
            move, arrived = geo.step_towards(walk, pos, orient, cell, {other})
            act = Action.INTERACT if arrived else (move or Action.STAY)

        # mutual-block breaker: if we meant to move and have not moved for two
        # ticks, step aside. Two agents can otherwise wait on each other forever.
        if act is not Action.INTERACT and pos == self._last_pos:
            self._stuck += 1
            if self._stuck >= 2:
                act = geo.sidestep(walk, pos, {other}, self._rng) or Action.STAY
                self._stuck = 0
        else:
            self._stuck = 0
        self._last_pos = pos

        self.t += 1
        self.log.append(self.last_subtask)
        return act, {"subtask": self.last_subtask}

    def _explore(self, pos, orient, other=None):
        """Head for the nearest frontier. Turning counts - it is one tick and
        reveals a whole new cone. PATIENCE stops us fixating on one goal."""
        self.explore_ticks += 1
        if self.explore_ticks > self.patience:
            self.abandoned.add(pos)
            self.explore_ticks = 0
        walk = (self.view.walkable | {pos}) - ({other} if other else set())
        goals = [c for c in self.view.frontier() if c not in self.abandoned]
        if goals:
            path = geo.astar(walk, pos, goals)
            if path and len(path) > 1:
                nxt = path[1]
                return (nxt[0] - pos[0], nxt[1] - pos[1])
        i = geo.DIRECTIONS.index(orient) if orient in geo.DIRECTIONS else 0
        return geo.DIRECTIONS[(i + 1) % 4]      # spin in place

    # -- plumbing so this drops into the existing harnesses ------------------
    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp
