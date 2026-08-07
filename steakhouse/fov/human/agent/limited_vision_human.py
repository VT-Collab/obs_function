"""
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
import math
import random
from collections import namedtuple

from overcooked_ai_py.mdp.overcooked_mdp import Action

FORGET_HORIZON = 12
UNKNOWN = "unknown"
Belief = namedtuple("Belief", ["value", "seen_at"])

# Terrain symbol -> station kind. Applied to ONE tile at a time, only when that
# tile is inside the vision cone.
TERRAIN_KIND = {'P': 'pot', 'B': 'board', 'W': 'sink', 'M': 'meat',
                'O': 'onion', 'D': 'dish', 'S': 'serve', 'X': 'counter'}
# The ONLY terrain that blocks line of sight (see _clear_los). Deliberately NOT
# in TERRAIN_KIND: a wall is not a station, so it is never a subtask target and
# never gets a belief - it is only ever scenery that light cannot cross.
WALL = '#'
# How far the agent can resolve a tile at all. Bounds the per-step scan and
# means the agent never reasons about the world beyond its own senses.
SIGHT_RADIUS = 8
# Turning is only worth a step if it reveals at least this many new tiles.
TURN_GAIN_THRESHOLD = 2

COMMITTED = {"deliver", "dump_item", "pickup_meat", "pickup_onion", "pickup_plate",
             "drop_meat", "drop_onion", "drop_plate",
             "pickup_steak", "pickup_garnish", "pickup_washed_plate"}

# --- subtask-sampling policy constants ------------------------------------
# The advancing subtasks that carry the sampling freedom - everything an empty-
# handed agent might CHOOSE to start/continue. Looks (check_*/explore), waits and
# held-object drops are NOT here: those are forced, not chosen.
SAMPLING_SUBTASKS = frozenset({
    "pickup_meat", "pickup_onion", "chop_onion",
    "pickup_plate", "heat_washed_plate", "pickup_washed_plate",
})
# Preference score per advancing subtask - higher = a more sensible next move.
# P(tau) proportional to exp(score / temperature) over the AVAILABLE set. The two
# prep lines (start the steak / start the garnish) are EQUAL top priority:
# starving the garnish deadlocks assembly. Assembling a ready order is highest.
PRIORITY = {
    "pickup_washed_plate": 5,   # ASSEMBLE a ready steak+garnish - finish the order
    "pickup_meat": 4,        # start the steak   (critical-path prep)
    "pickup_onion": 4,       # start the garnish (critical-path prep, EQUAL to meat)
    "chop_onion": 4,         # finish an in-progress chop
    "heat_washed_plate": 3,     # finish an in-progress wash
    "pickup_plate": 3,       # start a plate washing
}
DEFAULT_TEMPERATURE = 0.5    # the default sampler temperature (see docstring)
                             # ~50/50 among EQUALLY-helpful tasks (paper-faithful)
                             # while rarely choosing a clearly-worse one

# --- teammate (robot) awareness -------------------------------------------
# The one thing a VISIBLE teammate reveals that its effect on a STATION does not
# already show is what it is CARRYING - i.e. which FETCH it is about to make
# redundant. Map: teammate's SEEN held item -> the human fetch it makes redundant.
# Used at BOTH selection (_weights soft down-weight) and commitment
# (_robot_redundant abandon). NOT included: steak/dish (act on the human's OWN
# held item, never redundant) and chop_onion / heat_washed_plate (progress an item
# already committed to the human's OWN station). All of it is FOV-gated (fires
# only on beliefs[ROBOT], written only while the teammate is in the cone) and
# SOFT - see the "caution learned the hard way" paragraph in the module docstring
# for why a hard version inverted the gradient and this one does not.
ROBOT_HELD_FETCH = {
    "meat": "pickup_meat",
    "onion": "pickup_onion",
    "plate": "pickup_plate",
    "washed_plate": "pickup_washed_plate",
}

# Teammate POSITION channel (station-yield): the station the teammate is SEEN
# FACING -> the human tasks that USE that station, which the human yields this
# tick so it does not compete/collide for a station the teammate is already at.
# This is a second, purely POSITIONAL receptiveness channel the held-object
# channel above cannot provide - a teammate standing at the EMPTY pot about to
# cook shows nothing in its hands yet, but SEEING it there is enough to defer.
# Only visible by seeing the teammate's pose, so it grows with FOV; validated as
# the sole channel (of assembly-suppress / complementary-boost / lookahead-prep /
# station-yield) that fires correctly-gated on all 12 layouts while preserving
# team-win, P2 and behavioural divergence.
STATION_TASKS = {
    "pot":   frozenset({"pickup_meat"}),
    "board": frozenset({"pickup_onion", "chop_onion"}),
    "sink":  frozenset({"pickup_plate", "heat_washed_plate"}),
}

# A fetch/drop that RAN TO COMPLETION only to find its destination station already
# occupied was a WASTED TRIP - redundant, usually because the teammate got there
# first. Maps each such subtask to the station whose non-emptiness wastes it, so
# n_wasted_commits tallies all three lines (meat->pot, onion->board, plate->sink)
# and their drops, not just meat. Diagnostic only - no metric depends on it.
WASTED_IF_OCCUPIED = {
    "pickup_meat": "pot", "drop_meat": "pot",
    "pickup_onion": "board", "drop_onion": "board",
    "pickup_plate": "sink", "drop_plate": "sink",
}


class LimitedVisionSteakHuman:
    """Steak-task human whose subtask choice depends on FOV-gated, decaying
    beliefs about station states."""

    def __init__(self, mdp, fov, planner, agent_index=1,
                 forget_horizon=FORGET_HORIZON, temperature=DEFAULT_TEMPERATURE,
                 avoid_robot=False, occlude=False):
        # temperature of the subtask sampler P(tau) ~ exp(PRIORITY / temperature)
        # over the available advancing set. Lower -> more sensible / less random
        # (temperature -> 0 always picks the top priority); higher -> more
        # exploratory (-> inf is uniform). See the module docstring.
        self.temperature = temperature
        # NOTE: no `mlp`. The MediumLevelPlanner used to be held here for
        # mlp.mp.get_plan() pathfinding, which searched the FULL map and was the
        # single largest cheat in this agent. Routing is now the agent's own BFS
        # over observed floor only (_bfs_step), so the planner argument below is
        # used ONLY to map a subtask name to the station kind it targets.
        self.mdp = mdp
        self.fov = fov
        self.planner = planner
        self.agent_index = agent_index
        self.forget_horizon = forget_horizon
        # OPT-IN traffic-avoidance (DEFAULT FALSE -> no avoidance logic at all,
        # behaviour byte-identical to before). When True the human detours around
        # the teammate's seen cell in _bfs_step. Left off by default because its
        # benefit is layout/FOV-dependent (helps wide-cone / open layouts, can add
        # costly detours in narrow-cone / tight ones); intended to be switched on
        # later only if end-to-end coordination needs it.
        self.avoid_robot = avoid_robot
        # OPT-IN occlusion / line-of-sight (DEFAULT FALSE -> no LOS, behaviour
        # identical to before). When True a tile is visible only if the straight
        # line to it clears every wall/counter, so the cone cannot see THROUGH
        # walls: blind spots (P2) grow (even at 360 deg) and discovery drops. It
        # only ever RESTRICTS sight, so it adds no illegal belief writes. Off by
        # default: it is a perception-model change needing its own full
        # re-validation and can over-blind the human on tight layouts. Provided
        # for occlusion experiments (contention layouts are the influence lever).
        self.occlude = occlude
        # MISHA NEW CHANGE - the REAL attribute names, verified against the mdp.
        # The first version read mdp.cook_time / mdp.chop_time, NEITHER of which
        # exists, so both silently fell back to defaults (3 and 10). steak_island
        # actually uses chopping_time=5, so the agent believed a garnish was ready
        # at 3 while the game required 5: it tried pickup_garnish, the INTERACT
        # legally did nothing, and it looped on that subtask for the rest of the
        # episode. That single typo is why EVERY fov delivered 0.
        self.cook_time = getattr(mdp, "steak_cooking_time", 10)
        self.chop_time = getattr(mdp, "chopping_time", 5)
        self.wash_time = getattr(mdp, "wash_time", 5)
        self.reset()

    # -- vision -------------------------------------------------------------

    def visible(self, state, loc):
        """Can the agent see `loc`? The cone/range test (_visible_cone), AND - if
        the opt-in occlude flag is on - a line-of-sight test (no wall between).
        occlude off (default) -> exactly the cone test, unchanged."""
        if not self._visible_cone(state, loc):
            return False
        if not self.occlude:
            return True
        p = state.players[self.agent_index].position
        return self._clear_los((p[0], p[1]), (loc[0], loc[1]))

    #for occlusion only
    def _clear_los(self, p, loc):
        """True iff the straight line p->loc crosses no WALL tile (endpoints
        excluded). Reads TRUE terrain: a physical wall blocks light whatever the
        agent knows - not cheating (the agent still never reads a hidden station
        timer through a wall; it just cannot SEE past the wall).

        MISHA NEW CHANGE - only '#' occludes. This used to block on any non-floor
        tile (`!= ' '`), which made every counter and every station a sight
        barrier, so occlusion was a side effect of wherever the furniture
        happened to sit. Counters and stations are now waist-high and see-through;
        walls are the ONLY occluder, and exist purely to partition the FOV. That
        makes occlusion a deliberate property of a layout instead of an accident
        of its worktop plan."""
        x0, y0 = p; x1, y1 = loc
        dx = x1 - x0; dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps <= 1:
            return True
        mtx = self.mdp.terrain_mtx
        for i in range(1, steps):
            cx = int(round(x0 + dx * i / steps)); cy = int(round(y0 + dy * i / steps))
            if (cx, cy) == (x0, y0) or (cx, cy) == (x1, y1):
                continue
            if mtx[cy][cx] == WALL:
                return False
        return True

    def _visible_cone(self, state, loc):
        """

        check whether or not you can see the passed in loc grid cell depending on current visibility
        Cone test, mirroring the simulator's geometry (agent.py:906) but
        WITHOUT the unconditional adjacency exemption.
        
        The thing is that y increasing downward
        
        Cone test, mirroring the simulator's geometry (agent.py:906) but
        WITHOUT the unconditional adjacency exemption.

        The only freebie is the single tile directly faced - you can see the
        station you are interacting with. The stock model exempts both
        perpendicular neighbours too, which combined with always standing beside
        the active station is a large part of why FOV stopped mattering there.
        """
        player = state.players[self.agent_index]
        
        px, py = player.position
        ox, oy = player.orientation
        
        #the one tile directly in front of the human you can always see
        if (px+ox, py+oy) == tuple(loc):
            return True
        
        #see everything
        if self.fov >= 360:
            return True
        
        #vector from player position to this loc
        dx, dy = loc[0] - px, loc[1] - py
        
        #can always see current standing gird
        if dx == 0 and dy == 0:
            return True
        
        #define forward direction; make the grid as related to that direction
        #vector freom the agent to the taget tile
        #facing up
        if (ox, oy) == (0, -1):
            rx, ry = dx, dy
        #facing down
        elif (ox, oy) == (0, 1):
            rx, ry = -dx, -dy
        #facing right
        elif (ox, oy) == (1, 0):
            rx, ry = dy, -dx
        #facing left
        else:
            rx, ry = -dy, dx
            
        # behind the facing direction
        if ry >=0:
            return False
        #Picture the cone as a triangle opening forward. At depth |ry| ahead of you, the cone is tan(θ/2) · |ry| wide on each side. So the tile is inside iff its sideways distance |rx| fits within that width
        #centerline (straight ahead, rx=0)
        #           |
        #     \     |     /   ← cone edges, slope = tan(fov/2)
        #      \    |    /
        #       \   |   /
        # depth  \  |  /   at this depth |ry|, half-width = tan(fov/2)·|ry|
        #  |ry|   \ | /
        #          \|/
        #      agent (facing up)

        return abs(rx) <= math.tan(math.radians(self.fov / 2.0)) * abs(ry) + 1e-9
        
    # -- belief -------------------------------------------------------------

    def reset(self):
        # -- clock --------------------------------------------------------
        self.t = 0                   # internal step counter, +1 each action().
                                     # It is the timestamp written into every
                                     # belief's seen_at, so it is what drives all
                                     # belief freshness and decay (see observe()).

        # -- trajectory record --------------------------------------------
        self.prev_chosen_subtask = None  # the subtask chosen on the PREVIOUS tick
                                     # (== subtask_log[-1]); kept only for readout.
        self.subtask_log = []        # ordered history of EVERY subtask chosen.
                                     # Condition (A) - "FOV changes the subtask
                                     # sequence" - is measured off this list.

        # -- diagnostic counters (several are read by the eval harnesses) --
        self.n_checks = 0            # ticks spent on a check_* look. The agent
                                     # looks because a belief EXPIRED to UNKNOWN
                                     # (aged past forget_horizon), NOT because it
                                     # detected the value was wrong - it cannot.
                                     # Falls as FOV widens: a wide cone re-sees
                                     # stations incidentally and rarely forgets.
        self.n_wasted_commits = 0    # fetch/drop trips (meat->pot, onion->board,
                                     # plate->sink + their drops) that RAN TO
                                     # COMPLETION only to find the destination
                                     # station already occupied - a redundant round
                                     # trip, spent on a stale-but-trusted belief or
                                     # beaten to it by the teammate. See
                                     # WASTED_IF_OCCUPIED. The delivery-cost of
                                     # narrow vision.
        self.n_delivered = 0         # dishes THIS human personally delivered (a
                                     # 'deliver' errand that arrived), NOT the team
                                     # total - crediting the partner's deliveries
                                     # would mask the human's own FOV effect.
        self.n_abandoned = 0         # commitments dropped mid-errand because the
                                     # cone REVEALED the errand was pointless
                                     # (belief-sensitive abandonment, decide()).
        self.n_explore = 0           # ticks spent exploring - mapping an unseen
                                     # kitchen or hunting a route through known
                                     # floor. The dominant FOV signal: a narrow
                                     # cone spends a long opening phase just
                                     # discovering where the stations are.

        # -- live commitment state ----------------------------------------
        self._current = None         # the subtask currently committed to, or None
                                     # when free to re-plan. Set back to None the
                                     # tick an errand completes (arrival), so the
                                     # next tick re-decides from current belief.
        # -- sampling state -----------------------------------------------
        self._sampled = None         # advancing subtask currently sampled+committed
        # Private, per-episode-reproducible RNG, seeded from the (harness-seeded)
        # global stream. Because the seed of THIS draw is identical across FOVs at
        # a fixed episode seed, the sampler makes the SAME draws for every cone -
        # so cross-FOV subtask divergence is caused by beliefs (FOV), never by the
        # RNG. That is what makes condition (A) attributable to vision.
        self._rng = random.Random(random.random())
        # MISHA NEW CHANGE - the agent starts knowing NOTHING about the kitchen,
        # not even where the stations ARE. Previously station locations were read
        # straight off the mdp at reset ("familiar kitchen" spatial memory), which
        # is still knowledge never acquired by looking. Now every location must be
        # DISCOVERED by having the tile fall inside the vision cone.
        #
        # This makes vision matter more, not less: a narrow cone needs far longer
        # to even map the kitchen before it can begin cooking, and must explore to
        # find a station it has never seen.
        # MISHA NEW CHANGE - NOTHING is precomputed from the mdp. Previously the
        # agent built complete station sets at reset and merely refrained from
        # consulting them until a tile was visible; the data still sat in memory.
        # Now a tile's terrain is read ONLY at the moment it is looked at, via
        # _classify() below, exactly like turning your head and seeing a pot.
        self.stations = {k: [] for k in TERRAIN_KIND.values()}
        self.seen_cells = set()
        # Terrain the agent has actually LOOKED at. Everything else is unknown -
        # it does not even know whether an unseen tile is floor or wall, so it
        # cannot path through it.
        self.known_terrain = {}
        self.beliefs = {}
        # The agent does NOT know the size of the world. It scans only the tiles
        # within its own sight radius of its own position; anything beyond that
        # simply does not exist to it yet.
        self._h = len(self.mdp.terrain_mtx)
        self._w = len(self.mdp.terrain_mtx[0])
        # last-SEEN teammate pose (pos, orientation, t) for the station-yield
        # channel. Written only while the teammate is in the cone (observe), and
        # read only within forget_horizon, so it is as FOV-gated and as forgetful
        # as every other belief. None until the teammate has ever been seen.
        self._robot_seen = None

    ROBOT = "__teammate__"     # a perceivable entity, stored like any other
    ROBOT_SUPPRESS = 0.3       # weight multiplier on the fetch the teammate is
                               # SEEN carrying (soft; renormalised, never zeroed)

    def observe(self, state):
        """
        Update beliefs about the world based on what is currently visible from current state. Beliefs
        about stations decay after forget_horizon steps, so the human must look
        at a station to keep its belief current. The teammate is treated as a
        station: it is only perceived when it falls inside the cone, and its
        memory decays on the same horizon. No privileged channel - if the human is
        facing away, it simply does not know what its partner is carrying.
        """
        # The teammate is part of the world, so it is perceived exactly like a
        # station: only when it falls inside the cone, and the memory of it
        # decays on the same horizon. No privileged channel - if the human is
        # facing away, it simply does not know what its partner is carrying.
        
        #teammate
        rp = state.players[1 - self.agent_index]
        if self.visible(state, rp.position):
            held = rp.held_object.name if rp.held_object else "none"
            self.beliefs[self.ROBOT] = Belief(held, self.t)
            # remember WHERE it is and which way it faces (for station-yield,
            # _robot_faced_kind). Same cone gate as the held-object belief above.
            self._robot_seen = (tuple(rp.position), tuple(rp.orientation), self.t)
        elif self.ROBOT in self.beliefs and \
                self.t - self.beliefs[self.ROBOT].seen_at > self.forget_horizon:
            self.beliefs[self.ROBOT] = Belief(UNKNOWN, self.beliefs[self.ROBOT].seen_at)

        # DISCOVERY: any tile inside the cone becomes known - both that it exists
        # and, if it is a station, what kind. Nothing outside the cone is ever
        # consulted. This is the ONLY place terrain is read, and only for tiles
        # that pass visible() - i.e. exactly the tiles the agent is looking at.
        px, py = state.players[self.agent_index].position   # agent = centre of the scan
        # sweep the square sight window (+/- SIGHT_RADIUS) around the agent
        for dy in range(-SIGHT_RADIUS, SIGHT_RADIUS + 1):
            for dx in range(-SIGHT_RADIUS, SIGHT_RADIUS + 1):
                cell = (px + dx, py + dy)
                if not (0 <= cell[0] < self._w and 0 <= cell[1] < self._h):
                    continue                                # tile is off the map
                if not self.visible(state, cell):
                    continue                                # tile is outside the FOV cone
                # --- this tile is genuinely in view: record it ---
                self.seen_cells.add(cell)                   # "I have looked here"
                terrain = self.mdp.terrain_mtx[cell[1]][cell[0]]   # note: mtx is [y][x]
                self.known_terrain[cell] = terrain          # remember floor/wall (for BFS)
                kind = TERRAIN_KIND.get(terrain)            # is it a station? which kind?
                if kind and cell not in self.stations[kind]:
                    # first time this station has ever fallen in the cone: remember
                    # WHERE it is. Locations are spatial memory - never forgotten.
                    self.stations[kind].append(cell)
                    if kind in ("pot", "board", "sink"):
                        # a dynamic station: its LOCATION is now known but its STATE
                        # is not. Seed an UNKNOWN belief with an ancient seen_at so it
                        # reads as "never observed"; the refresh loop just below fills
                        # in the true state this same tick (the tile is in view now).
                        self.beliefs[cell] = Belief(UNKNOWN, -10 ** 9)

        for loc in list(self.beliefs):
            if loc == self.ROBOT:
                continue
            if self.visible(state, loc):
                self.beliefs[loc] = Belief(self._true_state_of(state, loc), self.t)
            #not seen it, make human forget
            elif self.t - self.beliefs[loc].seen_at > self.forget_horizon:
                self.beliefs[loc] = Belief(UNKNOWN, self.beliefs[loc].seen_at)

    def _true_state_of(self, state, loc):
        """Ground-truth station state, for stations actually in view.

        MISHA NEW CHANGE - parses the ACTUAL encodings this environment uses,
        verified by probing a live episode. The first version guessed and was
        wrong on all three, which deadlocked the agent at every FOV:

            POT   name='steak'      state=('steak', n_items, cook_timer)  <- TUPLE
            BOARD name='garnish'    state=chop_timer                      <- int
            SINK  name='washed_plate'  state=wash_timer                      <- int

        `is_cooking` / `is_ready` are None on all of them, so the original
        `"cooking" if obj.is_cooking else "ready"` always said "ready".
        """
        obj = state.objects.get(loc)
        if obj is None:
            return "empty"
        name = getattr(obj, "name", "")
        st = getattr(obj, "state", None)
        # Prefer the mdp's OWN readiness predicates - re-deriving the thresholds
        # by hand is exactly what went wrong before.
        try:
            if name == "steak":
                return "ready" if self.mdp.steak_ready_at_location(state, loc) else "cooking"
            if name in ("garnish", "onion"):
                return "ready" if self.mdp.garnish_ready_at_location(state, loc) else "chopping"
            if name in ("washed_plate", "dish", "plate"):
                return "ready" if self.mdp.plate_washed_at_location(state, loc) else "washing"
        except Exception:
            pass
        if name == "steak":
            timer = st[2] if isinstance(st, (tuple, list)) and len(st) > 2 else 0
            return "ready" if timer >= self.cook_time else "cooking"
        if name in ("garnish", "onion"):
            timer = st if isinstance(st, int) else 0
            return "ready" if timer >= self.chop_time else "chopping"
        if name in ("washed_plate", "dish", "plate"):
            timer = st if isinstance(st, int) else 0
            return "ready" if timer >= self.wash_time else "washing"
        return "occupied"

    def believed(self, kind):
        if not self.stations.get(kind):
            return UNKNOWN          # never even found one of these yet
        vals = [self.beliefs[l].value for l in self.stations[kind] if l in self.beliefs]
        for pref in ("ready", "cooking", "chopping", "washing", "occupied", "empty"):
            if pref in vals:
                return pref
        return UNKNOWN

    def _robot_redundant(self, subtask):
        """Is `subtask` a FETCH the teammate is (believed) already carrying the
        ingredient for? FOV-gated: beliefs[ROBOT] holds a concrete item only
        while the teammate is, or recently was, inside the cone (observe()); it is
        UNKNOWN otherwise, and ROBOT_HELD_FETCH.get(UNKNOWN) is None, so a human
        that cannot see its teammate reacts to nothing. Only the 4 fetches ever
        match - see ROBOT_HELD_FETCH. This is the SOFT teammate reaction's
        commitment half; the selection half is in _weights."""
        rb = self.beliefs.get(self.ROBOT, Belief(UNKNOWN, -10 ** 9)).value
        return ROBOT_HELD_FETCH.get(rb) == subtask

    def _robot_faced_kind(self):
        """The station kind the SEEN teammate is currently FACING (and thus about
        to use), or None - the trigger for station-yield in _available_advancing.
        FOV-gated (self._robot_seen is written only in-cone) and decaying (stale
        after forget_horizon, so a teammate glimpsed long ago stops blocking a
        station). Uses ONLY the teammate's SEEN pose and the human's own
        DISCOVERED station cells - never a full-map lookup, so it cannot cheat."""
        if self._robot_seen is None:
            return None
        pos, orient, t = self._robot_seen
        if self.t - t > self.forget_horizon:
            return None
        faced = (pos[0] + orient[0], pos[1] + orient[1])
        for kind in ("pot", "board", "sink"):
            if faced in self.stations.get(kind, []):
                return kind
        return None

    def _commit_still_useful(self, subtask):
        """Paper-faithful abandonment test: is the committed ERRAND still worth
        finishing under what the agent currently BELIEVES?

        The agent abandons a commitment ONLY when it has OBSERVED the errand is
        pointless. An observation is a concrete (non-UNKNOWN) belief: a station
        is set to a real value only while it is inside the vision cone
        (observe(), the `if self.visible(...)` branch), so a concrete
        CONTRADICTING value means the agent has actually seen the change. A
        FORGOTTEN belief (UNKNOWN) is NOT an observation and never abandons -
        dropping an errand because a belief went stale is dithering, not
        re-planning, and would collapse the FOV effect back to timing jitter.

        Because the trigger is a concrete belief and concrete beliefs are
        FOV-gated, the vision cone decides WHO abandons early: a wide cone keeps
        the pot / board / sink incidentally in view and sees the invalidation
        mid-trip; a narrow cone, pointed at whatever it is fetching, does not,
        and completes the wasted trip. That asymmetry IS the FOV effect.
        """
        # Teammate awareness (commitment half): drop a FETCH the teammate is SEEN
        # already carrying - a fait accompli in progress, FOV-gated, only the 4
        # fetches. If this fetch is the human's ONLY option it re-picks it next
        # tick (self-sufficient); dropping only bites when a non-redundant move
        # exists to switch to. drop_*/assembly never match ROBOT_HELD_FETCH, so
        # they fall through to the belief logic below.
        if self._robot_redundant(subtask):
            return False
        pot = self.believed("pot")
        board = self.believed("board")
        sink = self.believed("sink")
        # The trigger is a FAIT ACCOMPLI, not a prediction. The paper's clause is
        # "the robot has ALREADY finished it": abandon only when the errand's own
        # destination is OBSERVED to already hold the thing the errand would
        # produce, so completing it would literally no-op (INTERACT does nothing
        # into an occupied station). We deliberately do NOT abandon on a guess
        # that the teammate MIGHT do it (e.g. seeing it carry meat): deferring to
        # a partner that may not follow through strands the human waiting on a pot
        # that never gets filled, and - because the guess is FOV-gated - it made
        # MORE vision perform WORSE. Removing it restores the correct gradient.
        # PLACING errands (drop_*): pointless (a literal INTERACT no-op) once the
        # destination is SEEN occupied. Always belief-sensitive - this is the
        # deadlock break that replaces the old step-count backstop, and it fires
        # at the destination, which the agent always faces, so it needs no wide
        # cone. No deferral: the human already holds the item.
        if subtask == "drop_meat":
            return pot in ("empty", UNKNOWN)
        if subtask == "drop_onion":
            return board in ("empty", UNKNOWN)
        if subtask == "drop_plate":
            return sink in ("empty", UNKNOWN)
        # FETCHING errands (pickup_meat / pickup_onion): also abandoned mid-trip
        # once the cone reveals the destination is already occupied. This is the
        # paper's "the robot has already finished it" applied while still en
        # route, and it is the term the vision cone gates most directly.
        if subtask == "pickup_meat":
            return pot in ("empty", UNKNOWN)
        if subtask == "pickup_onion":
            return board in ("empty", UNKNOWN)
        # Everything else (pickup_steak / pickup_garnish / pickup_washed_plate /
        # pickup_plate / deliver / dump_item) runs to completion: a station not
        # ready YET is a WAIT, not a redundancy, and arrival ends the errand (a
        # no-op INTERACT still re-decides next tick), so no deadlock and no
        # dithering.
        return True

    # -- decision -----------------------------------------------------------

    def _forced_held(self, held):
        """The subtask FORCED by whatever is in hand, or None if empty-handed.
        There is no ordering freedom in what to do with an item already held, so
        this is deterministic and SHARED by both modes (and by the sampling
        likelihood, subtask_distribution)."""
        # MISHA NEW CHANGE - if the destination for what we are holding is
        # occupied, put the item down on a counter instead of jamming.
        #
        # Without this the agent DEADLOCKS and every FOV delivers zero: it picks
        # up meat, walks to the pot, finds the teammate already filled it,
        # INTERACT silently no-ops, and decide() returns drop_meat again forever
        # because that branch was unconditional on holding meat. Expiring the
        # commitment did not help - re-deciding produced the same subtask. The
        # agent needs a legal alternative, not more patience.
        if held == "meat":
            return "drop_meat" if self.believed("pot") in ("empty", UNKNOWN) else "dump_item"
        if held == "onion":
            return "drop_onion" if self.believed("board") in ("empty", UNKNOWN) else "dump_item"
        # MISHA NEW CHANGE - 'dish' is the FINISHED steak+garnish plate, created by
        # the garnish pickup (overcooked_mdp.py:2481 ObjectState(..., 'dish', pos)).
        # It was previously lumped in with the clean plate, so the completed order
        # was carried to the SINK to be washed instead of served - the agent ran the
        # entire pipeline correctly and then threw the result away, delivering 0 at
        # every FOV. A name collision, not a logic error.
        if held == "dish":
            return "deliver"
        if held == "plate":
            return "drop_plate" if self.believed("sink") in ("empty", UNKNOWN) else "dump_item"
        if held == "washed_plate":
            # If the pot is SEEN empty there is no steak coming and none can be
            # started with both hands full, so set the plate down and go cook
            # rather than looping check_pot forever holding it.
            if self.believed("pot") == "ready":
                return "pickup_steak"
            return "dump_item" if self.believed("pot") == "empty" else "check_pot"
        if held in ("steak_dish", "dish_with_steak"):
            # Symmetric to the washed_plate case: if the board is SEEN empty there is
            # no garnish coming and none can be chopped with a steak in hand, so
            # set the steak down and go make one rather than looping check_board
            # forever holding it (a residual deadlock when the garnish was taken
            # between plating up and reaching the board).
            if self.believed("board") == "ready":
                return "pickup_garnish"
            return "dump_item" if self.believed("board") == "empty" else "check_board"
        # MISHA NEW CHANGE - anything else in hand is a FINISHED dish: deliver it.
        # Without this branch the agent reached pickup_garnish, ended up holding a
        # completed plate, matched no branch, and jammed. The full-vision agent got
        # FURTHEST down the pipeline and therefore jammed hardest - which is why
        # measured performance came out INVERTED in FOV (360 delivered 0, 30
        # delivered 2). The 2 were the teammate's deliveries, not the human's.
        if held is not None:
            return "deliver"
        return None

    def decide(self, state):
        """Choose a subtask - the paper's tau_t ~ P(.|o_{0:t}).

        Held-object handling is FORCED (_forced_held): there is no choice in what
        to do with an item already in hand. EMPTY-HANDED, the subtask is SAMPLED
        among the believed-available advancing subtasks, weighted toward a
        sensible order, and committed until observed unhelpful (_decide_sampling).

        There is NO fixed recipe order. Two cones produce genuinely different
        subtask SEQUENCES because their FOV-gated beliefs - hence their available
        sets, and their need to stop and LOOK - differ. That FOV-driven
        divergence is condition (A), and it is attributable to VISION alone: at a
        fixed random seed the sampler draws identically across cones, so any
        divergence between two FOVs comes from their beliefs, not the RNG.
        """
        held = state.players[self.agent_index].held_object
        forced = self._forced_held(held.name if held else None)
        if forced is not None:
            return forced
        pot, board, sink = self.believed("pot"), self.believed("board"), self.believed("sink")
        return self._decide_sampling(pot, board, sink)

    # -- the sampling policy: tau ~ P(.|o) ----------------------------------
    #
    # SAMPLE among the subtasks that are believed available and task-advancing,
    # then COMMIT to the sample while it is still perceived helpful (drop it the
    # instant belief shows it is not). The human is SELF-SUFFICIENT - it starts
    # its own meat and onion whenever a station is free rather than deferring to
    # a teammate it sees carrying an ingredient (deferral was tried and INVERTED
    # the FOV effect: both agents deferred the meat to each other, neither cooked
    # it, the better-sighted human starved). Team delivery is the success metric
    # (the human's prep feeding a teammate's delivery is still a team win); the
    # FOV effect lives in WHICH subtasks it chooses and when it must stop to LOOK.
    # The distribution is known in closed form (subtask_distribution), which is
    # what makes exact Bayesian FOV inference possible.

    def _unfound(self, *kinds):
        return any(not self.stations.get(k) for k in kinds)

    def _weights(self, cands):
        """P(tau) over the available advancing set: softmax of PRIORITY at the
        agent's temperature, then a SOFT teammate-awareness down-weight on the
        fetch the teammate is SEEN carrying (renormalised, never zeroed - so the
        human stays self-sufficient). A KNOWN categorical, so the robot can read
        exact likelihoods off it - and because subtask_distribution() calls THIS
        same function, the teammate reaction is reflected in the inference
        distribution automatically (no separate bookkeeping)."""
        T = self.temperature
        if T <= 0:                       # deterministic limit: top priority only
            best = max(cands, key=lambda c: PRIORITY.get(c, 0))
            w = {c: (1.0 if c is best else 0.0) for c in cands}
        else:
            w = {c: math.exp(PRIORITY.get(c, 0) / T) for c in cands}
        # teammate awareness (selection half), FOV-gated: rb is a concrete item
        # only while the teammate is in the cone. Soft-suppress the redundant
        # fetch, then renormalise - a lone option renormalises back to 1.
        rb = self.beliefs.get(self.ROBOT, Belief(UNKNOWN, -10 ** 9)).value
        redundant = ROBOT_HELD_FETCH.get(rb)
        if redundant in w:
            w[redundant] *= self.ROBOT_SUPPRESS
        z = sum(w.values())
        return {c: wc / z for c, wc in w.items()} if z > 0 else \
            {c: 1.0 / len(cands) for c in cands}

    def _available_advancing(self, pot, board, sink):
        """Support of the sampling distribution: subtasks an EMPTY-HANDED agent
        could start now that both have their precondition BELIEVED satisfied and
        move the recipe forward, COLLECTED into the sampling support. The two
        prep lines (start the steak, start the garnish) are EQUAL top priority in
        PRIORITY: starving the garnish deadlocks assembly."""
        c = []
        if pot == "empty":
            c.append("pickup_meat")            # start the steak (bottleneck)
        if board == "empty":
            c.append("pickup_onion")           # start the garnish
        if board == "chopping":
            c.append("chop_onion")             # advance the garnish
        if sink == "empty":
            c.append("pickup_plate")           # start a plate washing
        if sink == "washing":
            c.append("heat_washed_plate")         # advance the wash
        # ASSEMBLY (pickup_washed_plate -> pickup_steak -> pickup_garnish -> deliver)
        # is only offered once BOTH components are actually ready: garnish done
        # (board ready), plate washed (sink ready), and a steak ready/cooking.
        # Flexible ordering must STATE this dependency explicitly, else it can
        # pick up a steak with no garnish and deadlock (it cannot chop with full
        # hands). Grabbing EARLIER was tried and is worse - it just deadlocks a
        # different way.
        if sink == "ready" and board == "ready" and pot in ("ready", "cooking"):
            c.append("pickup_washed_plate")
        # STATION-YIELD (teammate-receptiveness, POSITION channel): drop the tasks
        # that use a station the teammate is SEEN standing at / facing, so the
        # human does not compete or collide for a station its partner is already
        # working - but ONLY if a non-blocked task remains (soft: never strands
        # the human, so self-sufficiency holds). FOV-gated via _robot_faced_kind.
        # Placed here, in the sampling support, so subtask_distribution() - which
        # calls this - reflects the yield and the robot's FOV inference stays exact.
        block = STATION_TASKS.get(self._robot_faced_kind(), frozenset())
        if block:
            kept = [x for x in c if x not in block]
            if kept:
                c = kept
        return c

    def _sample_still_helpful(self, subtask, pot, board, sink):
        """'Commit only as long as it is perceived helpful.' True while the
        sampled errand's precondition still holds under CURRENT belief. UNKNOWN
        (a forgotten belief) counts as still-helpful - forgetting is not
        observing, the same rule as _commit_still_useful.

        Teammate awareness: the sampler has its own stickiness, so the redundant-
        fetch drop must be applied HERE too, or action()'s abandonment via
        _commit_still_useful is undone the same tick by a sticky re-sample."""
        if self._robot_redundant(subtask):
            return False
        if subtask == "pickup_meat":
            return pot in ("empty", UNKNOWN)
        if subtask == "pickup_onion":
            return board in ("empty", UNKNOWN)
        if subtask == "chop_onion":
            return board in ("chopping", UNKNOWN)
        if subtask == "pickup_plate":
            return sink in ("empty", UNKNOWN)
        if subtask == "heat_washed_plate":
            return sink in ("washing", UNKNOWN)
        if subtask == "pickup_washed_plate":
            return sink in ("ready", UNKNOWN) or (pot == "ready" and board == "ready")
        return True

    def _need_info_lookup(self, pot, board, sink):
        """No advancing subtask is believed available -> no free CHOICE, the
        agent must LOOK for what blocks it, PER NEED and in pipeline order.

        Two distinctions matter here. (1) A never-FOUND station (no known
        location) can only be resolved by EXPLORE - walking to check_ a place you
        have never seen has no target and spins forever (this once cost fov60 on
        steak_island 211 look-steps and 0 deliveries). A DECAYED belief (known
        location, stale value) is resolved by check_. (2) Looking is gated PER
        NEED: the agent looks for the pot before it has found the serving hatch,
        rather than demanding a complete map before doing any work."""
        if self._unfound("pot", "meat"):
            return "explore"
        if pot == UNKNOWN:
            return "check_pot"
        if self._unfound("board", "onion"):
            return "explore"
        if board == UNKNOWN:
            return "check_board"
        if self._unfound("sink", "dish"):
            return "explore"
        if sink == UNKNOWN:
            return "check_sink"
        if self._unfound("serve"):
            return "explore"
        return "wait"

    def _decide_sampling(self, pot, board, sink):
        """Empty-handed choice in sampling mode. Stick with the current sample
        while helpful; else (re-)sample among what is available; else go look."""
        if self._sampled is not None and \
                self._sample_still_helpful(self._sampled, pot, board, sink):
            return self._sampled
        self._sampled = None
        cands = self._available_advancing(pot, board, sink)
        if cands:
            w = self._weights(cands)
            self._sampled = self._rng.choices(list(w), weights=list(w.values()), k=1)[0]
            return self._sampled
        return self._need_info_lookup(pot, board, sink)

    def subtask_distribution(self, state, committed=None):
        """The known categorical P(tau | o_{0:t}, theta) this policy uses RIGHT
        NOW, as {subtask: prob} - what makes EXACT Bayesian FOV inference
        possible (the robot reads the probability of the observed subtask off it,
        no epsilon smoothing).

        `committed` is the advancing subtask the human is believed to be pursuing
        (the robot passes the last observed one). While it stays helpful the
        policy continues it deterministically; otherwise the mass is the softmax
        over the available advancing set (a fresh draw), or the single forced
        look if none is available. Mirrors _decide_sampling without drawing."""
        held = state.players[self.agent_index].held_object
        forced = self._forced_held(held.name if held else None)
        if forced is not None:
            return {forced: 1.0}
        pot, board, sink = self.believed("pot"), self.believed("board"), self.believed("sink")
        if committed in SAMPLING_SUBTASKS and \
                self._sample_still_helpful(committed, pot, board, sink):
            return {committed: 1.0}
        cands = self._available_advancing(pot, board, sink)
        if cands:
            return self._weights(cands)
        return {self._need_info_lookup(pot, board, sink): 1.0}

    def _bfs_step(self, state, goal_cells, adjacent=True):
        """Route toward goal_cells, optionally detouring around the teammate.

        With avoid_robot on (opt-in, DEFAULT OFF) and the teammate's CURRENT cell
        seen this tick, first try a route with that cell blocked; only if no such
        route exists fall back to routing through it, so traffic-avoidance never
        starves the human. With avoid_robot off this is exactly _bfs_step_core -
        the validated routing is untouched by default."""
        if self.avoid_robot and self._robot_seen is not None and self._robot_seen[2] == self.t:
            pos = self._robot_seen[0]
            if self.known_terrain.get(pos) == ' ' and pos != state.players[self.agent_index].position:
                saved = self.known_terrain[pos]
                self.known_terrain[pos] = 'X'          # temporarily impassable
                try:
                    step = self._bfs_step_core(state, goal_cells, adjacent)
                finally:
                    self.known_terrain[pos] = saved
                if step is not None:
                    return step                        # a detour exists -> take it
                # no detour: fall through and route through the teammate (no starve)
        return self._bfs_step_core(state, goal_cells, adjacent)

    def _bfs_step_core(self, state, goal_cells, adjacent=True):
        """First action on a shortest path, searched over ONLY the cells this
        agent has seen and knows to be floor.

        MISHA NEW CHANGE - this replaces mlp.mp.get_plan(), which was the largest
        remaining cheat in the agent: the medium-level planner holds the FULL
        map, so the agent was routing optimally through rooms it had never laid
        eyes on, and mdp.get_valid_player_positions() told it which unseen tiles
        were even walkable. Now an unseen tile is not known to be floor OR wall,
        so it cannot be traversed - the agent must literally see a route before
        it can follow one.

        adjacent=True searches for a cell NEXT TO a goal (you interact with a
        station from beside it); False searches for the goal cell itself.
        """
        start = state.players[self.agent_index].position
        goals = set()
        for g in goal_cells:
            if adjacent:
                for d in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                    goals.add((g[0] - d[0], g[1] - d[1]))
            else:
                goals.add(g)
        if not goals:
            return None
        walk = {c for c, t in self.known_terrain.items() if t == ' '}
        walk.add(start)
        if start in goals:
            return "ARRIVED"
        frontier = [(start, None)]
        seen = {start}
        while frontier:
            nxt = []
            for cell, first in frontier:
                for d in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                    nb = (cell[0] + d[0], cell[1] + d[1])
                    if nb in seen or nb not in walk:
                        continue
                    step = first if first is not None else d
                    if nb in goals:
                        return step
                    seen.add(nb)
                    nxt.append((nb, step))
            frontier = nxt
        return None

    def _cone_gain(self, state, pos, orient):
        """How many currently-unseen tiles a given pose would reveal.

        Cheap information-gain estimate used to choose between turning on the
        spot and walking somewhere. A narrow cone gains far more from ROTATING
        (a 30-degree agent sweeps four disjoint wedges by turning in place) than
        a wide one does, which is why this makes exploration FOV-dependent
        rather than a fixed frontier walk.
        """
        saved = state.players[self.agent_index].orientation
        state.players[self.agent_index].orientation = orient
        gain = 0
        px, py = pos
        try:
            for dy in range(-SIGHT_RADIUS, SIGHT_RADIUS + 1):
                for dx in range(-SIGHT_RADIUS, SIGHT_RADIUS + 1):
                    c = (px + dx, py + dy)
                    if c in self.seen_cells:
                        continue
                    if not (0 <= c[0] < self._w and 0 <= c[1] < self._h):
                        continue
                    if self.visible(state, c):
                        gain += 1
        finally:
            state.players[self.agent_index].orientation = saved
        return gain

    def _explore_action(self, state):
        """Information-seeking exploration.

        Two changes over a plain frontier walk, both of which matter more the
        narrower the cone:

        1. TURN IF LOOKING IS CHEAPER THAN WALKING. Rotating costs one step and
           for a narrow cone can reveal a whole new wedge. Only if no rotation
           reveals anything does the agent spend steps travelling.
        2. GIVE UP WHEN EXPLORING STOPS PAYING. If several consecutive explore
           steps reveal nothing new, the agent stops searching and works with
           the stations it has already found. Without this a 30-degree human
           spends the entire episode wandering (measured: its whole subtask
           trace was ['explore'] and it delivered 0), which is not sensible
           behaviour - a real person gives up and makes do.
        """
        player = state.players[self.agent_index]
        pos, orient = player.position, player.orientation

        # 1. is turning worth it?
        best_turn, best_gain = None, self._cone_gain(state, pos, orient)
        for d in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            if d == orient:
                continue
            g = self._cone_gain(state, pos, d)
            if g > best_gain:
                best_turn, best_gain = d, g
        if best_turn is not None and best_gain >= TURN_GAIN_THRESHOLD:
            return best_turn

        # 2. otherwise walk to the frontier of our OWN known map
        frontier = [c for c, t in self.known_terrain.items() if t == ' '
                    and any((c[0] + dd[0], c[1] + dd[1]) not in self.seen_cells
                            for dd in ((0, -1), (0, 1), (1, 0), (-1, 0)))]
        if frontier:
            step = self._bfs_step(state, frontier, adjacent=False)
            if step and step != "ARRIVED":
                return step

        order = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        return order[(order.index(orient) + 1) % 4] if orient in order else (0, -1)

    # -- acting -------------------------------------------------------------

    def execute(self, state, subtask, explore_act=None):
        """Turn a CHOSEN subtask into a concrete action. PURE: writes nothing on
        self and consumes no RNG, so it is safe to call speculatively.

        Returns (action, arrived, used_explore).
          arrived       the errand completed this tick (action() resets the
                        commitment on it)
          used_explore  an exploration action was emitted, so the CALLER should
                        bump n_explore. The counter lives outside on purpose -
                        it is the one side effect this branch used to have, and
                        keeping it here would make the method unusable for the
                        speculative queries below.

        MISHA NEW CHANGE - split out of action() unchanged so the FOV inference
        (robot/policy/old/inference/bayes_fov_sampling.py) can ask "what action
        would this agent emit if it were executing tau?" for many hypothetical
        tau per tick. It cannot go through action() to ask: that samples from
        _rng and writes _current / _sampled / t / subtask_log, which would both
        desynchronise a shadow from the real trajectory and correlate its draws
        with the real human's. The filter previously kept its own copy of this
        logic, which worked but would silently drift the first time the routing
        here changed; now there is one implementation.

        No new information is reachable from here: routing uses only the agent's
        OWN discovered stations and known_terrain, exactly as before.

        `explore_act` is a pure optimisation. _explore_action() depends only on
        (state, this agent's beliefs), so within a single tick it is the same for
        every subtask; a caller evaluating many subtasks may compute it once and
        pass it in. Leave it None and it is computed on demand, as action() does.
        """
        if subtask == "explore":
            act = self._explore_action(state) if explore_act is None else explore_act
            return act, False, True

        # MISHA NEW CHANGE - movement is planned by the AGENT's own BFS over
        # cells it has seen, not by planner.act()'s mlp.mp.get_plan(), which
        # searches the full map. The planner is now used only to look up
        # which station kind a subtask targets.
        targets = [t for t in self.stations.get(
            self.planner.target_kind(subtask), [])]
        player = state.players[self.agent_index]
        faced = (player.position[0] + player.orientation[0],
                 player.position[1] + player.orientation[1])
        if faced in targets:
            return (Action.STAY if subtask.startswith("check_")
                    else Action.INTERACT), True, False

        step = self._bfs_step(state, targets, adjacent=True)
        if step == "ARRIVED":
            # standing beside it but facing elsewhere: turn to face it
            tgt = min(targets, key=lambda g: abs(g[0] - player.position[0])
                      + abs(g[1] - player.position[1]))
            act = (tgt[0] - player.position[0], tgt[1] - player.position[1])
            if act not in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                act = Action.STAY
            return act, False, False
        if step is None:
            # no route through KNOWN floor - go look for one
            act = self._explore_action(state) if explore_act is None else explore_act
            return act, False, True
        return step, False, False

    def action(self, state):
        self.observe(state)

        # Commitment WITH BELIEF-SENSITIVE ABANDONMENT (see _commit_still_useful
        # and the module docstring). EXACTLY the paper's rule: the agent keeps
        # pursuing its current errand until it either COMPLETES or it OBSERVES,
        # through its FOV-gated beliefs, that the errand is pointless - e.g. it
        # sees the pot the teammate already filled and stops fetching meat for
        # it. There is NO step-count patience. It does NOT drop an errand just
        # because a belief was forgotten (UNKNOWN); that would be dithering.
        # No deadlock results: arriving at a station ends the errand below (even
        # a no-op INTERACT sets _current=None and re-decides next tick), and an
        # occupied destination is caught here on sight. Because the abandonment
        # trigger is FOV-gated, a wide cone notices invalidation early and
        # re-plans while a narrow cone completes the wasted trip - that asymmetry
        # IS the FOV effect.
        if self._current in COMMITTED \
                and self._commit_still_useful(self._current):
            subtask = self._current
        else:
            if self._current in COMMITTED and \
                    not self._commit_still_useful(self._current):
                self.n_abandoned += 1     # dropped: OBSERVED no longer useful
            subtask = self.decide(state)

        act, arrived, used_explore = self.execute(state, subtask)
        if used_explore:
            self.n_explore += 1

        if subtask.startswith("check_"):
            self.n_checks += 1
            if arrived:
                self._current = None      # look complete; re-decide next tick
            else:
                self._current = subtask
        elif arrived:
            # Errand finished. If what we find contradicts the belief that sent
            # us here, count it as wasted - that is the performance cost of
            # narrow vision, and the reason wide FOV should deliver more.
            if subtask == "deliver":
                self.n_delivered += 1
            wk = WASTED_IF_OCCUPIED.get(subtask)
            if wk is not None and self.believed(wk) not in ("empty", UNKNOWN):
                self.n_wasted_commits += 1
            self._current = None
        else:
            self._current = subtask

        self.prev_chosen_subtask = subtask
        self.subtask_log.append(subtask)
        self.t += 1
        return act, {"subtask": subtask}
