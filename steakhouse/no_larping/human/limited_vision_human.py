"""The limited-vision human. Everything it knows, it had to walk over and look at.

ONE SENTENCE. Every tick it observes through a cone, asks common/tasks.py what
its BELIEFS say is legal and reachable, takes the best of those on the ladder,
and walks one step of an A* laid over ground it has actually seen. Nothing is
planned ahead and nothing is carried over except a small amount of stickiness,
so "sees something better and switches" is the default path rather than a
special case.

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
"""
# =============================================================================
# IMPLEMENTATION
# =============================================================================
import os
import sys

sys.path.insert(0, os.environ.get(
    "STEAK_ROOT", "/Users/mishafu/Desktop/obs_function/steakhouse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overcooked_ai_py.mdp.actions import Action              # noqa: E402
from common import geometry as geo                            # noqa: E402
from common.tasks import (legal_subtasks, FETCH_OF, T_EXPLORE,  # noqa: E402
                          TIER_NAME)
from common.views import BeliefView                            # noqa: E402

# Ticks before what is ON a station decays back to unknown. This is the whole of
# the staleness channel: shorten it and even a 360 cone has to keep re-checking,
# lengthen it and a 30 cone stops paying for its blind spots.
FORGET_HORIZON = 30
# Ticks of UNBROKEN exploring before the tile we are stuck at is blacklisted as a
# frontier target and the committed goal is dropped. It is the only timer in the
# agent, and it is a way out of "the only unknown left is behind a wall I cannot
# see through", not a general-purpose give-up. Reset the instant anything on the
# ladder becomes legal again.
PATIENCE = 30


class LimitedVisionHuman:
    """Ladder-driven, FOV-gated teammate. See the module docstring for the design.

    Every tick: observe -> ask the ladder what BELIEF says is legal and reachable
    -> take the best of those -> walk one step or INTERACT. No plan, no
    hysteresis, no scores. The only state that survives a tick is the belief
    itself, the last subtask (for within-tier stickiness) and the explore goal.
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
        # tiles PATIENCE gave up standing at, dropped from the frontier until the
        # cone finds new ground -- see observe(). Not a permanent blacklist: a
        # frontier that was hopeless from here may be trivial from ten tiles on.
        self.abandoned = set()
        self.last_subtask = None
        self._recent = []          # last 12 positions, so _explore can spot a loop
        self._goal = None          # the explore target we are committed to
        self.log = []

    # -- perception ---------------------------------------------------------
    def observe(self, state):
        """Fold this tick's cone into the belief, and feel the four tiles we touch.

        Seeing new ground clears `abandoned`: a goal PATIENCE gave up on was only
        hopeless given what we knew then, and the map having grown is exactly the
        evidence that it might not be any more. Without this the frontier only
        ever shrinks and a long episode ends up with nothing left to aim at.
        """
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
        return seen

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
        """
        me = state.players[self.agent_index]
        pos = tuple(me.position)
        held = me.held_object.name if me.held_object else None
        walk = self.view.walkable | {pos}
        robot_held = self.view.robot[2] if self.view.robot else None

        robot_pos = self.view.robot[0] if self.view.robot else None

        # one memo, used both to decide legality and to rank it -- these were two
        # separate judgements before, which is exactly how the human ended up
        # committed to a station it could see and never reach.
        _d = {}

        def dist(cell):
            if cell not in _d:
                _d[cell] = geo.path_len(walk, pos, cell)
            return _d[cell]

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
            if self.react_to_robot and robot_pos is not None:
                rd = geo.path_len(walk | {robot_pos}, robot_pos, cell)
                contested = int(rd is not None and rd < d)
            scored.append((tier, redundant + contested, d, cell, verb))
        scored.sort()
        return scored

    def decide(self, state):
        """The one subtask we are doing this tick, or EXPLORE if there is none.

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
        """
        self.observe(state)
        tier, _, _, cell, verb = self.decide(state)
        self.last_subtask = (TIER_NAME[tier], verb, cell)

        me = state.players[self.agent_index]
        pos, orient = tuple(me.position), tuple(me.orientation)

        walk = self.view.walkable | {pos}
        if verb == "explore":
            act = self._explore(pos, orient)
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
    def set_agent_index(self, i):
        self.agent_index = i
        self.other_index = 1 - i

    def set_mdp(self, mdp):
        self.mdp = mdp
