"""The two world-views the ladder is evaluated against.

TruthView   - everything, always. What the robot baselines get.
BeliefView  - only what has been SEEN, with contents that expire. What the
              human gets.

Both expose the identical query surface, so common/tasks.py cannot tell them
apart. Every behavioural difference between the human and the robot comes from
which of these it is holding, and nothing else.
"""
from .tasks import POT, BOARD, SINK, COUNTER, STATION_CHARS, EMPTY

# how a station's occupant is read off the mdp
_READY_FN = {POT: "steak_ready_at_location",
             BOARD: "garnish_ready_at_location",
             SINK: "plate_washed_at_location"}


def _station_status(mdp, state, cell, char):
    """(name_or_EMPTY, ready) for a dynamic station cell."""
    obj = state.objects.get(cell)
    if obj is None:
        return EMPTY, False
    fn = _READY_FN.get(char)
    ready = bool(getattr(mdp, fn)(state, cell)) if fn else False
    return obj.name, ready


class TruthView:
    """Full observability. No FOV anywhere in this class - by design."""

    def __init__(self, mdp, state):
        self.mdp, self.state = mdp, state
        self.terrain = mdp.terrain_mtx
        self._stations = {}
        for y, row in enumerate(self.terrain):
            for x, ch in enumerate(row):
                if ch in STATION_CHARS:
                    self._stations.setdefault(ch, []).append((x, y))
        self.walkable = {(x, y) for y, row in enumerate(self.terrain)
                         for x, ch in enumerate(row) if ch == " "}

    # -- queries the ladder uses -------------------------------------------
    def stations(self, char):
        return list(self._stations.get(char, []))

    def _status(self, char, cell):
        return _station_status(self.mdp, self.state, cell, char)

    def ready(self, char):
        return [c for c in self.stations(char) if self._status(char, c)[1]]

    def in_progress(self, char):
        return [c for c in self.stations(char)
                if self._status(char, c)[0] not in (EMPTY,) and not self._status(char, c)[1]]

    def empty(self, char):
        return [c for c in self.stations(char) if self._status(char, c)[0] is EMPTY]

    def counters_holding(self, name):
        out = []
        for c in self.stations(COUNTER):
            obj = self.state.objects.get(c)
            if obj is not None and obj.name == name:
                out.append(c)
        return out

    def free_counters(self):
        return [c for c in self.stations(COUNTER) if c not in self.state.objects]


class BeliefView:
    """FOV-gated memory. Written ONLY from cells currently visible (plus felt terrain).

    Five stores, and which of them decays is the whole design:

      known_terrain {cell: char}      PERMANENT. Walls and worktops do not move.
      _stations     {char: {cell}}    PERMANENT. A station once found is never
                                      un-found - but one never SEEN is not in
                                      here at all, so no subtask can name it.
                                      That is the discovery channel.
      contents      {cell: (name, ready, t)}   DECAYS after forget_horizon,
                                      counters included: a counter not looked at
                                      recently is genuinely unknown, not assumed
                                      empty. Counters are where a handoff lands,
                                      so this is the perishable, decision-
                                      relevant half of the belief.
      seen_count    {cell: ticks}     Never decays. Ticks each cell has spent
                                      inside the cone; the human's explore uses
                                      it as a staleness proxy once the map is
                                      closed. See the note on the field.
      robot         (pos, orient, held, t)     TWO CLOCKS - see below.

    Plus `optimistic`, a flag the CALLER toggles to run the ladder twice; see
    _maybe() for what it does and why one pass is not enough.

    The belief about the robot has TWO halves on two clocks, because they are
    falsified by different things. Looking at the cell they were in and finding
    it empty refutes their POSITION on the spot; it says nothing at all about
    their HANDS, which only ever expire with time. So the human can be in the
    entirely reasonable state of knowing the robot is carrying meat while having
    no idea where it went.
    """

    def __init__(self, forget_horizon):
        self.forget_horizon = forget_horizon
        self.known_terrain = {}
        self._stations = {}

        # cell -> (name_or_EMPTY, ready, t). The three fields are, in order: WHAT
        # is on the tile, whether it is FINISHED, and the tick we last actually
        # LOOKED. Stations and counters share the record because the ladder asks
        # both the same questions -- a counter simply never has ready True, since
        # nothing on a worktop is mid-timer.
        #
        #   (3, 1): ("steak", False, 12)   grill: steak on it, still cooking
        #   (5, 1): (None,    False, 12)   counter: observed bare
        #   (7, 4): ("onion", False, 30)   counter: somebody left an onion
        #
        # The timestamp is the load-bearing field. Absence from this dict and a
        # record older than forget_horizon mean the SAME thing -- unknown -- and
        # _fresh() collapses the two, which is what stops anything downstream
        # reading a stale record as fact. Throughout this file `rec` is one of
        # these tuples, straight out of _fresh().
        self.contents = {}
        # how many ticks each cell has spent inside the cone. Cheap to keep, and
        # it is what the human's explore falls back on ONCE THE MAP IS CLOSED:
        # with no frontier left, "where should I look" has no unknown to aim at,
        # and least-seen is very nearly most-out-of-date. It is deliberately NOT
        # used while there is still map to find -- biasing that phase by seen
        # count walks past near unknowns to reach far ones and measured strictly
        # worse (1589 distinct tiles covered fell to 1258). Never decays: it is a
        # census of attention, not a belief.
        self.seen_count = {}
        # THE SECOND PASS. When False, a station whose contents have decayed is
        # dropped instead of guessed at, so the caller can ask "what do I KNOW"
        # and "what am I willing to GUESS" as two separate questions and prefer
        # the first. Owned by the caller, not by this class; see _maybe() for why
        # one pass in either setting is worse than two.
        self.optimistic = True
        # (pos, orient, held, t). pos and orient go None on their own, ahead of
        # held, when we look at where they were and they have moved on: see
        # observe(). held is the slow half of this belief, position the fast one.
        self.robot = None
        self.t = 0

    # -- perception ---------------------------------------------------------
    def observe(self, mdp, state, seen_cells, t, other_idx=None, felt=()):
        """Write one tick of perception in. The ONLY way anything gets in here.

        `seen_cells` is the cone, `felt` the four tiles being touched, and the
        two are treated very differently - see below. Nothing else about `state`
        is read, so an agent holding this view genuinely cannot act on anything
        it has not looked at.
        """
        self.t = t
        terrain = mdp.terrain_mtx
        # TOUCH, not sight. You can always tell whether the tile you are standing
        # against is floor or furniture without looking at it, and an agent that
        # cannot is not blind so much as paralysed: with a 30 cone it could not
        # plan a single step sideways, because unseen ground is not walkable, so
        # it span on the spot instead of exploring.
        #
        # ONLY the terrain is felt. Not what is on it, and NOT onto the station
        # list -- a pot you have brushed past but never looked at stays
        # undiscovered, so no subtask can target it. That is the line this must
        # not cross: "adjacency is free" is precisely what made the stock model
        # FOV-blind (it let an agent act on any station it was standing next to),
        # and feeling the floor buys mobility without buying any of that.
        for (x, y) in felt:
            if 0 <= y < len(terrain) and 0 <= x < len(terrain[0]):
                self.known_terrain.setdefault((x, y), terrain[y][x])
        for (x, y) in seen_cells:
            ch = terrain[y][x]
            self.seen_count[(x, y)] = self.seen_count.get((x, y), 0) + 1
            self.known_terrain[(x, y)] = ch
            if ch in STATION_CHARS:
                self._stations.setdefault(ch, set()).add((x, y))
            if ch in (POT, BOARD, SINK):
                self.contents[(x, y)] = _station_status(mdp, state, (x, y), ch) + (t,)
            elif ch == COUNTER:
                obj = state.objects.get((x, y))
                self.contents[(x, y)] = ((obj.name if obj else EMPTY), False, t)
        if other_idx is not None:
            p = state.players[other_idx]
            pos = tuple(p.position)
            if pos in seen_cells:
                held = p.held_object.name if p.held_object else None
                self.robot = (pos, tuple(p.orientation), held, t)
            elif self.robot and self.robot[0] is not None and self.robot[0] in seen_cells:
                # We are looking straight at where we last saw them and they are
                # not there. NOT seeing is evidence: WHERE they are becomes
                # unknown at once, instead of a confidently wrong position that
                # sits there until forget_horizon and misroutes the contention
                # check. WHAT they were holding survives -- an empty cell says
                # nothing about their hands, and that memory is the whole of
                # Channel A. The timestamp is NOT refreshed, so the hands belief
                # goes on expiring on the schedule it was already on; staring at
                # an empty tile must not renew it.
                self.robot = (None, None, self.robot[2], self.robot[3])
        if self.robot and self.t - self.robot[3] > self.forget_horizon:
            self.robot = None

    def _fresh(self, cell):
        """The record for `cell`, or None meaning UNKNOWN.

        Never-looked-at and looked-at-too-long-ago collapse to the same answer on
        purpose. The distinction the callers must never lose is UNKNOWN versus
        EMPTY: EMPTY is an observation ("I looked, there was nothing there") and
        is a perfectly good reason to walk over with an onion, while UNKNOWN is
        the absence of one. Returning a stale record here would let the ladder
        treat the second as the first.
        """
        rec = self.contents.get(cell)
        if rec is None or self.t - rec[2] > self.forget_horizon:
            return None                    # UNKNOWN: never claim "empty"
        return rec

    # -- identical query surface to TruthView -------------------------------
    #
    # OPTIMISM UNDER DECAY. A station we have FOUND but whose status has expired
    # is offered as available rather than dropped. Pessimism was worse in a way
    # that looked like a different bug entirely: with everything stale, nothing
    # was legal, so the agent fell through to EXPLORE and went off re-mapping
    # territory it already knew - measured on annex at fov 30, 210 of 400 ticks
    # exploring, on 92% of which it knew exactly where the board was and simply
    # did not know what was on it. It never walked the three tiles back to look.
    #
    # Optimism sends it instead, and on arrival the cone tells it the truth. A
    # wrong guess costs a round trip, which is the third FOV channel the design
    # asks for and the one that was missing: a narrow cone pays in wasted walking
    # rather than in nothing happening at all. It cannot loop, because arriving
    # refreshes the belief - the second decision is made on a fresh observation.
    #
    # DISCOVERY is untouched and is still the strongest channel: a station never
    # seen is not in stations() at all, so it cannot be guessed at. You may only
    # be optimistic about something you have actually found.

    def _maybe(self, char, test):
        """Stations passing `test`, optionally guessing at the decayed ones.

        `test(rec)` is the caller's question about a FRESH record -- ready,
        occupied, empty. A station with no fresh record never reaches it, and is
        instead decided by `optimistic` alone.

        `optimistic` is toggled by the caller for a TWO-PASS decision: act on
        what you actually KNOW first, and only guess if that leaves you with
        nothing to do. Guessing on equal footing with knowledge is worse than
        either, and in a way that looks like a different bug: a stale sink is
        assumed ready, collect_plate (tier 4) outranks get_plate (tier 7), so the
        human walks to the sink, sees it is not ready, switches to the dispenser,
        lets the sink belief decay on the way, and flips straight back. Seen on
        banquet_pass at fov 30 as an endless pick-plate / collect-plate shuffle
        in which no plate was ever picked up.
        """
        out = []
        for c in self.stations(char):
            rec = self._fresh(c)
            if test(rec) if rec is not None else self.optimistic:
                out.append(c)
        return out

    def stations(self, char):
        return sorted(self._stations.get(char, ()))

    def ready(self, char):
        return self._maybe(char, lambda rec: rec[1])

    def in_progress(self, char):
        return self._maybe(char, lambda rec: rec[0] is not EMPTY and not rec[1])

    def empty(self, char):
        return self._maybe(char, lambda rec: rec[0] is EMPTY)

    def counters_holding(self, name):
        # NOT optimistic: you cannot hallucinate an item onto a worktop. An
        # unknown counter might hold anything, and treating it as holding
        # whatever we happen to need would have the agent chasing imaginary
        # garnish across the kitchen.
        out = []
        for c in self.stations(COUNTER):
            rec = self._fresh(c)
            if rec and rec[0] == name:
                out.append(c)
        return out

    def free_counters(self):
        # optimistic, like the stations: go and try to put it down. If it turns
        # out to be occupied we have lost a trip, not the use of the tier.
        out = []
        for c in self.stations(COUNTER):
            rec = self._fresh(c)
            if (rec[0] is EMPTY) if rec is not None else self.optimistic:
                out.append(c)
        return out

    # -- pathing ------------------------------------------------------------
    @property
    def walkable(self):
        """Only ground actually seen. A corridor never looked at cannot be
        planned through - this is the second FOV channel after discovery."""
        return {c for c, ch in self.known_terrain.items() if ch == " "}

    def frontier(self):
        """Known floor bordering something never seen - where to go looking."""
        out = []
        for c, ch in self.known_terrain.items():
            if ch != " ":
                continue
            for d in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                if (c[0] + d[0], c[1] + d[1]) not in self.known_terrain:
                    out.append(c)
                    break
        return sorted(out)
