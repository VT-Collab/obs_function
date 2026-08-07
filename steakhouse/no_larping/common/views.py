"""The two world-views the ladder is evaluated against.

TruthView   - everything, always. What the robot baselines get.
BeliefView  - only what has been SEEN, with contents that expire. What the
              human gets.

Both expose the identical query surface, so common/tasks.py cannot tell them
apart. Every behavioural difference between the human and the robot comes from
which of these it is holding, and nothing else.
"""
from .tasks import (POT, BOARD, SINK, COUNTER, STATION_CHARS, UNKNOWN, EMPTY)

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

    def any_ready(self, char):
        return bool(self.ready(char))

    def any_empty(self, char):
        return bool(self.empty(char))

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
    """FOV-gated memory. Written ONLY from cells currently visible.

    known_terrain / stations are PERMANENT - walls and worktops do not move, and
    a station once found is never un-found. contents DECAY to UNKNOWN after
    forget_horizon, counters included: a counter not looked at recently is
    genuinely unknown, not assumed empty.
    """

    def __init__(self, forget_horizon):
        self.forget_horizon = forget_horizon
        self.known_terrain = {}
        self._stations = {}
        self.contents = {}                 # cell -> (name_or_EMPTY, ready, t)
        self.robot = None                  # (pos, orient, held, t)
        self.t = 0

    # -- perception ---------------------------------------------------------
    def observe(self, mdp, state, seen_cells, t, other_idx=None):
        self.t = t
        terrain = mdp.terrain_mtx
        for (x, y) in seen_cells:
            ch = terrain[y][x]
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
            if tuple(p.position) in seen_cells:
                held = p.held_object.name if p.held_object else None
                self.robot = (tuple(p.position), tuple(p.orientation), held, t)
        if self.robot and self.t - self.robot[3] > self.forget_horizon:
            self.robot = None

    def _fresh(self, cell):
        rec = self.contents.get(cell)
        if rec is None or self.t - rec[2] > self.forget_horizon:
            return None                    # UNKNOWN: never claim "empty"
        return rec

    # -- identical query surface to TruthView -------------------------------
    def stations(self, char):
        return sorted(self._stations.get(char, ()))

    def ready(self, char):
        out = []
        for c in self.stations(char):
            rec = self._fresh(c)
            if rec and rec[1]:
                out.append(c)
        return out

    def in_progress(self, char):
        out = []
        for c in self.stations(char):
            rec = self._fresh(c)
            if rec and rec[0] is not EMPTY and not rec[1]:
                out.append(c)
        return out

    def empty(self, char):
        out = []
        for c in self.stations(char):
            rec = self._fresh(c)
            if rec and rec[0] is EMPTY:
                out.append(c)
        return out

    def any_ready(self, char):
        return bool(self.ready(char))

    def any_empty(self, char):
        return bool(self.empty(char))

    def counters_holding(self, name):
        out = []
        for c in self.stations(COUNTER):
            rec = self._fresh(c)
            if rec and rec[0] == name:
                out.append(c)
        return out

    def free_counters(self):
        out = []
        for c in self.stations(COUNTER):
            rec = self._fresh(c)
            if rec and rec[0] is EMPTY:
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
