"""Vision cone, line of sight, and pathing. No agent logic lives here.

The one rule that matters: '#' is the ONLY terrain that blocks sight. Counters,
pots, sinks, boards and dispensers are all see-through. Walls exist purely to
partition the field of view, so occlusion is a property of the LAYOUT rather
than of where the furniture happens to sit.
"""
import collections
from math import cos, radians, hypot
from heapq import heappush, heappop

WALL = "#"
FLOOR = " "
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # N, S, W, E (Direction order)


def los_clear(terrain, a, b):
    """True iff the straight line a->b crosses no wall. Endpoints excluded."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps <= 1:
        return True
    for i in range(1, steps):
        cx = int(round(x0 + dx * i / steps))
        cy = int(round(y0 + dy * i / steps))
        if (cx, cy) == a or (cx, cy) == b:
            continue
        if terrain[cy][cx] == WALL:
            return False
    return True


# (pos, orient, fov) -> frozenset, per terrain. The cone is a pure function of
# those four and the terrain never changes during an episode, while an agent
# revisits the same tile facing the same way constantly -- and every rollout tick
# of every candidate re-asks the question. The entry holds a reference to its own
# terrain so the id() key cannot be reused by a later object at the same address.
_VIS_CACHE = {}
_VIS_CACHE_MAX = 4000


def visible_cells(terrain, pos, orient, fov):
    """Every cell the agent can see this tick, as a frozenset. Memoised.

    in_cone AND los_clear. There is deliberately NO sight radius: the cone runs
    to the edge of the map and only angle and walls limit it. At fov=360 the
    cone test is vacuous but LOS is not - omnidirectional is never omniscient.

    Returns a FROZENset so a caller that tried to mutate the shared answer would
    raise here rather than silently corrupt every later lookup.
    """
    key = id(terrain)
    ent = _VIS_CACHE.get(key)
    if ent is None or ent[0] is not terrain:
        ent = (terrain, {})
        _VIS_CACHE[key] = ent
    sub = ent[1]
    kk = (pos, orient, fov)
    got = sub.get(kk)
    if got is None:
        got = frozenset(_visible_cells_uncached(terrain, pos, orient, fov))
        # BOUNDED, and it has to be. Unbounded, several worker processes each
        # holding every (cell, facing, cone) it ever asked for drove this machine
        # to 13.9G of 14.3G swap and the OS killed the run mid-grid. The live set
        # is one agent's recent poses, so a flat cap with a wholesale clear costs
        # a handful of recomputes and cannot leak.
        if len(sub) >= _VIS_CACHE_MAX:
            sub.clear()
        sub[kk] = got
    return got


def _visible_cells_uncached(terrain, pos, orient, fov):
    h, w = len(terrain), len(terrain[0])
    ax, ay = pos
    fx, fy = orient
    cut = cos(radians(fov / 2.0))
    seen = {(ax, ay)}
    for y in range(h):
        for x in range(w):
            dx, dy = x - ax, y - ay
            if dx == 0 and dy == 0:
                continue
            if fov < 360:
                dot = dx * fx + dy * fy
                if dot <= 0:
                    continue
                if dot / hypot(dx, dy) < cut:
                    continue
            if los_clear(terrain, (ax, ay), (x, y)):
                seen.add((x, y))
    return seen


def astar(walkable, start, goals):
    """Shortest path over `walkable` cells from start to the nearest goal.

    `goals` are the cells to stand ON. Returns [start, ..., goal] or None.
    Ties are broken by cell coordinate so runs are reproducible.
    """
    if not goals:
        return None
    if start in goals:
        return [start]
    goals = set(goals)

    def h(c):
        return min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goals)

    open_ = [(h(start), 0, start, None)]
    came, best = {}, {start: 0}
    while open_:
        _, g, cur, parent = heappop(open_)
        if cur in came:
            continue
        came[cur] = parent
        if cur in goals:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        for dx, dy in DIRECTIONS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt not in walkable or nxt in came:
                continue
            ng = g + 1
            if ng < best.get(nxt, 1 << 30):
                best[nxt] = ng
                heappush(open_, (ng + h(nxt), ng, nxt, cur))
    return None


def adjacent_standing_cells(walkable, target):
    """Cells you can stand on to interact with `target`."""
    return [(target[0] + dx, target[1] + dy) for dx, dy in DIRECTIONS
            if (target[0] + dx, target[1] + dy) in walkable]


def step_towards(walkable, pos, orient, target):
    """One action that makes progress towards interacting with `target`.

    Returns (action, arrived). `arrived` means we are standing beside the target
    AND facing it, so the caller should INTERACT instead of moving.
    A movement action into a non-walkable cell only turns the agent, which is
    exactly how we aim at a station.

    NOTHING HERE ROUTES AROUND A TEAMMATE, and there is no way to ask it to. The
    old signature took a `blocked` cell and planned around it, because a move
    into an occupied tile silently fails in this env and a path through the other
    agent livelocks. That cannot happen in this suite: every layout is two rooms
    joined only by pass-through counters, so the two agents share no floor and
    were measured adjacent on 0 of 4800 ticks. The parameter, and the sidestep
    helper that went with it, fired zero times and were deleted rather than left
    as decoration. If a layout ever puts both cooks in one room again, this is
    the function that has to grow the rule back.
    """
    facing = (pos[0] + orient[0], pos[1] + orient[1])
    if facing == target:
        return None, True
    stands = adjacent_standing_cells(walkable, target)
    if pos in stands:                      # beside it but looking elsewhere: turn
        return (target[0] - pos[0], target[1] - pos[1]), False
    path = astar(walkable, pos, stands)
    if path is not None and len(path) >= 2:
        nxt = path[1]
        return (nxt[0] - pos[0], nxt[1] - pos[1]), False
    return None, False


def path_len(walkable, pos, target):
    """Steps needed to get beside `target`, or None if unreachable."""
    stands = adjacent_standing_cells(walkable, target)
    if pos in stands:
        return 0
    path = astar(walkable, pos, stands)
    return None if path is None else len(path) - 1


def dist_field(walkable, start):
    """BFS distance from `start` to every reachable cell. {cell: steps}.

    One sweep answers every path_len query that shares a start, which is what
    the rankers actually do: legal_subtasks hands back tens of candidates a tick
    and each one used to pay for its own A*. Measured at 75% of the filter's
    runtime, ~1900 A* calls a tick, 71% of them from the human's dist() memo.

    Equivalent to astar() for this purpose and not approximately: the grid is
    4-connected and every edge costs 1, so BFS order IS shortest-path order.
    astar's coordinate tie-break decides WHICH shortest path comes back, and
    path_len throws the path away and keeps the length, so the tie-break cannot
    be observed through this door. test_geometry_bfs.py checks that exhaustively.
    """
    d = {start: 0}
    q = collections.deque([start])
    while q:
        c = q.popleft()
        nd = d[c] + 1
        for dx, dy in DIRECTIONS:
            n = (c[0] + dx, c[1] + dy)
            if n in walkable and n not in d:
                d[n] = nd
                q.append(n)
    return d


def path_len_in(field, walkable, target):
    """path_len(walkable, start, target) for the `start` that built `field`.

    Standing ON a stand cell gives field[start] == 0, so the pos-in-stands case
    the A* version special-cased falls out of the min for free.
    """
    best = None
    for dx, dy in DIRECTIONS:
        s = (target[0] + dx, target[1] + dy)
        if s in walkable:
            v = field.get(s)
            if v is not None and (best is None or v < best):
                best = v
    return best
