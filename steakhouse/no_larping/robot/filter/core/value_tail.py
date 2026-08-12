"""tail(state) -- BOUNDED A* for the ticks the team still needs, on the ladder graph.

The second half of C_theta = t_end + tail(state at t_end). The head is a faithful
joint simulation for T ticks; this searches everything after it, so the score is
total completion time rather than "what happened in the next T ticks".

IT RUNS ON THE LADDER GRAPH, NEVER THE RAW STATE GRAPH. The true joint state
carries both positions, both held items, station contents, timers and the human's
beliefs, and half of every transition is the human's closed-loop choice, which a
search cannot make on their behalf. What CAN be searched is the recipe: which leg
still has to happen, who does it, in what order. Nodes are sets of completed legs
plus each agent's (position, free-at); edges are one leg assigned to one agent or
relayed between both, costed with station-to-station path_len plus the fixed work
and cook durations; the objective is makespan.

WHY A SEARCH AND NOT A GREEDY SCHEDULE. The first version of this file assigned
legs greedily in a fixed order. That is not merely suboptimal, it is UNSTABLE: one
tick of state change flips an assignment and the estimate jumps by tens of ticks.
Measured, it moved with a standard deviation of 13-16 ticks per tick against a
true signal of about one tick per tick, and every controller built on it thrashed
-- 43-74% of ticks deviating, and a baseline that scored 2 dishes reduced to 0. An
optimum does not flip on a near-tie, and that stability is the only thing that
makes the number comparable between candidates.

WHAT THE SEARCH IS BUILT AROUND. On every layout in this suite the two agents
stand in DISCONNECTED rooms and neither can reach all seven station types --
back_bar's robot has no pot, no sink and no serve; divide's robot has no meat, no
onion and no serve; and the robot reaches a serve hatch on NO layout at all, so
no dish is deliverable by one agent anywhere. Every one crosses the divide
through a counter both can stand at. So a leg is not always one agent walking: it
may be a RELAY, one agent carrying to a pass counter and the other collecting.
That is the whole reason a stash CELL has a price, and why the start node is the
head's end configuration. `python -m robot.filter.analysis.layout_facts` prints the table;
read it from there rather than by eye, which is how the numbers in this paragraph
were wrong the first time.

The human's half of a relay is priced through THEIR OWN VIEW: an item left where
their cone has not been is charged `blind` extra ticks, because they will not
collect what they do not know is there until a sweep finds it. Nobody writes
"prefer visible counters" anywhere; it falls out of that.
"""
import collections
import heapq
import os
import sys

_NL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(_NL)))
sys.path.insert(0, _NL)

from common import geometry as geo                                 # noqa: E402
from common.views import TruthView                                 # noqa: E402

POT, BOARD, SINK, MEAT, ONION, PLATE, SERVE, COUNTER = \
    "P", "B", "W", "M", "O", "D", "S", "X"
PRESS = 1
BIG = 10 ** 6
MAX_EXPAND = 1200          # the "bounded" in bounded A*
PASS_K = 4                 # pass counters considered per leg, cheapest first


class Geo:
    """Per-layout travel oracle: rooms, pass counters, BFS distances. Cached."""

    def __init__(self, mdp, state):
        v = TruthView(mdp, state)
        self.mdp, self.walk = mdp, set(v.walkable)
        self.stations = {c: set(v.stations(c)) for c in
                         (POT, BOARD, SINK, MEAT, ONION, PLATE, SERVE)}
        self._f, self._rooms = {}, {}
        r = 0
        for c in sorted(self.walk):
            if c in self._rooms:
                continue
            for cell in geo.dist_field(self.walk, c):
                self._rooms[cell] = r
            r += 1
        self.counters = [(x, y) for y in range(mdp.height) for x in range(mdp.width)
                         if mdp.get_terrain_type_at_pos((x, y)) == COUNTER]
        self.passes = [c for c in self.counters if len(self.rooms_of(c)) > 1]

    def field(self, pos):
        f = self._f.get(pos)
        if f is None:
            f = self._f[pos] = geo.dist_field(self.walk, pos)
        return f

    def rooms_of(self, cell):
        return {self._rooms[s] for d in geo.DIRECTIONS
                for s in [(cell[0] + d[0], cell[1] + d[1])] if s in self.walk}

    def room(self, pos):
        return self._rooms.get(pos, -1)

    def stand(self, cell, room):
        """A walkable tile beside `cell` in `room` -- where an agent ENDS UP.

        An agent's position in the search has to stay walkable: stations and
        counters are not, so storing the target cell as the new position made
        room(pos) = -1 and every subsequent leg looked unreachable. Deterministic
        (min by coordinate) so the search stays reproducible.
        """
        cs = [c for d in geo.DIRECTIONS
              for c in [(cell[0] + d[0], cell[1] + d[1])]
              if c in self.walk and self._rooms[c] == room]
        return min(cs) if cs else None

    def steps(self, pos, cell):
        d = geo.path_len_in(self.field(pos), self.walk, cell)
        return BIG if d is None else d + PRESS


def _geo(mdp, state):
    g = getattr(mdp, "_tail_geo2", None)
    if g is None or g.mdp is not mdp:
        g = Geo(mdp, state)
        try:
            mdp._tail_geo2 = g
        except Exception:
            pass
    return g


def _timer_left(mdp, state, cell, kind):
    """Ticks until the occupant of `cell` is ready; 0 if ready, None if empty.

    The pot cooks UNATTENDED so its remainder is a wall-clock floor that overlaps
    anything else the team does. The board and the sink advance only inside an
    INTERACT, so their time occupies an agent instead. env.txt is the authority
    and the difference is what makes chop and wash worth scheduling at all.
    """
    obj = state.objects.get(cell)
    if obj is None:
        return None
    if kind == POT:
        if mdp.steak_ready_at_location(state, cell):
            return 0
        t = obj.state[-1] if isinstance(obj.state, (tuple, list)) else obj.state
        return max(0, mdp.steak_cooking_time - int(t))
    if kind == BOARD:
        return 0 if mdp.garnish_ready_at_location(state, cell) \
            else max(0, mdp.chopping_time - int(obj.state))
    if kind == SINK:
        return 0 if mdp.plate_washed_at_location(state, cell) \
            else max(0, mdp.wash_time - int(obj.state))
    return None


class Leg:
    """Fetch from `src`, act at `dst`, occupy `work` ticks. `after` must precede.

    `ready` is a wall-clock floor -- a pot still cooking -- that no amount of
    walking can beat.
    """

    __slots__ = ("key", "src", "dst", "work", "after", "ready")

    def __init__(self, key, src, dst, work=0.0, after=(), ready=0.0):
        self.key, self.src, self.dst = key, src, dst
        self.work, self.after, self.ready = float(work), tuple(after), float(ready)


def _ways(g, leg, pos, free, blind, view, hi):
    """Every way to execute `leg`: each agent alone, or a relay through a pass.

    Enumerated in a fixed order; the caller takes the min over all of them, so
    the answer cannot depend on iteration order -- the defect that made the
    greedy version unusable.
    """
    src, dst, work = leg.src, leg.dst, leg.work
    out = []

    def pen(i, cell):
        if i != hi or view is None:
            return 0.0
        if cell in view.known_terrain and view._fresh(cell) is not None:
            return 0.0
        return float(blind)

    def can(i, cell):
        return g.room(pos[i]) in g.rooms_of(cell)

    for i in (0, 1):
        if (src is not None and not can(i, src)) or not can(i, dst):
            continue
        t, p = max(free[i], leg.ready), pos[i]
        if src is not None:
            t += g.steps(p, src) + pen(i, src)
            p = src
        t += g.steps(p, dst)
        if t >= BIG:
            continue
        t += work
        end = g.stand(dst, g.room(pos[i]))
        if end is None:
            continue
        np_, nf = list(pos), list(free)
        np_[i], nf[i] = end, t
        out.append((t, tuple(np_), tuple(nf)))

    if src is not None:
        # THE COST KNOB. Every pass counter times both orderings was up to 44
        # successors per leg and 130 ms a call, which at ~36 candidates a tick is
        # nine seconds. The cheapest few relays dominate the optimum in practice,
        # so the search sees those; this is the "bounded" doing real work rather
        # than a silent truncation.
        near = sorted(g.passes,
                      key=lambda c: (g.steps(src, c) + g.steps(c, dst), c))[:PASS_K]
        for c in near:
            for i in (0, 1):
                j = 1 - i
                if not can(i, src) or g.room(pos[i]) not in g.rooms_of(c):
                    continue
                if g.room(pos[j]) not in g.rooms_of(c) or not can(j, dst):
                    continue
                ta = max(free[i], leg.ready) + g.steps(pos[i], src) + pen(i, src) \
                    + g.steps(src, c)
                if ta >= BIG:
                    continue
                tb = max(free[j], ta) + g.steps(pos[j], c) + pen(j, c) \
                    + g.steps(c, dst)
                if tb >= BIG:
                    continue
                tb += work
                ei = g.stand(c, g.room(pos[i]))
                ej = g.stand(dst, g.room(pos[j]))
                if ei is None or ej is None:
                    continue
                np_, nf = list(pos), list(free)
                np_[i], nf[i] = ei, ta
                np_[j], nf[j] = ej, tb
                out.append((tb, tuple(np_), tuple(nf)))
    return out


def _legs(mdp, g, state, held):
    """The recipe legs still outstanding, with precedences and wall-clock floors.

    Existing progress short-circuits the front of a stream: a pot already cooking
    is not re-cooked, a washed plate on a counter is not re-washed, an item in a
    hand skips its fetch. None means this state cannot be finished at all, which
    is a real answer and not an error.
    """
    def counters_with(name):
        return sorted(c for c, o in state.objects.items()
                      if o.name == name and mdp.get_terrain_type_at_pos(c) == COUNTER)

    def stream(kind, raw_name, raw_ch, done_name, work, unattended, key):
        for i in (0, 1):
            if held[i] == done_name:
                return None, 0.0, None
        loose = counters_with(done_name)
        if loose:
            return None, 0.0, loose[0]
        best = None
        for c in sorted(g.stations[kind]):
            left = _timer_left(mdp, state, c, kind)
            if left is not None and (best is None or left < best[0]):
                best = (left, c)
        if best is not None:
            if unattended:
                return None, float(best[0]), best[1]
            return Leg(key, None, best[1], float(best[0])), 0.0, best[1]
        empty = sorted(c for c in g.stations[kind] if state.objects.get(c) is None) \
            or sorted(g.stations[kind])
        if not empty:
            return None, None, None
        dst = empty[0]
        held_raw = any(held[i] == raw_name for i in (0, 1))
        srcs = sorted(g.stations[raw_ch]) + counters_with(raw_name)
        if not held_raw and not srcs:
            return None, None, None
        src = None if held_raw else srcs[0]
        return (Leg(key, src, dst, 0.0 if unattended else work),
                float(work) if unattended else 0.0, dst)

    # A FINISHED DISH SHORT-CIRCUITS EVERYTHING. `stream` already skips a stream
    # whose product exists -- a washed plate is not re-washed -- but nothing did
    # that for the finished article, so a `dish` on a counter still had the tail
    # costing a plate, a steak, a garnish and two merges. That is what made
    # `get_meat` read as progress in a kitchen already holding what it owed.
    done = [c for c, o in state.objects.items()
            if o.name == "dish" and mdp.get_terrain_type_at_pos(c) == COUNTER]
    if any(held[i] == "dish" for i in (0, 1)) or done:
        serve0 = sorted(g.stations[SERVE])
        if not serve0:
            return None
        src = None if any(held[i] == "dish" for i in (0, 1)) else sorted(done)[0]
        return [Leg("deliver", src, serve0[0], 0.0)]

    # HALF-BUILT COMPOSITES SHORT-CIRCUIT THE STREAMS THEY CONTAIN. `stream` only
    # ever recognised its OWN product -- a loose `washed_plate` stops the wash, a
    # loose `steak` stops the cook -- so a `steak_dish` sitting on a counter was
    # invisible to both and to their merge. The tail then billed a plate, a steak
    # and the plate_steak merge for a dish that already had all three, which made
    # fetching a plate score as well as chopping the onion the dish actually needed.
    #
    # A steak_dish IS a plate and a steak, already merged. What remains is the
    # garnish, the merge that adds it, and the walk to the hatch.
    def composite(name):
        if any(held[i] == name for i in (0, 1)):
            return True, None
        loose = counters_with(name)
        return (True, sorted(loose)[0]) if loose else (False, None)

    # BOTH ASSEMBLY ORDERS EXIST, so both composites short-circuit. overcooked_mdp
    # spec 1.c: a garnish_dish carried to a pot holding a ready steak becomes a
    # dish, exactly as a steak_dish carried to a chopped board does. So a
    # garnish_dish is a plate AND a garnish and what it still needs is the STEAK --
    # which is why `2 garnish_dish` should make cooking meat the cheapest thing on
    # the board, and `2 steak_dish` should make chopping the onion cheapest.
    have_gd, at_gd = composite("garnish_dish")
    have_sd, at_sd = composite("steak_dish")
    if have_gd and not have_sd:
        serve0 = sorted(g.stations[SERVE])
        if not serve0:
            return None
        st2, f_st2, c_st2 = stream(POT, "meat", MEAT, "steak",
                                   mdp.steak_cooking_time, True, "steak")
        if f_st2 is None:
            return None
        out = [x for x in (st2,) if x is not None]
        out.append(Leg("add_steak", at_gd,
                       c_st2 if c_st2 is not None else serve0[0], 0.0,
                       tuple(x.key for x in (st2,) if x), ready=f_st2))
        out.append(Leg("deliver", None, serve0[0], 0.0, ("add_steak",)))
        return out
    if have_sd:
        serve0 = sorted(g.stations[SERVE])
        if not serve0:
            return None
        ga2, f_ga2, c_ga2 = stream(BOARD, "onion", ONION, "garnish",
                                   mdp.chopping_time, False, "garnish")
        if f_ga2 is None:
            return None
        out = [x for x in (ga2,) if x is not None]
        out.append(Leg("add_garnish", at_sd,
                       c_ga2 if c_ga2 is not None else serve0[0], 0.0,
                       tuple(x.key for x in (ga2,) if x), ready=f_ga2))
        out.append(Leg("deliver", None, serve0[0], 0.0, ("add_garnish",)))
        return out

    st, f_st, c_st = stream(POT, "meat", MEAT, "steak", mdp.steak_cooking_time, True, "steak")
    ga, f_ga, c_ga = stream(BOARD, "onion", ONION, "garnish", mdp.chopping_time, False, "garnish")
    pl, f_pl, c_pl = stream(SINK, "plate", PLATE, "washed_plate", mdp.wash_time, False, "plate")
    if None in (f_st, f_ga, f_pl):
        return None
    serve = sorted(g.stations[SERVE])
    if not serve:
        return None

    legs = [x for x in (st, ga, pl) if x is not None]
    legs.append(Leg("plate_steak", c_pl, c_st if c_st is not None else serve[0],
                    0.0, tuple(x.key for x in (pl, st) if x), ready=f_st))
    legs.append(Leg("add_garnish", None, c_ga if c_ga is not None else serve[0],
                    0.0, ("plate_steak",) + tuple(x.key for x in (ga,) if x),
                    ready=f_ga))
    legs.append(Leg("deliver", None, serve[0], 0.0, ("add_garnish",)))
    return legs


def _inventory(mdp, state, held):
    """{component: how many dishes the kitchen can already supply it for}.

    THE ACCOUNTING THAT LETS ONE A* PRICE MANY ORDERS. Every dish consumes exactly
    one steak, one garnish and one washed plate, so a component can serve at most
    ONE dish -- two steaks cover two orders and no more. Counting them is therefore
    enough to say how much of the remaining work is already paid for, without
    replicating the legs per order and squaring the search.

    Half-built and finished articles are counted as the components they CONTAIN,
    because that is what they can no longer be used for anything else as:

        steak_dish    = a plate and a steak
        garnish_dish  = a plate and a garnish
        dish          = all three

    Items resident at a station count as well: a pot part-way through its timer is
    a steak that exists, it just is not collectable yet, and the A* prices that
    delay for the FIRST dish through `_timer_left`. Double-counting it here would
    be wrong only if the same object also sat on a counter, which it cannot.
    """
    n = collections.Counter()
    for i in (0, 1):
        if held[i]:
            n[held[i]] += 1
    for _c, o in state.objects.items():
        n[o.name] += 1
    return {"steak": n["steak"] + n["steak_dish"] + n["dish"],
            "garnish": n["garnish"] + n["garnish_dish"] + n["dish"],
            "plate": (n["washed_plate"] + n["steak_dish"] + n["garnish_dish"]
                      + n["dish"])}


# WEIGHT ON OUTSTANDING FETCHES. The tail is a MAKESPAN, so work that fits inside
# somebody else's slack is free by construction: with a steak 10 ticks off the
# grill, a 5-tick chop costs nothing and fetching the onion early scores exactly
# zero. That is schedule-optimal and operationally useless -- every candidate ties
# at zero and `t_end` alone decides, so the robot wanders to whatever is nearest
# instead of to the thing that has to happen and has not started.
#
# So makespan is the primary objective and this is a SECONDARY one: among schedules
# of equal length, prefer the state with fewer ingredients still to fetch. One tick
# per outstanding fetch is far below the ~10-tick gaps between genuinely different
# schedules, so it can only break ties -- it can never talk the layer out of a
# faster plan.
FETCH_W = 1.0


def _pending(legs):
    """Legs that still need something FETCHED. Holding it already zeroes its src."""
    return sum(1 for l in legs if l.src is not None)


def _rest(mdp, g, need, inv):
    """Ticks for the orders AFTER the one the A* priced, given what is in stock.

    The A* consumed one of each component for its own dish, so the stock available
    to the remaining `need - 1` orders is one less of each. Whatever is still short
    has to be made:

        short(c) = max(0, extra - (stock(c) - 1))

    and the three streams are charged as a MAX rather than a sum, because they run
    at different stations and overlap -- the same reason `h(mask)` in the search is
    a max over outstanding work. That keeps this a lower bound on the parallel work
    rather than a pessimistic serial total.

    Every remaining order also has to be carried and merged whatever its stock, so
    `extra` laps of the kitchen are added on top. THAT is the term that makes a
    fetch costly once the components are all in hand: the work drops out, the
    carrying does not, and nothing about picking up more meat reduces the carrying.
    """
    extra = max(0, need - 1)
    if not extra:
        return 0.0
    short = {c: max(0, extra - max(0, inv.get(c, 0) - 1))
             for c in ("steak", "garnish", "plate")}
    work = max(short["steak"] * float(mdp.steak_cooking_time),
               short["garnish"] * float(mdp.chopping_time),
               short["plate"] * float(mdp.wash_time))
    return work + extra * _lap(mdp, g)


def _lap(mdp, g):
    """One trip across the kitchen. The part of a dish nobody can stock up on."""
    span = 0
    for a in list(g.walk)[:1]:
        fld = g.field(a)
        span = max(fld.values()) if fld else 0
    return float(span)


def _marginal(mdp, g):
    """Ticks for each dish AFTER the next one. A per-layout CONSTANT, on purpose.

    The search prices the NEXT dish only. Estimating all remaining dishes at once
    multiplies the chain length, the branch points and therefore the jitter --
    and worse, it SCALES that jitter by the number of orders left. Every
    candidate scored from one state shares the same `need`, so a constant per
    dish cancels out of the comparison entirely and contributes no noise, while
    still keeping the number monotone in orders remaining.

    Cheapest sane estimate of a pipeline dish: the cook, which is the one timer
    nobody can overlap away, plus a lap of the kitchen.
    """
    m = getattr(mdp, "_tail_marginal", None)
    if m is None:
        span = 0
        for a in list(g.walk)[:1]:
            fld = g.field(a)
            span = max(fld.values()) if fld else 0
        m = float(mdp.steak_cooking_time + span)
        try:
            mdp._tail_marginal = m
        except Exception:
            pass
    return m


def tail_ticks(mdp, state, human_view=None, human_index=1, blind=8.0, cap=BIG):
    """Bounded-A* ticks for the next dish, plus an inventory-priced term for the rest.

    Need-aware twice over. `orders_remaining` is len(order_list) - 1, because
    is_terminal fires at <= 1 and the last order is never deliverable, exactly as
    Phi counts. And the orders after the next one are priced against WHAT IS IN
    STOCK rather than at a flat constant each: each dish takes one steak, one
    garnish and one plate, so two steaks already made cover two orders and the cook
    is not charged twice. See `_inventory` and `_rest`.

    That split is what stops a fetch from reading as progress in a kitchen that
    already holds everything it owes. Searching only the next dish is still what
    keeps the number dense and monotone: every tick of real progress towards it
    takes it down by about one.
    """
    from robot.filter.core.progress import orders_remaining
    need = orders_remaining(mdp, state)
    if need <= 0:
        return 0.0
    g = _geo(mdp, state)
    held = {i: (state.players[i].held_object.name
                if state.players[i].held_object else None) for i in (0, 1)}
    inv = _inventory(mdp, state, held)
    legs = _legs(mdp, g, state, held)
    if not legs:
        return cap
    idx = {l.key: k for k, l in enumerate(legs)}
    full = (1 << len(legs)) - 1
    hi = human_index

    def h(mask):
        """Admissible: work still owed that no schedule can avoid paying."""
        return max((l.work for k, l in enumerate(legs) if not mask >> k & 1),
                   default=0.0)

    start = (0, (tuple(state.players[0].position),
                 tuple(state.players[1].position)), (0.0, 0.0))
    best = {}
    pq = [(h(0), 0.0, start)]
    n, floor = 0, None
    while pq and n < MAX_EXPAND:
        f, cost, (mask, pos, free) = heapq.heappop(pq)
        # A* pops in f order, so the FIRST pop is the tightest lower bound this
        # search will ever have. Remember it: running out of budget must degrade
        # to an estimate, never to `cap`. A cap spike is a 10^6-tick jump in the
        # middle of a comparison and it poisons every candidate around it.
        if floor is None:
            floor = f
        if mask == full:
            return (float(cost) + _rest(mdp, g, need, inv)
                    + FETCH_W * _pending(legs))
        k = (mask, pos, free)
        if best.get(k, BIG) < cost:
            continue
        best[k] = cost
        n += 1
        for j, l in enumerate(legs):
            if mask >> j & 1:
                continue
            if any(not mask >> idx[a] & 1 for a in l.after if a in idx):
                continue
            nm = mask | 1 << j
            for t, np_, nf in _ways(g, l, pos, free, blind, human_view, hi):
                c2 = max(t, cost)
                kk = (nm, np_, nf)
                if best.get(kk, BIG) <= c2:
                    continue
                best[kk] = c2
                heapq.heappush(pq, (c2 + h(nm), c2, kk))
    if floor is not None:
        return (float(floor) + _rest(mdp, g, need, inv)
                + FETCH_W * _pending(legs))
    return cap
