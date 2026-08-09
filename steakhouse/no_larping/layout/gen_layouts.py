"""Layouts for the wall-occlusion FOV experiment. GENERATED from specs, not drawn.

    python gen_layouts.py layouts          regenerate all of them (prunes strays)
    python gen_layouts.py --audit layouts  check what is on disk
    python gen_layouts.py --show layouts   print them as ascii, to LOOK at them

Legend: # wall (blocks sight AND movement)   X counter (see-through, stages items)
        P grill  B board  W sink  M meat  O onion  D plate  S serve
        1 player index 0 = the ROBOT      2 player index 1 = the HUMAN

    The digits read backwards: the mdp seats '1' as index 0, and everywhere else
    in no_larping the limited-vision seat is index 1. So '2' is the human.

================================================================================
WHY THESE ARE GENERATED, AND WHY THEY LOOK LIKE KITCHENS NOW
================================================================================
An earlier version hand-drew some layouts, inherited the rest from an imported
suite, and retrofitted the rules onto both. The rules held but the kitchens were
a mess: only 7 of 21 would accept a non-straight dividing wall, and the counter
pass lined EVERY wall-adjacent floor tile, so each interior pillar grew its own
fringe of worktop and the grids came out as noise.

Three things fix it, and all three are structural rather than cosmetic:

  * Every layout is BUILT from a spec - a SCENARIO (what the two rooms are FOR),
    size, which axis the room divides on, the SHAPE of that divide, the blocks in
    each room, who gets which station. Variety is a parameter instead of
    something negotiated with somebody else's walls.
  * The HUMAN gets no private worktop at all. Rule 3 forbids one anyway, so the
    old pass built human-side counters and then converted every one of them back
    to '#' - which is where the ragged interior wall blobs came from. The human's
    staging surface is the PASS: the shared counter wall is their worktop, which
    is what a pass is in a real kitchen.
  * The ROBOT gets a BENCH, not a fringe: one or two contiguous runs of worktop
    laid along the wall beside its own stations, budgeted per layout, with the
    tiles you have to stand on to work a station left clear so a run always ends
    at a station instead of burying it.

Stations are worked INTO the runs the way hand-drawn divide.layout and
pantry.layout do it, rather than parked on the border in a neat row: 'mid'
embeds a station in the pass itself (a sink in the middle of a run of worktop),
'island' hangs one off an interior block out in the room.

================================================================================
THREE RULES. make() refuses to ship a layout that breaks any of them.
================================================================================
1. THE ROBOT CANNOT REACH THE SERVING HATCH.        robot_reaches_serve()
   No walkable route from '1' to any tile beside an S. The env has no doors, so
   this means every layout is TWO ROOMS joined only by pass-through counters.
   Every order crosses the wall and only the human can finish one.

2. THE HUMAN CANNOT FINISH ALONE.                   human_self_sufficient()
   The human is always missing a station the recipe needs (M->P cook, D->W wash,
   O->B chop, assemble, S serve). This is the rule REDUNDANCY can quietly break,
   so it is asserted after the passes rather than argued about up front.

3. EVERY COUNTER THE HUMAN CAN REACH, THE ROBOT CAN REACH TOO.  counter_reach()
   Nothing the human puts down is ever out of the robot's reach, so a handoff is
   never stranded and the only open question about a drop is whether the other
   one SAW it - which is what this suite exists to ask. One-directional on
   purpose: the robot keeps a private bench, since it can never serve.

================================================================================
THE SEVEN THINGS THAT VARY, AND WHY EACH IS THERE
================================================================================
SCENARIO   What the two rooms ARE - a larder, a scullery, a prep room, a cold
           room, a servery, a wash-up. Not flavour text: the scenario is what
           picks the split and the shape, so every layout has a reason the two
           cooks need each other rather than being a shape exercise with
           stations sprinkled on it.

SPLIT      Which stations sit on which side. Twelve schemes, not permutations of
           each other: the number of times an item must cross the wall runs from
           1 to 5.

REDUNDANT  A kind on BOTH sides ("red" in a spec). Without it the two agents can
           never want the same job, so duplicated effort does not exist and the
           coordination channel has nothing to prevent. With a board each, one
           can waste a trip chopping an onion the other is already chopping -
           and Channel A demoting a fetch it can SEE the robot doing is exactly
           what should stop that. A cone-blind pair pays that cost more often,
           which is what makes it measurable.
           How MUCH redundancy varies too: some layouts have none at all, some
           duplicate two or three kinds, and which kinds it is changes - a
           regular one-duplicate-per-layout rule is itself a confound.
           The KEYSTONE is whatever is still robot-only once the middle stations
           are handed over - the grill in every split here, the sink alongside it
           in most - and it is never duplicated away entirely, or the human
           becomes self-sufficient and rule 2 dies. S is never duplicated either,
           or the robot can serve and rule 1 dies. _check_spec() refuses both up
           front, by name, instead of letting make() fail three passes later.

MIDDLE     Stations embedded IN the divide, reachable from both rooms. Dividing
           the kitchen deleted every contention decision - agents that can never
           reach the same station never negotiate one. A shared board or sink
           puts it back: both want it, one at a time, from opposite sides. A
           layout may have one or two.

DIVIDE     straight, one_bay, two_bays, deep_nook, notch, elbow, shoulder,
           wander - the run-length profiles the SPECS below actually use.
           PROFILES also defines step_in and lean, which nothing currently asks
           for; they are kept because a profile is cheap and a new scenario often
           wants a shape that already exists. Not decoration. A straight wall is
           uniformly visible from the human's room, so a drop is noticed wherever
           it lands and the robot's choice of counter is worth nothing. A pocket
           puts a counter at the back of an alcove, visible from a narrow set of
           positions, while a counter on a protruding tooth is visible from most
           of the room. That SPREAD is the experiment: with it, the robot must
           pick a counter the human will actually look at, and a cone-blind robot
           drops into the blind spot.

SIGHT LINE Which of the human's OWN stations is a window into the robot's room.
           '#' is the only thing that blocks sight, so a station facing a long
           run of pass counter lets the human watch the robot work while they
           chop, and a station tucked behind a block or set back in a pocket does
           not. 'watch' in a spec names the kind that should be the window and
           place_stations aims for it; the other human stations are aimed AWAY
           from the robot's room, so the view is a property of WHERE YOU ARE
           STANDING rather than something the layout gives away everywhere.
           Some layouts watch from the board, some only from the serving pass,
           some from nowhere at all (watch=None aims every station away).
           station_sightlines() measures it and --audit prints it.

BLOCKS     Pillars, shelving and benches inside the rooms: occlusion a cone must
           be walked around rather than pointed. They are also what makes a
           low-visibility station possible, so they are placed with the sight
           lines in mind rather than scattered.

hatch_visibility() reports the spread over the pass counters; --audit prints it.
A layout whose shared counters are all equally visible is one where FOV cannot
matter.

================================================================================
WHAT STOPPED MATTERING
================================================================================
This file used to be organised around a two-tile-width rule, because the env
reverts BOTH agents on a collision (overcooked_mdp.py _handle_collisions) - same
tile or a swap and neither moves, the innocent one included - so a one-tile
corridor was a wall with a person in it. That cannot happen now: the two agents
are never in the same room. The one place they meet is a MIDDLE station, and
they stand at it from opposite sides. single_file() is still reported as a shape
check, but it is not a correctness property any more.
"""
import os
from math import cos, radians, hypot

D4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
FLOORCH = " 123456789"
MIN_RUN = 3           # a bench run shorter than this reads as scattered noise
KINDS = "PBWMODS"
NEED = {"M", "P", "D", "W", "O", "B", "S"}      # everything one order requires

# ---------------------------------------------------------------- the splits
# R robot only, H human only. Redundancy ("red" in a spec) widens a kind to both.
SPLITS = {                                                        # crossings
    "grill_only":     dict(M="R", P="R", D="H", W="H", O="H", B="H", S="H"),  # 1
    "steak_line":     dict(M="R", P="R", D="R", W="R", O="H", B="H", S="H"),  # 2
    "dish_pit":       dict(M="R", P="R", D="H", W="H", O="R", B="R", S="H"),  # 2
    "larder":         dict(M="R", P="R", D="R", W="H", O="R", B="H", S="H"),  # 2
    "hot_side":       dict(M="R", P="R", D="H", W="R", O="H", B="H", S="H"),  # 2
    "scullery":       dict(M="R", P="R", D="R", W="R", O="R", B="H", S="H"),  # 3
    "pinned":         dict(M="H", P="R", D="R", W="R", O="R", B="H", S="H"),  # 3
    "onion_return":   dict(M="R", P="R", D="H", W="R", O="R", B="H", S="H"),  # 4
    "plate_return":   dict(M="R", P="R", D="H", W="R", O="R", B="R", S="H"),  # 4
    "machine_pair":   dict(M="H", P="R", D="H", W="R", O="R", B="R", S="H"),  # 4
    "fire_and_board": dict(M="H", P="R", D="R", W="R", O="H", B="R", S="H"),  # 4
    "machine_room":   dict(M="H", P="R", D="H", W="R", O="H", B="R", S="H"),  # 5
}


def _tup(v):
    """A set of kinds, written either as "OB" or as ("O", "B")."""
    return tuple(v) if v else ()


def S_(name, w, h, axis, at, shape, split, why, mid=(), red=(), amp=3,
       blocks=(), watch=None, bench=8, island=(), run=6):
    """axis 'v': wall runs top-to-bottom, robot LEFT. 'h': robot TOP.

    why     one line: what the two rooms are and why the cooks need each other.
    mid     kinds embedded in the divide, reachable from both sides.
    island  kinds hung off an interior block instead of the outer border.
    watch   the HUMAN-side kind that should see into the robot's room, or None
            for a layout where the human can never watch.
    bench   how many tiles of private worktop the robot gets, total.
    run     longest single bench run, so the bench cannot wrap the whole room.
    """
    return dict(name=name, w=w, h=h, axis=axis, at=at, shape=shape, split=split,
                why=why, mid=_tup(mid), red=_tup(red), amp=amp,
                blocks=tuple(blocks), watch=watch, bench=bench,
                island=_tup(island), run=run)


# Hand-authored, NOT generated and NEVER pruned. These are drawn by hand and are
# the reference for what the generated ones should feel like. write() leaves them
# completely alone - it does not rewrite them and does not delete them - so a
# regeneration can never eat somebody's hand-tuned kitchen. They are still
# audited like everything else.
HAND_AUTHORED = {"divide", "pantry"}

SPECS = [
    # ---------------------------------------------------------------- stores
    S_("larder", 23, 12, "v", 8, "straight", "larder",
       "dry store: the robot has the goods and the fire, the human chops and serves",
       mid="B", watch="W", bench=8, run=6,
       blocks=[(4, 4, 5, 6), (13, 3, 14, 8)]),

    S_("cold_room", 25, 13, "v", 9, "two_bays", "steak_line",
       "walk-in: the robot fetches everything cold, the human owns the greens",
       mid="B", red="O", island="O", watch=None, amp=3, bench=8, run=6,
       blocks=[(3, 4, 4, 8), (14, 3, 18, 4), (14, 9, 18, 10)]),

    S_("goods_in", 25, 15, "h", 6, "one_bay", "machine_pair",
       "receiving bay: deliveries land with the human, every machine is the robot's",
       mid="W", red="OB", island="M", watch="M", amp=3, bench=9, run=6,
       blocks=[(4, 2, 8, 3), (5, 11, 8, 12), (16, 11, 19, 12)]),

    # -------------------------------------------------------------- wash-ups
    S_("scullery", 25, 13, "v", 13, "deep_nook", "scullery",
       "wash-up room: the robot cooks and washes, the human only chops and plates up",
       mid="W", red="O", watch="S", amp=4, bench=8, run=7,
       blocks=[(3, 3, 6, 4), (19, 7, 20, 11)]),

    S_("dish_pit", 23, 15, "h", 5, "elbow", "dish_pit",
       "dish pit under the line: plates and water are the human's, the greens the robot's",
       mid="W", watch="D", amp=4, bench=7, run=5,
       blocks=[(14, 11, 17, 12)]),

    S_("pot_wash", 25, 14, "v", 11, "notch", "onion_return",
       "pot wash: the human washes nothing, so every plate goes back over the wall",
       mid="B", red="DO", island="D", watch="O", amp=2, bench=9, run=6,
       blocks=[(3, 4, 4, 5), (17, 4, 20, 5), (17, 10, 20, 11)]),

    # ------------------------------------------------------------ prep rooms
    S_("prep_room", 25, 15, "h", 6, "one_bay", "fire_and_board",
       "prep room: the human trims and passes, the robot has the fire and the board",
       mid="B", red="O", watch="M", amp=3, bench=8, run=6,
       blocks=[(16, 2, 19, 3), (6, 11, 9, 12), (17, 11, 20, 12)]),

    S_("butchery", 25, 14, "v", 11, "one_bay", "pinned",
       "butchery: the meat is on the human's side and nothing else the robot needs is",
       mid="W", red="MOD", island="M", watch=None, amp=2, bench=8, run=5,
       blocks=[(3, 4, 4, 5), (16, 2, 21, 3), (16, 9, 21, 11)]),

    S_("garde_manger", 25, 13, "v", 10, "elbow", "hot_side",
       "cold larder: the human owns the cold station, the robot everything hot and wet",
       mid="B", red="O", watch="D", amp=5, bench=8, run=6,
       blocks=[(16, 2, 19, 3), (20, 8, 21, 10)]),

    # -------------------------------------------------- the fire is walled in
    S_("grill_house", 25, 12, "v", 6, "straight", "grill_only",
       "grill house: one hot room, one big kitchen, the fire shared through the wall",
       mid="P", watch="B", bench=6, run=6,
       blocks=[(12, 1, 13, 5), (12, 8, 13, 10), (17, 4, 21, 5)]),

    S_("hot_line", 25, 15, "h", 6, "shoulder", "machine_room",
       "hot line: every machine is behind the wall, the human only fetches and finishes",
       mid="W", red="D", watch="S", amp=3, bench=9, run=6,
       blocks=[(5, 2, 8, 3), (6, 11, 10, 12), (16, 11, 20, 12)]),

    # ----------------------------------------------------------- the pass
    S_("servery", 21, 13, "h", 6, "straight", "machine_room",
       "servery: a narrow serving strip, everything that cooks is through the hatch",
       mid="W", watch="S", bench=7, run=7,
       blocks=[(4, 9, 7, 10), (12, 9, 15, 10)]),

    S_("banquet_pass", 27, 13, "h", 6, "one_bay", "plate_return",
       "banquet pass: a long shallow pass, plates come back over it as fast as they go out",
       mid="W", red="OB", watch="B", amp=2, bench=10, run=6,
       blocks=[(5, 2, 10, 3), (17, 10, 21, 11)]),

    S_("still_room", 25, 14, "v", 12, "wander", "plate_return",
       "still room: the robot keeps the plates and the water, the human the fire's output",
       mid="B", red="O", island="O", watch=None, amp=2, bench=9, run=6,
       blocks=[(3, 4, 4, 5), (7, 9, 8, 10), (19, 2, 20, 5), (19, 8, 20, 11)]),

    S_("chefs_table", 23, 15, "h", 6, "one_bay", "pinned",
       "chef's table: the human works the counter in front of guests and can cook nothing",
       mid="WB", red="MOD", watch="O", amp=3, bench=9, run=5,
       blocks=[(4, 2, 5, 4), (15, 2, 16, 4), (4, 11, 6, 12), (16, 11, 18, 12)]),

    S_("back_bar", 23, 14, "h", 6, "notch", "onion_return",
       "back bar: a returns bench behind the line, onions and plates both come back",
       mid="B", red="D", watch="D", amp=3, bench=8, run=6,
       blocks=[(4, 10, 7, 11), (15, 10, 18, 11)]),
]


# --------------------------------------------------------------- basic geometry
def _floor(rows):
    H, W = len(rows), len(rows[0])
    return {(x, y) for y in range(H) for x in range(W) if rows[y][x] in FLOORCH}


def _flood(rows, start, floor=None):
    floor = _floor(rows) if floor is None else floor
    if start not in floor:
        return set()
    seen, q = {start}, [start]
    while q:
        c = q.pop()
        for d in D4:
            n = (c[0] + d[0], c[1] + d[1])
            if n in floor and n not in seen:
                seen.add(n)
                q.append(n)
    return seen


def _components(rows):
    floor, seen, out = _floor(rows), set(), []
    for c in sorted(floor):
        if c in seen:
            continue
        comp = _flood(rows, c, floor)
        seen |= comp
        out.append(frozenset(comp))
    return out


def _cells_of(rows, ch):
    H, W = len(rows), len(rows[0])
    return [(x, y) for y in range(H) for x in range(W) if rows[y][x] == ch]


def _nbrs(c):
    return [(c[0] + d[0], c[1] + d[1]) for d in D4]


def _sides(rows):
    """(robot floor, human floor)."""
    H, W = len(rows), len(rows[0])
    st = {rows[y][x]: (x, y) for y in range(H) for x in range(W)
          if rows[y][x] in "12"}
    return (_flood(rows, st["1"]) if "1" in st else set(),
            _flood(rows, st["2"]) if "2" in st else set())


def kinds_beside(rows, cells):
    H, W = len(rows), len(rows[0])
    out = set()
    for (x, y) in cells:
        for d in D4:
            nx, ny = x + d[0], y + d[1]
            if 0 <= nx < W and 0 <= ny < H and rows[ny][nx] in KINDS:
                out.add(rows[ny][nx])
    return out


# ------------------------------------------------------------------- the rules
def robot_reaches_serve(rows):
    """RULE 1. Serving tiles the ROBOT can walk to. Empty is what we want."""
    rob, _ = _sides(rows)
    return sorted(c for c in _cells_of(rows, "S")
                  if any((c[0] + d[0], c[1] + d[1]) in rob for d in D4))


def human_self_sufficient(rows):
    """RULE 2. True if the human could finish an order with no robot at all."""
    _, hum = _sides(rows)
    return NEED <= kinds_beside(rows, hum)


def counter_reach(rows):
    """(counters the robot can use, counters the human can use)."""
    rob, hum = _sides(rows)
    r, h = set(), set()
    for c in _cells_of(rows, "X"):
        nb = set(_nbrs(c))
        if nb & rob:
            r.add(c)
        if nb & hum:
            h.add(c)
    return r, h


def stranded_counters(rows):
    """Counters with no floor beside them: nobody can ever use one."""
    floor = _floor(rows)
    return [c for c in _cells_of(rows, "X")
            if not any(n in floor for n in _nbrs(c))]


def single_file(rows):
    """Floor in no 2x2 block of floor. Reported as a shape check, not a rule."""
    floor = _floor(rows)
    return [c for c in sorted(floor)
            if not any(all((c[0] + dx + i, c[1] + dy + j) in floor
                           for i in (0, 1) for j in (0, 1))
                       for dx in (-1, 0) for dy in (-1, 0))]


def _keystone(spec):
    """The kinds that keep rule 2 alive: robot-only under this split, so
    duplicating the last one of them would hand the human a whole recipe."""
    split = SPLITS[spec["split"]]
    return {k for k in NEED if split[k] == "R"} - set(spec["mid"])


def _human_only(spec):
    """The kinds the human has a station of that the robot cannot reach - the
    ones whose PLACEMENT is free enough to aim a sight line with."""
    split = SPLITS[spec["split"]]
    return ({k for k in NEED if split[k] == "H"} | set(spec["red"])) \
        - set(spec["mid"])


def _check_spec(spec):
    """Refuse a spec that asks for redundancy the rules cannot survive, up front
    and by name, rather than letting make() report 'the human can finish alone'
    from three passes downstream."""
    red = set(spec["red"])
    if "S" in red:
        return "S is duplicated: the robot could serve and rule 1 dies"
    if red & set(spec["mid"]):
        return "%s is both middle and duplicated" % "".join(sorted(red & set(spec["mid"])))
    keys = _keystone(spec)
    if not keys:
        return "no robot-only station: the human is self-sufficient by construction"
    if keys <= red:
        return "the keystone %s is duplicated: rule 2 dies" % "".join(sorted(keys))
    if spec["watch"] and spec["watch"] not in _human_only(spec):
        return ("watch=%s is not a station of the human's own (a middle station "
                "is a window by construction)" % spec["watch"])
    return None


# ------------------------------------------------------------------ visibility
def los_clear(rows, a, b):
    """'#' is the ONLY thing that blocks sight - counters are see-through."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps <= 1:
        return True
    for i in range(1, steps):
        cx = int(round(x0 + dx * i / steps))
        cy = int(round(y0 + dy * i / steps))
        if (cx, cy) in (a, b):
            continue
        if rows[cy][cx] == "#":
            return False
    return True


def _seer(rows, target):
    """Memoised 'what of `target` can be seen from this tile'."""
    cache = {}

    def see(p):
        if p not in cache:
            cache[p] = {q for q in target if los_clear(rows, p, q)}
        return cache[p]
    return see


def station_sightlines(rows):
    """THE SIGHT-LINE MEASURE. For every station the HUMAN can stand at, the
    fraction of the ROBOT's floor visible from the tiles you have to stand on to
    USE it. Counters and stations are see-through and '#' is not, so a station
    facing a long run of pass counter is a window into the other kitchen and one
    set back behind a block is not - and the difference is a property of where
    the human's job puts them, which is the thing this suite is about.

    Two dicts, because a station IN the divide is a window by construction - you
    are standing at the wall to use it - and would drown out the interesting
    number. {human-only station: fraction}, {middle station: fraction}.
    """
    rob, hum = _sides(rows)
    own, mid = {}, {}
    if not rob or not hum:
        return own, mid
    see = _seer(rows, rob)
    H, W = len(rows), len(rows[0])
    for y in range(H):
        for x in range(W):
            if rows[y][x] not in KINDS:
                continue
            nb = _nbrs((x, y))
            stand = [n for n in nb if n in hum]
            if not stand:
                continue
            seen = set()
            for p in stand:
                seen |= see(p)
            f = len(seen) / float(len(rob))
            d = mid if any(n in rob for n in nb) else own
            k = rows[y][x]
            d[k] = max(d.get(k, 0.0), f)
    return own, mid


def hatch_visibility(rows, fov=90):
    """Per shared counter, the fraction of the human's room it can be seen from,
    and the SPREAD across them - which is the number that matters. A wall whose
    counters are all equally visible gives the robot no choice worth making."""
    _, hum = _sides(rows)
    r_ct, h_ct = counter_reach(rows)
    shared = sorted(r_ct & h_ct)
    if not shared or not hum:
        return {}, 0.0
    cut = cos(radians(fov / 2.0))
    per = {}
    for c in shared:
        n = 0
        for p in hum:
            if not los_clear(rows, p, c):
                continue
            dx, dy = c[0] - p[0], c[1] - p[1]
            r = hypot(dx, dy) or 1.0
            if any((dx * fx + dy * fy) > 0 and (dx * fx + dy * fy) / r >= cut
                   for fx, fy in D4):
                n += 1
        per[c] = n / float(len(hum))
    return per, max(per.values()) - min(per.values())


# --------------------------------------------------------------- the generator
# Hand-authored wall profiles: a list of (how many steps, how far offset).
#
# These used to be formulas - comb every 3 rows, a pocket at exactly one third -
# and periodic geometry reads as machine-made however you tune the period. A
# kitchen wall that repeats on a fixed beat looks fake because nothing built by
# anybody does that. So the runs are irregular and asymmetric on purpose: a long
# straight stretch, one deep bay, a short jog back, a shallow return. The point
# is not prettiness - a wall with one memorable deep alcove and one shallow nook
# gives the robot a real CHOICE of counter, where an even comb gives it fifteen
# interchangeable ones.
#
# Kept deterministic (no rng) so a layout is the same every time it is generated.
PROFILES = {
    "straight":   [(1, 0)],
    "one_bay":    [(4, 0), (4, 3), (5, 0)],
    "deep_nook":  [(5, 0), (3, 4), (2, 1), (4, 0)],
    "wander":     [(3, 0), (2, 1), (4, 0), (3, 2), (2, 0), (3, 1)],
    "two_bays":   [(2, 0), (3, 2), (4, 0), (2, 4), (3, 0)],
    "step_in":    [(3, 0), (3, 1), (3, 2), (4, 3)],
    "notch":      [(6, 0), (2, 2), (5, 0), (2, 1)],
    "lean":       [(4, 1), (5, 0), (3, 3), (2, 0)],
    "elbow":      [(6, 0), (6, 4)],
    "shoulder":   [(3, 2), (5, 0), (4, 2), (3, 0)],
}


def _profile(shape, n, amp):
    """Offset of the wall at each step, from an irregular run-length spec."""
    runs = PROFILES.get(shape, PROFILES["straight"])
    out, i = [], 0
    while len(out) < n:
        count, off = runs[i % len(runs)]
        out.extend([min(off, amp)] * count)
        i += 1
    out = out[:n]
    # a lone offset step is a one-tile nub sticking out of the wall: flatten it
    for i in range(n):
        if ((i == 0 or out[i - 1] != out[i])
                and (i == n - 1 or out[i + 1] != out[i])):
            out[i] = out[i - 1] if i else (out[i + 1] if n > 1 else 0)
    return out


def _wall_cells(spec):
    """The divide, as a connected barrier from one border to the opposite one."""
    W, H, at, amp = spec["w"], spec["h"], spec["at"], spec["amp"]
    vert = spec["axis"] == "v"
    span = list(range(1, (H if vert else W) - 1))
    prof = _profile(spec["shape"], len(span), amp)
    lo, hi = (3, W - 4) if vert else (3, H - 4)
    cells, prev = [], None
    for k, i in enumerate(span):
        c = max(lo, min(hi, at + prof[k]))
        if prev is not None and c != prev:          # a jog needs its corner
            a, b = sorted((prev, c))
            for m in range(a, b + 1):
                cells.append((m, i) if vert else (i, m))
        cells.append((c, i) if vert else (i, c))
        prev = c
    return cells


def build(spec):
    """Spec -> rooms, wall and player starts. Stations come later."""
    W, H = spec["w"], spec["h"]
    g = [["#"] * W for _ in range(H)]
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            g[y][x] = " "
    for (x0, y0, x1, y1) in spec["blocks"]:
        for y in range(max(1, y0), min(H - 2, y1) + 1):
            for x in range(max(1, x0), min(W - 2, x1) + 1):
                g[y][x] = "#"
    wall = _wall_cells(spec)
    # a block that touches the divide merges into it and the pair reads as one
    # ragged blob rather than "a wall and a pillar". Keep them a tile apart.
    near = {(w[0] + dx, w[1] + dy) for w in wall
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    for (x, y) in list(near):
        if 0 < x < W - 1 and 0 < y < H - 1 and g[y][x] == "#" and (x, y) not in wall:
            g[y][x] = " "
    for (x, y) in wall:
        g[y][x] = "#"

    rows = ["".join(r) for r in g]
    comps = _components(rows)
    if len(comps) != 2:
        return None, "the divide made %d rooms" % len(comps)
    key = (lambda c: min(p[0] for p in c)) if spec["axis"] == "v" \
        else (lambda c: min(p[1] for p in c))
    lo, hi = sorted(comps, key=key)
    if min(len(lo), len(hi)) < 24:
        return None, "a room is too small (%d/%d)" % (len(lo), len(hi))
    for comp, ch in ((lo, "1"), (hi, "2")):
        far = max(sorted(comp),
                  key=lambda p: min(abs(p[0] - w[0]) + abs(p[1] - w[1])
                                    for w in wall))
        g[far[1]][far[0]] = ch
    return ["".join(r) for r in g], None


def share_counters(rows):
    """The divide becomes worktop: every wall tile touching BOTH rooms turns
    into a counter, so the shared surface is a whole wall rather than a few
    hatches. Runs FIRST, before the stations, for two reasons: a 'mid' station
    is placed ON one of these, and sight lines are measured through them, so
    they have to exist before anything is aimed at them."""
    H, W = len(rows), len(rows[0])
    g = [list(r) for r in rows]
    rob, hum = _sides(rows)
    for y in range(H):
        for x in range(W):
            if g[y][x] != "#":
                continue
            nb = set(_nbrs((x, y)))
            if (nb & rob) and (nb & hum):
                g[y][x] = "X"
    return ["".join(r) for r in g]


def place_stations(rows, spec):
    """Stations onto wall tiles, on the side the split assigns them.

    A station has to sit where the floor of the RIGHT room touches it, or it is
    unusable or usable by the wrong agent - both have shipped from this file
    before, so it is checked rather than assumed.

    Three loci, because a kitchen does not park everything on the outer border:
      edge    the outer wall (the default, and where a run of bench ends)
      mid     IN the divide, touching both rooms - a sink in the middle of the
              pass, the way pantry.layout does it
      island  an interior block face, so a dispenser stands out in the room

    Human-side placement is also where the SIGHT LINE is decided: the 'watch'
    kind is put where its standing tiles see the most of the robot's room, and
    every other human station is put where they see the least, so the view is
    somewhere specific rather than everywhere.
    """
    H, W = len(rows), len(rows[0])
    g = [list(r) for r in rows]
    rob, hum = _sides(rows)
    see = _seer(rows, rob)
    split = dict(SPLITS[spec["split"]])
    for k in spec["red"]:
        split[k] = "RH"
    for k in spec["mid"]:
        split[k] = "mid"
    locus = {k: "island" for k in spec["island"]}

    used = []

    def hosts(want, where):
        out = []
        for y in range(H):
            for x in range(W):
                ch = g[y][x]
                if where == "mid":
                    if ch != "X":
                        continue
                else:
                    if ch != "#":
                        continue
                    border = x in (0, W - 1) or y in (0, H - 1)
                    if where == "edge" and not border:
                        continue
                    if where == "island" and border:
                        continue
                touch = set()
                for n in _nbrs((x, y)):
                    if n in rob:
                        touch.add("R")
                    if n in hum:
                        touch.add("H")
                if touch != want:
                    continue
                out.append((x, y))
        return out

    def pick(cand, kind, want):
        """Spread stations out; aim the human's ones at or away from the robot."""
        aim = 0.0
        if "H" in want and rob:
            aim = 1.0 if kind == spec["watch"] else -1.0
        best, bestkey = None, None
        for (x, y) in sorted(cand):
            spread = min([abs(x - u[0]) + abs(y - u[1]) for u in used] or [99])
            v = 0.0
            if aim:
                stand = [n for n in _nbrs((x, y)) if n in hum]
                seen = set()
                for p in stand:
                    seen |= see(p)
                v = len(seen) / float(len(rob))
            key = min(spread, 8) / 8.0 + 1.6 * aim * v
            if bestkey is None or key > bestkey:
                best, bestkey = (x, y), key
        return best

    order = sorted(split, key=lambda k: {"mid": 0, "RH": 1}.get(split[k], 2))
    for k in order:
        where = split[k]
        wants = ([({"R", "H"}, "mid")] if where == "mid" else
                 [({"R"}, locus.get(k, "edge")), ({"H"}, locus.get(k, "edge"))]
                 if where == "RH" else
                 [({"R"}, locus.get(k, "edge"))] if where == "R" else
                 [({"H"}, locus.get(k, "edge"))])
        for want, loc in wants:
            cand = []
            for tryloc in ([loc] if loc == "mid" else [loc, "edge", "any"]):
                cand = hosts(want, tryloc)
                if cand:
                    break
            if not cand:
                return None, "nowhere to put %s on %s" % (k, "".join(sorted(want)))
            x, y = pick(cand, k, want)
            g[y][x] = k
            used.append((x, y))
    return ["".join(r) for r in g], None


# ------------------------------------------------------------- counter passes
def _runs(cells):
    """Maximal D4-connected groups of a set of tiles."""
    pool, out = set(cells), []
    while pool:
        p = min(pool)
        comp, q = {p}, [p]
        while q:
            c = q.pop()
            for n in _nbrs(c):
                if n in pool and n not in comp:
                    comp.add(n)
                    q.append(n)
        pool -= comp
        out.append(comp)
    return out


def _walk(seg, start):
    """Breadth-first order through a segment, so a run is laid down along the
    wall in order and stops in one piece rather than growing a hole."""
    out, seen, q = [], {start}, [start]
    while q:
        c = q.pop(0)
        out.append(c)
        for n in _nbrs(c):
            if n in seg and n not in seen:
                seen.add(n)
                q.append(n)
    return out


def robot_bench(rows, spec):
    """The robot's private worktop: one or two DELIBERATE runs beside its own
    stations, budgeted, and nothing anywhere else.

    Stashing is the ladder's only escape for an item with no current use, so a
    kitchen with a handful of counters deadlocks as soon as they fill - the robot
    needs bench. But the old version of this pass converted every wall-adjacent
    floor tile on BOTH sides, so each room grew a fringe of worktop and the
    scattered leftovers read as arbitrary rather than as a working surface.

    So: only the robot's side, only tiles against the outer wall, never a tile
    somebody has to stand on to work a station or the pass, runs of MIN_RUN or
    more only, taken from the segments nearest the robot's own stations first,
    each capped at spec['run'] so one wall cannot eat the whole budget, and
    stopping at spec['bench']. What comes out is a bench that starts at the grill
    and ends at the meat, which is what the robot would have built.
    """
    H, W = len(rows), len(rows[0])
    g = [list(r) for r in rows]
    rob, _ = _sides(rows)
    keep, stat = set(), []
    for y in range(H):
        for x in range(W):
            if g[y][x] in KINDS or g[y][x] == "X":
                keep |= set(_nbrs((x, y)))
            if g[y][x] in KINDS and any(n in rob for n in _nbrs((x, y))):
                stat.append((x, y))
    ring = {p for p in rob if (p[0] in (1, W - 2) or p[1] in (1, H - 2))
            and g[p[1]][p[0]] == " "}      # never pave over a player's start
    segs = _runs(ring - keep)

    def dist(p):
        return min([abs(p[0] - s[0]) + abs(p[1] - s[1]) for s in stat] or [0])

    segs.sort(key=lambda s: (min(dist(p) for p in s), -len(s), min(s)))
    budget = spec["bench"]
    for seg in segs:
        if budget < MIN_RUN:
            break
        if len(seg) < MIN_RUN:
            continue
        anchor = min(sorted(seg), key=dist)
        walk = _walk(seg, anchor)
        laid = []
        for (x, y) in walk:
            if len(laid) >= min(budget, spec["run"]):
                break
            g[y][x] = "X"
            trial = ["".join(r) for r in g]
            if len(_components(trial)) != 2 or stranded_counters(trial):
                g[y][x] = " "
                break
            laid.append((x, y))
        if len(laid) < MIN_RUN:     # a stub of one or two reads as noise, not bench
            for (x, y) in laid:
                g[y][x] = " "
            laid = []
        budget -= len(laid)
    return ["".join(r) for r in g]


def drop_human_only_counters(rows):
    """RULE 3: a counter only the human can stand at becomes plain wall. With the
    human's private worktop gone this is a no-op backstop rather than a pass that
    does work - which is the point, because when it DID do work it left a run of
    reverted counters behind as an interior wall blob."""
    g = [list(r) for r in rows]
    r_ok, h_ok = counter_reach(rows)
    for (x, y) in h_ok - r_ok:
        g[y][x] = "#"
    return ["".join(r) for r in g]


# -------------------------------------------------------------------- pipeline
TEMPLATE = '''{{
    "grid":  """{grid}""",
    "start_order_list": ['steak', 'steak', 'steak'],
    "cook_time": 15,
    "delivery_reward": 20,
    'num_items_for_steak': 1,
    'chop_time': 5,
    'wash_time': 5,
    "rew_shaping_params": None
}}
'''


def reseat_players(rows):
    """Move each player to the most OPEN tile of its own room.

    build() has to drop the digits early, because every pass after it works out
    which room is which by flooding from them - but that means the starts are
    chosen before the counter passes exist, and a tile that was in the middle of
    the floor can end up wedged into a run of worktop by the time we are done.
    That is how a robot came to start inside a counter run with a wall at its
    back. So: place them provisionally, build the kitchen, then sit them down
    properly on a tile with floor on as many sides as possible.
    """
    g = [list(r) for r in rows]
    rob, hum = _sides(rows)
    floor = _floor(rows)
    for comp, ch in ((rob, "1"), (hum, "2")):
        old = next((c for c in comp if g[c[1]][c[0]] == ch), None)
        if old is None:
            continue
        cx = sum(p[0] for p in comp) / float(len(comp))
        cy = sum(p[1] for p in comp) / float(len(comp))

        def score(p):
            open_n = sum(1 for n in _nbrs(p) if n in floor)
            return (open_n, -(abs(p[0] - cx) + abs(p[1] - cy)))
        best = max(sorted(comp), key=score)
        g[old[1]][old[0]] = " "
        g[best[1]][best[0]] = ch
    return ["".join(r) for r in g]


def make(spec):
    """The whole pipeline, with every rule checked at the end.

    The ORDER is forced, not a preference:

      _check_spec      refuse impossible redundancy by name, before three passes
                       of work turn it into "the human can finish alone"
      build            rooms, divide, provisional player digits. Everything after
                       this works out which room is which by flooding from those
                       digits, which is why they have to exist this early.
      share_counters   the divide becomes worktop FIRST, because a 'mid' station
                       is placed ON one of these tiles and sight lines are
                       measured THROUGH them
      place_stations   onto wall tiles, on the side the split says, aiming the
                       'watch' station at the robot's room and the rest away
      robot_bench      private worktop for the robot only, in deliberate runs
      drop_human_only_counters   rule 3 backstop
      reseat_players   only now can we know which tiles ended up open, so the
                       provisional starts get moved somewhere sensible

    Then every rule is asserted rather than argued about: a pass that quietly
    breaks rule 1 or 2 produces a kitchen that LOOKS right and silently turns a
    forced-handoff experiment into a solo one.
    """
    bad = _check_spec(spec)
    if bad:
        return None, "spec: " + bad
    rows, err = build(spec)
    if err:
        return None, err
    rows = share_counters(rows)
    rows, err = place_stations(rows, spec)
    if err:
        return None, err
    rows = robot_bench(rows, spec)
    rows = drop_human_only_counters(rows)
    rows = reseat_players(rows)

    if len(_components(rows)) != 2:
        return None, "not two rooms after the counter passes"
    bad = robot_reaches_serve(rows)
    if bad:
        return None, "RULE 1: robot reaches serve at %s" % bad
    if human_self_sufficient(rows):
        return None, "RULE 2: the human can finish alone"
    r_ct, h_ct = counter_reach(rows)
    if h_ct - r_ct:
        return None, "RULE 3: human-only counters %s" % sorted(h_ct - r_ct)
    if stranded_counters(rows):
        return None, "stranded counters %s" % stranded_counters(rows)
    missing = NEED - kinds_beside(rows, _floor(rows))
    if missing:
        return None, "unreachable stations %s" % sorted(missing)

    for ch in "12":
        if sum(r.count(ch) for r in rows) != 1:
            return None, "player %s is not on the grid exactly once" % ch

    rob, hum = _sides(rows)
    _, spread = hatch_visibility(rows)
    own, midv = station_sightlines(rows)
    best = max(own, key=lambda k: (own[k], k)) if own else "-"
    return rows, dict(shape=spec["shape"], split=spec["split"],
                      mid="".join(spec["mid"]) or "-",
                      red="".join(spec["red"]) or "-",
                      shared=len(r_ct & h_ct), spread=spread,
                      bench=len(r_ct - h_ct),
                      view=own, midview=midv, best=best,
                      bestv=own.get(best, 0.0),
                      vspread=(max(own.values()) - min(own.values()))
                      if own else 0.0,
                      watch=spec["watch"] or "-",
                      ok=(spec["watch"] is None or spec["watch"] == best),
                      why=spec["why"],
                      robot="".join(sorted(kinds_beside(rows, rob))),
                      human="".join(sorted(kinds_beside(rows, hum))),
                      rooms="%d/%d" % (len(rob), len(hum)))


def write(outdir, prune=True):
    """Generate every spec. Anything else in the directory is removed: a stray
    .layout is one nothing has checked, and it may quietly hand the robot a route
    to the pass and turn a forced-handoff experiment into a solo one.

    HAND_AUTHORED names are never written and never pruned."""
    os.makedirs(outdir, exist_ok=True)
    gen = {s["name"] for s in SPECS}
    assert not (gen & HAND_AUTHORED), \
        "a spec is named after a hand-drawn layout: %s" % sorted(gen & HAND_AUTHORED)
    names = gen | HAND_AUTHORED
    if prune:
        for f in sorted(os.listdir(outdir)):
            if f.endswith(".layout") and f[:-len(".layout")] not in names:
                os.remove(os.path.join(outdir, f))
                print("  [prune] removed %s" % f)
    out = []
    for spec in SPECS:
        if spec["name"] in HAND_AUTHORED:
            continue
        rows, rep = make(spec)
        if rows is None:
            print("  [SKIP] %-13s %s" % (spec["name"], rep))
            continue
        grid = ("\n" + " " * 16).join(rows)
        with open(os.path.join(outdir, spec["name"] + ".layout"), "w") as f:
            f.write(TEMPLATE.format(grid=grid))
        out.append((spec["name"], len(rows[0]), len(rows), rep))
    return out


def read_grid(path):
    import re
    m = re.search(r'"grid":\s*"""(.*?)"""', open(path).read(), re.S)
    return [r.strip() for r in m.group(1).split("\n") if r.strip()]


def audit_dir(d):
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".layout"):
            continue
        rows = read_grid(os.path.join(d, f))
        r_ct, h_ct = counter_reach(rows)
        _, spread = hatch_visibility(rows)
        own, midv = station_sightlines(rows)
        best = max(own, key=lambda k: (own[k], k)) if own else "-"
        view = " ".join("%s%.2f" % (k, own[k]) for k in sorted(own))
        out.append((f[:-len(".layout")], len(rows[0]), len(rows),
                    len(_components(rows)), len(r_ct & h_ct), spread,
                    "%s %.2f" % (best, own.get(best, 0.0)), view,
                    bool(robot_reaches_serve(rows)), human_self_sufficient(rows),
                    bool(h_ct - r_ct), len(single_file(rows))))
    return out


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--audit":
        print("%-14s %-8s %-5s %-6s %-6s %-8s %-5s %-9s %s"
              % ("layout", "size", "rooms", "shared", "spread", "watch",
                 "thin", "R1 R2 R3", "sight lines, human's own stations"))
        for (n, w, h, c, sh, sp, bk, vw, r1, r2, r3, thin) in audit_dir(sys.argv[2]):
            print("%-14s %3dx%-4d %-5d %-6d %-6.2f %-8s %-5d %s %s %s  %s"
                  % (n, w, h, c, sh, sp, bk, thin,
                     "FAIL" if r1 else "ok  ", "FAIL" if r2 else "ok  ",
                     "FAIL" if r3 else "ok", vw))
    elif arg == "--show":
        for f in sorted(os.listdir(sys.argv[2])):
            if f.endswith(".layout"):
                print("=== %s ===" % f[:-len(".layout")])
                print("\n".join(read_grid(os.path.join(sys.argv[2], f))))
                print()
    else:
        res = write(sys.argv[1], prune="--keep-strays" not in sys.argv)
        print("%-14s %-8s %-10s %-14s %-4s %-4s %-6s %-5s %-6s %-8s %s"
              % ("layout", "size", "divide", "split", "mid", "red", "shared",
                 "bench", "spread", "watch", "robot | human"))
        for n, w, h, r in res:
            print("%-14s %3dx%-4d %-10s %-14s %-4s %-4s %-6d %-5d %-6.2f %-8s %s | %s%s"
                  % (n, w, h, r["shape"], r["split"], r["mid"], r["red"],
                     r["shared"], r["bench"], r["spread"],
                     "%s %.2f" % (r["best"], r["bestv"]),
                     r["robot"], r["human"],
                     "" if r["ok"] else "  [warn: wanted watch=%s]" % r["watch"]))
        print("\nsight lines - fraction of the robot's floor visible from the"
              " tiles you work each station from")
        for n, w, h, r in res:
            print("  %-14s own %-28s pass %-14s  %s"
                  % (n,
                     " ".join("%s%.2f" % (k, r["view"][k])
                              for k in sorted(r["view"])) or "-",
                     " ".join("%s%.2f" % (k, r["midview"][k])
                              for k in sorted(r["midview"])) or "-",
                     r["why"]))
