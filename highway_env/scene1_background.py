"""Watch scene 1's background traffic actually drive, not just sit there.

    python scene1_background.py
    python scene1_background.py --count 20 --seed 7
    python scene1_background.py --steps 600 --dt 0.05

This is the animated version of the idea in the enrich_scene.py snippet
from chat: that one only placed IDMVehicle instances at a random starting
position/heading and rendered a single static frame -- nothing was ever
stepping the simulation, so "background traffic" just meant "some cars
sitting on the road." Here the loop actually calls road.act() (each
IDMVehicle decides an action -- accelerate, brake, change lanes, via the
Intelligent Driver Model + MOBIL) then road.step(dt) (integrates the
physics + handles collisions) every frame, so the cars visibly car-follow
and react to each other over time.

Lives outside highway_env/layout/ on purpose (nothing in that folder is
touched) and reaches into it only to reuse display_all.py's loader/renderer
helpers and _real_scene.py's road import for scene 1.

Caveat worth knowing: the real imported road (see _real_scene.py) is a set
of disconnected lane polylines, not a connected routing graph -- Waymo's
raw map data doesn't ship one. A vehicle that reaches the end of its own
(often short) lane segment has no graph edge telling it what comes next,
even where one obviously should exist -- real roads (including turns) get
chopped into many separate polyline fragments in this data, so "off the
end of this lane" usually just means "the next fragment of the same road,
possibly curving, is a different lane index," not "actually off real
road." find_continuation() looks for a fragment starting within 8m in
roughly the vehicle's current heading and hops it over -- which is what
makes a car actually turn at a branch point instead of driving straight
through and colliding with cross traffic. Only when no such fragment
exists (a genuine dead end/map edge) does off_road() despawn the vehicle,
so the count can still drop over a long enough run, just far less than
before.

Keys: q / ESC / close window -> quit.
"""
import argparse
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "layout"))
import display_all as d  # noqa: E402
from highway_env.road.graphics import RoadGraphics, WorldSurface  # noqa: E402
from highway_env.vehicle.behavior import IDMVehicle  # noqa: E402

SCENE = "real_001_rebuilt"


class NoResnapIDMVehicle(IDMVehicle):
    """Plain IDMVehicle, minus the part of on_state_update() that re-finds
    the closest lane in the WHOLE network on every single physics step.

    That built-in behavior is cheap on a typical highway-env scenario (a
    handful of lanes), but our imported real map has ~115 separate lane
    fragments (see module docstring), so it was doing ~115 lane-distance
    checks per vehicle per step regardless of what any of the functions
    below do -- confirmed by profiling: it alone accounted for roughly a
    third of total runtime. It's also actively wrong for us: we manage
    lane transitions ourselves in advance_vehicles(), picking a real
    continuation that matches the vehicle's heading; the built-in version
    would silently overwrite that choice with whatever's geometrically
    NEAREST regardless of heading, right after every physics step.
    """

    def on_state_update(self) -> None:
        pass


def off_road(vehicle):
    """True the instant a vehicle is past the end of its own (short, real,
    disconnected -- see module docstring) lane segment and would be
    extrapolating into empty space rather than actually continuing on real
    road. No tolerance margin: this fires right at the boundary, not after
    it's drifted further -- a car that's off real road shouldn't still be
    shown driving on nothing for a few more frames."""
    try:
        s, lat = vehicle.lane.local_coordinates(vehicle.position)
    except Exception:
        return True
    return s < 0 or s > vehicle.lane.length or abs(lat) > vehicle.lane.width_at(np.clip(s, 0, vehicle.lane.length))


def all_lane_indexes(road):
    return [(f, t, i) for f, tos in road.network.graph.items()
            for t, lanes in tos.items() for i in range(len(lanes))]


def _lane_successors(road):
    """{lane_index: {lane_indexes that continue directly from its own 'to'
    node}} -- plain one-hop graph adjacency, cached on `road` (same pattern
    as find_continuation's own _continuation_geom_cache: lane GEOMETRY/
    TOPOLOGY is fixed once a road is built, only which vehicles sit on it
    changes tick to tick, so this is worth computing once, not per call)."""
    cache = getattr(road, "_lane_successors_cache", None)
    if cache is not None:
        return cache
    succ = {}
    for f, tos in road.network.graph.items():
        for t, lanes in tos.items():
            for i in range(len(lanes)):
                s = set()
                for t2, lanes2 in road.network.graph.get(t, {}).items():
                    for j in range(len(lanes2)):
                        s.add((t, t2, j))
                succ[(f, t, i)] = s
    road._lane_successors_cache = succ
    return succ


def _lane_predecessors(road):
    """{lane_index: {lane_indexes that lead directly INTO its own 'from'
    node}} -- the reverse of _lane_successors, same caching pattern."""
    cache = getattr(road, "_lane_predecessors_cache", None)
    if cache is not None:
        return cache
    pred = {idx: set() for idx in _lane_successors(road)}
    for a, succs in _lane_successors(road).items():
        for b in succs:
            pred.setdefault(b, set()).add(a)
    road._lane_predecessors_cache = pred
    return pred


def _walk_back(road, lane_index, lon, distance):
    """(lane_index, lon) reached by walking `distance` meters BACKWARD
    along the lane graph from (lane_index, lon) -- hopping onto a
    predecessor lane (arbitrarily the first one found, by construction;
    good enough for a bounded, best-effort retreat, not a claim that it's
    the "right" branch at a real fork) once the walk would go past the
    CURRENT lane's own start, instead of clamping at lon=0 and staying
    there.

    Without this, _retreat_if_safe's own backward search silently wastes
    most of a long `retreat` distance the moment the vehicle happens to
    be on a short lane (common here -- a turn arc or a GAP=12m bridge is
    often well under 12m): every candidate `step_back` past that point
    clips to the exact same lon=0 position, so trying a LARGER retreat
    distance changes nothing, confirmed directly (retreat=12 -> 40
    produced almost no improvement in success rate on a dense scene).
    Walking across the lane boundary instead genuinely reaches further
    back in real space.

    Stops (returns lon=0 on whichever lane it's on) if it runs out of
    predecessors before covering the full distance -- a genuine network
    dead-end behind the vehicle, not a bug to route around further.
    """
    pred = _lane_predecessors(road)
    remaining = distance
    idx = lane_index
    cur_lon = lon
    while remaining > cur_lon:
        remaining -= cur_lon
        preds = pred.get(idx)
        if not preds:
            return idx, 0.0
        idx = next(iter(preds))
        cur_lon = road.network.get_lane(idx).length
    return idx, cur_lon - remaining


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, c, tol=1e-6):
    return (min(a[0], b[0]) - tol <= c[0] <= max(a[0], b[0]) + tol
            and min(a[1], b[1]) - tol <= c[1] <= max(a[1], b[1]) + tol)


def _segments_intersect(p1, p2, p3, p4, tol=1e-6):
    """True if 2D segments p1-p2 and p3-p4 share a point, endpoints
    included -- a standard orientation test for a genuine crossing, plus
    an explicit colinear/touching check so two paths that MEET exactly at
    a shared endpoint (e.g. a real merge point, where two lanes' own
    centerlines end at the identical physical position) count too, not
    just an X-shaped crossing in the middle."""
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    if ((d1 > tol) != (d2 > tol)) and ((d3 > tol) != (d4 > tol)) and \
       (abs(d1) > tol or abs(d2) > tol) and (abs(d3) > tol or abs(d4) > tol):
        return True
    if abs(d1) <= tol and _on_segment(p3, p4, p1, tol):
        return True
    if abs(d2) <= tol and _on_segment(p3, p4, p2, tol):
        return True
    if abs(d3) <= tol and _on_segment(p1, p2, p3, tol):
        return True
    if abs(d4) <= tol and _on_segment(p1, p2, p4, tol):
        return True
    return False


def lane_conflict_table(road, sample_step=1.0):
    """{lane_index: {other lane_indexes whose own path GEOMETRICALLY
    crosses or merges with this one}} -- a real, static fact about the
    network's own topology (every junction/roundabout/merge here is built
    from a known, finite set of primitives -- add_four_way, add_three_way,
    round_about, merge -- so every crossing/merging conflict between any
    two lanes is fully determined by their own fixed geometry, computable
    once, not re-guessed every tick from live heading/position).

    Replaces a live heading-difference heuristic (a car turning right
    counted an unrelated car going straight on a DIFFERENT, non-crossing
    lane as a same-ish-heading-passing-close-by "conflict" purely because
    their instantaneous headings and lateral offsets happened to line up
    near the junction, not because their actual PATHS ever cross) with an
    exact question: do these two lanes' own centerlines ever occupy the
    same point in space. Two lane pairs are deliberately excluded from
    that test, both real, both confirmed to otherwise produce false
    positives:

      1. Directly graph-adjacent pairs (one is the other's own successor,
         sharing 'a is right in front of b' via a common node) -- that's
         find_front_vehicle's own job (car ahead on MY OWN path), not a
         crossing hazard between two DIFFERENT paths.
      2. Two lanes that share the same 'from' node (a fork: both start at
         the identical physical point, e.g. a straight and a turn option
         leaving the same approach) -- sharing a START point isn't a
         hazard, they're diverging AWAY from each other from there; only
         skips comparing their own very first sampled segment (still
         index >= 1 onward), so a genuine later re-crossing between two
         lanes that happen to also share a start is still caught.

    A shared 'to' node (two lanes CONVERGING to the same physical point --
    a real merge) is deliberately NOT excluded: that endpoint touch IS the
    conflict, the same way a crossing point in the middle of two lanes is.

    Cached on `road` (same pattern as find_continuation's own geometry
    cache) -- O(lanes^2 * samples^2) worst case, but a bounding-box reject
    per pair keeps this fast in practice, and it only ever runs once per
    road, not per tick.
    """
    cache = getattr(road, "_lane_conflict_table", None)
    if cache is not None:
        return cache

    indexes = all_lane_indexes(road)
    successors = _lane_successors(road)
    polylines = {}
    for idx in indexes:
        lane = road.network.get_lane(idx)
        n = max(2, int(lane.length / sample_step) + 1)
        polylines[idx] = [np.asarray(lane.position(s, 0)) for s in np.linspace(0, lane.length, n)]

    conflicts = {idx: set() for idx in indexes}
    for ai in range(len(indexes)):
        a = indexes[ai]
        pa = polylines[a]
        ax = [p[0] for p in pa]
        ay = [p[1] for p in pa]
        for bi in range(ai + 1, len(indexes)):
            b = indexes[bi]
            if b in successors.get(a, ()) or a in successors.get(b, ()):
                continue
            pb = polylines[b]
            bx = [p[0] for p in pb]
            by = [p[1] for p in pb]
            if max(ax) < min(bx) or min(ax) > max(bx) or max(ay) < min(by) or min(ay) > max(by):
                continue
            shared_start = a[0] == b[0]
            hit = False
            for i in range(1 if shared_start else 0, len(pa) - 1):
                for j in range(1 if shared_start else 0, len(pb) - 1):
                    if _segments_intersect(pa[i], pa[i + 1], pb[j], pb[j + 1]):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                conflicts[a].add(b)
                conflicts[b].add(a)

    road._lane_conflict_table = conflicts
    return conflicts


def find_continuation(road, lane_indexes, vehicle, max_dist=8.0, max_heading_diff_deg=60.0):
    """A lane fragment whose start sits near the vehicle's current position
    and continues in roughly its current heading -- the real map chops one
    continuous physical road (including a turn) into many separate
    polyline fragments, so 'off the end of this lane' usually just means
    'the next fragment of the same road is a different lane index', not
    'actually off real road'. Restricting to a heading match keeps this
    from grabbing an unrelated crossing lane that just happens to start
    nearby (a real intersection has several of those).

    Caches each lane's own (position(0,0), heading_at(0)) on `road` itself
    (road._continuation_geom_cache) the first time this runs for a given
    road, instead of recomputing it from every one of ~115 lane fragments
    on every call. Lane GEOMETRY is fixed once a road is built (only which
    vehicles sit on it changes tick to tick), so that recomputation was
    pure waste -- confirmed as the single largest per-tick cost in a real
    profiling run: on this real map's short (~20-40m) fragments, nearly
    every vehicle sits within find_front_vehicle's own 30m lookahead on
    nearly every tick, so this ran roughly once per vehicle per tick, each
    doing a full ~115-lane linear scan -- 2.6 million np.linalg.norm calls
    across just 200 ticks with ~43 vehicles, directly behind reports of
    the whole simulation running in jittery slow motion in the actual
    interactive interface (a fixed per-tick sim dt with no real-time
    catch-up means any tick this expensive makes the WHOLE scene lag).

    Attached to `road` (not a module-level dict keyed by id()) specifically
    because `road` is the one object here guaranteed to live exactly as
    long as the geometry it's caching stays valid -- a module-level
    id()-keyed cache would risk a stale hit if an old road gets garbage
    collected and Python reuses its memory address for an unrelated new
    one, a real risk here since this codebase's own test harnesses build
    many road instances in a loop within one process.
    """
    cache = getattr(road, "_continuation_geom_cache", None)
    if cache is None:
        cache = {}
        road._continuation_geom_cache = cache

    best_idx, best_d = None, max_dist
    for idx in lane_indexes:
        if idx == vehicle.lane_index:
            continue
        geom = cache.get(idx)
        if geom is None:
            lane = road.network.get_lane(idx)
            geom = (lane.position(0, 0), lane.heading_at(0))
            cache[idx] = geom
        start_pos, start_heading = geom
        d = np.linalg.norm(start_pos - vehicle.position)
        if d >= best_d:
            continue
        heading_diff = abs((np.degrees(start_heading - vehicle.heading) + 180) % 360 - 180)
        if heading_diff > max_heading_diff_deg:
            continue
        best_idx, best_d = idx, d
    return best_idx


def nearby_vehicles(road, vehicle, radius):
    """Cheap Euclidean-distance pre-filter -- local_coordinates() involves
    a search over a lane's own polyline and is not O(1), so calling it for
    every (vehicle, other) pair regardless of distance is what made the
    2-minute/100-vehicle test take 5+ minutes wall-clock. Almost every
    pair is obviously irrelevant on a real map spanning hundreds of
    meters; this cuts them out with a single cheap norm() first."""
    return [o for o in road.vehicles if o is not vehicle
            and np.linalg.norm(o.position - vehicle.position) <= radius]


def find_front_vehicle(road, vehicle, lane_indexes, candidates, lookahead=30.0, lateral_tol=3.0):
    """A better front_vehicle for IDM than road.neighbour_vehicles() gives:
    that call only considers vehicles on the exact same lane_index, but our
    real lane graph is fragmented (see find_continuation's docstring) --
    the car directly ahead of you, on what looks like the same lane, is
    very often nominally on a *different* lane_index (the next fragment),
    and a car about to cross your path at an intersection is on a
    genuinely different lane_index too. Neither is "seen" by the same-
    lane-only check, which is the actual bug behind cars driving straight
    into a car that is, physically, right in front of them.

    Projects every candidate onto ego's own current lane (catches both a
    same-fragment leader and a crossing vehicle passing near the conflict
    point) and, if ego is near the end of its lane, also onto the real
    continuation fragment (catches a leader that already transitioned onto
    the next segment). Returns whichever such vehicle is closest ahead
    within `lookahead`, laterally within `lateral_tol` -- or None.
    """
    try:
        s0, _ = vehicle.lane.local_coordinates(vehicle.position)
    except Exception:
        return None

    best, best_gap = None, lookahead
    for other in candidates:
        s_other, lat_other = vehicle.lane.local_coordinates(other.position)
        gap = s_other - s0
        if 0 < gap < best_gap and abs(lat_other) < lateral_tol:
            best, best_gap = other, gap

    remaining = vehicle.lane.length - s0
    if remaining < lookahead:
        nxt = find_continuation(road, lane_indexes, vehicle)
        if nxt is not None:
            nxt_lane = road.network.get_lane(nxt)
            for other in candidates:
                s_other, lat_other = nxt_lane.local_coordinates(other.position)
                gap = remaining + s_other
                if 0 < gap < best_gap and abs(lat_other) < lateral_tol:
                    best, best_gap = other, gap
    return best


def crossing_conflict_brake(road, vehicle, candidates, heads=frozenset(), danger_dist=22.0, lateral_tol=6.0,
                             stopped_threshold=1.0, safe_release_dist=10.0):
    """A harder brake for a genuine crossing conflict. WHICH pairs of
    vehicles even get checked is decided by lane_conflict_table (see its
    own docstring) -- a real, precomputed structural fact about whether
    `other`'s own lane ever geometrically crosses or merges with
    `vehicle`'s own lane at all, replacing an earlier live heading-
    difference/lateral-tolerance heuristic that had no notion of which
    lanes actually intersect (a car turning right and a car on a
    different, non-crossing lane could satisfy it by coincidence near a
    junction). WHETHER the two are close enough RIGHT NOW to matter is
    still purely instantaneous, no simulated future path -- IDM itself
    never predicts anything either, it reacts to the current gap and
    relative velocity to whatever's currently in front. Same idea as
    find_front_vehicle (project a vehicle's CURRENT position onto a lane
    and check the CURRENT gap/lateral offset), just checked in both
    directions.

    Checks two things, both purely instantaneous, for every candidate
    lane_conflict_table already says CAN conflict:
      1. other's current position projected onto ego's own lane (a
         crossing vehicle sitting close to ego's path right now).
      2. ego's current position projected onto other's lane (the
         symmetric case: ego is close to entering a path other already
         occupies).
    Returns a hard deceleration if either check trips within
    `danger_dist`/`lateral_tol`, else None -- EXCEPT for the tie-break
    below, which can release a vehicle from an otherwise-tripped check.

    Tie-break: two vehicles that both see each other as this kind of
    conflict, both brake to a stop, and then just stay there forever --
    each one's own gap/relative-velocity reading never changes once both
    are stationary, so nothing in a purely reactive rule ever gives either
    one a reason to move again. That's a real, confirmed failure mode (see
    module docstring / git history: a whole scene can lock up permanently
    this way, invisibly, while the crash count stays at 0).

    A first attempt released the tied "winner" by treating `other` as an
    ordinary IDM front_vehicle at the current (already tiny, by
    construction -- that's WHY they stopped this close) gap. That's
    self-defeating: IDM's own braking term saturates hard for a gap well
    under its ~10m comfort distance regardless of who's nominally "in
    front", so the "release" still computed a near-zero/negative
    acceleration and nothing actually moved -- confirmed by testing:
    avg speed across the whole scene still collapsed to a permanent 0.00.

    This version instead does a real, unconditional release (`continue`,
    no acceleration contribution from this pair at all) once BOTH:
      1. both vehicles are essentially stopped (`stopped_threshold`) --
         this is what makes releasing safe at all: a still-decelerating,
         still-closing vehicle (see below) is a real collision risk, a
         vehicle that's already stationary and STAYING stationary is not,
         since IDM's own free-flow acceleration ramps up gradually from 0;
      2. the real (euclidean, not lane-projected) distance between them is
         at least `safe_release_dist`, matched to IDM's own comfortable
         following distance (DISTANCE_WANTED, ~10m). A much smaller floor
         (3m -- just past physically touching) was also tried, on the
         theory that release is already gated on both being fully
         stopped so a small gap should be fine. It let more vehicles
         release, including heads that had won every one of their ties,
         but made things measurably worse, not better: 8 new collisions
         over a 2-minute/100-vehicle test (vs 0 at 10m), and the scene
         still eventually froze solid anyway -- so the closer releases
         bought a temporarily higher average speed at the cost of actual
         crashes, without even fixing the permanent freeze. Reverted to
         the full comfort distance.

         RE-TESTED after the tie-break itself was fixed (see below --
         `_stopped_ticks` replacing raw id()), on the theory that the
         earlier crashes were caused by the OLD, unreliable winner
         selection releasing genuinely-not-ready vehicles, not by
         proximity itself. Verified false: even with the corrected
         tie-break, 6m freed 2 of 5 real queue-front vehicles in a 15-
         vehicle scene (up from 0 of 5 at 10m) but the whole scene still
         fully re-froze within another ~60s, and 0m (gate effectively
         removed) produced 2 new crashes on top of still not fixing the
         freeze. So closing this specific gap is not what's actually
         gating the remaining freeze; the tie-break/distance-gate design
         in this function appears to be functioning as intended now, and
         the remaining permanent-freeze cases traced back to it involve a
         crossing candidate close enough that forcing a release there is a
         genuine safety tradeoff, not just a parameter to tune away.
         Still an open problem as of this writing.
    A pair that's tripped this check while already CLOSER than that
    keeps the ordinary hard -8.0 brake even for the "winner" -- release
    only ever loosens a conflict that has room to loosen safely, never
    all of them at once. In practice this means a released winner is
    usually one that hasn't fully closed to a stop this close in the
    first place, not a vehicle that's already sitting right next to the
    other.

    Exactly one side of a tied pair gets released. A pure id()-based total
    order (higher id never yields to a lower one, anywhere in the scene)
    is enough to prevent cycles, but turned out NOT to be enough to
    prevent gridlock: it only guarantees the single scene-wide-highest-id
    vehicle is never held by a crossing conflict, and that one vehicle is
    usually not a same-lane queue's own front vehicle (front_vehicle is
    already someone else, ahead of it, unrelated to id order) -- so it
    stays stuck behind an ordinary, undramatic same-lane leader anyway,
    while every actual queue-front vehicle (`heads`, no same-lane leader
    of its own) can easily still lose one of ITS OWN several simultaneous
    ties by pure id-luck and never release either. Confirmed by testing:
    every queue-front vehicle in a frozen scene still had at least one
    crossing candidate with a higher id, so NONE of them ever released,
    even though the tie-break code was firing constantly elsewhere.

    So `heads` (each vehicle with no same-lane leader right now --
    computed once per apply_better_car_following() call, since it already
    does this same front_vehicle lookup for every vehicle anyway) gets
    first say: a head always wins against a non-head it's tied with, on
    the real-world logic that "already first in your own lane's queue"
    is a stronger, more physically meaningful claim than an arbitrary
    memory address.

    STILL NOT ENOUGH ON ITS OWN, confirmed by testing (see LimitedVisionHuman
    validation, highway_env/human/): head-vs-non-head is resolved correctly,
    but a head tied against ANOTHER head -- or a non-head against another
    non-head -- fell through to raw id() comparison, which is exactly the
    pairing found holding a real scene frozen solid at every FOV width
    tested, including 360 degrees (so not an FOV bug): neither vehicle in
    the pair was a head, id() said the same loser every single tick, and
    nothing ever changed. Every queue-front vehicle in that frozen scene
    could independently be shown to still lose to at least one of its own
    ties, by the same mechanism.

    THE ACTUAL FIX: within a head/non-head tier, break the tie by
    `_stopped_ticks` (how many CONSECUTIVE steps a vehicle has already been
    observed below `stopped_threshold` -- maintained in
    apply_better_car_following(), pure history, nothing simulated forward,
    same category of bookkeeping as `.crashed`) instead of id(). Whoever has
    been stopped LONGER wins -- first-come-first-served, the same
    convention an unsignalized real intersection actually runs on, rather
    than an arbitrary memory address that has no relationship to how long
    anyone has actually been waiting. id() is now only the tie-break of
    LAST resort, for the near-impossible case of two vehicles with
    identical head-status AND identical stopped-tick counts.

    Still exactly as cycle-proof as plain id() was: the combined key
    (is_head, stopped_ticks, id) is a genuine total order over every
    vehicle at a given instant (three totally-ordered fields, compared
    lexicographically), and a total order cannot contain a cycle by
    construction -- A beats B beats C beats A is impossible under any
    consistent ordering, whatever it's built from. Swapping id() for a more
    meaningful primary/secondary key doesn't touch that guarantee; it only
    changes WHICH vehicle wins, from "unpredictable" to "whoever has
    actually been waiting longest." The loser keeps braking until ITS OWN
    reading of the conflict changes (the winner moved far/fast enough to
    clear it), which happens naturally as a side effect of the winner
    moving -- no coordination between the two calls needed.

    (An even earlier version released unconditionally, with no
    stopped_threshold gate at all -- so it could release against a
    vehicle that was still moving/closing, not just stopped -- and made
    things worse (new collisions). It was also tested at the same time as
    an unrelated stuck-timeout in advance_vehicles() that was
    independently harmful (see that function's docstring), confounding
    the result. This version adds both the stopped-gate and the distance
    gate, and was tested in isolation, without that timeout.)
    """
    try:
        s0, _ = vehicle.lane.local_coordinates(vehicle.position)
    except Exception:
        return None
    my_conflicts = lane_conflict_table(road).get(vehicle.lane_index, ())
    result = None
    for other in candidates:
        if other.crashed:
            continue
        if other.lane_index not in my_conflicts:
            # lane_conflict_table (see its own docstring) is a real,
            # precomputed structural fact: does `other`'s own lane ever
            # geometrically cross or merge with `vehicle`'s own lane at
            # all, anywhere -- not a live heading/distance guess. Same-ish
            # direction traffic (find_front_vehicle/IDM's own job) and
            # oncoming traffic in a parallel lane both correctly have no
            # entry here (their lanes run alongside, never crossing), same
            # as the heading-difference heuristic this replaced was
            # trying to approximate -- but exactly, from the network's own
            # geometry, instead of guessing from instantaneous heading and
            # lateral offset (which a car turning right and a car going
            # straight on a DIFFERENT, non-crossing lane could satisfy by
            # coincidence near a junction even though their actual paths
            # never intersect -- confirmed the source of false positives
            # this table exists to remove).
            continue

        conflict = False
        s_other, lat_other = vehicle.lane.local_coordinates(other.position)
        g = s_other - s0
        if 0 < g < danger_dist and abs(lat_other) < lateral_tol:
            conflict = True

        if not conflict:
            try:
                s_ego_on_other, lat_ego_on_other = other.lane.local_coordinates(vehicle.position)
                other_s, _ = other.lane.local_coordinates(other.position)
            except Exception:
                continue
            g2 = s_ego_on_other - other_s
            if 0 < g2 < danger_dist and abs(lat_ego_on_other) < lateral_tol:
                conflict = True

        if not conflict:
            continue

        vehicle_is_head = id(vehicle) in heads
        other_is_head = id(other) in heads
        if vehicle_is_head != other_is_head:
            wins = vehicle_is_head
        else:
            v_ticks = getattr(vehicle, "_stopped_ticks", 0)
            o_ticks = getattr(other, "_stopped_ticks", 0)
            wins = v_ticks > o_ticks if v_ticks != o_ticks else id(vehicle) > id(other)
        if (vehicle.speed < stopped_threshold and other.speed < stopped_threshold
                and wins
                and np.linalg.norm(other.position - vehicle.position) >= safe_release_dist):
            continue  # released: both stopped, already at a safe distance, and we win the tie

        result = -8.0 if result is None else min(result, -8.0)
    return result


def vehicle_status_label(road, vehicle, lane_indexes, radius=35.0, stopped_threshold=0.5):
    """Short debug string describing WHY `vehicle` is currently behaving as
    it is -- for an on-screen debug overlay (watch.py's --debug-status),
    never consulted by any actual driving decision. Re-derives its answer
    from the exact same checks find_front_vehicle/crossing_conflict_brake/
    off_road/find_continuation already make internally, so what's on
    screen is guaranteed to match what the vehicle is actually reacting to
    -- not a separate, potentially-drifting heuristic.

    Returns None while `vehicle` is moving above `stopped_threshold` (no
    label needed -- ordinary driving isn't a debugging question), else one
    of "CRASHED", "END OF ROAD", "SAME LANE <Nm>" (find_front_vehicle: a
    real leader close ahead in vehicle's OWN lane -- ordinary car-
    following, same direction of travel, ends the instant that leader
    moves), "CROSS TRAFFIC" (crossing_conflict_brake: someone on a
    GENUINELY DIFFERENT heading whose own path crosses this vehicle's,
    e.g. another approach at a junction -- a different category of hazard
    from a same-lane leader even though both look like "stopped, blocked
    by something" from outside), or "STOPPED" (neither matched -- e.g.
    still accelerating from a stop, or blocked by something further up a
    chain this single-vehicle check doesn't trace) -- each suffixed with
    `vehicle.lane_index` itself (the exact (from, to, lane_id) tuple the
    vehicle's own driving logic currently believes it's on -- the same
    value find_front_vehicle/off_road/etc. above all read), so a lane-
    transition bug (vehicle visibly on one physical lane, .lane_index
    still pointing at a different one) shows up directly on screen instead
    of only being discoverable by print-debugging separately.
    """
    lane_str = f"{vehicle.lane_index[0]}->{vehicle.lane_index[1]}#{vehicle.lane_index[2]}" \
        if getattr(vehicle, "lane_index", None) is not None else "no lane_index"

    if vehicle.crashed:
        return f"CRASHED [{lane_str}]"
    if vehicle.speed >= stopped_threshold:
        return None

    candidates = nearby_vehicles(road, vehicle, radius)
    front = find_front_vehicle(road, vehicle, lane_indexes, candidates)

    if off_road(vehicle) and find_continuation(road, lane_indexes, vehicle) is None:
        return f"END OF ROAD [{lane_str}]"
    if front is not None:
        gap = float(np.linalg.norm(np.asarray(front.position) - np.asarray(vehicle.position)))
        return f"SAME LANE {gap:.0f}m [{lane_str}]"

    heads = {id(o) for o in candidates
             if find_front_vehicle(road, o, lane_indexes, nearby_vehicles(road, o, radius)) is None}
    if front is None:
        heads.add(id(vehicle))
    if crossing_conflict_brake(road, vehicle, candidates, heads=heads) is not None:
        return f"CROSS TRAFFIC [{lane_str}]"

    return f"STOPPED [{lane_str}]"


def apply_better_car_following(road, lane_indexes, dt, radius=35.0):
    """Recompute each vehicle's acceleration using find_front_vehicle() and
    crossing_conflict_brake() instead of whatever road.act() (via
    road.neighbour_vehicles(), same-lane_index only) just gave it. Reuses
    IDM's own acceleration formula -- vehicle.acceleration() -- just fed a
    front_vehicle that isn't blind to fragment boundaries and crossing
    traffic, plus a harder override for genuine crossing conflicts IDM's
    own formula isn't built for. Both checks are purely reactive
    (current-state only, no simulated future path), same as IDM itself.

    The hard -8.0 override in crossing_conflict_brake doesn't know the
    vehicle's own speed, so a few consecutive steps of it can drive speed
    past zero into reversing -- confirmed as the actual cause of same-
    direction "rear-end" collisions in testing: a vehicle pushed into
    reverse by this override then drives backward into whatever's behind
    it, which find_front_vehicle has no way to see (it only ever looks
    for a leader ahead, gap > 0). IDM's own acceleration() is naturally
    bounded and doesn't do this; the fix is to floor the FINAL combined
    acceleration here so nothing this function does can take speed below
    zero, regardless of which check produced it.

    Call after road.act() and before road.step(dt) -- `dt` must be the
    same timestep passed to that road.step() call.

    Two passes: the first just finds each vehicle's own front_vehicle
    (needed for the IDM override below regardless), and along the way
    builds `heads` -- the set of vehicles with no same-lane leader right
    now -- for crossing_conflict_brake's tie-break to use in the second
    pass. Free (no extra lookups): it's the same find_front_vehicle call
    this function already had to make for every vehicle anyway.
    """
    per_vehicle = {}
    for v in road.vehicles:
        if v.crashed or not hasattr(v, "acceleration"):
            continue
        # Bookkeeping for crossing_conflict_brake's tie-break: how many
        # CONSECUTIVE steps this vehicle has already been observed stopped.
        # Pure history (v.speed is last step's already-integrated result,
        # nothing simulated forward), same category of state as .crashed.
        v._stopped_ticks = getattr(v, "_stopped_ticks", 0) + 1 if v.speed < 1.0 else 0
        candidates = nearby_vehicles(road, v, radius)
        front = find_front_vehicle(road, v, lane_indexes, candidates)
        per_vehicle[id(v)] = (v, candidates, front)
    heads = {vid for vid, (_, _, front) in per_vehicle.items() if front is None}

    for v, candidates, front in per_vehicle.values():
        if front is not None:
            v.action["acceleration"] = min(
                v.action.get("acceleration", 0.0),
                v.acceleration(ego_vehicle=v, front_vehicle=front, rear_vehicle=None),
            )
        conflict_override = crossing_conflict_brake(road, v, candidates, heads=heads)
        if conflict_override is not None:
            v.action["acceleration"] = min(v.action.get("acceleration", 0.0), conflict_override)

        v.action["acceleration"] = max(v.action["acceleration"], -v.speed / dt)


def advance_vehicles(road, lane_indexes):
    """Step every vehicle onto its lane's real continuation if it just ran
    off the end of one, or remove it if there's genuinely nowhere to go.

    A stuck-for-N-seconds timeout was also tried right here, to remove
    vehicles that go completely motionless. It made things worse: a
    92-vehicle scene on a real, unsignalized, no-traffic-light map
    naturally has cars queued and briefly stationary for a long time as
    ordinary congestion, not as a bug, and "stationary for N seconds"
    can't tell that apart from a genuine permanent stall -- mass-removing
    "slow" vehicles mid-queue opened sudden gaps that destabilized the
    survivors into new collisions. Deliberately NOT reintroduced here; the
    actual permanent-stall fix (mutual-deadlock tie-break) lives in
    crossing_conflict_brake() instead, since that's a case this function
    can't distinguish from normal congestion by vehicle state alone.
    """
    survivors = []
    for v in road.vehicles:
        if off_road(v):
            next_idx = find_continuation(road, lane_indexes, v)
            if next_idx is None:
                continue  # genuinely off real road -- despawn, per off_road()'s own docstring
            v.lane_index = next_idx
            v.lane = road.network.get_lane(next_idx)
            v.target_lane_index = next_idx
        survivors.append(v)
    road.vehicles = survivors


def _retreat_if_safe(road, vehicle, retreat=12.0, retreat_safe_distance=7.0, retreat_step=0.5,
                      moving_safe_distance=18.0, moving_speed_threshold=1.0):
    """Reposition `vehicle` backward along its own current lane until it
    clears a safe distance from every OTHER vehicle on the road -- tries
    the full `retreat` distance first, then less in `retreat_step`
    increments (fine enough that a clear gap narrower than one step can't
    be stepped over and missed), down to no movement at all if nothing
    along that stretch is clear. Returns whether a safe spot was found;
    `vehicle` is left exactly where it was on a False (never moving is
    always at least as safe as its own current, already-non-crashed
    position).

    Two DIFFERENT things can make a center-to-center distance unsafe, so
    there are two separate thresholds, not one flat number:

    `retreat_safe_distance` (against a vehicle that's ALSO essentially
    stopped, speed < moving_speed_threshold): must exceed Vehicle.LENGTH
    (5.0m) by a real margin, not just clear it -- two vehicle BODIES, not
    points, and a straight retreat along a lane tends to land ego roughly
    nose-to-tail with whatever's already there. Confirmed as a real, not
    theoretical, bug: an earlier version of this used exactly 5.0 (equal
    to LENGTH, zero margin) and a 60-vehicle test on real_001_rebuilt
    produced 4 new collisions -- traced directly to two STOPPED vehicles
    (both speed 0.00) left a center-to-center 5.18m apart, just over the
    old threshold but well inside their own combined body length once
    nose-to-tail. 7.0 leaves a real ~2m body gap instead of ~0.

    `moving_safe_distance` (against a vehicle that's still moving): wider
    still, since it can close even a real, non-overlapping gap within a
    couple of ticks well before reacting to a vehicle that just appeared
    there -- 18.0, comfortably past this scene's own 8-14 m/s spawn-speed
    reaction+braking distance.

    Re-running the same 60-vehicle/1500-step test with both fixes
    produced 0 new collisions and no permanent freeze.

    The bounded, one-time "someone backs the car up" intervention every
    stuck-resolution function in this codebase uses instead of deleting a
    vehicle -- never loosens any of crossing_conflict_brake's own safety
    gates, just clears room for its EXISTING, unmodified release logic to
    actually see clearance on a later tick.

    The search itself walks the lane GRAPH backward (_walk_back), not
    just the vehicle's own current lane -- see that function's own
    docstring for why a same-lane-only search silently wastes most of
    `retreat` the moment the vehicle is on a short lane (common here).
    Landing on a different lane than the vehicle started on updates its
    full state (lane_index/lane/target_lane_index/heading), not just
    position -- otherwise its own next local_coordinates()-based
    reasoning (find_front_vehicle, crossing_conflict_brake, ...) would
    read against a lane object that no longer matches where it visually
    and physically is.
    """
    lon, lat = vehicle.lane.local_coordinates(vehicle.position)
    others = [(np.asarray(v.position), v.speed) for v in road.vehicles if v is not vehicle]
    for step_back in np.arange(retreat, retreat_step - 1e-9, -retreat_step):
        candidate_idx, candidate_lon = _walk_back(road, vehicle.lane_index, lon, step_back)
        candidate_lane = road.network.get_lane(candidate_idx)
        candidate_pos = candidate_lane.position(candidate_lon, lat)
        safe = True
        for p, speed in others:
            required = moving_safe_distance if speed >= moving_speed_threshold else retreat_safe_distance
            if np.linalg.norm(candidate_pos - p) < required:
                safe = False
                break
        if safe:
            vehicle.position = candidate_pos
            if candidate_idx != vehicle.lane_index:
                vehicle.heading = candidate_lane.heading_at(candidate_lon)
                vehicle.lane_index = candidate_idx
                vehicle.lane = candidate_lane
                vehicle.target_lane_index = candidate_idx
            return True
    return False


def unstick_stalled_traffic(road, dt, timeout_s=25.0, retreat=12.0, retreat_safe_distance=7.0):
    """Break a genuine permanent standoff WITHOUT ever deleting a vehicle.

    crossing_conflict_brake's own tie-break (see its docstring) resolves
    most mutual standoffs by letting one side win and move, but its own
    safe_release_dist gate can leave even the rightful winner stuck
    forever when the conflict geometry itself never puts the pair that
    far apart -- confirmed directly: a 60-vehicle run on real_001_rebuilt
    (apply_better_car_following + advance_vehicles only, no other
    intervention) collapses to a permanent, exact 0.00 average speed by
    step ~400 and never recovers over 1500 steps tested. `_stopped_ticks`
    (already maintained by apply_better_car_following for the tie-break's
    own use) is what distinguishes this from ordinary queuing: a vehicle
    whose front gap ever opens even slightly shows nonzero IDM creep and
    resets its own counter, so `timeout_s` straight seconds of EXACT zero
    movement is strong evidence of a genuine deadlock, not a long but
    ordinary wait in a busy queue -- same reasoning this codebase's other
    stuck-resolution functions already rely on.

    Call once per tick, after apply_better_car_following/road.step -- the
    general form of what limit_vision_human.py's own _unstick_if_frozen/
    _unstick_frozen_background/_resolve_stuck_route_pair do for the human/
    robot specifically (see those for the fuller history); this is the
    version usable by scene1_background.py's OWN main() loop, which has
    no human/robot and previously had no unstick mechanism of any kind --
    the exact gap that produced the permanent freeze above.

    Resets `_stopped_ticks` to 0 after a successful retreat: repositioning
    doesn't touch `.speed`, so without this the same vehicle would still
    read as freshly stopped on the very next tick, before IDM has had any
    chance to actually accelerate it away, and get retreated again and
    again every tick thereafter.

    HONEST LIMIT, not swept under the rug: retreat only helps where a
    genuinely SAFE spot exists to retreat TO. In an extremely saturated
    local cluster (confirmed directly: user_study/test_dense_scene.py's
    own 35-background+3-robot+seed_maneuver_traffic scene, deliberately
    named "dense", most of a compact ~160-lane network sitting stopped at
    once) `_retreat_if_safe`'s own success rate measured as low as ~2% --
    there is often nowhere within reach that clears every other vehicle's
    own safety margin, same as a real car genuinely cannot always find
    room to back up in real gridlock. This function's predecessor
    "solved" that same scene by deleting vehicles outright (confirmed:
    swapping ONLY the delete-based version back in, same everything else,
    recovers the old throughput) -- an option deliberately not available
    here. What retreat DOES still guarantee, verified directly on that
    same scene: zero new collisions and no vehicle ever removed, even
    while progress in that one extreme scenario is genuinely slower than
    the old delete-based version's own. It is not a substitute for
    reducing how many vehicles a network is asked to hold at once if a
    given scene turns out to need more capacity than it has.
    """
    for v in road.vehicles:
        if v.crashed:
            continue
        if getattr(v, "_stopped_ticks", 0) * dt < timeout_s:
            continue
        if _retreat_if_safe(road, v, retreat, retreat_safe_distance):
            v._stopped_ticks = 0


def add_background_traffic(road, count=100, seed=0, speed_range=(8.0, 14.0), safe_distance=10.0, max_tries=20,
                            lane_indexes=None):
    """Spawn up to `count` IDMVehicle, each on a random real lane at a
    random position -- but never on top of / overlapping a vehicle already
    placed, OR any vehicle already on `road` before this call (e.g. a
    LimitedVisionHuman spawned via add_human_vehicle first -- its own fixed
    spawn point has no say in where random background traffic lands, so it
    needs the same protection placed-this-call vehicles already got). Each
    candidate spot is checked against every already-placed vehicle; if it's
    within `safe_distance`, that spot is rejected and a different random
    lane/position is tried instead (up to `max_tries`). If no safe spot
    turns up after that many tries, that vehicle is simply skipped rather
    than spawned into a guaranteed collision -- so `count` is an upper
    bound, not a guarantee, once the road is dense enough that safe spots
    run out.

    lane_indexes: candidate lanes to spawn on (default: every lane in the
    network, the original behavior). Pass a restricted list -- e.g.
    mega_scene.route_adjacent_lane_indexes() -- to bias density toward
    where a route-following vehicle actually drives instead of spreading it
    uniformly over lanes it may never visit (a small synthetic network has
    a much higher fraction of those, e.g. a roundabout's own unused stub
    arms, than a large real recorded map does).
    """
    rng = np.random.default_rng(seed)
    lane_indexes = lane_indexes if lane_indexes is not None else all_lane_indexes(road)
    placed = [v.position for v in road.vehicles]
    for _ in range(count):
        for _ in range(max_tries):
            f, t, i = lane_indexes[rng.integers(len(lane_indexes))]
            lane = road.network.get_lane((f, t, i))
            longitudinal = rng.uniform(0, lane.length)
            position = lane.position(longitudinal, 0)
            if all(np.linalg.norm(position - p) >= safe_distance for p in placed):
                break
        else:
            continue  # no safe spot found in max_tries -- skip this one, don't spawn it into a collision
        vehicle = NoResnapIDMVehicle(
            road, position,
            heading=lane.heading_at(longitudinal),
            speed=rng.uniform(*speed_range),
        )
        road.vehicles.append(vehicle)
        placed.append(position)
    return road


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=100, help="number of background vehicles (default 12)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--dt", type=float, default=1 / 15, help="simulation timestep in seconds (default 1/15)")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many steps (default: run until closed)")
    args = parser.parse_args()

    module = d.load_layout(SCENE)
    road = module.build_road()
    add_background_traffic(road, count=args.count, seed=args.seed)
    print(f"{SCENE}: {len(road.vehicles)} background vehicles, stepping at dt={args.dt:.3f}s")

    pygame.init()
    desktop = pygame.display.Info()
    w, h = int(desktop.current_w * 0.9), int(desktop.current_h * 0.85)
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"scene1 background traffic ({len(road.vehicles)} vehicles)")

    surface = WorldSurface((w, h - d.LABEL_H), 0, pygame.Surface((w, h - d.LABEL_H)))
    min_x, max_x, min_y, max_y = d.road_bounding_box(road)
    surface.scaling = min(w / ((max_x - min_x) * 1.15), (h - d.LABEL_H) / ((max_y - min_y) * 1.15))
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    surface.origin = center - np.array([w / 2, (h - d.LABEL_H) / 2]) / surface.scaling

    human_route = getattr(module, "HUMAN_ROUTE", None)
    robot_route = getattr(module, "ROBOT_ROUTE", None)
    font = pygame.font.SysFont(None, 22)
    lane_indexes = all_lane_indexes(road)

    running = True
    step_count = 0
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        road.act()
        apply_better_car_following(road, lane_indexes, args.dt)
        road.step(args.dt)
        advance_vehicles(road, lane_indexes)
        unstick_stalled_traffic(road, args.dt)
        step_count += 1
        if args.steps is not None and step_count >= args.steps:
            running = False

        surface.fill(surface.GREY)
        RoadGraphics.display(road, surface)
        d.draw_lane_arrows(surface, road)
        RoadGraphics.display_traffic(road, surface, simulation_frequency=round(1 / args.dt), offscreen=True)
        if human_route:
            d.draw_routes(surface, road, human_route, [d.HUMAN_COLOR], -d.ROUTE_LATERAL_OFFSET, SCENE, "human")
        if robot_route:
            d.draw_routes(surface, road, robot_route, d.ROBOT_PALETTE, d.ROUTE_LATERAL_OFFSET, SCENE, "robot")

        window.fill(d.BG)
        window.blit(surface, (0, d.LABEL_H))
        window.blit(font.render(f"{SCENE}  step {step_count}  ({len(road.vehicles)} vehicles)  [q/ESC quit]",
                                 True, d.LABEL), (10, 2))
        pygame.display.flip()
        clock.tick(round(1 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
