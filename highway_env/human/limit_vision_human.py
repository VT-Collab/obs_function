"""A driver whose perception is genuinely FOV-limited: an angular cone AND
occlusion by other vehicles' bodies, with ZERO memory of anything once it
leaves that cone. Drives a fixed path through scene 1 (HUMAN_ROUTE).

Everything this feature needs lives in this file and this directory --
scene1_background.py is used strictly as an unmodified library (imported,
never edited), by explicit instruction, so this experimental feature cannot
put the now-stable, hard-won background-traffic simulation at risk. Two
pieces of logic below (route_aware_continuation, apply_human_aware_car_
following/advance_vehicles_with_route) are therefore small, deliberate
duplicates of scene1_background.find_continuation()/apply_better_car_
following()/advance_vehicles() rather than parameters added to those
functions -- see each one's docstring for exactly what it reuses vs repeats.

--------------------------------------------------------------------------
PERCEPTION -- what "FOV-faithful" means here
--------------------------------------------------------------------------
visible(other) = in_cone(other) AND NOT occluded(other), evaluated fresh
every tick from current positions/headings only. No sight radius of its own
-- the cone/occlusion filter is applied to whatever nearby_vehicles(radius)
already handed it, so distance is already bounded by that (35m default),
comfortably past the 22m/10m ranges the rest of the machinery cares about.

Occlusion is checked against every OTHER nearby vehicle's actual rotated
rectangle (its real LENGTH/WIDTH), not a point -- a vehicle sitting behind a
truck is invisible even if its own center would otherwise be in the cone and
in range. This is deliberately the ONLY kind of occluder: there are no
buildings/static obstacles in the imported map data, and adding synthetic
ones is out of scope for this pass.

NO MEMORY, at all, of a vehicle once it leaves the visible set -- not even a
last-known position for one tick. Two reasons, not one:
  1. It was asked for directly: "occluded and does not sense and know and
     act on anything outside of FOV."
  2. It would quietly reintroduce prediction. This project has a separate,
     hard, pre-existing rule that IDM/crossing-brake react to CURRENT state
     only, never a simulated future (see crossing_conflict_brake's own
     docstring in scene1_background.py). A remembered position of a MOVING
     vehicle, reacted to as if still current, is an extrapolation of exactly
     that kind -- and at highway speeds (8-14 m/s) a memory even 1-2 seconds
     stale is already wrong by more than the distances (10-22m) this module
     decides on, so it would be actively misleading, not "slightly outdated
     but useful." (Overcooked's belief-decay design is sound for THAT domain
     specifically because station contents change slowly between glances;
     that reasoning does not transfer to a moving hazard.)

--------------------------------------------------------------------------
ROUTE -- a fixed path, not a destination node
--------------------------------------------------------------------------
HUMAN_ROUTE (real_001.py) is a plain list of (x, y) world points, not a
lane-graph destination -- stock ControlledVehicle.plan_route_to() needs a
connected road-network graph to BFS over, which this fragmented real map
(~115 disconnected polyline fragments) doesn't have; that's the whole reason
find_continuation()/advance_vehicles() exist as a custom replacement in
scene1_background.py in the first place. So route-following here is NOT the
stock .route/plan_route_to machinery -- it's a bias on the SAME fork-choice
find_continuation() already makes at the end of a fragment: among whatever
candidates pass the existing safety gates (max_dist, max_heading_diff_deg,
UNCHANGED), prefer the one closest ahead on HUMAN_ROUTE instead of purely the
nearest start-point. See route_aware_continuation()'s own docstring for why
this matters beyond convenience: a route-free fork choice has no protection
against a small local loop (the human circling the same few fragments
forever) since it has no memory of where it's been either; progress along an
external fixed path can't loop that way by construction.

Progress along the route is recomputed from scratch every call (nearest
point on the polyline to the vehicle's CURRENT position) rather than tracked
as a stored waypoint index/cursor -- so a vehicle nudged off-line by traffic
just re-projects from wherever it actually is next tick. No cursor to get
stuck.

--------------------------------------------------------------------------
VALIDATION -- what "done" means (see test_limited_vision_human.py)
--------------------------------------------------------------------------
Two things this has to demonstrate, MEASURED rather than assumed (this
project has already been burned once, elsewhere, by a metric -- crash count
alone -- that looked fine while the scene was completely gridlocked):

  (A) BEHAVIOR GENUINELY DIFFERS BY FOV. Narrower cones should show a
      measurably shorter reaction distance to a conflict, and/or more/harder
      braking events, and/or slower route progress -- this is the actual
      signal a later Bayesian FOV-inference step needs to exist at all.
  (B) FUNCTIONAL AT EVERY FOV. Never permanently stuck (nonzero average
      speed over the whole run) and no worse a crash rate than an
      unrestricted driver, at every cone width tested.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # highway_env/
import scene1_background as sb  # noqa: E402
from common.geometry import in_cone, is_occluded, segment_intersects_rotated_rect  # noqa: E402,F401

FOV_DEG_DEFAULT = 120.0


# -- geometry: cone + occlusion -- lives in highway_env/common/geometry.py
# (pure functions of positions/headings/extents only, no route/task
# knowledge, a sibling of human/ and robot/ so neither depends on the
# other), re-exported above unchanged so nothing importing them from here
# breaks.


# -- route progress: stateless nearest-point-on-polyline projection --------

def _route_progress(route_points, position):
    """(arc_length, distance) of the nearest point on the route_points
    polyline to `position`. Stateless -- no stored index, recomputed fresh
    from the actual current position every call, so a vehicle knocked
    sideways by traffic just re-projects from wherever it really is next
    time rather than reading a stale cursor."""
    pos = np.asarray(position, dtype=float)
    pts = np.asarray(route_points, dtype=float)
    best_dist, best_arc, arc = None, 0.0, 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = b - a
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            continue
        t = np.clip(np.dot(pos - a, seg) / (seg_len ** 2), 0.0, 1.0)
        dist = np.linalg.norm(pos - (a + t * seg))
        if best_dist is None or dist < best_dist:
            best_dist, best_arc = dist, arc + t * seg_len
        arc += seg_len
    return best_arc, (best_dist if best_dist is not None else float("inf"))


def route_total_length(route_points):
    pts = np.asarray(route_points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def route_aware_continuation(road, lane_indexes, vehicle, route_points,
                              max_dist=8.0, max_heading_diff_deg=60.0):
    """A copy of scene1_background.find_continuation()'s candidate search
    (same two safety gates, byte-for-byte: max_dist on start-point distance,
    max_heading_diff_deg on heading match), but re-ranked by progress along
    `route_points` instead of purely by nearest-start-point distance, when a
    route is given.

    WHY A SEPARATE COPY rather than a `route=` parameter added to the real
    find_continuation(): this feature's explicit scope is "keep all changes
    in highway_env/human" -- scene1_background.py is not edited for this.

    WHY THIS MATTERS BEYOND CONVENIENCE: plain find_continuation() has no
    memory of which fragments a vehicle has already visited, so nothing
    structurally prevents it from settling into a small local loop (finding
    itself back in the same few fragments' neighborhood and repeatedly
    making the same nearest-start-point choice). Progress along a FIXED
    external path can't loop that way by construction -- revisiting the same
    fragments would mean no forward arc-length gain, so this ranking keeps
    pushing toward whatever fork actually advances.

    FALLBACK RE-APPROACHES THE ROUTE RATHER THAN IGNORING IT. When no
    candidate shows forward progress, the choice among them is by distance
    from the CANDIDATE'S START POINT TO THE ROUTE POLYLINE (pulling back
    toward the path), not by distance to the vehicle's current position
    (plain find_continuation()'s own metric, which has no notion of the
    route at all). This was found to matter empirically, not just in
    theory: an earlier version used the vehicle-distance metric for the
    fallback, and a vehicle that strayed from the route at one fork kept
    compounding the drift at every later fork (each fallback choice being
    "nearest to wherever I've wandered to" rather than "nearest back to
    where I'm supposed to be"), ending in a genuine dead end far from the
    route with no viable continuation at all.

    NEVER WORSE than plain find_continuation() in the sense that it never
    returns None when find_continuation() would have found something --
    same two safety gates, just re-ranked -- but see advance_vehicles_
    with_route() for what happens on the genuine "no candidate at all"
    case, which for a route-following vehicle is handled differently than
    for background traffic.
    """
    if route_points is None:
        return sb.find_continuation(road, lane_indexes, vehicle, max_dist, max_heading_diff_deg)

    candidates = []
    for idx in lane_indexes:
        if idx == vehicle.lane_index:
            continue
        lane = road.network.get_lane(idx)
        start = lane.position(0, 0)
        d = np.linalg.norm(start - vehicle.position)
        if d >= max_dist:
            continue
        heading_diff = abs((np.degrees(lane.heading_at(0) - vehicle.heading) + 180) % 360 - 180)
        if heading_diff > max_heading_diff_deg:
            continue
        candidates.append((idx, start))

    if not candidates:
        return None

    veh_progress, _ = _route_progress(route_points, vehicle.position)
    scored = []
    for idx, start in candidates:
        prog, dist_to_route = _route_progress(route_points, start)
        scored.append((idx, prog, dist_to_route))

    # ON-PATH GATE, checked before the progress-tier split below: a
    # candidate's own "prog" value is the arc-length of whichever route
    # point happens to be NEAREST it, which is only a meaningful "how far
    # along am I" reading when the candidate is actually close to the
    # route to begin with. On a real multi-lane road (e.g.
    # real_001_rebuilt), a PARALLEL lane's own next fragment can sit a
    # full lane-width off the route (dist_to_route several meters) while
    # still nominally projecting to a LARGER prog than the correct,
    # right-there (dist_to_route ~= 0) fragment -- confirmed directly: a
    # fragment 0.55m away with prog=211.83 (veh_progress=212.00, so just
    # BEHIND -- tier 1) lost to one 5.92m away with prog=215.83 (tier 0),
    # even though the close one was obviously the real next segment of
    # the road already being driven. Picking the far one for one tick
    # produces a real, visible heading snap (confirmed: -260.67 ->
    # -250.67 degrees) as the vehicle jumps onto a parallel lane and back
    # -- exactly the "shakes while turning" symptom. Requiring a
    # candidate's own dist_to_route to be within ON_PATH_EPSILON of
    # whatever the CLOSEST candidate achieves before it's even allowed to
    # win on progress fixes this without touching the ranking for the
    # ordinary (single close candidate, or several genuinely comparable
    # ones) case, where every real candidate already has a small
    # dist_to_route and this gate never excludes anything.
    ON_PATH_EPSILON = 1.0  # meters
    best_dist_to_route = min(d for _, _, d in scored)

    best_idx, best_key = None, None
    for idx, prog, dist_to_route in scored:
        on_path = dist_to_route <= best_dist_to_route + ON_PATH_EPSILON
        # (0, prog) always outranks (1, dist_to_route): any real forward
        # progress beats the fallback, and among progressing candidates the
        # SMALLEST forward step wins (the very next point ahead, not a
        # jump). Among non-progressing (or off-path) candidates, prefer
        # whichever pulls closest back toward the route itself.
        key = (0, prog) if (on_path and prog > veh_progress + 0.5) else (1, dist_to_route)
        if best_key is None or key < best_key:
            best_key, best_idx = key, idx
    return best_idx


# -- the vehicle -------------------------------------------------------------

class LimitedVisionHuman(sb.NoResnapIDMVehicle):
    """A NoResnapIDMVehicle (so it keeps the fragment-hop lane fix, same as
    every background vehicle) whose OWN acceleration/braking decisions --
    computed in apply_human_aware_car_following(), not here -- only ever see
    `visible_candidates()`'s output, never the full nearby-vehicle list.

    enable_fov / enable_occlusion are flags on this ONE class, not separate
    subclasses -- with both False, visible_candidates() is the identity
    function and this drives exactly like an ordinary NoResnapIDMVehicle
    with a route bias (see test_limited_vision_human.py's ablation check).
    One class, flags for ablation, per the Overcooked precedent's own
    documented reason: a feature split into a subclass is a feature a
    forecast/shadow-construction site can silently build without.
    """

    def __init__(self, *args, fov_deg=FOV_DEG_DEFAULT, route_points=None,
                 enable_fov=True, enable_occlusion=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fov_deg = fov_deg
        self.route_points = route_points
        self.enable_fov = enable_fov
        self.enable_occlusion = enable_occlusion

    def visible_candidates(self, candidates):
        """What this vehicle can actually perceive RIGHT NOW: in cone (if
        enabled) AND not occluded by another vehicle's body (if enabled).
        Pure function of the current tick's positions/headings -- nothing
        here is ever stored, so there is no memory to fall back on once
        something leaves this set."""
        out = []
        for c in candidates:
            if self.enable_fov and not in_cone(self.position, self.heading, c.position, self.fov_deg):
                continue
            if self.enable_occlusion:
                blockers = [o for o in candidates if o is not c]
                if is_occluded(self.position, c.position, blockers):
                    continue
            out.append(c)
        return out


def add_human_vehicle(road, route_points, fov_deg=FOV_DEG_DEFAULT, speed=10.0, **kwargs):
    """Spawn a LimitedVisionHuman on the real lane nearest route_points[0],
    facing along that lane -- mirrors add_background_traffic's own
    lane.position()/heading_at() spawn pattern, one-time O(all lanes) cost
    via get_closest_lane_index (fine at spawn; NoResnapIDMVehicle is what
    stops this from being paid again every step)."""
    start = np.asarray(route_points[0], dtype=float)
    lane_index = road.network.get_closest_lane_index(start)
    lane = road.network.get_lane(lane_index)
    longitudinal, _ = lane.local_coordinates(start)
    longitudinal = max(longitudinal, 0.0)
    position = lane.position(longitudinal, 0)
    heading = lane.heading_at(longitudinal)
    human = LimitedVisionHuman(road, position, heading=heading, speed=speed,
                                fov_deg=fov_deg, route_points=route_points, **kwargs)
    road.vehicles.append(human)
    return human


# -- per-step orchestration, entirely additive on top of scene1_background --

def apply_human_aware_car_following(road, lane_indexes, dt, radius=35.0):
    """Runs scene1_background.apply_better_car_following() FIRST, unmodified
    -- every vehicle, including any LimitedVisionHuman on the road, gets
    exactly the same fragment-aware/crossing-conflict-aware action background
    traffic already gets. Then, for each LimitedVisionHuman, OVERWRITES just
    its own acceleration using ONLY visible_candidates() -- strictly additive
    on top of the existing, already-hardened per-step logic; nothing about
    how background traffic is computed changes, and this file never touches
    scene1_background.py itself.

    Call in place of (not in addition to) apply_better_car_following().

    A HUMAN'S FINAL ACCELERATION DELIBERATELY REPLACES, RATHER THAN
    REFINES-VIA-MIN, WHATEVER apply_better_car_following() JUST SET. That
    function's own result also folds in whatever road.act() (stock
    IDMVehicle.act(), which every vehicle including this one runs first as
    part of road.act()) already computed via the UNFILTERED road.
    neighbour_vehicles() -- a same-exact-lane-index check that doesn't go
    through nearby_vehicles()/visible_candidates() at all, so it can never
    be made FOV-aware without reimplementing it too. Taking a min() with
    that value here would let an occluded vehicle the human "can't see"
    still constrain it through this back door, which is exactly the literal
    requirement this module exists to satisfy ("does not sense and know and
    act on anything outside of FOV"). No real coverage is lost: find_front_
    vehicle() is a strict superset of what that stock check finds (same
    lane AND the fragment-continuation near its end, vs. exact-lane-index
    only -- see find_front_vehicle's own docstring in scene1_background.py).
    """
    sb.apply_better_car_following(road, lane_indexes, dt, radius=radius)
    _unstick_frozen_background(road, dt)

    humans = [v for v in road.vehicles if isinstance(v, LimitedVisionHuman) and not v.crashed]
    for h in humans:
        candidates = sb.nearby_vehicles(road, h, radius)
        visible = h.visible_candidates(candidates)

        front = sb.find_front_vehicle(road, h, lane_indexes, visible)
        accel = h.acceleration(ego_vehicle=h, front_vehicle=front, rear_vehicle=None)

        # `heads` for crossing_conflict_brake's tie-break, computed ONLY over
        # what this human can see (a vehicle it can't see can't be reasoned
        # about as a queue-head either) -- deliberately not the same `heads`
        # apply_better_car_following() computed globally above, since that
        # one is built from every vehicle's UNFILTERED perception and using
        # it here would let the human's tie-break priority depend on traffic
        # relationships it cannot itself observe.
        heads = {id(v) for v in visible
                  if sb.find_front_vehicle(road, v, lane_indexes, sb.nearby_vehicles(road, v, radius)) is None}
        if front is None:
            heads.add(id(h))

        conflict = sb.crossing_conflict_brake(h, visible, heads=heads)
        if conflict is not None:
            accel = min(accel, conflict)

        h.action["acceleration"] = max(accel, -h.speed / dt)
        _unstick_if_frozen(road, h, front, visible, dt)

    _resolve_stuck_route_pair(road, dt)


STALL_TIMEOUT_S = 20.0  # how long the human can be genuinely motionless before its own blocker is cleared


def _unstick_if_frozen(road, human, front, visible, dt):
    """crossing_conflict_brake's own tie-break (scene1_background.py) is
    extensively tuned to resolve most head-on/crossing standoffs without
    ever forcing an unsafe release, but its own docstring documents real,
    remaining cases -- two vehicles from different lanes both stopped right
    at a shared merge point -- that it deliberately does NOT force through,
    since forcing it there traded permanent freezes for new collisions in
    testing. That's an acceptable risk to leave open for anonymous
    background traffic; it is not acceptable for the one human this whole
    module exists to study getting stuck behind it forever (see
    advance_vehicles_with_route's own "never strand the human" rule).

    Scoped as narrowly as possible to avoid the exact destabilization
    scene1_background.py's advance_vehicles() docstring already warned
    about for a broader "remove anything stationary" rule: only ever
    removes ONE vehicle -- the human's own immediate front_vehicle if it
    has one, else (measured necessary: a human repeatedly re-triggering
    crossing_conflict_brake against a DIFFERENT nearby vehicle each tick
    has no front_vehicle at all, so was never being cleared by the
    front_vehicle-only version of this function and stalled permanently,
    confirmed by testing out to 400s of sim time with zero progress) the
    single closest other vehicle it can currently see -- and only after the
    human itself has been essentially stationary for STALL_TIMEOUT_S
    straight seconds, long past any ordinary queue wait. This behaves the
    same as a real-world "someone gets out and moves the stalled car"
    outcome rather than an invisible teleport of the human itself.
    """
    human._stalled_ticks = human._stalled_ticks + 1 if human.speed < 0.5 else 0
    if human._stalled_ticks * dt < STALL_TIMEOUT_S:
        return
    blocker = front
    if blocker is None and visible:
        blocker = min(visible, key=lambda v: np.linalg.norm(np.asarray(v.position) - np.asarray(human.position)))
    # A route-following blocker (the robot) can't be despawned -- that case
    # is a genuine human-vs-robot standoff, not "stuck behind background
    # traffic", and is handled separately by _resolve_stuck_route_pair
    # (called once per tick regardless of whether this function fires at
    # all, so it isn't gated on human._stalled_ticks specifically).
    if blocker is not None and getattr(blocker, "route_points", None) is None:
        road.vehicles.remove(blocker)
    human._stalled_ticks = 0


def _resolve_stuck_route_pair(road, dt, retreat=12.0, clear_dist=10.0, retreat_safe_distance=5.0):
    """The route-following (human/robot) counterpart to _unstick_if_frozen's
    background-blocker case: crossing_conflict_brake's tie-break correctly
    decides WHO should go first between two mutually-stopped vehicles (its
    own `_stopped_ticks`/id ordering -- see that function's docstring), but
    its safe_release_dist=10.0 gate can leave even the rightful winner stuck
    forever if the merge point itself puts them closer than that. Observed
    exactly this way in testing: the human's left turn and the robot's
    straight-through movement converge on mega_scene's tw_il2_0 node only
    ~6m apart -- both approaches genuinely end at the same physical point by
    construction (add_three_way/add_four_way build every turn/straight
    option at a corner to land on the same subsequent lane), so a real
    real-world stop-line offset never got built in here -- held motionless
    380+s straight in testing: the tie-break picked a winner correctly
    (confirmed: the human's own _stopped_ticks was consistently higher) but
    that winner still needed dist>=10 to actually be released, which this
    particular convergence geometry never reaches on its own.

    A LOOSENED release distance was tried first (as low as 3.0, scoped only
    to an already-timed-out route-vehicle pair) and rejected: it did get the
    winner moving, but at this geometry "close enough to release" and "close
    enough to actually collide" turned out to be nearly the same number --
    confirmed by testing (a real crash at min_gap=3.72, both vehicles ~5m
    long, converging at a 68 degree relative heading -- their rotated
    rectangles can overlap well before a raw center-to-center distance gets
    anywhere near what crossing_conflict_brake's own docstring found unsafe
    to relax globally). Loosening a proximity-based safety check at exactly
    the range where two vehicle bodies can overlap is not a threshold to
    tune away.

    THIS VERSION NEVER LOOSENS A SAFETY CHECK. It repositions the LOSER
    backward along its own current lane instead -- the same category of
    intervention as _unstick_if_frozen's own "someone gets out and moves the
    stalled car" (a bounded, one-time nudge, not an ongoing exemption) --
    until the EXISTING, already-validated crossing_conflict_brake release
    (safe_release_dist=10.0, completely unmodified) sees real clearance and
    releases the winner through its own normal logic on a later tick. Only
    ever moves a vehicle that's already fully stopped and going nowhere
    anyway (the loser, by definition, is the one NOT released), so there is
    no discontinuity in the winner's own trajectory to reason about, and
    nothing about crossing_conflict_brake or find_front_vehicle changes for
    anyone.

    The retreat distance is capped by `retreat_safe_distance` clearance from
    EVERY other vehicle on the road, not just the winner of this pair --
    confirmed as a real, not theoretical, gap: a blind `retreat` meters
    backward considers only the two vehicles in this standoff, so it can
    walk the loser straight into a completely unrelated third vehicle
    (background traffic, say) that happens to already be sitting at the
    destination point. Measured directly: a route vehicle retreated the
    full 12m landed 2.7m from a stopped background vehicle it had never
    interacted with and crashed into it the very next tick -- an artificial
    collision from this function's own teleport, not from any ordinary
    driving decision. Tries the full requested retreat first, then less, in
    0.5m steps (fine enough that a clear gap narrower than 1m can't be
    stepped over and missed entirely), down to no movement at all if
    nothing along that stretch is clear -- never moving is always at least
    as safe as the vehicle's own current, already-non-crashed position.

    retreat_safe_distance=5.0 (~one vehicle length, comfortably more than
    the 2.7m gap that produced the confirmed crash above) rather than
    something closer to crossing_conflict_brake's own safe_release_dist=10.0
    -- that larger a bar was tried first and made things WORSE, not just
    redundant: in the dense multi-robot/background traffic this module is
    meant to run under, a full 10m of clearance from EVERY nearby vehicle is
    often simply unavailable anywhere within a 12m retreat, so the search
    fell back to "don't move" almost every time -- confirmed directly: a
    scene that reached 100% route progress with the plain (unsafe) retreat
    dropped to stalling around 25% once retreat_safe_distance=8.0 made the
    search fail this often. 5.0 is the smallest value that still
    categorically prevents the specific failure this exists to fix (two
    vehicle bodies overlapping) without also defeating the retreat's own
    purpose.

    If the DESIGNATED loser still has nowhere safe to go (a genuinely
    saturated local cluster -- confirmed directly: a human and two robots
    all mutually within a few meters, each stopped 300+ seconds straight,
    the assigned loser's own 12m stretch never once clear), this falls back
    to trying the WINNER instead. The winner is, by definition, also
    currently stopped and non-crashed here (both members of a pair must be
    to reach this point at all), so nudging it back is exactly as safe a
    move as nudging the loser -- it just means whichever of the two
    actually has room to move is the one that does, rather than the tie-
    break's own priority silently deciding "neither ever moves" purely
    because its designated pick happened to have the more crowded side.

    ALSO retreats the winner (in addition to the loser, not instead of)
    once the winner's own _stopped_ticks passes EXTREME_STALL_S. The
    winner/loser choice below is inherited from crossing_conflict_brake's
    own fairness convention -- whoever's waited LONGER wins, exactly like a
    real 4-way-stop -- which is the right call for a BRIEF mutual standoff,
    but stops making sense once one side has been stopped for MINUTES: that
    length of stall is itself evidence the "winner" was never actually
    waiting on THIS pair's own clearance in the first place, it's stuck on
    something else entirely, and retreating only the loser forever is
    futile -- confirmed directly: a robot stopped 3235+ ticks (215+
    seconds) straight, completely unmoving, was chosen as "winner" every
    single cycle purely because the human's own count kept resetting
    (normal IDM behavior: it re-approaches, stops at a safe following
    distance behind the still-frozen robot, gets retreated again 20s later,
    repeat) -- 20+ cycles of retreating the human accomplished nothing
    because the robot was never going to move regardless.
    """
    EXTREME_STALL_S = 3 * STALL_TIMEOUT_S

    def retreat_if_safe(vehicle):
        lon, lat = vehicle.lane.local_coordinates(vehicle.position)
        others = [np.asarray(v.position) for v in road.vehicles if v is not vehicle]
        for step_back in np.arange(retreat, 0.5 - 1e-9, -0.5):
            candidate_lon = max(0.0, lon - step_back)
            candidate_pos = vehicle.lane.position(candidate_lon, lat)
            if all(np.linalg.norm(candidate_pos - p) >= retreat_safe_distance for p in others):
                vehicle.position = candidate_pos
                return True
        return False

    actors = [v for v in road.vehicles if getattr(v, "route_points", None) is not None and not v.crashed]
    for i, a in enumerate(actors):
        for b in actors[i + 1:]:
            if a.speed >= 1.0 or b.speed >= 1.0:
                continue
            a_ticks, b_ticks = getattr(a, "_stopped_ticks", 0), getattr(b, "_stopped_ticks", 0)
            if a_ticks * dt < STALL_TIMEOUT_S or b_ticks * dt < STALL_TIMEOUT_S:
                continue
            if np.linalg.norm(np.asarray(a.position) - np.asarray(b.position)) >= clear_dist:
                continue  # already clear -- not what's blocking the existing release logic

            winner, loser = (a, b) if (a_ticks > b_ticks or (a_ticks == b_ticks and id(a) > id(b))) else (b, a)
            winner_ticks = a_ticks if winner is a else b_ticks
            moved_loser = retreat_if_safe(loser)
            if not moved_loser or winner_ticks * dt >= EXTREME_STALL_S:
                retreat_if_safe(winner)


BACKGROUND_STALL_TIMEOUT_S = 25.0  # longer than STALL_TIMEOUT_S -- the human's own unstick gets first say


def _unstick_frozen_background(road, dt, timeout_s=BACKGROUND_STALL_TIMEOUT_S):
    """Background-traffic analogue of _unstick_if_frozen -- same idea, wider
    net. crossing_conflict_brake's tie-break (scene1_background.py)
    resolves nearly every standoff, but its own docstring documents a real
    remaining case it deliberately leaves open for anonymous traffic: two
    vehicles from different lanes/rings both stopped right at a shared
    merge point, e.g. mega_scene's roundabout entries once
    route_adjacent_lane_indexes started biasing spawn density onto the
    route the human's own path takes THROUGH the roundabout -- exactly
    where this was actually observed freezing.

    _unstick_if_frozen already prevents this from stranding the human
    itself, but does nothing for a standoff between two background vehicles
    that never touches the human at all. This generalizes the same
    signal -- `_stopped_ticks`, already maintained by
    apply_better_car_following for crossing_conflict_brake's own tie-break,
    incremented only while GENUINELY stopped (speed < 1.0) and reset the
    instant a vehicle moves above that even briefly -- so ordinary
    stop-and-go congestion (which keeps creeping forward) never accumulates
    this; only a vehicle that hasn't moved AT ALL for a long time can. On
    this traffic-light-free map, a vehicle whose front gap ever opened even
    slightly would show some nonzero IDM creep and reset its own counter,
    so `timeout_s` straight seconds of exact zero movement is strong
    evidence of a genuine permanent deadlock, not a long but ordinary queue
    wait -- same reasoning as _unstick_if_frozen's own STALL_TIMEOUT_S, just
    applied network-wide instead of only to the human's immediate blocker.

    Never removes a route-following vehicle (route_points is not None --
    the human or the robot): only anonymous background traffic, exactly
    like _unstick_if_frozen's own guard against removing one of those.

    Iterates a SNAPSHOT (list(road.vehicles)), not road.vehicles itself --
    removing from a list while a plain `for` loop walks it shifts every
    later element back one slot, which the loop's own advancing index then
    steps past, silently skipping whatever just shifted into the removed
    slot. Confirmed as a real, not theoretical, failure: with two or more
    long-stopped vehicles adjacent in road.vehicles (exactly a genuine
    multi-vehicle jam at a busy junction -- the case this function exists
    to break), the one right after each removed vehicle could be skipped
    on every single call, forever, since each call started a fresh
    iteration that reproduced the same skip. Measured directly: a single
    seeded vehicle's own _stopped_ticks climbed past 4600 (over 300
    seconds, 12x the 25s timeout) while parked directly in a turn span's
    own conflict lane, never once removed, permanently blocking that
    maneuver for the rest of a whole test run.
    """
    for v in list(road.vehicles):
        if getattr(v, "route_points", None) is not None or v.crashed:
            continue
        if getattr(v, "_stopped_ticks", 0) * dt >= timeout_s:
            road.vehicles.remove(v)


def advance_vehicles_with_route(road, lane_indexes):
    """A copy of scene1_background.advance_vehicles(), byte-for-byte
    identical in structure, except the fork-choice at the end of a fragment
    uses route_aware_continuation() (falls back to plain find_continuation()
    for any vehicle without a route_points attribute, i.e. every background
    vehicle) instead of calling find_continuation() directly. Kept as a full
    copy rather than a shared helper so scene1_background.py needs no
    changes for this feature.

    Call in place of (not in addition to) advance_vehicles().

    A ROUTE-FOLLOWING VEHICLE (route_points is not None -- both the human
    AND a robot: play.py/interface.py/evaluate.py all set robot.route_points
    directly to opt a robot into route_aware_continuation too, so this is
    NOT the same set as "LimitedVisionHuman instances") NEVER GETS
    DESPAWNED AT A DEAD END, unlike background traffic. Background despawn-
    at-a-genuine-dead-end is fine for anonymous traffic (scene1_background.
    py's own advance_vehicles() docstring says as much), but silently
    deleting the human this whole feature exists to study -- or a robot a
    live participant can see driving around -- is a much worse failure than
    a vehicle that's temporarily off the map edge. When no continuation is
    found, it simply keeps its current lane_index and retries next step --
    combined with route_aware_continuation's fallback pulling back toward
    the route (see that function's docstring), this was sufficient in
    testing to recover rather than genuinely strand.

    A CRASHED BACKGROUND VEHICLE (crashed, not a LimitedVisionHuman -- see
    the isinstance check below, not a route_points check) IS ALWAYS
    REMOVED, unlike scene1_background.advance_vehicles() which leaves it in
    place. That's fine on real_001's own multi-lane-equivalent fragment
    graph (a wreck rarely owns the only path forward), but a route-following
    human here can be confined to a single lane through a junction (see
    mega_scene.py's _prune_to_route -- needed to make its own fork choice
    unambiguous), where a permanently-stopped wreck ahead would otherwise
    trap the human behind it forever, contradicting the "never despawn/never
    strand the human" guarantee above just as badly as an outright despawn
    would. A crashed vehicle can't be re-collided with (it's already
    crashed), so removing it costs nothing but a moment of "the wreck
    disappeared" visually.
    """
    survivors = []
    for v in road.vehicles:
        # Only an actual LimitedVisionHuman is protected from crash-removal
        # (see this function's own docstring: never despawn/strand the
        # human this whole module exists to study). A ROBOT also carries
        # route_points is not None (interface.py/play.py set it directly to
        # opt into route_aware_continuation), but is not a LimitedVisionHuman
        # instance, so a crashed one is removed exactly like crashed
        # background traffic -- confirmed as a genuine, previously-open gap:
        # a crashed robot sitting in the human's own lane was never removed
        # by ANY existing mechanism (this check kept it forever regardless
        # of route_points; _unstick_frozen_background and _unstick_if_frozen
        # both deliberately never touch a route vehicle; _resolve_stuck_
        # route_pair explicitly excludes crashed vehicles from its own
        # actors list) -- so the human was left permanently, unrecoverably
        # blocked, confirmed directly: progress froze solid at the exact
        # tick a robot crashed one lane-length ahead of it and never moved
        # again for the rest of a 20000-step run.
        if v.crashed and not isinstance(v, LimitedVisionHuman):
            continue
        if sb.off_road(v):
            route_points = getattr(v, "route_points", None)
            next_idx = route_aware_continuation(road, lane_indexes, v, route_points)
            if next_idx is None:
                if route_points is not None:
                    survivors.append(v)  # never despawn a route-following human
                    continue
                continue  # background traffic: genuine dead end, despawn
            v.lane_index = next_idx
            v.lane = road.network.get_lane(next_idx)
            v.target_lane_index = next_idx
        survivors.append(v)
    road.vehicles = survivors
