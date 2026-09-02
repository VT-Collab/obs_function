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

FOV_DEG_DEFAULT = 120.0


# -- geometry: cone + occlusion, both pure functions of current state only --

def in_cone(ego_position, ego_heading, target_position, fov_deg):
    """True iff target_position lies within fov_deg of ego_heading, as seen
    from ego_position. fov_deg >= 360 is vacuously true (matches the
    Overcooked precedent's stance: no separate radius/omniscience baked in
    here either -- occlusion still applies at 360, see is_occluded)."""
    if fov_deg >= 360:
        return True
    delta = np.asarray(target_position, dtype=float) - np.asarray(ego_position, dtype=float)
    dist = np.linalg.norm(delta)
    if dist < 1e-6:
        return True
    heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    cos_angle = np.dot(delta, heading_vec) / dist
    # small epsilon: cos(radians(90)) is ~6e-17, not exactly 0, so an exact
    # boundary case (e.g. fov=180, target exactly perpendicular) would
    # otherwise fail this comparison by pure floating-point noise.
    return cos_angle >= np.cos(np.radians(fov_deg / 2.0)) - 1e-9


def segment_intersects_rotated_rect(a, b, center, length, width, angle):
    """True iff the line segment a->b crosses the rectangle centered at
    `center` (dimensions length x width, rotated by `angle` radians, same
    rotation convention as highway_env.utils.point_in_rotated_rectangle).

    Exact segment-vs-axis-aligned-box test (Liang-Barsky clipping) done in
    the rectangle's own unrotated frame, rather than reusing
    highway_env.utils.rotated_rectangles_intersect via a near-zero-width
    degenerate rectangle: that function is corner-sampling (does any corner
    of one rectangle land inside the other), which is a reasonable
    approximation for two comparably-sized rectangles (what it's built for
    elsewhere in highway-env) but unreliable for a thin ray THROUGH the
    middle of a blocker -- exactly the common "car hidden behind another
    car" case this module exists to detect, where neither rectangle's
    corners land inside the other at all.
    """
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    center = np.asarray(center, dtype=float)
    la = rot @ (np.asarray(a, dtype=float) - center)
    lb = rot @ (np.asarray(b, dtype=float) - center)
    dx, dy = lb[0] - la[0], lb[1] - la[1]
    half_l, half_w = length / 2.0, width / 2.0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, la[0] - (-half_l)), (dx, half_l - la[0]),
                 (-dy, la[1] - (-half_w)), (dy, half_w - la[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False  # parallel to this edge and outside it
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def is_occluded(ego_position, target_position, blockers):
    """True iff any vehicle in `blockers` sits on the straight line between
    ego and target. Crashed/stationary vehicles are included "for free" --
    blockers is whatever the caller passes (typically every other nearby
    vehicle, crashed or not), and wreckage should still occlude."""
    for other in blockers:
        length = getattr(other, "LENGTH", 5.0)
        width = getattr(other, "WIDTH", 2.0)
        if segment_intersects_rotated_rect(ego_position, target_position,
                                            other.position, length, width, other.heading):
            return True
    return False


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
    best_idx, best_key = None, None
    for idx, start in candidates:
        prog, dist_to_route = _route_progress(route_points, start)
        # (0, prog) always outranks (1, dist_to_route): any real forward
        # progress beats the fallback, and among progressing candidates the
        # SMALLEST forward step wins (the very next point ahead, not a
        # jump). Among non-progressing candidates, prefer whichever pulls
        # closest back toward the route itself.
        key = (0, prog) if prog > veh_progress + 0.5 else (1, dist_to_route)
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


def advance_vehicles_with_route(road, lane_indexes):
    """A copy of scene1_background.advance_vehicles(), byte-for-byte
    identical in structure, except the fork-choice at the end of a fragment
    uses route_aware_continuation() (falls back to plain find_continuation()
    for any vehicle without a route_points attribute, i.e. every background
    vehicle) instead of calling find_continuation() directly. Kept as a full
    copy rather than a shared helper so scene1_background.py needs no
    changes for this feature.

    Call in place of (not in addition to) advance_vehicles().

    A ROUTE-FOLLOWING VEHICLE (route_points is not None, i.e. any
    LimitedVisionHuman) IS NEVER DESPAWNED, unlike background traffic.
    Background despawn-at-a-genuine-dead-end is fine for anonymous traffic
    (scene1_background.py's own advance_vehicles() docstring says as much),
    but silently deleting the one human this whole feature exists to study
    is a much worse failure than a vehicle that's temporarily off the map
    edge -- there is no plausible use of this module where "the human
    vanished partway through" is an acceptable outcome. When no
    continuation is found, it simply keeps its current lane_index and
    retries next step (cheap: this only runs for the human, not the whole
    background population) -- combined with route_aware_continuation's
    fallback pulling back toward the route (see that function's docstring),
    this was sufficient in testing to recover rather than genuinely strand.

    A CRASHED BACKGROUND VEHICLE (crashed, route_points is None) IS ALWAYS
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
        if v.crashed and getattr(v, "route_points", None) is None:
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
