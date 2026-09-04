"""Classifies a layout's own HUMAN_ROUTE/ROBOT_ROUTE lane list (the
_HUMAN_ROUTE_LANES-style (from, to, lane_id) sequence every
layout/layouts/real_NNN_rebuilt.py already builds, to construct its own
public HUMAN_ROUTE/ROBOT_ROUTE point polylines) into discrete
drive-by-subtask segments -- forward, turn, lane_change, merge_in (entering
a roundabout), merge_out (leaving one) -- each with a [start, end) progress
range along the route's own polyline.

NO NEW PER-LAYOUT METADATA: this is derived entirely from the SAME
lane-index list every real_NNN_rebuilt.py already exposes (as
_HUMAN_ROUTE_LANES / _ROBOT_ROUTE_LANES -- a leading underscore is a
convention, not enforced privacy; see route_lanes_for() below) plus
build_scene.py's own consistent node-naming convention, verified directly
against real_001_rebuilt.py's own docstring and code:
  - a junction turn:            {prefix}ir{c}_{i}  ->  {prefix}il{c'}_{i},
                                 AND the lane object is a CircularLane --
                                 add_four_way/add_three_way use this exact
                                 same ir->il node pattern for the STRAIGHT-
                                 THROUGH movement too (StraightLane, opp_c
                                 instead of prev_c/next_c), so the node
                                 names alone don't distinguish an actual
                                 turn from driving straight across a box;
                                 only the lane's own type does (verified
                                 directly: real_001_rebuilt.py's own first
                                 "ir0_0 -> il2_0" is the straight-through
                                 movement at its 4-way, and its lane came
                                 back as a StraightLane, not a
                                 CircularLane, before this check was added).
  - merging ONTO a roundabout:  {prefix}farin{k}_{i} -> bendin{k}_{i}
                                 {prefix}bendin{k}_{i} -> entry{k}_{i}
  - driving the ring itself:    {prefix}entry{k}_{i}  -> exit{k'}_{i}
  - merging OFF a roundabout:   {prefix}exit{k}_{i}  -> bendout{k}_{i}
                                 {prefix}bendout{k}_{i} -> farout{k}_{i}
  - an explicit lane change:    any edge whose own lane object is a
                                 SineLane (_layout_utils.lane_change_to/
                                 lane_change's own signature) -- round_about
                                 never emits one, and a route never touches
                                 merge()'s own on-ramp SineLane (the route
                                 only ever rides the main lanes through a
                                 merge zone, per real_001_rebuilt.py's own
                                 docstring), so within a HUMAN_ROUTE/
                                 ROBOT_ROUTE lane list this is unambiguous.
Anything else (an ordinary StraightLane/CircularLane bridge or approach, or
an ir->il straight-through movement) is "forward".
"""
import importlib
import re

import numpy as np

from highway_env.road.lane import CircularLane, SineLane
from scene1_background import NoResnapIDMVehicle

_TURN_FROM = re.compile(r"_ir\d+_\d+$")
_TURN_TO = re.compile(r"_il\d+_\d+$")

DISPLAY_NAME = {
    "forward": "go forward",
    "turn": "turn",
    "lane_change": "change lane",
    "merge_in": "merge into roundabout",
    "merge_out": "exit roundabout",
}


def classify_edge(net, lane_index):
    """One of "lane_change", "turn", "merge_in", "merge_out", "ring", or
    "forward" for a single (from, to, i) lane edge. "ring" (driving the
    roundabout's own arc between an entry and exit point) is a distinct
    return value from "forward" here so callers that care can tell them
    apart, but segment_route() below folds it into "forward" -- there is
    no separate subtask for it in the interface's own list (go forward,
    change lanes, merge in, exit, turn, wait), it's simply more forward
    driving that happens to be curved."""
    frm, to, _ = lane_index
    lane = net.get_lane(lane_index)
    if isinstance(lane, SineLane):
        return "lane_change"
    if isinstance(lane, CircularLane) and _TURN_FROM.search(frm) and _TURN_TO.search(to):
        return "turn"
    if (("_farin" in frm) or ("_bendin" in frm)) and (("_bendin" in to) or ("_entry" in to)):
        return "merge_in"
    if "_entry" in frm and "_exit" in to:
        return "ring"
    if (("_exit" in frm) or ("_bendout" in frm)) and (("_bendout" in to) or ("_farout" in to)):
        return "merge_out"
    return "forward"


def segment_route(net, route_lanes):
    """[(kind, start_progress, end_progress, lane_indexes), ...] in route
    order. Consecutive edges of the same kind (with "ring" folded into
    "forward" -- see classify_edge's own docstring) are merged into one
    subtask span, so a multi-edge maneuver (e.g. merge_in's own
    farin->bendin->entry pair) or a long straight stretch made of several
    short bridge edges is one decision point, not one per lane edge.

    Progress is cumulative lane length along `route_lanes`, in the SAME
    units and origin as limit_vision_human._route_progress(route_points,
    ...) reports for a point exactly on the route -- route_points is built
    by _layout_utils.polyline() sampling these exact same lanes, in this
    exact same order, from lon=0 to lon=lane.length, so summing lane
    lengths here reproduces that parameterization exactly rather than
    resampling and risking drift.
    """
    raw_kinds = [classify_edge(net, idx) for idx in route_lanes]
    kinds = ["forward" if k == "ring" else k for k in raw_kinds]

    starts = []
    acc = 0.0
    for idx in route_lanes:
        starts.append(acc)
        acc += net.get_lane(idx).length

    spans = []
    i = 0
    n = len(route_lanes)
    while i < n:
        kind = kinds[i]
        j = i
        while j + 1 < n and kinds[j + 1] == kind:
            j += 1
        end_progress = starts[j] + net.get_lane(route_lanes[j]).length
        spans.append((kind, starts[i], end_progress, tuple(route_lanes[i:j + 1])))
        i = j + 1
    return spans


def route_lanes_for(module, which):
    """The private _HUMAN_ROUTE_LANES / _ROBOT_ROUTE_LANES list a
    layout/layouts/real_NNN_rebuilt.py module builds internally to
    construct its own public HUMAN_ROUTE/ROBOT_ROUTE polyline -- a leading
    underscore is a naming convention in Python, not enforced privacy, and
    every layout in this batch follows the identical private-list-then-
    polyline pattern (verified directly against real_001_rebuilt.py), so
    this is a stable, if informally-named, part of each layout's own
    interface. `which` is "human" or "robot"."""
    name = f"_{which.upper()}_ROUTE_LANES"
    if not hasattr(module, name):
        raise AttributeError(
            f"{module.__name__} has no {name} -- subtask segmentation needs the private "
            f"lane-index list every real_NNN_rebuilt.py layout builds, not just the public "
            f"point polyline")
    return getattr(module, name)


def build_subtasks(module, which="human"):
    """segment_route() for a layout module's own human or robot route,
    rebuilding a fresh network via module.build_road() (never the shared
    one a live sim is stepping)."""
    net = module.build_road().network
    route_lanes = route_lanes_for(module, which)
    return segment_route(net, route_lanes)


def seed_maneuver_traffic(road, subtasks_list, seed=0, speed_range=(1.2, 2.5), n_per_lane=2, safe_distance=10.0,
                           max_tries=10):
    """Spawn dedicated IDMVehicle(s) directly onto EACH non-forward span's
    own upcoming lane(s) -- turn/lane_change/merge_in/merge_out -- partway
    along it, at a modest speed. Call AFTER human/robots are already in
    road.vehicles and BEFORE scene1_background.add_background_traffic, so
    (a) this can skip a spot too close to an already-placed vehicle, and
    (b) add_background_traffic's own placement check (which reads
    `road.vehicles` fresh) naturally avoids stacking random traffic on top
    of what this just placed.

    WHY THIS EXISTS: ApproximateLimitVisionHuman.gated()'s own
    _lane_conflict check only ever looks at vehicles on THESE specific
    lanes (the maneuver's own upcoming lane(s), not a blanket radius --
    see that method's docstring). add_background_traffic spawns uniformly
    at random over route_adjacent_lane_indexes(), which is dozens of lanes
    covering the WHOLE route; any one maneuver's own lane is typically
    15-35m out of that, so a uniformly-random spawn essentially never
    lands there and stays long enough to matter. Measured directly: a
    full 400s run with 35 background vehicles + 3 robots produced ZERO
    _lane_conflict detections across 294 ticks spent in turn/merge_in/
    merge_out spans combined (the only detections that did occur were
    during a single lane_change episode, and that one didn't even
    discriminate between FOV widths) -- meaning turn/merge availability
    was NEVER actually FOV-gated in that run, regardless of the human's
    true FOV: exactly the "no situation where fov matters, posterior is
    terrible" failure this fixes.

    Deliberately separate from add_background_traffic (which stays a
    general-purpose, uniformly-random traffic generator): this takes the
    already-computed subtask spans directly instead of re-deriving
    anything, and is specific to making THIS human's own maneuvers
    genuinely testable rather than changing traffic generation broadly.

    speed_range=(1.2, 2.5) (slow, not the earlier (3.0, 7.0)) and
    n_per_lane=2 (not 1): a live participant takes real human reaction/
    decision time to notice a button and click it, unlike a scripted test
    that authorizes the instant something becomes legal. At the old
    speed, a seeded vehicle crosses a ~30m danger zone in about 5-6
    seconds -- reliable for an automated test polling every tick, but a
    real player who takes even 10-15 seconds to look and click has a real
    chance of the zone having already cleared (or not yet been reached)
    by the time they act, so the maneuver looks FOV-independent purely by
    bad timing, not because FOV genuinely doesn't matter. At the new
    speed a single vehicle lingers for roughly 13-27 seconds, and two
    staggered ones (independently placed along the lane) mean at least
    one is very likely still there across a realistic range of human
    response times.

    The floor of 1.2 (not slower still, which would linger even longer)
    is deliberate: limit_vision_human._unstick_frozen_background retreats
    (see its own docstring -- no longer removes) ANY non-route vehicle
    once its speed has read below 1.0 for 25 straight seconds, network-
    wide, regardless of why -- a seeded vehicle genuinely cruising at,
    say, 0.8 would eventually get nudged back along its own lane by that
    same mechanism before a slow-to-react player ever reached it, still
    present but no longer exactly where it was placed. Staying safely
    above that 1.0 cutoff means this vehicle is never mistaken for a
    stalled one in the first place.
    """
    rng = np.random.default_rng(seed)
    spawned = []
    for kind, _, _, lane_indexes in subtasks_list:
        if kind == "forward":
            continue
        for lane_idx in lane_indexes:
            lane = road.network.get_lane(lane_idx)
            for _ in range(n_per_lane):
                placed = [v.position for v in road.vehicles]
                for _ in range(max_tries):
                    longitudinal = rng.uniform(0.15, 0.85) * lane.length
                    position = lane.position(longitudinal, 0)
                    if all(np.linalg.norm(position - p) >= safe_distance for p in placed):
                        break
                else:
                    continue  # no safe spot on this lane -- skip, don't spawn into a collision
                vehicle = NoResnapIDMVehicle(road, position, heading=lane.heading_at(longitudinal),
                                              speed=rng.uniform(*speed_range))
                road.vehicles.append(vehicle)
                spawned.append(vehicle)
    return spawned
