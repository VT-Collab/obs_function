"""Mega scene: one connected drive spanning every build_scene.py primitive --
a 4-way intersection, a T-junction, a 90-degree turn, a roundabout, and a
highway merge, chained end to end -- built entirely from
highway_env/layout/build_scene.py's own functions (add_four_way,
add_three_way, turn_corner, round_about, merge, connect_junctions), not
hand-rolled lane geometry.

Different build_scene functions space their lanes differently through a
turn (e.g. add_three_way's per-lane turn radius grows with lane index,
turn_corner's shared-circle radius shrinks with it), so there's no single
rigid transform that keeps every lane of a multi-lane junction aligned with
every lane of a different-shaped one across a bridge. Rather than force
that, this reduces to one lane for the turn_corner/round_about/merge
stretch (a country road narrowing outside the built-up junction, which is
realistic anyway) and bridges every junction pair with an explicit
StraightLane between their own actual (queried, not hand-computed) endpoint
positions -- exactly how the old hand-built district.py/mega_scene.py did
their own bridging, just computed from build_scene's returned network
instead of a bespoke "tips" dict.

HUMAN_ROUTE drives the whole thing: spawns south of the 4-way, right turn
onto the bridge to the T-junction, left turn onto the T's north (stem-to-
crossbar) arm, through the turn, halfway around the roundabout, and out
onto the highway through the merge zone.

ROBOT_ROUTE drops in from the T-junction's south arm and takes the exact
same straight-through movement onto that same north-arm node the human's
own turn lands on -- from there the two routes are identical: turn, half
the roundabout, and the whole merge stretch, shared start to finish.

HUMAN_ROUTE/ROBOT_ROUTE are exposed as plain (x, y) world-point polylines
(sampled every ~3m along the lane-index path below), not (from, to,
lane_id) tuples: highway_env/human/limit_vision_human.py's add_human_vehicle
and route_aware_continuation -- built for the hand-clicked/matched-to-road
real_*.py scenes -- only understand that format (they use route progress
along a dense point polyline to pick the correct fork at a junction, not
graph traversal), and display_all.py's own route_polyline() renders points
just as well as lane tuples, so this format works everywhere the tuple form
did without changing anything outside this file.
"""
import numpy as np

from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork

from build_scene import LANE_WIDTH, add_four_way, add_three_way, connect_junctions, merge, round_about, turn_corner

C = LineType.CONTINUOUS

FW_CENTER = (0.0, 0.0)
TW_CENTER = (220.0, 0.0)
TC_ACCESS = 60.0
RA_RADIUS, RA_ACCESS, RA_ALPHA, RA_MERGE_RADIUS = 20.0, 40.0, 24.0, 15.0
BRIDGE_GAP = 30.0

# Which ring gap (k, base_deg = -90*k) the turn_corner feeds into, and which
# one the drive exits from -- fixed by the geometry below (verified once,
# not re-derived per import): the turn_corner's outgoing heading always
# points the same way relative to these fixed centers.
RA_IN_K = 3
RA_OUT_K = 1
RA_EXIT_K = (RA_IN_K + 1) % 4  # the single ring gap passed through on the way from RA_IN_K to RA_OUT_K


def _extend_center(far_point, heading_rad, lane_j, access_length, od, gap):
    """Invert the `far = center + rot(heading) @ (turn_lat_j, access_length +
    od)` convention shared by turn_corner's in_j and add_four_way/
    add_three_way's o{corner}_j, to find the `center` that places that same
    lane's far point exactly `gap` meters further along `heading_rad` from
    a known point."""
    turn_lat = LANE_WIDTH / 2 + lane_j * LANE_WIDTH
    rot = np.array([[np.cos(heading_rad), -np.sin(heading_rad)], [np.sin(heading_rad), np.cos(heading_rad)]])
    new_far = np.array(far_point) + gap * np.array([np.cos(heading_rad), np.sin(heading_rad)])
    return new_far - rot @ np.array([turn_lat, access_length + od])


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    right_turn_radius_base = LANE_WIDTH + 5

    # Single lane everywhere: route_aware_continuation (highway_env/human/
    # limit_vision_human.py) picks a junction's exit fork by nearest point on
    # a dense route polyline, not by lane_index -- a second, parallel lane
    # only ~4m away (well inside its max_dist=8.0 gate) is genuinely
    # ambiguous to that logic and was observed sending the human onto the
    # wrong lane at both junctions. The route only ever used lane 0 anyway.
    add_four_way(net, center=FW_CENTER, n_vertical=1, n_horizontal=1, access_length=60.0, prefix="fw_")

    add_three_way(net, center=TW_CENTER, n_stem=1, n_cross=1, access_length=60.0, prefix="tw_", missing_corner=3)
    connect_junctions(net, "fw_", FW_CENTER, 60.0, 3, "tw_", TW_CENTER, 60.0, 1, 1)

    # tw_'s own north exit (corner 2), lane 0 -- narrows to one lane here,
    # into turn_corner.
    tw_exit_lane = net.get_lane(("tw_il2_0", "tw_o2_0", 0))
    tw_exit_far = tw_exit_lane.position(tw_exit_lane.length, 0)
    tw_exit_heading = tw_exit_lane.heading_at(tw_exit_lane.length)

    tc_od = right_turn_radius_base + LANE_WIDTH / 2  # n_lanes=1 -> base == right_turn_radius_base
    tc_center = _extend_center(tw_exit_far, tw_exit_heading, 0, TC_ACCESS, tc_od, gap=BRIDGE_GAP)
    turn_corner(net, center=tc_center, heading_deg=np.degrees(tw_exit_heading), n_lanes=1,
                access_length=TC_ACCESS, prefix="tc_")
    tc_in_pos = net.get_lane(("tc_in_0", "tc_mid_0", 0)).position(0, 0)
    net.add_lane("tw_o2_0", "tc_in_0", StraightLane(tw_exit_far, tc_in_pos, line_types=[C, C]))

    # turn_corner's own out_0 far endpoint continues into round_about.
    tc_out_lane = net.get_lane(("tc_mid2_0", "tc_out_0", 0))
    tc_out_far = tc_out_lane.position(tc_out_lane.length, 0)
    tc_out_heading = tc_out_lane.heading_at(tc_out_lane.length)

    ra_center = np.array(tc_out_far) + (RA_ACCESS + RA_RADIUS + BRIDGE_GAP) * np.array(
        [np.cos(tc_out_heading), np.sin(tc_out_heading)])
    round_about(net, center=ra_center, radius=RA_RADIUS, n_lanes=1, access_length=RA_ACCESS,
                alpha=RA_ALPHA, merge_radius=RA_MERGE_RADIUS, prefix="ra_")
    ra_in_far = net.get_lane((f"ra_farin{RA_IN_K}_0", f"ra_bendin{RA_IN_K}_0", 0)).position(0, 0)
    net.add_lane("tc_out_0", f"ra_farin{RA_IN_K}_0", StraightLane(tc_out_far, ra_in_far, line_types=[C, C]))

    ra_out_lane = net.get_lane((f"ra_bendout{RA_OUT_K}_0", f"ra_farout{RA_OUT_K}_0", 0))
    ra_out_far = ra_out_lane.position(ra_out_lane.length, 0)
    ra_out_heading = ra_out_lane.heading_at(ra_out_lane.length)

    mg_center = np.array(ra_out_far) + BRIDGE_GAP * np.array([np.cos(ra_out_heading), np.sin(ra_out_heading)])
    merge(net, center=mg_center, heading_deg=np.degrees(ra_out_heading), n_lanes=1,
          before_length=50.0, taper_length=40.0, merge_length=40.0, after_length=60.0, ramp_side=1, prefix="mg_")
    net.add_lane(f"ra_farout{RA_OUT_K}_0", "mg_a_0", StraightLane(ra_out_far, mg_center, line_types=[C, C]))

    _prune_to_route(net, _HUMAN_ROUTE_LANES, _ROBOT_ROUTE_LANES)
    return net


def _prune_to_route(net, *lane_routes):
    """Remove every outgoing edge from a route's own fork nodes except the
    one the route actually takes.

    Needed because highway_env/human/limit_vision_human.py's
    route_aware_continuation ranks fork candidates by how far each
    candidate's OWN start point sits along the route polyline -- but
    add_four_way/add_three_way/round_about's turn/exit options at one
    junction all literally start at the same shared node (the reference
    IntersectionEnv/RoundaboutEnv's own convention: one stop-line node feeds
    right/left/straight, one ring-gap node feeds both "continue" and
    "leave"), so every alternative at a real fork ties EXACTLY with the
    intended one and the tie-break silently keeps whichever happened to be
    enumerated first -- observed sending the human the wrong way at both the
    3-way's left/right fork and the roundabout's continue/leave fork. Real
    map fragments (real_*.py) don't share start points this way, which is
    why this only shows up on a build_scene-built network. Only prunes the
    specific nodes the routes pass through; every other arm/movement (background
    traffic's own turns elsewhere) is untouched."""
    keep = {}
    for route in lane_routes:
        for f, t, _i in route:
            keep.setdefault(f, set()).add(t)
    for f, tos in keep.items():
        for t in list(net.graph.get(f, {})):
            if t not in tos:
                del net.graph[f][t]


def build_road() -> Road:
    return Road(network=_build_network())


def _polyline(net, lane_route, step=3.0):
    """World-space points tracing `lane_route` (a list of (from, to,
    lane_id) lane indices), sampled every `step` meters along each lane's
    own centerline -- same approach as display_all.py's route_polyline(),
    just local to this file rather than importing a viewer utility from a
    layout module."""
    points = []
    for lane_index in lane_route:
        lane = net.get_lane(lane_index)
        for lon in np.linspace(0, lane.length, max(2, int(lane.length / step))):
            points.append(tuple(lane.position(lon, 0)))
    return points


_SHARED_TAIL = [
    ("tw_o2_0", "tc_in_0", 0),
    ("tc_in_0", "tc_mid_0", 0), ("tc_mid_0", "tc_mid2_0", 0), ("tc_mid2_0", "tc_out_0", 0),
    ("tc_out_0", f"ra_farin{RA_IN_K}_0", 0),
    (f"ra_farin{RA_IN_K}_0", f"ra_bendin{RA_IN_K}_0", 0), (f"ra_bendin{RA_IN_K}_0", f"ra_entry{RA_IN_K}_0", 0),
    (f"ra_entry{RA_IN_K}_0", f"ra_exit{RA_EXIT_K}_0", 0), (f"ra_exit{RA_EXIT_K}_0", f"ra_entry{RA_EXIT_K}_0", 0),
    (f"ra_entry{RA_EXIT_K}_0", f"ra_exit{RA_OUT_K}_0", 0),
    (f"ra_exit{RA_OUT_K}_0", f"ra_bendout{RA_OUT_K}_0", 0), (f"ra_bendout{RA_OUT_K}_0", f"ra_farout{RA_OUT_K}_0", 0),
    (f"ra_farout{RA_OUT_K}_0", "mg_a_0", 0),
    ("mg_a_0", "mg_b_0", 0), ("mg_b_0", "mg_c_0", 0), ("mg_c_0", "mg_d_0", 0),
]

_HUMAN_ROUTE_LANES = [
    ("fw_o0_0", "fw_ir0_0", 0), ("fw_ir0_0", "fw_il3_0", 0),  # spawn south, right turn -> east arm
    ("fw_il3_0", "fw_o3_0", 0),
    ("fw_o3_0", "tw_o1_0", 0),
    ("tw_o1_0", "tw_ir1_0", 0), ("tw_ir1_0", "tw_il2_0", 0),  # left turn onto north arm
    ("tw_il2_0", "tw_o2_0", 0),
] + _SHARED_TAIL

_ROBOT_ROUTE_LANES = [
    ("tw_o0_0", "tw_ir0_0", 0), ("tw_ir0_0", "tw_il2_0", 0), ("tw_il2_0", "tw_o2_0", 0),
] + _SHARED_TAIL

_route_net = _build_network()
HUMAN_ROUTE = _polyline(_route_net, _HUMAN_ROUTE_LANES)
ROBOT_ROUTE = _polyline(_route_net, _ROBOT_ROUTE_LANES)
