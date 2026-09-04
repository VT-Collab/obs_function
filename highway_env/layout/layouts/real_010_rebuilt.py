"""Real scene 010, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way x2, add_three_way, round_about, merge), not real
map data.

Chained tightly, short gap to short gap: a 4-way (convergence), a LEFT turn
at a second 4-way right after it, a LEFT turn at a T-junction, a
quarter-turn around a roundabout, and a highway on-ramp merge on the
opposite side (ramp_side=-1).

An earlier version used turn_corner for the third bend instead of a
T-junction -- dropped for the same reason real_001_rebuilt.py's own
turn_corner was (see that file's docstring): the identical curve reads as
a sharp, isolated "elbow" alone in a corridor, but reads as smooth,
ordinary intersection geometry once embedded inside a real junction.

FW1<->FW2<->TW uses bridge_corners (both lanes) -- see real_001_rebuilt.py's
own docstring for why a single-lane bridge leaves the outer lane
disconnected (TW<->RA stays single-lane, since the roundabout itself only
has one). The route makes two lane changes total: one right after the
first 4-way (0 -> 1), riding lane 1's own pre-built turn geometry through
the second 4-way and the T-junction before dropping back to lane 0 via
the roundabout bridge itself, and one more onto the merge highway's own
lane 1, ridden to the end rather than changed back (see
real_001_rebuilt.py's own docstring on why an earlier "out and back" pair
on the merge highway left the outer lane looking cut).

HUMAN_ROUTE: south arm of the first 4-way, straight through.
ROBOT_ROUTE: west arm, left turn -- joining the human immediately at the
first 4-way. From there both share every remaining maneuver.

Every add_four_way/add_three_way call here uses EQUAL n_vertical/n_horizontal
(or n_stem/n_cross) -- see real_001_rebuilt.py's docstring for why unequal
counts make the roads look like they don't line up.
"""
import numpy as np

from highway_env.road.road import Road, RoadNetwork

from build_scene import LANE_WIDTH, RIGHT_TURN_RADIUS_EXTRA, add_four_way, add_three_way, round_about, merge
from _layout_utils import (bridge, bridge_corners, bridge_roundabout, center_ahead_corner, exit_far,
                            lane_change_to, ring_gap_for_heading, polyline, prune_to_route,
                            route_adjacent_lane_indexes as _route_adjacent)

RIGHT_TURN_RADIUS_BASE = LANE_WIDTH + RIGHT_TURN_RADIUS_EXTRA

FW1_CENTER = (0.0, 0.0)
FW1_ACCESS = 25.0
N_FW1 = 2

FW2_ACCESS = 25.0
N_FW2 = 2

TW_ACCESS = 25.0
N_TW = 2
TW_MISSING_CORNER = 1  # stem = corner (1+2)%4 = 3 (east), matching fw2_'s own west exit

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 18.0
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)

MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 20.0, 20.0, 30.0

GAP = 12.0

# Deterministic from the fixed geometry (a LEFT turn from TW's east stem
# lands on heading 90, south) -- computed once here so _SHARED_TAIL can
# reference the same values _build_network uses.
_RA_IN_K = ring_gap_for_heading(np.pi / 2)
_RA_OUT_K = (_RA_IN_K + 1) % 4  # quarter-turn


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    add_four_way(net, center=FW1_CENTER, n_vertical=N_FW1, n_horizontal=N_FW1,
                 access_length=FW1_ACCESS, prefix="fw1_")

    # fw1_'s north exit (heading -90) -> fw2_'s south arm.
    fw1_far, fw1_heading = exit_far(net, "fw1_", 2, 0)
    fw2_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2
    fw2_center = center_ahead_corner(fw1_far, fw1_heading, 0, FW2_ACCESS, fw2_od, GAP, LANE_WIDTH)
    add_four_way(net, center=fw2_center, n_vertical=N_FW2, n_horizontal=N_FW2,
                 access_length=FW2_ACCESS, prefix="fw2_")
    bridge_corners(net, "fw1_", 2, "fw2_", 0, N_FW1)

    # LEFT turn onto fw2_'s west arm (heading 180) -> TW's east arm (stem).
    fw2_far, fw2_heading = exit_far(net, "fw2_", 1, 0)
    tw_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2
    tw_center = center_ahead_corner(fw2_far, fw2_heading, 0, TW_ACCESS, tw_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw_center, n_stem=N_TW, n_cross=N_TW, access_length=TW_ACCESS,
                  prefix="tw_", missing_corner=TW_MISSING_CORNER)
    bridge_corners(net, "fw2_", 1, "tw_", 3, N_FW2)

    # LEFT turn onto TW's south arm (heading 90) -> roundabout, entering
    # from its north-facing gap. Queried on lane 1 -- the route rides lane
    # 1 through fw2_ and TW (see the lane change below) -- though the
    # bridge itself only carries the one lane the route uses.
    tw_far, tw_heading = exit_far(net, "tw_", 0, 1)
    ra_center = np.array(tw_far) + (RA_ACCESS + RA_RADIUS + GAP) * np.array(
        [np.cos(tw_heading), np.sin(tw_heading)])
    round_about(net, center=ra_center, radius=RA_RADIUS, n_lanes=1, access_length=RA_ACCESS,
                alpha=RA_ALPHA, merge_radius=RA_MERGE_RADIUS, prefix="ra_")
    bridge_roundabout(net, "ra_", _RA_IN_K, "tw_", 0, j_lane_i=1)

    # Quarter-turn around the ring -> highway merge on the opposite ramp side.
    ra_out_lane = net.get_lane((f"ra_bendout{_RA_OUT_K}_0", f"ra_farout{_RA_OUT_K}_0", 0))
    ra_out_far = ra_out_lane.position(ra_out_lane.length, 0)
    ra_out_heading = ra_out_lane.heading_at(ra_out_lane.length)
    mg_center = np.array(ra_out_far) + GAP * np.array([np.cos(ra_out_heading), np.sin(ra_out_heading)])
    merge(net, center=mg_center, heading_deg=np.degrees(ra_out_heading), n_lanes=2,
          before_length=MG_BEFORE, taper_length=MG_TAPER, merge_length=MG_MERGE, after_length=MG_AFTER,
          ramp_side=-1, prefix="mg_")
    bridge(net, f"ra_farout{_RA_OUT_K}_0", "mg_a_0", ra_out_far, mg_center)

    # Lane change #1: fw1_'s own north exit (0 -> 1).
    p = net.get_lane(("fw1_il2_0", "fw1_o2_0", 0)).position(0, 0)
    h = net.get_lane(("fw1_il2_0", "fw1_o2_0", 0)).heading_at(0)
    target = net.get_lane(("fw1_il2_1", "fw1_o2_1", 0))
    lane_change_to(net, "fw1_il2_0", "fw1_o2_1", p, h, target.position(target.length, 0), LANE_WIDTH)

    # Lane change #2: onto the merge's own "before" stretch (0 -> 1),
    # ridden all the way to the end on lane 1 -- the true outer lane --
    # without any further changes or forks away from it.
    p = net.get_lane(("mg_a_0", "mg_b_0", 0)).position(0, 0)
    h = net.get_lane(("mg_a_0", "mg_b_0", 0)).heading_at(0)
    target = net.get_lane(("mg_a_1", "mg_b_1", 0))
    lane_change_to(net, "mg_a_0", "mg_b_1", p, h, target.position(target.length, 0), LANE_WIDTH)

    prune_to_route(net, _HUMAN_ROUTE_LANES, _ROBOT_ROUTE_LANES)
    return net


def build_road() -> Road:
    return Road(network=_build_network())


_SHARED_TAIL = [
    ("fw1_il2_0", "fw1_o2_1", 0),                                 # lane change #1 (0->1)
    ("fw1_o2_1", "fw2_o0_1", 0),                                  # bridge, lane 1
    ("fw2_o0_1", "fw2_ir0_1", 0), ("fw2_ir0_1", "fw2_il1_1", 0),  # left turn onto the west arm, lane 1
    ("fw2_il1_1", "fw2_o1_1", 0), ("fw2_o1_1", "tw_o3_1", 0),     # bridge, lane 1
    ("tw_o3_1", "tw_ir3_1", 0), ("tw_ir3_1", "tw_il0_1", 0),      # left turn onto the south arm, lane 1
    ("tw_il0_1", "tw_o0_1", 0), ("tw_o0_1", f"ra_farin{_RA_IN_K}_0", 0),
    (f"ra_farin{_RA_IN_K}_0", f"ra_bendin{_RA_IN_K}_0", 0), (f"ra_bendin{_RA_IN_K}_0", f"ra_entry{_RA_IN_K}_0", 0),
    (f"ra_entry{_RA_IN_K}_0", f"ra_exit{_RA_OUT_K}_0", 0),
    (f"ra_exit{_RA_OUT_K}_0", f"ra_bendout{_RA_OUT_K}_0", 0), (f"ra_bendout{_RA_OUT_K}_0", f"ra_farout{_RA_OUT_K}_0", 0),
    (f"ra_farout{_RA_OUT_K}_0", "mg_a_0", 0),
    ("mg_a_0", "mg_b_1", 0),                                      # lane change #2 (0->1)
    ("mg_b_1", "mg_c_1", 0),                                      # merge/taper zone, lane 1
    ("mg_c_1", "mg_d_1", 0),                                      # after zone, lane 1 (true outer lane, no further change)
]

_HUMAN_ROUTE_LANES = [
    ("fw1_o0_0", "fw1_ir0_0", 0), ("fw1_ir0_0", "fw1_il2_0", 0),
] + _SHARED_TAIL

# West arm, left turn: next_c of corner 1 is 2 -> lands on il2_0.
_ROBOT_ROUTE_LANES = [
    ("fw1_o1_0", "fw1_ir1_0", 0), ("fw1_ir1_0", "fw1_il2_0", 0),
] + _SHARED_TAIL

_route_net = _build_network()
HUMAN_ROUTE = polyline(_route_net, _HUMAN_ROUTE_LANES)
ROBOT_ROUTE = polyline(_route_net, _ROBOT_ROUTE_LANES)


def route_adjacent_lane_indexes(radius=15.0):
    return _route_adjacent(_build_network, HUMAN_ROUTE, radius=radius)
