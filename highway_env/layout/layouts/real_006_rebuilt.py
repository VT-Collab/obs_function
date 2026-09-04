"""Real scene 006, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way, add_three_way x2, round_about, merge), not real
map data.

Chained tightly, short gap to short gap: a 4-way (convergence), a LEFT turn
at one T-junction, a quarter-turn around a roundabout (replacing what was
originally a second, RIGHT-turning T-junction -- same net turn, west-facing
entry to north-facing exit, so everything downstream is unaffected), a
LEFT turn at a third T-junction, and a highway on-ramp merge on the
opposite side (ramp_side=-1).

An earlier version used turn_corner for the third bend instead of a third
T-junction -- dropped for the same reason real_001_rebuilt.py's own
turn_corner was (see that file's docstring): the identical curve reads as
a sharp, isolated "elbow" alone in a corridor, but reads as smooth,
ordinary intersection geometry once embedded inside a real junction.

FW<->TW1 uses bridge_corners (both lanes) -- see real_001_rebuilt.py's own
docstring for why a single-lane bridge leaves the outer lane disconnected
(TW1<->RA and RA<->TW3 stay single-lane, since the roundabout itself only
has one). The route makes two lane changes total: one right after the
4-way (0 -> 1), riding lane 1's own pre-built turn geometry through TW1
before dropping back to lane 0 via the roundabout bridge itself, and one
more onto the merge highway's own lane 1, ridden to the end rather than
changed back (see real_001_rebuilt.py's own docstring on why an earlier
"out and back" pair on the merge highway left the outer lane looking cut).

HUMAN_ROUTE: south arm of the 4-way, straight through.
ROBOT_ROUTE: west arm, left turn -- joining the human immediately at the
4-way. From there both share every remaining maneuver.

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

FW_CENTER = (0.0, 0.0)
FW_ACCESS = 25.0
N_FW = 2

TW1_ACCESS = 25.0
N_TW1 = 2
TW1_MISSING_CORNER = 2  # stem = corner 0 (south), matching FW's north-exit heading

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 18.0
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)

TW3_ACCESS = 25.0
N_TW3 = 2
TW3_MISSING_CORNER = 2  # stem = corner 0 (south), matching the roundabout's own north exit

MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 20.0, 20.0, 30.0

GAP = 12.0

# Deterministic from the fixed geometry (TW1's own west exit is always
# heading 180) -- computed once here so _SHARED_TAIL can reference the same
# values _build_network uses.
RA_IN_K = ring_gap_for_heading(np.pi)
RA_OUT_K = (RA_IN_K + 1) % 4  # quarter-turn: west-facing entry -> north-facing exit, same net turn as the RIGHT turn this roundabout replaces


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    add_four_way(net, center=FW_CENTER, n_vertical=N_FW, n_horizontal=N_FW,
                 access_length=FW_ACCESS, prefix="fw_")

    # FW's north exit (heading -90) -> TW1's south arm (stem).
    fw_far, fw_heading = exit_far(net, "fw_", 2, 0)
    tw1_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2
    tw1_center = center_ahead_corner(fw_far, fw_heading, 0, TW1_ACCESS, tw1_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw1_center, n_stem=N_TW1, n_cross=N_TW1, access_length=TW1_ACCESS,
                  prefix="tw1_", missing_corner=TW1_MISSING_CORNER)
    bridge_corners(net, "fw_", 2, "tw1_", 0, N_FW)

    # LEFT turn onto TW1's west arm (heading 180) -> roundabout, entering
    # from its east-facing gap. Queried on lane 1 -- the route rides lane 1
    # through TW1 (see the lane change below) -- though the bridge itself
    # only carries the one lane the route uses.
    tw1_far, tw1_heading = exit_far(net, "tw1_", 1, 1)
    ra_center = np.array(tw1_far) + (RA_ACCESS + RA_RADIUS + GAP) * np.array(
        [np.cos(tw1_heading), np.sin(tw1_heading)])
    round_about(net, center=ra_center, radius=RA_RADIUS, n_lanes=1, access_length=RA_ACCESS,
                alpha=RA_ALPHA, merge_radius=RA_MERGE_RADIUS, prefix="ra_")
    bridge_roundabout(net, "ra_", RA_IN_K, "tw1_", 1, j_lane_i=1)

    # Quarter-turn around the ring (west gap -> north gap, the same net
    # turn as the RIGHT turn this roundabout replaces) -> TW3's south arm
    # (stem).
    ra_out_lane = net.get_lane((f"ra_bendout{RA_OUT_K}_0", f"ra_farout{RA_OUT_K}_0", 0))
    ra_out_far = ra_out_lane.position(ra_out_lane.length, 0)
    ra_out_heading = ra_out_lane.heading_at(ra_out_lane.length)
    tw3_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2
    tw3_center = center_ahead_corner(ra_out_far, ra_out_heading, 0, TW3_ACCESS, tw3_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw3_center, n_stem=N_TW3, n_cross=N_TW3, access_length=TW3_ACCESS,
                  prefix="tw3_", missing_corner=TW3_MISSING_CORNER)
    bridge_roundabout(net, "ra_", RA_OUT_K, "tw3_", 0)

    # LEFT turn onto TW3's west arm -> highway merge on the opposite ramp side.
    tw3_far, tw3_heading = exit_far(net, "tw3_", 1, 0)
    mg_center = np.array(tw3_far) + GAP * np.array([np.cos(tw3_heading), np.sin(tw3_heading)])
    merge(net, center=mg_center, heading_deg=np.degrees(tw3_heading), n_lanes=2,
          before_length=MG_BEFORE, taper_length=MG_TAPER, merge_length=MG_MERGE, after_length=MG_AFTER,
          ramp_side=-1, prefix="mg_")
    bridge(net, "tw3_o1_0", "mg_a_0", tw3_far, mg_center)

    # Lane change #1: FW's own north exit (0 -> 1), landing on lane 1's own
    # pre-built far point so bridge_corners' own lane-1 bridge -- and then
    # lane 1's own complete pre-built turn geometry through TW1 -- carries
    # the route to the roundabout bridge above without any further changes.
    p = net.get_lane(("fw_il2_0", "fw_o2_0", 0)).position(0, 0)
    h = net.get_lane(("fw_il2_0", "fw_o2_0", 0)).heading_at(0)
    target = net.get_lane(("fw_il2_1", "fw_o2_1", 0))
    lane_change_to(net, "fw_il2_0", "fw_o2_1", p, h, target.position(target.length, 0), LANE_WIDTH)

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
    ("fw_il2_0", "fw_o2_1", 0),                                   # lane change #1 (0->1)
    ("fw_o2_1", "tw1_o0_1", 0),                                   # bridge, lane 1
    ("tw1_o0_1", "tw1_ir0_1", 0), ("tw1_ir0_1", "tw1_il1_1", 0),  # left turn onto the west arm, lane 1
    ("tw1_il1_1", "tw1_o1_1", 0), ("tw1_o1_1", f"ra_farin{RA_IN_K}_0", 0),
    (f"ra_farin{RA_IN_K}_0", f"ra_bendin{RA_IN_K}_0", 0), (f"ra_bendin{RA_IN_K}_0", f"ra_entry{RA_IN_K}_0", 0),
    (f"ra_entry{RA_IN_K}_0", f"ra_exit{RA_OUT_K}_0", 0),
    (f"ra_exit{RA_OUT_K}_0", f"ra_bendout{RA_OUT_K}_0", 0), (f"ra_bendout{RA_OUT_K}_0", f"ra_farout{RA_OUT_K}_0", 0),
    (f"ra_farout{RA_OUT_K}_0", "tw3_o0_0", 0),
    ("tw3_o0_0", "tw3_ir0_0", 0), ("tw3_ir0_0", "tw3_il1_0", 0),  # left turn onto the west arm
    ("tw3_il1_0", "tw3_o1_0", 0), ("tw3_o1_0", "mg_a_0", 0),
    ("mg_a_0", "mg_b_1", 0),                                      # lane change #2 (0->1)
    ("mg_b_1", "mg_c_1", 0),                                      # merge/taper zone, lane 1
    ("mg_c_1", "mg_d_1", 0),                                      # after zone, lane 1 (true outer lane, no further change)
]

_HUMAN_ROUTE_LANES = [
    ("fw_o0_0", "fw_ir0_0", 0), ("fw_ir0_0", "fw_il2_0", 0),
] + _SHARED_TAIL

# West arm, left turn: next_c of corner 1 is 2 -> lands on il2_0.
_ROBOT_ROUTE_LANES = [
    ("fw_o1_0", "fw_ir1_0", 0), ("fw_ir1_0", "fw_il2_0", 0),
] + _SHARED_TAIL

_route_net = _build_network()
HUMAN_ROUTE = polyline(_route_net, _HUMAN_ROUTE_LANES)
ROBOT_ROUTE = polyline(_route_net, _ROBOT_ROUTE_LANES)


def route_adjacent_lane_indexes(radius=15.0):
    return _route_adjacent(_build_network, HUMAN_ROUTE, radius=radius)
