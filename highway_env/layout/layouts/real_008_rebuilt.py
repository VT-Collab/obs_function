"""Real scene 008, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way, round_about, add_three_way x2, merge), not real
map data.

Chained tightly, short gap to short gap: a 4-way (convergence), an
immediate half-turn around a roundabout, a LEFT turn at one T-junction, a
RIGHT turn at a second T-junction, and a highway on-ramp merge on the
opposite side (ramp_side=-1).

An earlier version used turn_corner for the first bend instead of a
T-junction -- dropped for the same reason real_001_rebuilt.py's own
turn_corner was (see that file's docstring): the identical curve reads as
a sharp, isolated "elbow" alone in a corridor, but reads as smooth,
ordinary intersection geometry once embedded inside a real junction.

TW1<->TW2 uses bridge_corners (both lanes) -- see real_001_rebuilt.py's
own docstring for why a single-lane bridge leaves the outer lane
disconnected (FW<->RA and RA<->TW1 stay single-lane, since the roundabout
itself only has one). The route makes two lane changes total: one at
TW1's own exit (0 -> 1), riding lane 1's own pre-built turn geometry
through TW2 before dropping back to lane 0 into the merge's own single
incoming bridge, and one more onto the merge highway's own lane 1, ridden
to the end rather than changed back (see real_001_rebuilt.py's own
docstring on why an earlier "out and back" pair on the merge highway left
the outer lane looking cut).

HUMAN_ROUTE: south arm of the 4-way, straight through.
ROBOT_ROUTE: west arm, left turn -- joining the human immediately at the
4-way. From there both share every remaining maneuver.

Every add_four_way/add_three_way call here uses EQUAL n_vertical/n_horizontal
(or n_stem/n_cross) -- see real_001_rebuilt.py's docstring for why unequal
counts make the roads look like they don't line up.
"""
import numpy as np

from highway_env.road.road import Road, RoadNetwork

from build_scene import LANE_WIDTH, RIGHT_TURN_RADIUS_EXTRA, add_four_way, round_about, add_three_way, merge
from _layout_utils import (bridge, bridge_corners, bridge_roundabout, center_ahead_corner, exit_far,
                            lane_change_to, ring_gap_for_heading, polyline, prune_to_route,
                            route_adjacent_lane_indexes as _route_adjacent)

RIGHT_TURN_RADIUS_BASE = LANE_WIDTH + RIGHT_TURN_RADIUS_EXTRA

FW_CENTER = (0.0, 0.0)
FW_ACCESS = 25.0
N_FW = 2

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 18.0
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)

TW1_ACCESS = 25.0
N_TW1 = 2
TW1_MISSING_CORNER = 2  # stem = corner 0 (south), matching the roundabout's own north exit

TW2_ACCESS = 25.0
N_TW2 = 2
TW2_MISSING_CORNER = 1  # stem = corner (1+2)%4 = 3 (east), matching TW1's own west exit

MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 20.0, 20.0, 30.0

GAP = 12.0

# Deterministic from the fixed geometry (FW exits north at -90) -- computed
# once here so _SHARED_TAIL can reference the same values _build_network uses.
_RA_IN_K = ring_gap_for_heading(-np.pi / 2)
_RA_HOP_K = (_RA_IN_K + 1) % 4
_RA_OUT_K = (_RA_IN_K + 2) % 4  # half-turn


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    add_four_way(net, center=FW_CENTER, n_vertical=N_FW, n_horizontal=N_FW,
                 access_length=FW_ACCESS, prefix="fw_")

    # FW's north exit (heading -90) -> roundabout, entering from the south gap.
    fw_far, fw_heading = exit_far(net, "fw_", 2, 0)
    ra_center = np.array(fw_far) + (RA_ACCESS + RA_RADIUS + GAP) * np.array(
        [np.cos(fw_heading), np.sin(fw_heading)])
    round_about(net, center=ra_center, radius=RA_RADIUS, n_lanes=1, access_length=RA_ACCESS,
                alpha=RA_ALPHA, merge_radius=RA_MERGE_RADIUS, prefix="ra_")
    bridge_roundabout(net, "ra_", _RA_IN_K, "fw_", 2)

    # Half-turn around the ring -> TW1's south arm (stem).
    ra_out_lane = net.get_lane((f"ra_bendout{_RA_OUT_K}_0", f"ra_farout{_RA_OUT_K}_0", 0))
    ra_out_far = ra_out_lane.position(ra_out_lane.length, 0)
    ra_out_heading = ra_out_lane.heading_at(ra_out_lane.length)
    tw_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2
    tw1_center = center_ahead_corner(ra_out_far, ra_out_heading, 0, TW1_ACCESS, tw_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw1_center, n_stem=N_TW1, n_cross=N_TW1, access_length=TW1_ACCESS,
                  prefix="tw1_", missing_corner=TW1_MISSING_CORNER)
    bridge_roundabout(net, "ra_", _RA_OUT_K, "tw1_", 0)

    # LEFT turn onto TW1's west arm (heading 180) -> TW2's east arm (stem).
    tw1_far, tw1_heading = exit_far(net, "tw1_", 1, 0)
    tw_center = center_ahead_corner(tw1_far, tw1_heading, 0, TW2_ACCESS, tw_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw_center, n_stem=N_TW2, n_cross=N_TW2, access_length=TW2_ACCESS,
                  prefix="tw_", missing_corner=TW2_MISSING_CORNER)
    bridge_corners(net, "tw1_", 1, "tw_", 3, N_TW1)

    # RIGHT turn onto the north arm -> highway merge on the opposite ramp
    # side. Queried on lane 1 -- the route rides lane 1 through TW2 (see
    # the lane change below) -- though the bridge itself only carries the
    # one lane the route uses.
    tw_far, tw_heading = exit_far(net, "tw_", 2, 1)
    mg_center = np.array(tw_far) + GAP * np.array([np.cos(tw_heading), np.sin(tw_heading)])
    merge(net, center=mg_center, heading_deg=np.degrees(tw_heading), n_lanes=2,
          before_length=MG_BEFORE, taper_length=MG_TAPER, merge_length=MG_MERGE, after_length=MG_AFTER,
          ramp_side=-1, prefix="mg_")
    bridge(net, "tw_o2_1", "mg_a_0", tw_far, mg_center)

    # Lane change #1: TW1's own exit (0 -> 1).
    p = net.get_lane(("tw1_il1_0", "tw1_o1_0", 0)).position(0, 0)
    h = net.get_lane(("tw1_il1_0", "tw1_o1_0", 0)).heading_at(0)
    target = net.get_lane(("tw1_il1_1", "tw1_o1_1", 0))
    lane_change_to(net, "tw1_il1_0", "tw1_o1_1", p, h, target.position(target.length, 0), LANE_WIDTH)

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
    ("fw_il2_0", "fw_o2_0", 0), ("fw_o2_0", f"ra_farin{_RA_IN_K}_0", 0),
    (f"ra_farin{_RA_IN_K}_0", f"ra_bendin{_RA_IN_K}_0", 0), (f"ra_bendin{_RA_IN_K}_0", f"ra_entry{_RA_IN_K}_0", 0),
    (f"ra_entry{_RA_IN_K}_0", f"ra_exit{_RA_HOP_K}_0", 0),
    (f"ra_exit{_RA_HOP_K}_0", f"ra_entry{_RA_HOP_K}_0", 0),  # pass through, don't leave
    (f"ra_entry{_RA_HOP_K}_0", f"ra_exit{_RA_OUT_K}_0", 0),
    (f"ra_exit{_RA_OUT_K}_0", f"ra_bendout{_RA_OUT_K}_0", 0), (f"ra_bendout{_RA_OUT_K}_0", f"ra_farout{_RA_OUT_K}_0", 0),
    (f"ra_farout{_RA_OUT_K}_0", "tw1_o0_0", 0),
    ("tw1_o0_0", "tw1_ir0_0", 0), ("tw1_ir0_0", "tw1_il1_0", 0),  # left turn onto the west arm
    ("tw1_il1_0", "tw1_o1_1", 0),                                 # lane change #1 (0->1)
    ("tw1_o1_1", "tw_o3_1", 0),                                   # bridge, lane 1
    ("tw_o3_1", "tw_ir3_1", 0), ("tw_ir3_1", "tw_il2_1", 0),      # right turn onto the north arm, lane 1
    ("tw_il2_1", "tw_o2_1", 0), ("tw_o2_1", "mg_a_0", 0),
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
