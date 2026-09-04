"""Real scene 003, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way, round_about, add_three_way x2, merge), not real
map data.

Chained tightly, short gap to short gap -- same philosophy as
real_001_rebuilt.py/real_002_rebuilt.py, a different order again: a 4-way
(convergence), a quarter-turn around a roundabout, a LEFT turn at one
T-junction, a LEFT turn at a second T-junction, and a highway on-ramp merge.

An earlier version used turn_corner for the second bend instead of a
second T-junction -- dropped for the same reason real_001_rebuilt.py's own
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
to the end rather than changed back -- both isolated, and both forking
away from lane 0 only, so lane 1's own true straight edges are never
pruned (see real_001_rebuilt.py's own docstring on why an earlier "out and
back" pair on the merge highway left the outer lane looking cut).

HUMAN_ROUTE: south arm of the 4-way, straight through.
ROBOT_ROUTE: east arm, right turn -- joining the human immediately at the
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

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 18.0
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)

TW1_ACCESS = 25.0
N_TW1 = 2

TW2_ACCESS = 25.0
N_TW2 = 2
TW2_MISSING_CORNER = 2  # stem points south (corner 0), matching TW1's own north exit

MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 20.0, 20.0, 30.0

GAP = 12.0

# FW's north exit always has heading -90 (add_four_way's own fixed corner-2
# convention), so this is deterministic -- computed once here (not
# re-derived inside _build_network) so _SHARED_TAIL below can reference the
# same values used to actually build the network.
_RA_IN_K = ring_gap_for_heading(-np.pi / 2)
_RA_OUT_K = (_RA_IN_K + 1) % 4  # quarter-turn around the ring


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    add_four_way(net, center=FW_CENTER, n_vertical=N_FW, n_horizontal=N_FW,
                 access_length=FW_ACCESS, prefix="fw_")

    # FW's north exit (heading -90) -> roundabout, entering from the south gap.
    fw_far, fw_heading = exit_far(net, "fw_", 2, 0)
    ra_in_k = _RA_IN_K
    ra_center = np.array(fw_far) + (RA_ACCESS + RA_RADIUS + GAP) * np.array(
        [np.cos(fw_heading), np.sin(fw_heading)])
    round_about(net, center=ra_center, radius=RA_RADIUS, n_lanes=1, access_length=RA_ACCESS,
                alpha=RA_ALPHA, merge_radius=RA_MERGE_RADIUS, prefix="ra_")
    bridge_roundabout(net, "ra_", ra_in_k, "fw_", 2)

    # Quarter-turn around the ring -> T-junction, entering its west arm (stem).
    ra_out_k = _RA_OUT_K
    ra_out_lane = net.get_lane((f"ra_bendout{ra_out_k}_0", f"ra_farout{ra_out_k}_0", 0))
    ra_out_far = ra_out_lane.position(ra_out_lane.length, 0)
    ra_out_heading = ra_out_lane.heading_at(ra_out_lane.length)
    tw1_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2
    tw1_center = center_ahead_corner(ra_out_far, ra_out_heading, 0, TW1_ACCESS, tw1_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw1_center, n_stem=N_TW1, n_cross=N_TW1, access_length=TW1_ACCESS,
                  prefix="tw1_", missing_corner=3)  # stem = corner (3+2)%4 = 1 (west)
    bridge_roundabout(net, "ra_", ra_out_k, "tw1_", 1)

    # LEFT turn onto the north arm (heading -90) -> second T-junction,
    # entering its south arm (stem).
    tw1_far, tw1_heading = exit_far(net, "tw1_", 2, 0)
    tw2_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2
    tw2_center = center_ahead_corner(tw1_far, tw1_heading, 0, TW2_ACCESS, tw2_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw2_center, n_stem=N_TW2, n_cross=N_TW2, access_length=TW2_ACCESS,
                  prefix="tw2_", missing_corner=TW2_MISSING_CORNER)
    bridge_corners(net, "tw1_", 2, "tw2_", 0, N_TW1)

    # LEFT turn onto the west arm -> highway merge. Queried on lane 1 --
    # the route rides lane 1 through TW2 (see the lane change below) --
    # though the bridge itself only carries the one lane the route uses.
    tw2_far, tw2_heading = exit_far(net, "tw2_", 1, 1)
    mg_center = np.array(tw2_far) + GAP * np.array([np.cos(tw2_heading), np.sin(tw2_heading)])
    merge(net, center=mg_center, heading_deg=np.degrees(tw2_heading), n_lanes=2,
          before_length=MG_BEFORE, taper_length=MG_TAPER, merge_length=MG_MERGE, after_length=MG_AFTER,
          ramp_side=1, prefix="mg_")
    bridge(net, "tw2_o1_1", "mg_a_0", tw2_far, mg_center)

    # Lane change #1: TW1's own exit (0 -> 1), now that bridge_corners
    # carries both lanes into TW2 -- lane 1's own complete pre-built turn
    # geometry then carries the route through TW2 to the merge bridge
    # above without any further changes in between.
    p = net.get_lane(("tw1_il2_0", "tw1_o2_0", 0)).position(0, 0)
    h = net.get_lane(("tw1_il2_0", "tw1_o2_0", 0)).heading_at(0)
    target = net.get_lane(("tw1_il2_1", "tw1_o2_1", 0))
    lane_change_to(net, "tw1_il2_0", "tw1_o2_1", p, h, target.position(target.length, 0), LANE_WIDTH)

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
    (f"ra_entry{_RA_IN_K}_0", f"ra_exit{_RA_OUT_K}_0", 0),
    (f"ra_exit{_RA_OUT_K}_0", f"ra_bendout{_RA_OUT_K}_0", 0), (f"ra_bendout{_RA_OUT_K}_0", f"ra_farout{_RA_OUT_K}_0", 0),
    (f"ra_farout{_RA_OUT_K}_0", "tw1_o1_0", 0),
    ("tw1_o1_0", "tw1_ir1_0", 0), ("tw1_ir1_0", "tw1_il2_0", 0),  # left turn onto the north arm
    ("tw1_il2_0", "tw1_o2_1", 0),                                 # lane change #1 (0->1)
    ("tw1_o2_1", "tw2_o0_1", 0),                                  # bridge, lane 1
    ("tw2_o0_1", "tw2_ir0_1", 0), ("tw2_ir0_1", "tw2_il1_1", 0),  # left turn onto the west arm, lane 1
    ("tw2_il1_1", "tw2_o1_1", 0), ("tw2_o1_1", "mg_a_0", 0),
    ("mg_a_0", "mg_b_1", 0),                                      # lane change #2 (0->1)
    ("mg_b_1", "mg_c_1", 0),                                      # merge/taper zone, lane 1
    ("mg_c_1", "mg_d_1", 0),                                      # after zone, lane 1 (true outer lane, no further change)
]

_HUMAN_ROUTE_LANES = [
    ("fw_o0_0", "fw_ir0_0", 0), ("fw_ir0_0", "fw_il2_0", 0),
] + _SHARED_TAIL

_ROBOT_ROUTE_LANES = [
    ("fw_o3_0", "fw_ir3_0", 0), ("fw_ir3_0", "fw_il2_0", 0),
] + _SHARED_TAIL

_route_net = _build_network()
HUMAN_ROUTE = polyline(_route_net, _HUMAN_ROUTE_LANES)
ROBOT_ROUTE = polyline(_route_net, _ROBOT_ROUTE_LANES)


def route_adjacent_lane_indexes(radius=15.0):
    return _route_adjacent(_build_network, HUMAN_ROUTE, radius=radius)
