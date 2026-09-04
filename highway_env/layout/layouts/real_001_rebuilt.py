"""Real scene 001, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way, add_three_way x2, round_about, merge), not real
map data.

Chained tightly, short gap to short gap, so the drive is dense with
maneuvers rather than long straight stretches: a 4-way (where HUMAN_ROUTE
and ROBOT_ROUTE converge), a LEFT turn at one T-junction, a LEFT turn at a
second T-junction, a quarter-turn around a roundabout, and a highway
on-ramp merge -- five distinct maneuvers back to back, all sharing one
lane's width the whole way (a country road narrowing outside each built-up
junction, same reasoning as mega_scene.py's own chain).

An earlier version used turn_corner for the second bend instead of a
second T-junction. Dropped: turn_corner's curve has the same radius as a
three-way's own turn, but turn_corner sits alone in an otherwise straight
corridor with nothing else nearby, so the SAME curve reads as a sharp,
isolated "elbow" -- the identical curve embedded inside a real junction
(another approach, cross traffic, a proper stop-line) reads as smooth,
ordinary intersection geometry instead. real_009_rebuilt.py never used
turn_corner at all and was the one that visibly read as smooth; every
scene in this batch now sticks to add_four_way/add_three_way/round_about/
merge only, matching that.

Every add_four_way/add_three_way junction here is 2 lanes each way, and
every junction-to-junction hop uses bridge_corners (not a single-lane
bridge()) to carry BOTH lanes across, in BOTH directions -- a single-lane
one-way bridge leaves the outer lane dangling (disconnected right at the
first junction's own edge, since nothing else in either primitive's own
construction reaches across the gap for it) and, worse, leaves the return
direction with no bridge at all (see _layout_utils.bridge_corners' own
docstring). That outer lane and the return direction aren't part of
HUMAN_ROUTE/ROBOT_ROUTE itself, just present and connected for background
traffic. TW2<->roundabout uses bridge_roundabout for the same reason, in
both directions (see its own docstring, and round_about's own on why
entry/exit needed to become two genuinely separate lanes first).

The route makes TWO lane changes total, each isolated and each forking
away from lane 0 (the INNER lane) only -- never lane 1, the true outer
lane: one right after the 4-way (0 -> 1), riding lane 1's own pre-built
turn geometry through both T-junctions before dropping back to lane 0 at
the roundabout, and one more onto the merge highway's own lane 1 (also
0 -> 1), which the route then rides to the very end rather than changing
back. An earlier version added a second merge-highway change back to lane
0 -- dropped because that fork started at a node lane 1's own true
straight edge also used, so prune_to_route deleted that real edge as an
unchosen alternative, leaving the outer lane looking cut for the length of
the "after" stretch (see _layout_utils.lane_change's own docstring).
Riding lane 1 to the end instead means the outer lane's own pre-built
straight edges are never pruned at all.

HUMAN_ROUTE: south arm of the 4-way, straight through.
ROBOT_ROUTE: east arm, right turn -- joining the human immediately at the
4-way. From there both share every remaining maneuver: a lane change onto
lane 1, left at each T (on lane 1's own turn geometry), back to lane 0
before the roundabout, a quarter of the roundabout, and one more lane
change onto lane 1 through the merge, ridden to the end.

Every add_four_way/add_three_way call here uses EQUAL n_vertical/n_horizontal
(or n_stem/n_cross): unequal counts make opposite corners' turn arcs
different sizes, which reads as the roads not lining up -- see this
codebase's other build_scene-based layouts, all of which follow the same
rule.
"""
import numpy as np

from highway_env.road.road import Road, RoadNetwork

from build_scene import LANE_WIDTH, RIGHT_TURN_RADIUS_EXTRA, add_four_way, add_three_way, round_about, merge
from _layout_utils import (bridge, bridge_corners, bridge_roundabout, center_ahead_corner, exit_far,
                            lane_change_to, polyline, prune_to_route, route_adjacent_lane_indexes as _route_adjacent)

RIGHT_TURN_RADIUS_BASE = LANE_WIDTH + RIGHT_TURN_RADIUS_EXTRA  # matches build_scene.py's own internal constant

FW_CENTER = (0.0, 0.0)
FW_ACCESS = 25.0
N_FW = 2

TW1_ACCESS = 25.0
N_TW1 = 2
TW1_MISSING_CORNER = 2  # stem points south (corner 0), cross arms west/east (1/3)

TW2_ACCESS = 25.0
N_TW2 = 2
TW2_MISSING_CORNER = 1  # stem points east (corner 3), matching TW1's own west exit

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 18.0
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)
RA_IN_K = 1   # gap facing back north, toward TW2 (see build_scene.round_about: k=1 is north-facing)
RA_OUT_K = 2  # gap facing west -- a quarter-turn around the ring

MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 20.0, 20.0, 30.0

GAP = 12.0  # every bridge between junctions -- short, so maneuvers stay close together


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    add_four_way(net, center=FW_CENTER, n_vertical=N_FW, n_horizontal=N_FW,
                 access_length=FW_ACCESS, prefix="fw_")

    # FW's north exit (heading -90, i.e. north) -> TW1's south arm (corner 0).
    fw_far, fw_heading = exit_far(net, "fw_", 2, 0)
    tw1_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2  # n_stem/n_cross lane 0
    tw1_center = center_ahead_corner(fw_far, fw_heading, 0, TW1_ACCESS, tw1_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw1_center, n_stem=N_TW1, n_cross=N_TW1, access_length=TW1_ACCESS,
                  prefix="tw1_", missing_corner=TW1_MISSING_CORNER)
    bridge_corners(net, "fw_", 2, "tw1_", 0, N_FW)

    # TW1's west exit (left turn from the south stem, heading 180, i.e.
    # west) -> TW2's east arm (its stem).
    tw1_far, tw1_heading = exit_far(net, "tw1_", 1, 0)
    tw2_od = RIGHT_TURN_RADIUS_BASE + LANE_WIDTH / 2
    tw2_center = center_ahead_corner(tw1_far, tw1_heading, 0, TW2_ACCESS, tw2_od, GAP, LANE_WIDTH)
    add_three_way(net, center=tw2_center, n_stem=N_TW2, n_cross=N_TW2, access_length=TW2_ACCESS,
                  prefix="tw2_", missing_corner=TW2_MISSING_CORNER)
    bridge_corners(net, "tw1_", 1, "tw2_", 3, N_TW1)

    # TW2's south exit (left turn from the east stem, heading 90, i.e.
    # south) -> roundabout, entering from the north gap. Queried on lane 1
    # (see the lane change below -- the route rides lane 1 through both
    # T-junctions), though the bridge itself only carries the one lane the
    # route actually uses, same as every other junction-to-connector hop.
    tw2_far, tw2_heading = exit_far(net, "tw2_", 0, 1)
    ra_center = np.array(tw2_far) + (RA_ACCESS + RA_RADIUS + GAP) * np.array(
        [np.cos(tw2_heading), np.sin(tw2_heading)])
    round_about(net, center=ra_center, radius=RA_RADIUS, n_lanes=1, access_length=RA_ACCESS,
                alpha=RA_ALPHA, merge_radius=RA_MERGE_RADIUS, prefix="ra_")
    bridge_roundabout(net, "ra_", RA_IN_K, "tw2_", 0, j_lane_i=1)

    # A quarter-turn around the ring (north gap -> west gap) -> highway merge.
    ra_out_lane = net.get_lane((f"ra_bendout{RA_OUT_K}_0", f"ra_farout{RA_OUT_K}_0", 0))
    ra_out_far = ra_out_lane.position(ra_out_lane.length, 0)
    ra_out_heading = ra_out_lane.heading_at(ra_out_lane.length)
    mg_center = np.array(ra_out_far) + GAP * np.array([np.cos(ra_out_heading), np.sin(ra_out_heading)])
    merge(net, center=mg_center, heading_deg=np.degrees(ra_out_heading), n_lanes=2,
          before_length=MG_BEFORE, taper_length=MG_TAPER, merge_length=MG_MERGE, after_length=MG_AFTER,
          ramp_side=1, prefix="mg_")
    bridge(net, f"ra_farout{RA_OUT_K}_0", "mg_a_0", ra_out_far, mg_center)

    # Lane change #1: FW's own north exit (0 -> 1), landing on lane 1's own
    # pre-built far point so bridge_corners' own lane-1 bridge -- and then
    # lane 1's own complete pre-built turn geometry through both
    # T-junctions -- carries the route the rest of the way to the
    # roundabout without any further changes in between.
    p = net.get_lane(("fw_il2_0", "fw_o2_0", 0)).position(0, 0)
    h = net.get_lane(("fw_il2_0", "fw_o2_0", 0)).heading_at(0)
    target = net.get_lane(("fw_il2_1", "fw_o2_1", 0))
    lane_change_to(net, "fw_il2_0", "fw_o2_1", p, h, target.position(target.length, 0), LANE_WIDTH)

    # Lane change #2: onto the merge's own "before" stretch (0 -> 1),
    # landing on lane 1's own pre-built node so the taper/merge/after zone
    # (already built for both lanes) carries the route to the end on lane
    # 1 -- the true outer lane -- without any further changes or forks
    # away from it.
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
    ("tw1_il1_1", "tw1_o1_1", 0), ("tw1_o1_1", "tw2_o3_1", 0),    # bridge, lane 1
    ("tw2_o3_1", "tw2_ir3_1", 0), ("tw2_ir3_1", "tw2_il0_1", 0),  # left turn onto the south arm, lane 1
    ("tw2_il0_1", "tw2_o0_1", 0), ("tw2_o0_1", f"ra_farin{RA_IN_K}_0", 0),
    (f"ra_farin{RA_IN_K}_0", f"ra_bendin{RA_IN_K}_0", 0), (f"ra_bendin{RA_IN_K}_0", f"ra_entry{RA_IN_K}_0", 0),
    (f"ra_entry{RA_IN_K}_0", f"ra_exit{RA_OUT_K}_0", 0),
    (f"ra_exit{RA_OUT_K}_0", f"ra_bendout{RA_OUT_K}_0", 0), (f"ra_bendout{RA_OUT_K}_0", f"ra_farout{RA_OUT_K}_0", 0),
    (f"ra_farout{RA_OUT_K}_0", "mg_a_0", 0),
    ("mg_a_0", "mg_b_1", 0),                                      # lane change #2 (0->1)
    ("mg_b_1", "mg_c_1", 0),                                      # merge/taper zone, lane 1
    ("mg_c_1", "mg_d_1", 0),                                      # after zone, lane 1 (true outer lane, no further change)
]

# South arm, straight through: opp_c of corner 0 is 2 -> lands on il2_0.
_HUMAN_ROUTE_LANES = [
    ("fw_o0_0", "fw_ir0_0", 0), ("fw_ir0_0", "fw_il2_0", 0),
] + _SHARED_TAIL

# East arm, right turn: prev_c of corner 3 is 2 -> lands on il2_0, same
# node the human's straight movement lands on.
_ROBOT_ROUTE_LANES = [
    ("fw_o3_0", "fw_ir3_0", 0), ("fw_ir3_0", "fw_il2_0", 0),
] + _SHARED_TAIL

_route_net = _build_network()
HUMAN_ROUTE = polyline(_route_net, _HUMAN_ROUTE_LANES)
ROBOT_ROUTE = polyline(_route_net, _ROBOT_ROUTE_LANES)


def route_adjacent_lane_indexes(radius=15.0):
    return _route_adjacent(_build_network, HUMAN_ROUTE, radius=radius)
