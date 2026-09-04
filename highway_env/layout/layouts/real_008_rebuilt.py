"""Real scene 008, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way, round_about, add_three_way x2, merge), not real
map data.

Chained tightly, short gap to short gap: a 4-way (convergence), an
immediate half-turn around a roundabout, a LEFT turn at one T-junction, a
RIGHT turn at a second T-junction, and a highway on-ramp merge.

ramp_side=1, matching real_001_rebuilt.py's own value, not -1 as an
earlier version had it -- see real_004_rebuilt.py's own docstring for why
-1 doesn't work once bidirectional=True adds the reverse-direction lanes
(its own target lane sits on the opposite side of them from its own ramp).

An earlier version used turn_corner for the first bend instead of a
T-junction -- dropped for the same reason real_001_rebuilt.py's own
turn_corner was (see that file's docstring): the identical curve reads as
a sharp, isolated "elbow" alone in a corridor, but reads as smooth,
ordinary intersection geometry once embedded inside a real junction.

Every primitive here (add_four_way, round_about, add_three_way, merge)
uses the SAME lane count throughout (2 each direction), matching
real_001_rebuilt.py's own convention exactly -- no primitive narrows
relative to its neighbor, so every hop carries every lane across, in both
directions. The roundabout's own n_lanes=2 and RA_ACCESS=40.0 and the
merge's own bidirectional=True with MG_TAPER=40.0 (amplitude scaled the
same way) are the SAME numeric values real_001_rebuilt.py uses -- see its
own docstring for why each one is what it is.

The route makes ONE lane change: at TW1's own exit (0 -> 1), riding lane
1's own pre-built turn geometry through TW2 and onto the merge highway's
own lane 1 directly -- no second change needed, since every primitive
from there on shares the same lane count (see real_001_rebuilt.py's own
docstring on why an earlier version needed a second change here, before
every primitive matched).

HUMAN_ROUTE: south arm of the 4-way, straight through.
ROBOT_ROUTE: west arm, left turn -- joining the human immediately at the
4-way. From there both share every remaining maneuver.

Every add_four_way/add_three_way call here uses EQUAL n_vertical/n_horizontal
(or n_stem/n_cross) -- see real_001_rebuilt.py's docstring for why unequal
counts make the roads look like they don't line up.
"""
import numpy as np

from highway_env.road.road import Road, RoadNetwork

from build_scene import LANE_WIDTH, S, add_four_way, stitch_junction, stitch_roundabout, stitch_merge
from _layout_utils import lane_change_to, ring_gap_for_heading, polyline, prune_to_route, \
    route_adjacent_lane_indexes as _route_adjacent

FW_CENTER = (0.0, 0.0)
FW_ACCESS = 25.0
N_FW = 2

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 40.0  # matches real_001_rebuilt.py's own value exactly
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)

TW1_ACCESS = 25.0
N_TW1 = 2
TW1_MISSING_CORNER = 2  # stem = corner 0 (south), matching the roundabout's own north exit

TW2_ACCESS = 25.0
N_TW2 = 2
TW2_MISSING_CORNER = 1  # stem = corner (1+2)%4 = 3 (east), matching TW1's own west exit

# Short, not the reference highway_env MergeEnv's own 150/80/80/150 -- see
# real_001_rebuilt.py's own docstring on MG_BEFORE/MG_TAPER for the full
# reasoning; every value here is the SAME as real_001_rebuilt.py's own.
MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 40.0, 20.0, 30.0
MG_TAPER_AMPLITUDE = 3.25 / 80 * MG_TAPER  # see real_001_rebuilt.py's own docstring
MG_LANES = N_TW2  # bidirectional -- matches every other primitive here (2 lanes each way)

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

    # FW's north exit (heading -90) -> roundabout, entering from the south
    # gap. FW and the roundabout carry the SAME lane count (both 2), so
    # every lane bridges 1:1.
    stitch_roundabout(net, "fw_", 2, 0, "ra_", RA_RADIUS, RA_ACCESS, RA_ALPHA, RA_MERGE_RADIUS,
                       n_lanes=N_FW, gap=GAP, in_k=_RA_IN_K, bridge_lane_i=0)

    # Half-turn around the ring -> TW1's south arm (stem). Queried on lane
    # 0 (stitch_junction's own roundabout-source branch bridges starting
    # at this SAME lane index, so it must be 0 to cover both roundabout
    # lanes {0, 1}).
    stitch_junction(net, ("roundabout", "ra_", _RA_OUT_K, 0), "tw1_", TW1_ACCESS, N_TW1,
                     missing_corner=TW1_MISSING_CORNER, gap=GAP)

    # LEFT turn onto TW1's west arm (heading 180) -> TW2's east arm (stem).
    stitch_junction(net, ("junction", "tw1_", 1, 0), "tw_", TW2_ACCESS, N_TW2,
                     missing_corner=TW2_MISSING_CORNER, gap=GAP)

    # RIGHT turn onto the north arm -> highway merge on the opposite ramp
    # side, bidirectional (matching every other primitive here) and
    # exact_center=True since this bridges N_TW2 lanes in parallel.
    stitch_merge(net, ("junction", "tw_", 2, 1), "mg_", MG_LANES, gap=GAP, exact_center=True,
                 before_length=MG_BEFORE, taper_length=MG_TAPER, merge_length=MG_MERGE, after_length=MG_AFTER,
                 ramp_side=1, ramp_gap=0.0, amplitude=MG_TAPER_AMPLITUDE, bidirectional=True)

    # The merge lane (ramp) is NOT bridged to the roundabout -- see
    # real_001_rebuilt.py's own docstring for the full reasoning. It still
    # visibly merges into the highway a little further on, via one
    # lane-change run the rest of the way down the highway for a visibly
    # gentle curve -- same construction as real_001_rebuilt.py's own.
    ramp_taper = net.get_lane(("mg_k", "mg_b_ramp", 0))
    p0 = ramp_taper.position(ramp_taper.length, 0)
    h0 = ramp_taper.heading_at(ramp_taper.length)
    highway_outer = net.get_lane(("mg_b_1", "mg_c_1", 0))
    target = highway_outer.position(highway_outer.length, 0)
    lane_change_to(net, "mg_b_ramp", "mg_c_1", p0, h0, target, LANE_WIDTH, line_types=(S, S))
    del net.graph["mg_b_ramp"]["mg_c_ramp"]  # superseded by the lane-change above -- see real_001_rebuilt.py's own docstring

    # Lane change #1: TW1's own exit (0 -> 1) -- lane 1's own complete
    # pre-built turn geometry then carries the route through TW2 and onto
    # the merge's own lane 1 directly, without any further changes.
    p = net.get_lane(("tw1_il1_0", "tw1_o1_0", 0)).position(0, 0)
    h = net.get_lane(("tw1_il1_0", "tw1_o1_0", 0)).heading_at(0)
    target = net.get_lane(("tw1_il1_1", "tw1_o1_1", 0))
    lane_change_to(net, "tw1_il1_0", "tw1_o1_1", p, h, target.position(target.length, 0), LANE_WIDTH)

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
    ("tw_il2_1", "tw_o2_1", 0), ("tw_o2_1", "mg_a_1", 0),         # bridge, into the highway's own normal lane 1
    ("mg_a_1", "mg_b_1", 0),                                      # before zone, lane 1 -- no lane change needed,
    ("mg_b_1", "mg_c_1", 0),                                      # matches the merge's own lane 1 exactly
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
