"""Real scene 004, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way, add_three_way x2, round_about, merge), not real
map data.

Chained tightly, short gap to short gap: a 4-way (convergence), a RIGHT
turn at a T-junction, three-quarters of the way around a roundabout (the
long way around, unlike real_001_rebuilt.py's quarter-turn), a LEFT turn at
a second T-junction, and a highway on-ramp merge.

ramp_side=1, matching real_001_rebuilt.py's own value, not -1 as an
earlier version had it: for a bidirectional merge, ramp_side=-1's own
"beside lane 0" target sits on the OPPOSITE side of the reverse-direction
lanes from where that ramp_side's own taper places the ramp itself --
forward lane 0 is adjacent to the centerline, not to any open shoulder, so
reaching it without crossing the reverse lanes first isn't geometrically
possible once bidirectional=True adds them. ramp_side=1's own target
(the highest-index forward lane) sits on the SAME side as its own taper,
with nothing in between -- the only side that's actually open road.

An earlier version used turn_corner for the second bend instead of a
second T-junction -- dropped for the same reason real_001_rebuilt.py's own
turn_corner was (see that file's docstring): the identical curve reads as
a sharp, isolated "elbow" alone in a corridor, but reads as smooth,
ordinary intersection geometry once embedded inside a real junction.

Every primitive here (add_four_way, add_three_way, round_about, merge)
uses the SAME lane count throughout (2 each direction), matching
real_001_rebuilt.py's own convention exactly -- no primitive narrows
relative to its neighbor, so every hop carries every lane across, in both
directions. The roundabout's own n_lanes=2 and RA_ACCESS=40.0 and the
merge's own bidirectional=True with MG_TAPER=40.0 (amplitude scaled the
same way) are the SAME numeric values real_001_rebuilt.py uses -- see its
own docstring for why each one is what it is.

The route makes ONE lane change: right after the 4-way (0 -> 1), riding
lane 1's own pre-built turn geometry through TW1, the roundabout's own
three-quarter turn, and TW2, landing directly on the merge highway's own
lane 1 -- no second change needed, since every primitive from there on
shares the same lane count (see real_001_rebuilt.py's own docstring on
why an earlier version needed a second change here, before every
primitive matched).

HUMAN_ROUTE: south arm of the 4-way, straight through.
ROBOT_ROUTE: west arm, left turn -- joining the human immediately at the
4-way. From there both share every remaining maneuver.

Every add_four_way/add_three_way call here uses EQUAL n_vertical/n_horizontal
(or n_stem/n_cross) -- see real_001_rebuilt.py's docstring for why unequal
counts make the roads look like they don't line up.
"""
from highway_env.road.road import Road, RoadNetwork

from build_scene import LANE_WIDTH, S, add_four_way, stitch_junction, stitch_roundabout, stitch_merge
from _layout_utils import lane_change_to, ring_gap_for_heading, polyline, prune_to_route, \
    route_adjacent_lane_indexes as _route_adjacent

FW_CENTER = (0.0, 0.0)
FW_ACCESS = 25.0
N_FW = 2

TW1_ACCESS = 25.0
N_TW1 = 2
TW1_MISSING_CORNER = 2  # stem = corner (2+2)%4 = 0 (south), matching FW's north-exit heading

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 40.0  # matches real_001_rebuilt.py's own value exactly
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)

TW2_ACCESS = 25.0
N_TW2 = 2
TW2_MISSING_CORNER = 2  # stem points south (corner 0), matching the roundabout's own north exit

# Short, not the reference highway_env MergeEnv's own 150/80/80/150 -- see
# real_001_rebuilt.py's own docstring on MG_BEFORE/MG_TAPER for the full
# reasoning; every value here is the SAME as real_001_rebuilt.py's own.
MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 40.0, 20.0, 30.0
MG_TAPER_AMPLITUDE = 3.25 / 80 * MG_TAPER  # see real_001_rebuilt.py's own docstring
MG_LANES = N_TW2  # bidirectional -- matches every other primitive here (2 lanes each way)

GAP = 12.0

# Deterministic from the fixed geometry (FW always exits north at -90; a
# RIGHT turn from TW's south stem always lands on its east arm, exit heading
# 0) -- computed once here so _SHARED_TAIL can reference the same values
# _build_network uses.
_RA_IN_K = ring_gap_for_heading(0.0)
_RA_HOP1_K = (_RA_IN_K + 1) % 4
_RA_HOP2_K = (_RA_IN_K + 2) % 4
_RA_OUT_K = (_RA_IN_K + 3) % 4  # three-quarters around


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    add_four_way(net, center=FW_CENTER, n_vertical=N_FW, n_horizontal=N_FW,
                 access_length=FW_ACCESS, prefix="fw_")

    # FW's north exit (heading -90) -> TW1's south arm (stem).
    stitch_junction(net, ("junction", "fw_", 2, 0), "tw_", TW1_ACCESS, N_TW1,
                     missing_corner=TW1_MISSING_CORNER, gap=GAP)

    # RIGHT turn onto the east arm (heading 0) -> roundabout, entering from
    # its west-facing gap. TW1 and the roundabout carry the SAME lane
    # count (both 2), so every lane bridges 1:1.
    stitch_roundabout(net, "tw_", 3, 1, "ra_", RA_RADIUS, RA_ACCESS, RA_ALPHA, RA_MERGE_RADIUS,
                       n_lanes=N_TW1, gap=GAP, in_k=_RA_IN_K, bridge_lane_i=0)

    # Three-quarters around the ring -> second T-junction, entering its
    # south arm (stem). Queried on lane 0 (stitch_junction's own
    # roundabout-source branch bridges starting at this SAME lane index,
    # so it must be 0 to cover both roundabout lanes {0, 1}).
    stitch_junction(net, ("roundabout", "ra_", _RA_OUT_K, 0), "tw2_", TW2_ACCESS, N_TW2,
                     missing_corner=TW2_MISSING_CORNER, gap=GAP)

    # LEFT turn onto the west arm -> highway merge on the opposite ramp
    # side, bidirectional (matching every other primitive here) and
    # exact_center=True since this bridges N_TW2 lanes in parallel.
    stitch_merge(net, ("junction", "tw2_", 1, 1), "mg_", MG_LANES, gap=GAP, exact_center=True,
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

    # Lane change #1: FW's own north exit (0 -> 1), landing on lane 1's own
    # pre-built far point so bridge_corners' own lane-1 bridge -- and then
    # lane 1's own complete pre-built turn geometry through TW1, the
    # roundabout, and TW2 -- carries the route to the merge's own lane 1
    # without any further changes in between.
    p = net.get_lane(("fw_il2_0", "fw_o2_0", 0)).position(0, 0)
    h = net.get_lane(("fw_il2_0", "fw_o2_0", 0)).heading_at(0)
    target = net.get_lane(("fw_il2_1", "fw_o2_1", 0))
    lane_change_to(net, "fw_il2_0", "fw_o2_1", p, h, target.position(target.length, 0), LANE_WIDTH)

    prune_to_route(net, _HUMAN_ROUTE_LANES, _ROBOT_ROUTE_LANES)
    return net


def build_road() -> Road:
    return Road(network=_build_network())


_SHARED_TAIL = [
    ("fw_il2_0", "fw_o2_1", 0),                                   # lane change #1 (0->1)
    ("fw_o2_1", "tw_o0_1", 0),                                    # bridge, lane 1
    ("tw_o0_1", "tw_ir0_1", 0), ("tw_ir0_1", "tw_il3_1", 0),      # right turn onto the east arm, lane 1
    ("tw_il3_1", "tw_o3_1", 0), ("tw_o3_1", f"ra_farin{_RA_IN_K}_1", 0),
    (f"ra_farin{_RA_IN_K}_1", f"ra_bendin{_RA_IN_K}_1", 0), (f"ra_bendin{_RA_IN_K}_1", f"ra_entry{_RA_IN_K}_1", 0),
    (f"ra_entry{_RA_IN_K}_1", f"ra_exit{_RA_HOP1_K}_1", 0),
    (f"ra_exit{_RA_HOP1_K}_1", f"ra_entry{_RA_HOP1_K}_1", 0),  # pass through, don't leave
    (f"ra_entry{_RA_HOP1_K}_1", f"ra_exit{_RA_HOP2_K}_1", 0),
    (f"ra_exit{_RA_HOP2_K}_1", f"ra_entry{_RA_HOP2_K}_1", 0),  # pass through, don't leave
    (f"ra_entry{_RA_HOP2_K}_1", f"ra_exit{_RA_OUT_K}_1", 0),
    (f"ra_exit{_RA_OUT_K}_1", f"ra_bendout{_RA_OUT_K}_1", 0), (f"ra_bendout{_RA_OUT_K}_1", f"ra_farout{_RA_OUT_K}_1", 0),
    (f"ra_farout{_RA_OUT_K}_1", "tw2_o0_1", 0),
    ("tw2_o0_1", "tw2_ir0_1", 0), ("tw2_ir0_1", "tw2_il1_1", 0),  # left turn onto the west arm, lane 1
    ("tw2_il1_1", "tw2_o1_1", 0), ("tw2_o1_1", "mg_a_1", 0),      # bridge, into the highway's own normal lane 1
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
