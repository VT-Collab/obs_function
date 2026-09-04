"""Real scene 002, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way, add_three_way x2, round_about, merge), not real
map data.

Chained tightly, short gap to short gap, so the drive is dense with
maneuvers rather than long straight stretches -- same philosophy as
real_001_rebuilt.py, with a different order and different turns for
variety: a 4-way (where HUMAN_ROUTE and ROBOT_ROUTE converge), a LEFT turn
at one T-junction, a RIGHT turn at a second T-junction, a half-turn around
a roundabout (twice the arc of real_001_rebuilt.py's quarter-turn), and a
highway on-ramp merge.

ramp_side=1, matching real_001_rebuilt.py's own value, not -1 as an
earlier version had it -- see real_004_rebuilt.py's own docstring for why
-1 doesn't work once bidirectional=True adds the reverse-direction lanes
(its own target lane sits on the opposite side of them from its own ramp).

An earlier version used turn_corner for the first bend instead of a
T-junction -- dropped for the same reason real_001_rebuilt.py's own
turn_corner was: the identical curve reads as a sharp, isolated "elbow"
when it's a lone bend in an otherwise straight corridor, but reads as
smooth, ordinary intersection geometry once it's embedded inside a real
junction instead. See real_001_rebuilt.py's docstring for the fuller note.

Every primitive here (add_four_way, add_three_way, round_about, merge)
uses the SAME lane count throughout (2 each direction), matching
real_001_rebuilt.py's own convention exactly (see its own docstring for
the full reasoning) -- no primitive narrows relative to its neighbor, so
every hop carries every lane across, in both directions (bridge_corners/
bridge_roundabout, not a single-lane bridge()). The roundabout's own
n_lanes=2 and RA_ACCESS=40.0 and the merge's own bidirectional=True with
MG_TAPER=40.0 (amplitude scaled the same way) are the SAME numeric values
real_001_rebuilt.py uses -- see its own docstring for why each one is
what it is; nothing here re-derives that reasoning independently.

The route makes ONE lane change (0 -> 1) right after the 4-way, riding
lane 1's own pre-built turn geometry through both T-junctions, the
roundabout's own half-turn, and onto the merge highway's own lane 1
directly -- no second change needed, since every primitive from here on
shares the same lane count (see real_001_rebuilt.py's own docstring on
why an earlier version needed a second change here, before every
primitive matched).

HUMAN_ROUTE: south arm of the 4-way, straight through.
ROBOT_ROUTE: west arm, LEFT turn -- joining the human immediately at the
4-way (real_001_rebuilt.py's robot instead enters east and turns right; this
is the other way traffic can converge onto the same exit). From there both
share every remaining maneuver.

Every add_four_way/add_three_way call here uses EQUAL n_vertical/n_horizontal
(or n_stem/n_cross) -- see real_001_rebuilt.py's docstring for why unequal
counts make the roads look like they don't line up.
"""
from highway_env.road.road import Road, RoadNetwork

from build_scene import LANE_WIDTH, S, add_four_way, stitch_junction, stitch_roundabout, stitch_merge
from _layout_utils import lane_change_to, polyline, prune_to_route, route_adjacent_lane_indexes as _route_adjacent

FW_CENTER = (0.0, 0.0)
FW_ACCESS = 25.0
N_FW = 2

TW1_ACCESS = 25.0
N_TW1 = 2
TW1_MISSING_CORNER = 2  # stem points south (corner 0), matching FW's north-exit heading

TW2_ACCESS = 25.0
N_TW2 = 2
TW2_MISSING_CORNER = 1  # stem points east (corner 3), matching TW1's own west exit

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 40.0  # matches real_001_rebuilt.py's own value exactly
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)
RA_IN_K = 3   # gap facing back toward TW2 (see build_scene.round_about: k=3 is the south-facing gap)
RA_MID_K = 0  # ring gap passed straight through (not left) on the way around
RA_OUT_K = 1  # gap facing north -- a half-turn around the ring

# Short, not the reference highway_env MergeEnv's own 150/80/80/150 -- see
# real_001_rebuilt.py's own docstring on MG_BEFORE/MG_TAPER for the full
# reasoning; every value here is the SAME as real_001_rebuilt.py's own.
MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 40.0, 20.0, 30.0
MG_TAPER_AMPLITUDE = 3.25 / 80 * MG_TAPER  # see real_001_rebuilt.py's own docstring
MG_LANES = N_TW2  # bidirectional -- matches every other primitive here (2 lanes each way)

GAP = 12.0  # every bridge between junctions -- short, so maneuvers stay close together


def _build_network() -> RoadNetwork:
    net = RoadNetwork()
    add_four_way(net, center=FW_CENTER, n_vertical=N_FW, n_horizontal=N_FW,
                 access_length=FW_ACCESS, prefix="fw_")

    # FW's north exit (heading -90, i.e. north) -> TW1's south arm (corner 0).
    stitch_junction(net, ("junction", "fw_", 2, 0), "tw1_", TW1_ACCESS, N_TW1,
                     missing_corner=TW1_MISSING_CORNER, gap=GAP)

    # TW1's west exit (left turn from the south stem, heading 180, i.e.
    # west) -> TW2's east arm (its stem).
    stitch_junction(net, ("junction", "tw1_", 1, 0), "tw_", TW2_ACCESS, N_TW2,
                     missing_corner=TW2_MISSING_CORNER, gap=GAP)

    # TW's north exit (right turn from the east stem, heading -90, i.e.
    # north) -> roundabout, entering from the south gap. TW and the
    # roundabout carry the SAME lane count (both 2), so every lane bridges
    # 1:1 (see real_001_rebuilt.py's own docstring on bridge_roundabout's
    # own n>1).
    stitch_roundabout(net, "tw_", 2, 1, "ra_", RA_RADIUS, RA_ACCESS, RA_ALPHA, RA_MERGE_RADIUS,
                       n_lanes=N_TW2, gap=GAP, in_k=RA_IN_K, bridge_lane_i=0)

    # A half-turn around the ring (south gap -> west gap, passing straight
    # through the east gap -> north gap) -> highway merge on the far side,
    # bidirectional (matching every other primitive here) and
    # exact_center=True since this bridges N_TW2 lanes in parallel (see
    # stitch_merge's own docstring, and real_001_rebuilt.py's own on why
    # that matters once more than one lane bridges at once).
    stitch_merge(net, ("roundabout", "ra_", RA_OUT_K, 1), "mg_", MG_LANES, gap=GAP, exact_center=True,
                 before_length=MG_BEFORE, taper_length=MG_TAPER, merge_length=MG_MERGE, after_length=MG_AFTER,
                 ramp_side=1, ramp_gap=0.0, amplitude=MG_TAPER_AMPLITUDE, bidirectional=True)

    # The merge lane (ramp) is NOT bridged to the roundabout -- see
    # real_001_rebuilt.py's own docstring for the full reasoning (matches
    # the reference highway_env MergeEnv's own vehicle-spawned-directly-
    # on-the-ramp design). It still visibly merges into the highway a
    # little further on, via one lane-change run the rest of the way down
    # the highway for a visibly gentle curve -- same construction as
    # real_001_rebuilt.py's own, see its own docstring.
    ramp_taper = net.get_lane(("mg_k", "mg_b_ramp", 0))
    p0 = ramp_taper.position(ramp_taper.length, 0)
    h0 = ramp_taper.heading_at(ramp_taper.length)
    highway_outer = net.get_lane(("mg_b_1", "mg_c_1", 0))
    target = highway_outer.position(highway_outer.length, 0)
    lane_change_to(net, "mg_b_ramp", "mg_c_1", p0, h0, target, LANE_WIDTH, line_types=(S, S))
    del net.graph["mg_b_ramp"]["mg_c_ramp"]  # superseded by the lane-change above -- see real_001_rebuilt.py's own docstring

    # Lane change #1: FW's own north exit (0 -> 1), landing on lane 1's own
    # pre-built far point so bridge_corners' own lane-1 bridge -- and then
    # lane 1's own complete pre-built turn geometry through both
    # T-junctions, the roundabout, and onto the merge's own lane 1 --
    # carries the route the rest of the way without any further changes.
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
    ("fw_o2_1", "tw1_o0_1", 0),                                   # bridge, lane 1
    ("tw1_o0_1", "tw1_ir0_1", 0), ("tw1_ir0_1", "tw1_il1_1", 0),  # left turn onto the west arm, lane 1
    ("tw1_il1_1", "tw1_o1_1", 0), ("tw1_o1_1", "tw_o3_1", 0),     # bridge, lane 1
    ("tw_o3_1", "tw_ir3_1", 0), ("tw_ir3_1", "tw_il2_1", 0),      # right turn onto the north arm, lane 1
    ("tw_il2_1", "tw_o2_1", 0), ("tw_o2_1", f"ra_farin{RA_IN_K}_1", 0),
    (f"ra_farin{RA_IN_K}_1", f"ra_bendin{RA_IN_K}_1", 0), (f"ra_bendin{RA_IN_K}_1", f"ra_entry{RA_IN_K}_1", 0),
    (f"ra_entry{RA_IN_K}_1", f"ra_exit{RA_MID_K}_1", 0),
    (f"ra_exit{RA_MID_K}_1", f"ra_entry{RA_MID_K}_1", 0),  # pass straight through, don't leave here
    (f"ra_entry{RA_MID_K}_1", f"ra_exit{RA_OUT_K}_1", 0),
    (f"ra_exit{RA_OUT_K}_1", f"ra_bendout{RA_OUT_K}_1", 0), (f"ra_bendout{RA_OUT_K}_1", f"ra_farout{RA_OUT_K}_1", 0),
    (f"ra_farout{RA_OUT_K}_1", "mg_a_1", 0),                      # bridge, into the highway's own normal lane 1
    ("mg_a_1", "mg_b_1", 0),                                      # before zone, lane 1 -- no lane change needed,
    ("mg_b_1", "mg_c_1", 0),                                      # matches the roundabout's own lane 1 exactly
    ("mg_c_1", "mg_d_1", 0),                                      # after zone, lane 1 (true outer lane, no further change)
]

# South arm, straight through: opp_c of corner 0 is 2 -> lands on il2_0.
_HUMAN_ROUTE_LANES = [
    ("fw_o0_0", "fw_ir0_0", 0), ("fw_ir0_0", "fw_il2_0", 0),
] + _SHARED_TAIL

# West arm, left turn: next_c of corner 1 is 2 -> lands on il2_0, same
# node the human's straight movement lands on.
_ROBOT_ROUTE_LANES = [
    ("fw_o1_0", "fw_ir1_0", 0), ("fw_ir1_0", "fw_il2_0", 0),
] + _SHARED_TAIL

_route_net = _build_network()
HUMAN_ROUTE = polyline(_route_net, _HUMAN_ROUTE_LANES)
ROBOT_ROUTE = polyline(_route_net, _ROBOT_ROUTE_LANES)


def route_adjacent_lane_indexes(radius=15.0):
    return _route_adjacent(_build_network, HUMAN_ROUTE, radius=radius)
