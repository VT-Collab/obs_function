"""Real scene 001, rebuilt: built entirely from build_scene.py's own
primitives (add_four_way, add_three_way x2, round_about, merge), not real
map data.

Chained tightly, short gap to short gap, so the drive is dense with
maneuvers rather than long straight stretches: a 4-way (where HUMAN_ROUTE
and ROBOT_ROUTE converge), a LEFT turn at one T-junction, a LEFT turn at a
second T-junction, a quarter-turn around a roundabout, and a highway
on-ramp merge -- five distinct maneuvers back to back. Every primitive
here (add_four_way, add_three_way, round_about, merge) uses the SAME lane
count throughout (2 each direction) -- no primitive here narrows relative
to its neighbor, so every junction-to-junction hop carries every lane
across, in both directions (bridge_corners/bridge_roundabout, not a
single-lane bridge() -- see their own docstrings). The roundabout sits
between TWO exact, matched connections at once: its own north gap to
TW2, and its own east gap to a NEW direct link straight to FW's own west
arm (added purely for background connectivity -- HUMAN_ROUTE/ROBOT_ROUTE
never use FW's west arm, see _build_network's own comment there), placed
so both bridges land exactly straight simultaneously (ra_center's own
X taken from TW2's true axis, Y from FW's -- see its own comment).

The merge itself is short (before/taper/merge/after = 20/20/20/30, not
the reference highway_env MergeEnv's own 150/80/80/150 build_scene.py's
own demo uses -- see MG_BEFORE's own comment) and bidirectional (2 lanes
each way, matching every other primitive here -- see MG_LANES's own
comment), with the roundabout's own west gap bridged exactly into its
two NORMAL lanes only. The ramp itself is NOT bridged to the roundabout
at all -- it stays exactly as merge() built it, disconnected upstream,
originating from nothing (same as the reference's own vehicle-spawned-
directly-on-the-ramp design), but still visibly merges into the highway
via one lane-change (see its own comment) so the scene still reads as a
real on-ramp merge, even though neither route drives it.

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
traffic. TW2<->roundabout uses bridge_roundabout(n=N_TW2) for the same
reason -- both lanes, both directions, matching bridge_corners exactly
now that the roundabout carries the same lane count as TW2 itself (see
bridge_roundabout's own docstring on n>1, and round_about's own on why
entry/exit needed to become two genuinely separate lanes first).

The route makes ONE lane change: right after the 4-way (0 -> 1), riding
lane 1's own pre-built turn geometry through both T-junctions, the
roundabout's own quarter-turn, and onto the merge highway's own lane 1
(landing there directly, no second change needed, since the roundabout
and the merge highway carry the same lane count too) -- ridden to the
very end. An earlier version dropped back to lane 0 before the roundabout
(when the roundabout was single-lane, n_lanes=1, and needed one) and made
a second, separate change onto the merge highway's own lane 1; with every
primitive now at the same lane count end to end, lane 1 carries the route
the whole way after that first change, with nothing forcing a drop back
to lane 0 anywhere.

HUMAN_ROUTE: south arm of the 4-way, straight through.
ROBOT_ROUTE: east arm, right turn -- joining the human immediately at the
4-way. From there both share every remaining maneuver: one lane change
onto lane 1, left at each T (on lane 1's own turn geometry), a quarter of
the roundabout on lane 1, and onto the merge's own lane 1, ridden to the
end.

Every add_four_way/add_three_way call here uses EQUAL n_vertical/n_horizontal
(or n_stem/n_cross): unequal counts make opposite corners' turn arcs
different sizes, which reads as the roads not lining up -- see this
codebase's other build_scene-based layouts, all of which follow the same
rule.
"""
import numpy as np

from highway_env.road.road import Road, RoadNetwork

from build_scene import LANE_WIDTH, RIGHT_TURN_RADIUS_EXTRA, C, S, add_four_way, add_three_way, round_about, merge
from _layout_utils import (bridge, bridge_corners, bridge_roundabout, center_ahead_corner, exit_far,
                            exit_far_center, lane_change_to, polyline, prune_to_route,
                            route_adjacent_lane_indexes as _route_adjacent)

RIGHT_TURN_RADIUS_BASE = LANE_WIDTH + RIGHT_TURN_RADIUS_EXTRA  # matches build_scene.py's own internal constant

FW_CENTER = (0.0, 0.0)
FW_ACCESS = 47.5  # matches build_scene.py's own __main__ grid demo (_demo_four_way) exactly
N_FW = 2

TW1_ACCESS = 47.5  # matches build_scene.py's own __main__ grid demo (_demo_three_way) exactly
N_TW1 = 2
TW1_MISSING_CORNER = 2  # stem points south (corner 0), cross arms west/east (1/3)

TW2_ACCESS = 47.5  # matches build_scene.py's own __main__ grid demo (_demo_three_way) exactly
N_TW2 = 2
TW2_MISSING_CORNER = 1  # stem points east (corner 3), matching TW1's own west exit

RA_RADIUS = 22.0  # bumped from 10: highway_env draws curves as straight STRIPE_SPACING=4.33m chords -- fine on a wide highway curve, visibly faceted on a tight roundabout, so the ring needs a bigger radius to stay smooth-looking
RA_ACCESS = 40.0  # matches build_scene.py's own __main__ grid demo (_demo_round_about) exactly
RA_ALPHA = 24.0
RA_MERGE_RADIUS = 16.0  # bumped from 8, same reason (the access-to-ring bend arcs)
RA_IN_K = 1   # gap facing back north, toward TW2 (see build_scene.round_about: k=1 is north-facing)
RA_OUT_K = 2  # gap facing west -- a quarter-turn around the ring

# Short, not the reference highway_env MergeEnv's own 150/80/80/150 (what
# this used to be, matching build_scene.py's own __main__ grid demo exactly)
# -- that scale reads fine as a standalone demo of merge() in isolation, but
# is wildly disproportionate chained after four tightly-spaced junctions
# (GAP=12, ~25-47m access roads); real_001 wants short gap to short gap the
# whole way, merge included.
#
# MG_TAPER=40, not equal to MG_MERGE like the rest, so the taper (gentle
# lead-in) and the lane-change (sharp finish) read as two visibly
# different moves, not two similar bends blurring together.
MG_BEFORE, MG_TAPER, MG_MERGE, MG_AFTER = 20.0, 40.0, 20.0, 30.0
# The taper's own curve, rendered directly from highway_env's own real
# MergeEnv (not this project's own port -- literally instantiated and
# rendered to check, since "smooth" is hard to reason about from the
# formula alone) is nowhere near a visible S-wave: at its own real scale
# (amplitude=3.25 over taper_length=80) it reads as ALMOST a straight
# diagonal, curving noticeably only right at the very end where it
# levels out parallel to the highway. That's a LOW amplitude-to-length
# ratio (3.25/80 = 0.0406), not a special curve shape -- this codebase's
# own default amplitude=3.25 kept at MG_TAPER=40 (half the reference's own
# 80) is double that ratio, a visibly tighter bend than the reference's
# own, which is what still read as "weird" even after the length fix
# above. Scaling amplitude down to match the reference's own ratio at
# THIS taper_length (not the reference's own absolute amplitude, which
# would be disproportionate at half the length) reproduces the same
# "almost straight, curves into place" look without changing MG_TAPER's
# own length at all.
MG_TAPER_AMPLITUDE = 3.25 / 80 * MG_TAPER
# The merge's own main highway -- bidirectional=True (2 lanes each way,
# matching N_TW2/RA's own n_lanes=2 convention exactly: EVERY primitive in
# this scene is "n_lanes each direction," 4 total physical lanes, not a
# mix of "n_lanes total, one-way" for merge alone. The reference's own
# one-way-only design (bidirectional=False) was fine chained onto a
# single-lane-out roundabout, but reads as an unmatched, oddly-narrower
# road once the roundabout itself became 2-lanes-each-way.
MG_LANES = N_TW2

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

    # TW2's south exit -> roundabout, entering from the north gap. TW2 and
    # the roundabout now carry the SAME lane count (both 2, matching
    # build_scene.py's own __main__ grid demo exactly -- see FW_ACCESS's
    # own comment above), so bridge_roundabout(n=N_TW2) connects every lane
    # 1:1 (TW2 lane 0 <-> ra lane 0, lane 1 <-> lane 1) instead of the
    # single asymmetric pair (TW2's own lane 1 down into the roundabout's
    # only lane) an earlier, single-lane version of this roundabout needed
    # -- see bridge_roundabout's own docstring on why n=1 there was a
    # genuine narrowing, not a bug, and why n>1 is correct once the lane
    # counts actually match. The route itself keeps riding lane 1 (the
    # outer lane, same as through both T-junctions) straight through, on
    # the roundabout's own lane 1 now instead of dropping to lane 0.
    #
    # exit_far_center, not exit_far -- placing ra_center directly off
    # exit_far's own lane-1 point (as an earlier version of this did)
    # anchors the roundabout's own axis to TW2's lane-1 line, not its true
    # centerline, offsetting the WHOLE roundabout by that lane's own
    # turn_lat. Invisible with the single-lane bridge this used to be (one
    # line can silently absorb any angle); with both lanes now bridging in
    # parallel (n=N_TW2 just above) that offset compounds into a visible
    # diagonal twist across both. exit_far_center undoes lane 1's own
    # lateral offset first, so ra_center lands on TW2's real axis and both
    # lanes bridge in exactly, perfectly straight.
    #
    # ra_center satisfies BOTH exact alignments at once, not just one:
    # the roundabout's own NORTH gap (k=1) and EAST gap (k=0) are always
    # exactly vertical and exactly horizontal respectively (fixed cardinal
    # directions from round_about's own center, unconditionally) -- and
    # TW2's own south exit is always exactly vertical too, while FW's own
    # west arm is always exactly horizontal too (add_four_way/add_three_way
    # corners are always cardinal, never dependent on incoming heading).
    # Aligning ra_center's own X to TW2's true centerline and its own Y to
    # FW's true centerline independently -- rather than deriving ra_center
    # from only ONE of the two and letting the other come out wherever it
    # happens to land -- lands it EXACTLY on both axes simultaneously,
    # because the two constraints are on different coordinates and don't
    # conflict: verified directly, an earlier version derived ra_center
    # from FW alone and left the TW2 side ~3.5m off its own true axis
    # (small, but a real diagonal, not the "exactly straight" bridge
    # bridge_roundabout(n>1) needs to avoid a twist -- see its own
    # docstring, and exit_far_center's own, for why ANY nonzero axis
    # offset compounds across multiple lanes bridged in parallel).
    tw2_far, _ = exit_far_center(net, "tw2_", 0, 1, LANE_WIDTH)
    fw_west_far, _ = exit_far_center(net, "fw_", 1, 0, LANE_WIDTH)
    ra_center = np.array([tw2_far[0], fw_west_far[1]])
    round_about(net, center=ra_center, radius=RA_RADIUS, n_lanes=N_TW2, access_length=RA_ACCESS,
                alpha=RA_ALPHA, merge_radius=RA_MERGE_RADIUS, prefix="ra_")
    bridge_roundabout(net, "ra_", RA_IN_K, "tw2_", 0, j_lane_i=0, ra_lane_i=0, n=N_TW2)

    # New direct link: the roundabout's own east gap (k=0) <-> FW's west
    # arm (corner=1) -- background-traffic connectivity only (neither
    # HUMAN_ROUTE nor ROBOT_ROUTE ever uses FW's west arm, so
    # prune_to_route's own heading-aware pruning never touches this; see
    # its own docstring). Both sides carry N_FW==N_TW2 lanes, so
    # bridge_roundabout(n=N_FW) connects every lane 1:1, exactly straight,
    # same as the TW2 connection above -- "stitched together perfectly."
    bridge_roundabout(net, "ra_", 0, "fw_", 1, j_lane_i=0, ra_lane_i=0, n=N_FW)

    # A quarter-turn around the ring (north gap -> west gap) -> highway
    # merge. n_lanes=MG_LANES==N_TW2: the roundabout and the merge's own
    # forward direction now carry the SAME lane count, so (like the TW2
    # and FW connections above) the normal lanes bridge exactly, every
    # lane 1:1, not just a single connection.
    #
    # mg_center needs the same centerline correction ra_center's own
    # comment explains above -- round_about doesn't have its own
    # exit_far_center (that's built for add_four_way/add_three_way's own
    # il/o node naming), so this re-derives the same math directly: undo
    # lane 1's own turn_lat lateral offset to recover the gap's true
    # center axis. With only a single lane bridging (as it was before),
    # this offset was invisible (see exit_far_center's own docstring on
    # why); bridging N_TW2 lanes in parallel below is exactly the case
    # where it would otherwise show up as a twist.
    ra_out_lane = net.get_lane((f"ra_bendout{RA_OUT_K}_1", f"ra_farout{RA_OUT_K}_1", 0))
    ra_out_far = ra_out_lane.position(ra_out_lane.length, 0)
    ra_out_heading = ra_out_lane.heading_at(ra_out_lane.length)
    ra_out_direction = np.array([np.cos(ra_out_heading), np.sin(ra_out_heading)])
    ra_out_lateral = np.array([-ra_out_direction[1], ra_out_direction[0]])
    ra_out_turn_lat = LANE_WIDTH / 2 + 1 * LANE_WIDTH  # lane 1's own offset, to undo
    ra_out_center_far = ra_out_far - ra_out_turn_lat * ra_out_lateral
    mg_center = ra_out_center_far + GAP * ra_out_direction
    merge(net, center=mg_center, heading_deg=np.degrees(ra_out_heading), n_lanes=MG_LANES,
          before_length=MG_BEFORE, taper_length=MG_TAPER, merge_length=MG_MERGE, after_length=MG_AFTER,
          ramp_side=1, ramp_gap=0.0, amplitude=MG_TAPER_AMPLITUDE, bidirectional=True, prefix="mg_")

    # The normal lanes: every one of the roundabout's own west-gap lanes
    # bridged straight into the highway's own matching "before" lane
    # (mg_a_i), same per-lane line-type convention bridge_roundabout's own
    # n>1 case uses. This is ordinary background-traffic connectivity for
    # the highway itself, same as the TW2 and FW connections.
    for i in range(N_TW2):
        edge_left = S if i != 0 else C
        edge_right = C if i == N_TW2 - 1 else S
        ra_i_lane = net.get_lane((f"ra_bendout{RA_OUT_K}_{i}", f"ra_farout{RA_OUT_K}_{i}", 0))
        ra_i_far = ra_i_lane.position(ra_i_lane.length, 0)
        mg_a_i_pos = net.get_lane((f"mg_a_{i}", f"mg_b_{i}", 0)).position(0, 0)
        bridge(net, f"ra_farout{RA_OUT_K}_{i}", f"mg_a_{i}", ra_i_far, mg_a_i_pos,
               line_types=(edge_left, edge_right))

    # The return direction: merge()'s own reverse lanes (mg_rd_i, arriving
    # FROM the highway back toward the roundabout) bridged into the
    # roundabout's own entry (ra_farin{RA_OUT_K}_i) at this SAME gap --
    # without this, the reverse lanes just stop short of the roundabout by
    # exactly GAP (verified directly: mg_rd_0 ends 12m short of
    # ra_farin2_0, same Y both ends, just missing the bridge), a real gap
    # in the road surface, not merely an unconnected background lane
    # elsewhere. Mirrors bridge_roundabout's own two-direction pattern for
    # the TW2 and FW connections above.
    for i in range(N_TW2):
        edge_left = S if i != 0 else C
        edge_right = C if i == N_TW2 - 1 else S
        mg_rd_i_lane = net.get_lane((f"mg_ra_{i}", f"mg_rd_{i}", 0))
        mg_rd_i_pos = mg_rd_i_lane.position(mg_rd_i_lane.length, 0)
        ra_in_i_pos = net.get_lane((f"ra_farin{RA_OUT_K}_{i}", f"ra_bendin{RA_OUT_K}_{i}", 0)).position(0, 0)
        bridge(net, f"mg_rd_{i}", f"ra_farin{RA_OUT_K}_{i}", mg_rd_i_pos, ra_in_i_pos,
               line_types=(edge_left, edge_right))

    # The merge lane (ramp) is NOT bridged to the roundabout at all -- only
    # the two normal lanes above connect the roundabout to the highway,
    # exactly, lane for lane. The ramp stays exactly as merge() itself
    # built it: disconnected upstream, originating from nothing, the same
    # way the reference highway_env MergeEnv's own ramp is populated by a
    # vehicle spawned directly onto it (see merge()'s own docstring) rather
    # than driven there from elsewhere -- it disappears after merging
    # either way, so where it came from was never load-bearing.
    #
    # It still visibly merges INTO the highway a little further on -- ONE
    # lane-change starting directly off "mg_b_ramp" (the ramp taper's own
    # endpoint), same lane_change_to formula build_scene.py's own
    # __main__ demo (_demo_merge) uses for exactly this. This lane isn't
    # part of either route (see above -- it's a visual-only "ramp merges
    # into the highway" flourish), so it's free to run the rest of the way
    # down the highway -- through both the merge zone and the after zone,
    # landing on "mg_d_1" -- rather than stopping at "mg_c_1": the same
    # lateral shift spread over more distance reads as a visibly gentler
    # curve, the same reason the true highway_env reference's own long
    # taper (3.25m over 80m) reads as almost straight.
    ramp_taper = net.get_lane(("mg_k", "mg_b_ramp", 0))
    p0 = ramp_taper.position(ramp_taper.length, 0)
    h0 = ramp_taper.heading_at(ramp_taper.length)
    highway_outer = net.get_lane(("mg_c_1", "mg_d_1", 0))
    target = highway_outer.position(highway_outer.length, 0)
    lane_change_to(net, "mg_b_ramp", "mg_d_1", p0, h0, target, LANE_WIDTH, line_types=(S, S))

    # merge()'s own original mg_b_ramp->mg_c_ramp (running straight past
    # the lane-change above, parallel to the highway, then just stopping)
    # is superseded by the lane-change: without deleting it, the same
    # phantom "path that continues after the merge" build_scene.py's own
    # __main__ demo (_demo_merge) hit and fixed the same way -- two lanes
    # occupying the same stretch from the same node, only one of which
    # (the lane-change) actually goes anywhere once the ramp is
    # disconnected upstream, since prune_to_route only prunes alternatives
    # at nodes the ROUTE visits, and the route never visits "mg_b_ramp"
    # now that it rides the normal lanes instead. Delete just that one
    # edge (not the node itself -- mg_b_ramp still needs to exist as the
    # FROM end of the lane-change above).
    del net.graph["mg_b_ramp"]["mg_c_ramp"]

    # Lane change #1: FW's own north exit (0 -> 1), landing on lane 1's own
    # pre-built far point so bridge_corners' own lane-1 bridge -- and then
    # lane 1's own complete pre-built turn geometry through both
    # T-junctions -- carries the route the rest of the way to the
    # roundabout without any further changes in between.
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
    ("tw1_il1_1", "tw1_o1_1", 0), ("tw1_o1_1", "tw2_o3_1", 0),    # bridge, lane 1
    ("tw2_o3_1", "tw2_ir3_1", 0), ("tw2_ir3_1", "tw2_il0_1", 0),  # left turn onto the south arm, lane 1
    ("tw2_il0_1", "tw2_o0_1", 0), ("tw2_o0_1", f"ra_farin{RA_IN_K}_1", 0),
    (f"ra_farin{RA_IN_K}_1", f"ra_bendin{RA_IN_K}_1", 0), (f"ra_bendin{RA_IN_K}_1", f"ra_entry{RA_IN_K}_1", 0),
    (f"ra_entry{RA_IN_K}_1", f"ra_exit{RA_OUT_K}_1", 0),
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
