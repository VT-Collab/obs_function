"""
General functions for building up the scene
"""

import os
import numpy as np
import pygame
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, CircularLane, SineLane, LineType, AbstractLane
from highway_env.road.graphics import RoadGraphics, WorldSurface


LANE_WIDTH = AbstractLane.DEFAULT_WIDTH  # 4.0
N, C, S = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

# The "+5" every turn radius in this file is built from -- exported so every
# real_0XX_rebuilt.py's own local RIGHT_TURN_RADIUS_BASE = LANE_WIDTH + 5 (used
# to position downstream junctions relative to this one's own turn geometry)
# reads it from here instead of hardcoding its own separate copy of "5".
#
# lane_width + 5 = 9m at the default lane_width -- already a realistic,
# roughly AASHTO-standard curb-return radius for an arterial/collector road
# (real designs run ~4.5m for a tight residential corner up to ~9-15m for a
# bigger arterial), not an arbitrary placeholder. It IS too tight to render
# perfectly smoothly at extreme close-up: highway_env's own LaneGraphics
# renders any curve as straight chords spaced STRIPE_SPACING=4.33m apart, so
# a 90-degree turn at 9m radius gets only ~3 chords, visibly kinked once
# zoomed in far enough. Tried fixing that by growing the radius instead (20,
# then even 12): both moved AWAY from realistic proportions rather than
# toward them, and 20 was bigger than this project's own ~25m access roads,
# swallowing whole intersections into sprawling curves once every real_0XX
# scene's downstream junction re-tuned around it (confirmed directly).
# Left at the reference's own value: trading a real road's own realistic
# scale for a rendering-only artifact that's invisible at every zoom level
# this project actually views scenes at isn't a trade worth making.
RIGHT_TURN_RADIUS_EXTRA = 5.0


def add_four_way(net, center, n_vertical, n_horizontal, access_length=100.0, lane_width=LANE_WIDTH, prefix=""):
    """
    A standard 4-way intersection: highway_env's own reference construction
    (highway_env.envs.intersection_env.IntersectionEnv._make_road -- per
    lane, a right turn, a left turn, and a straight movement, all starting
    from the same point), generalized to a configurable number of lanes
    per road.

    Every lane gets its own independent circle for each turn (re-running
    the reference's own single-lane formula at that lane's own lateral
    position), rather than sharing one circle across a whole corner --
    a shared circle forces the outer lanes' radius toward zero, shrinking
    their turn to a barely-visible point. Independent circles also mean
    every lane can reach every direction, not just a single outer lane:
    with only one lane able to turn, the rest would have no route onto the
    cross street, making a multi-lane intersection a dead end.

    n_vertical: lane count for the South and North approaches.
    n_horizontal: lane count for the West and East approaches. When the two
    differ, a lane whose index has no counterpart on the road it's turning
    onto (e.g. lane 3 of a 4-lane road turning onto a 3-lane road) lands on
    that road's outermost lane instead (clamped).

    Corners are numbered 0=South, 1=West, 2=North, 3=East. Turning from
    South: right -> East, left -> West, straight -> North. Lane index 0 is
    closest to the centerline, index n-1 is the curb lane.
    """
    lanes = {0: n_vertical, 1: n_horizontal, 2: n_vertical, 3: n_horizontal}
    cx, cy = center
    right_turn_radius_base = lane_width + RIGHT_TURN_RADIUS_EXTRA

    rotation = {c: np.array([[np.cos(np.radians(90 * c)), -np.sin(np.radians(90 * c))],
                              [np.sin(np.radians(90 * c)), np.cos(np.radians(90 * c))]]) for c in range(4)}

    for corner in range(4):
        n = lanes[corner]
        angle = np.radians(90 * corner)
        rot = rotation[corner]
        prev_c, next_c, opp_c = (corner - 1) % 4, (corner + 1) % 4, (corner + 2) % 4
        n_prev, n_next = lanes[prev_c], lanes[next_c]
        # od(i) (below) grows with lane index -- necessary for the turn
        # circles (an outer lane's own right/left turn needs a bigger
        # radius), but it also anchors where "access_length further out"
        # is MEASURED FROM. Left uncorrected, that means access_length
        # away from a DIFFERENT stop-line per lane, so the far ends of a
        # multi-lane arm land at different distances from center instead
        # of a flat, perpendicular cut -- verified directly (every lane's
        # own far point, an exact affine function of its own od(i), for
        # n=2: 51.0m vs 55.3m from center, not the same 51.0m). od_far
        # fixes the reference to the OUTERMOST lane's own od (the one
        # that already sits furthest out) so every lane's own far point
        # wraps around the SAME distance, while od(i) itself -- and every
        # turn circle built from it -- stays exactly as it was.
        od_far = right_turn_radius_base + (lane_width / 2 + (n - 1) * lane_width)

        for i in range(n):
            turn_lat = lane_width / 2 + i * lane_width
            # od (this lane's stop-line distance) and right_turn_radius are
            # the reference's own single-lane formula, evaluated at this
            # lane's own turn_lat instead of a fixed lane_width / 2 -- an
            # independent circle per lane, not shared across the corner.
            # od depends only on the lane index, so the target road gets
            # the same value at the same index, which keeps both turns
            # exactly tangent regardless of which two roads are meeting.
            od = right_turn_radius_base + turn_lat
            right_turn_radius = right_turn_radius_base
            left_turn_radius = od + turn_lat

            o, ir = f"{prefix}o{corner}_{i}", f"{prefix}ir{corner}_{i}"
            # Only the true curb (the outermost lane's outer edge) is solid;
            # every other line -- including the boundary against the
            # opposing direction's own lanes -- is a dashed divider.
            edge_left, edge_right = S, (C if i == n - 1 else S)

            start = np.array([cx, cy]) + rot @ np.array([turn_lat, access_length + od_far])
            end = np.array([cx, cy]) + rot @ np.array([turn_lat, od])
            net.add_lane(o, ir, StraightLane(start, end, width=lane_width, line_types=[edge_left, edge_right]))

            # right turn: lands on the previous road's own index i --
            # clamped to its outermost lane if that index doesn't exist
            # there (only possible when the two roads have different lane
            # counts; exact tangency needs the un-clamped case, i == j).
            j = min(i, n_prev - 1)
            r_center = np.array([cx, cy]) + rot @ np.array([od, od])
            net.add_lane(ir, f"{prefix}il{prev_c}_{j}",
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      width=lane_width, line_types=[N, C if i == n - 1 else N]))

            # left turn: lands on the next road's own index i (clamped likewise).
            j = min(i, n_next - 1)
            l_center = np.array([cx, cy]) + rot @ np.array([-od, od])
            net.add_lane(ir, f"{prefix}il{next_c}_{j}",
                         CircularLane(l_center, left_turn_radius, angle, angle + np.radians(-90), clockwise=False,
                                      width=lane_width, line_types=[N, N]))

            # straight: same index on the opposite approach, which always
            # has the same lane count (it's the same road) and therefore
            # the same od(i) too.
            start_s = np.array([cx, cy]) + rot @ np.array([turn_lat, od])
            end_s = np.array([cx, cy]) + rot @ np.array([turn_lat, -od])
            net.add_lane(ir, f"{prefix}il{opp_c}_{i}",
                         StraightLane(start_s, end_s, width=lane_width, line_types=[edge_left, N]))

            # exit: this corner's own mirrored (-turn_lat) lane.
            start_e = np.array([cx, cy]) + rot @ np.array([-turn_lat, od])
            end_e = np.array([cx, cy]) + rot @ np.array([-turn_lat, access_length + od_far])
            net.add_lane(f"{prefix}il{corner}_{i}", o,
                         StraightLane(start_e, end_e, width=lane_width, line_types=[edge_left, edge_right]))


def add_three_way(net, center, n_stem, n_cross, access_length=100.0, lane_width=LANE_WIDTH,
                   prefix="", missing_corner=2):
    """
    A 3-way (T) intersection: exactly add_four_way's own per-lane
    construction, with one of the four corners simply absent. Whichever
    movement would have gone to or from that missing arm is just not
    created; the other three arms stay fully connected to each other (each
    crossing arm can turn onto the stem or go straight to the other
    crossing arm, and the stem can turn onto either crossing arm).

    missing_corner: which of add_four_way's corners (0=South, 1=West,
    2=North, 3=East) is absent. Default 2 (North) gives a T whose stem
    points South -- pass a different missing_corner to rotate it instead
    of re-deriving the geometry.
    n_stem: lane count for the stem (the arm opposite missing_corner).
    n_cross: lane count for the two arms either side of the missing one.
    """
    stem_corner = (missing_corner + 2) % 4
    corners = [c for c in range(4) if c != missing_corner]
    lanes = {c: (n_stem if c == stem_corner else n_cross) for c in corners}
    cx, cy = center
    right_turn_radius_base = lane_width + RIGHT_TURN_RADIUS_EXTRA

    rotation = {c: np.array([[np.cos(np.radians(90 * c)), -np.sin(np.radians(90 * c))],
                              [np.sin(np.radians(90 * c)), np.cos(np.radians(90 * c))]]) for c in range(4)}

    for corner in corners:
        n = lanes[corner]
        angle = np.radians(90 * corner)
        rot = rotation[corner]
        prev_c, next_c, opp_c = (corner - 1) % 4, (corner + 1) % 4, (corner + 2) % 4
        # See add_four_way's own docstring/comment on od_far: without it,
        # each lane's far end sits access_length from ITS OWN stop-line
        # (od(i), which grows with lane index for the turn circles), so a
        # multi-lane arm's far edge lands staggered instead of flat.
        od_far = right_turn_radius_base + (lane_width / 2 + (n - 1) * lane_width)

        for i in range(n):
            turn_lat = lane_width / 2 + i * lane_width
            od = right_turn_radius_base + turn_lat
            right_turn_radius = right_turn_radius_base
            left_turn_radius = od + turn_lat

            o, ir = f"{prefix}o{corner}_{i}", f"{prefix}ir{corner}_{i}"
            edge_left, edge_right = S, (C if i == n - 1 else S)

            start = np.array([cx, cy]) + rot @ np.array([turn_lat, access_length + od_far])
            end = np.array([cx, cy]) + rot @ np.array([turn_lat, od])
            net.add_lane(o, ir, StraightLane(start, end, width=lane_width, line_types=[edge_left, edge_right]))

            if prev_c in lanes:  # right turn -- omitted if it would target the missing arm
                j = min(i, lanes[prev_c] - 1)
                r_center = np.array([cx, cy]) + rot @ np.array([od, od])
                net.add_lane(ir, f"{prefix}il{prev_c}_{j}",
                             CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                          width=lane_width, line_types=[N, C if i == n - 1 else N]))

            if next_c in lanes:  # left turn -- omitted likewise
                j = min(i, lanes[next_c] - 1)
                l_center = np.array([cx, cy]) + rot @ np.array([-od, od])
                net.add_lane(ir, f"{prefix}il{next_c}_{j}",
                             CircularLane(l_center, left_turn_radius, angle, angle + np.radians(-90), clockwise=False,
                                          width=lane_width, line_types=[N, N]))

            if opp_c in lanes:  # straight -- omitted when the opposite arm is the missing one
                # Unlike add_four_way's straight movement, this uses the
                # full [edge_left, edge_right] treatment (not [edge_left,
                # N]): a T has no real intersection box on the far side of
                # this crossbar, so there's nothing to justify blanking out
                # its outer curb -- it should just stay a normal, fully
                # marked through-road all the way across.
                start_s = np.array([cx, cy]) + rot @ np.array([turn_lat, od])
                end_s = np.array([cx, cy]) + rot @ np.array([turn_lat, -od])
                net.add_lane(ir, f"{prefix}il{opp_c}_{i}",
                             StraightLane(start_s, end_s, width=lane_width, line_types=[edge_left, edge_right]))

            start_e = np.array([cx, cy]) + rot @ np.array([-turn_lat, od])
            end_e = np.array([cx, cy]) + rot @ np.array([-turn_lat, access_length + od_far])
            net.add_lane(f"{prefix}il{corner}_{i}", o,
                         StraightLane(start_e, end_e, width=lane_width, line_types=[edge_left, edge_right]))

def _bend(start, heading, radius, angle_change):
    """
    (center, radius, start_phase, end_phase, clockwise) for a CircularLane
    that starts at `start` heading `heading` (radians) and turns by
    `angle_change` (radians, signed) -- the general version of the fixed
    90-degree turns used elsewhere in this file.

    Derived from CircularLane's own heading_at (phase + pi/2 * direction)
    and length (radius * (end_phase - start_phase) * direction) formulas:
    those two force end_phase - start_phase == angle_change exactly
    (independent of direction), and force clockwise = angle_change > 0
    for the length to come out positive.
    """
    clockwise = angle_change > 0
    sign = 1.0 if clockwise else -1.0
    normal = np.array([np.cos(heading + sign * np.pi / 2), np.sin(heading + sign * np.pi / 2)])
    center = np.array(start) + radius * normal
    start_phase = heading - sign * np.pi / 2
    end_phase = start_phase + angle_change
    return center, radius, start_phase, end_phase, clockwise


def _solve_bend_radius(p_start, h_start, angle_change, axis_origin, lateral, target_offset):
    """The radius R such that _bend(p_start, h_start, R, angle_change)'s own
    far endpoint lands EXACTLY `target_offset` meters along `lateral` from
    axis_origin -- so the bend can be aimed directly at the lane's own
    final, safely-separated position instead of landing wherever it
    naturally falls and needing a second stage to correct it afterward.

    The far endpoint's own offset along `lateral` is an EXACTLY affine
    (linear) function of R: _bend's own center is p_start + R*(a fixed unit
    vector depending only on h_start and angle_change's sign), and the far
    point is that center plus R*(another fixed unit vector depending only
    on angle_change) -- both terms scale with R alone, nothing else varies
    with it. Verified directly by sampling multiple R and confirming a
    constant slope. Two sample radii therefore determine the line exactly;
    no iterative solver, no risk of it not converging.
    """
    def offset_at(R):
        c, r, sp, ep, _ = _bend(p_start, h_start, R, angle_change)
        p_far = c + r * np.array([np.cos(ep), np.sin(ep)])
        return np.dot(p_far - axis_origin, lateral)

    r1, r2 = 10.0, 30.0
    o1, o2 = offset_at(r1), offset_at(r2)
    slope = (o2 - o1) / (r2 - r1)
    return max(r1 + (target_offset - o1) / slope, 1.0)


def round_about(net, center, radius, n_lanes=2, access_length=100.0, lane_width=LANE_WIDTH,
                 alpha=24.0, merge_radius=30.0, prefix=""):
    """
    A roundabout: a generalized port of highway_env's own
    RoundaboutEnv._make_road. n_lanes concentric rings (radius stepped by
    lane_width per lane, same as the reference), each split at every
    cardinal direction into an "exit" point (alpha degrees before the
    direction) and an "entry" point (alpha degrees after it) -- the
    reference's own 8-arc-per-lane ring, instead of one point per
    direction: traffic circulating the ring passes exit_k then entry_k in
    that order (matching the reference's own se/ex/ee/nx/... sequence), a
    leaving car peels off at exit_k, and a new car merges in at entry_k.

    Line types match the reference's own `line = [[c, s], [n, c]]`: the
    innermost ring's inner edge (the island curb) and the outermost ring's
    outer edge (the true curb) are solid; every edge between two rings is
    a dashed divider, drawn once and left blank on the matching edge of
    the neighboring lane so it isn't drawn twice.

    Each cardinal side gets 2*n_lanes lanes total: n_lanes entering (merging
    in) and n_lanes leaving (peeling off), each its own physically separate,
    one-way lane -- never one lane object serving both directions, and
    never crossing paths with each other (see below). Getting from the
    ring's own tangent heading at entry_k/exit_k back to parallel with the
    cardinal axis uses a plain CircularLane bend (see _bend): each lane
    bends by exactly (90 - alpha) degrees, same as ever -- that part of the
    geometry isn't what was wrong.

    What was wrong: that bend's own two ends (entry and exit, mirrored
    around the cardinal axis at +/-alpha degrees on the ring) do NOT
    reliably end up a safe lane_width apart just because the geometry is
    mirrored -- verified directly by sweeping alpha and merge_radius: the
    resulting separation crosses back through zero at multiple seemingly-
    reasonable parameter combinations (it's a real geometric
    near-cancellation, not a rare edge case), so at the default alpha=24/
    merge_radius=16 combination used throughout this codebase's own
    scenes, entry and exit ended up only ~1.1m apart -- less than a real
    vehicle's own width, which is exactly what "cars butting against each
    other" looks like: a car merging in and a car leaving at the same gap,
    running nearly head-on for the whole access road. Tuning alpha/
    merge_radius to some OTHER combination that happens to test out safe
    isn't a real fix either (the same sweep shows plenty of OTHER
    seemingly-reasonable combinations that are just as broken).

    An earlier version left the bend's own natural (too-close) far point
    alone and added a SECOND stage afterward -- an explicit SineLane
    shifting each lane out to a safe offset before running straight the
    rest of the way. That worked geometrically (verified: tangent, exactly
    lane_width apart) but LOOKED wrong: entry and exit still visibly
    pinched together right where the first stage's own natural, too-close
    point was, before opening back out -- a "twist" instead of two lanes
    that just stay apart the whole time, the opposite of "smooth, natural,
    separated." The actual fix -- _solve_bend_radius, above -- aims the
    ORIGINAL bend itself at the final safe offset directly, by solving for
    the one merge_radius (per lane, per direction) that lands it there
    exactly. One continuous curve from the ring to the far edge, always
    lane_width apart or more along its entire length, nothing to pinch.

    Which SIDE each lane's target offset sits on isn't a free choice
    either: it has to land on the same +lateral side add_four_way's own
    approach/exit lanes always use (see add_four_way's own docstring --
    verified directly there: an outer lane's larger offset is always
    reached by walking further in THAT lane's own +lateral direction,
    never a "whichever side is closer" choice). Picking the sign from the
    bend's own natural (tiny) offset instead doesn't reliably agree with
    the adjacent junction's own fixed convention, and produces a
    roundabout-to-junction bridge that cuts across at a steep diagonal
    instead of running parallel to its own partner lane -- so
    target_offset below is always +turn_lat, unconditionally, in each
    lane's own lateral_in/lateral_out frame, exactly add_four_way's rule,
    never a per-gap guess.

    This is what farin/bendin/entry and exit/bendout/farout actually
    resolve to now; every caller that queries farin*/farout*'s own
    position (to bridge a neighboring junction to the roundabout) picks up
    the corrected, safely-separated point automatically, with no changes
    needed on its own end.
    """
    cx, cy = np.array(center)
    bend_angle = np.radians(90 - alpha)

    def point_at(angle_rad, r):
        return np.array([cx, cy]) + r * np.array([np.cos(angle_rad), np.sin(angle_rad)])

    def tangent_at(angle_rad):
        # Heading of ring traffic at this phase (see heading_at: phase +
        # pi/2 * direction, direction=-1 for clockwise=False below).
        return angle_rad - np.pi / 2

    for lane_i in range(n_lanes):
        r = radius + lane_i * lane_width
        edge = [C if lane_i == 0 else N, C if lane_i == n_lanes - 1 else S]
        for k in range(4):
            base_deg = -90 * k             # kept un-wrapped (not mod 360) so every arc's
            next_base_deg = -90 * (k + 1)  # phase keeps decreasing -- see the note below.
            a_exit = np.radians(base_deg + alpha)
            a_entry = np.radians(base_deg - alpha)
            a_next_exit = np.radians(next_base_deg + alpha)
            # CircularLane.length = radius * (end_phase - start_phase) *
            # direction -- both arcs go from a higher phase to a lower one,
            # which needs clockwise=False (direction=-1) to come out
            # positive; clockwise=True here would give a negative length,
            # which the renderer clips into nothing.
            net.add_lane(f"{prefix}exit{k}_{lane_i}", f"{prefix}entry{k}_{lane_i}",
                         CircularLane([cx, cy], r, a_exit, a_entry, clockwise=False,
                                      width=lane_width, line_types=edge))
            net.add_lane(f"{prefix}entry{k}_{lane_i}", f"{prefix}exit{(k + 1) % 4}_{lane_i}",
                         CircularLane([cx, cy], r, a_entry, a_next_exit, clockwise=False,
                                      width=lane_width, line_types=edge))

    for k in range(4):
        base_deg = -90 * k
        base_rad = np.radians(base_deg)
        a_entry = np.radians(base_deg - alpha)
        a_exit = np.radians(base_deg + alpha)
        axis_origin = point_at(base_rad, 0.0)
        h_far_entry = base_rad + np.pi  # verified: the bend's own far end is exactly axis-parallel, heading toward the ring
        h_far_exit = base_rad           # verified: exactly axis-parallel, heading away from the ring
        direction_in = np.array([np.cos(h_far_entry), np.sin(h_far_entry)])
        lateral_in = np.array([-direction_in[1], direction_in[0]])
        direction_out = np.array([np.cos(h_far_exit), np.sin(h_far_exit)])
        lateral_out = np.array([-direction_out[1], direction_out[0]])

        # Two passes, same reason add_four_way/add_three_way compute their
        # own od_far: each lane's bend lands its far point (p_far_entry/
        # p_far_exit) at a different depth along direction_in/out
        # (radius_in/radius_out isn't the same per lane -- see
        # _solve_bend_radius above), so cutting every lane's straight
        # segment access_length back from ITS OWN far point leaves the
        # arm's far edge staggered instead of a flat, perpendicular cut
        # (verified directly at this codebase's own alpha=24/
        # merge_radius=16, n_lanes=2: lane 0 and lane 1 landed 4.0m apart
        # along direction_in, not flush). Pass 1 computes every lane's own
        # bend geometry and how deep ITS OWN natural far point sits; pass 2
        # flushes every lane's actual far endpoint to the DEEPEST of those
        # (mirroring od_far's own use of the outermost lane's od: always
        # the value that keeps every lane's straight portion at least as
        # long as it naturally needs, never shorter) while keeping that
        # lane's own exact turn_lat lateral offset -- so the straight
        # segment stays parallel to direction_in/out, just cut flush at a
        # common depth instead of each lane's own.
        lane_geom = []
        for lane_i in range(n_lanes):
            r = radius + lane_i * lane_width
            edge_left, edge_right = S, (C if lane_i == n_lanes - 1 else S)
            turn_lat = lane_width / 2 + lane_i * lane_width  # add_four_way's own approach/exit separation convention

            # Merging in: the real (forward) arc runs from far away, heading
            # radially inward, bending to the ring's own tangent heading at
            # entry_k. _bend only derives an arc FORWARD from a known start,
            # and the far point isn't known yet -- so solve it backward
            # instead, starting at p_entry with heading reversed (h_entry +
            # pi) and turning by -bend_angle (a reversed turn is the
            # negative of the forward one). That gives the same physical
            # arc; p_entry is then this backward arc's OWN start (sp), and
            # the far point is its end (ep). The real forward lane is the
            # same center/radius with start/end swapped and clockwise
            # flipped (traversing one arc in opposite directions flips its
            # effective direction).
            #
            # Radius: ALWAYS _solve_bend_radius's own exact-target radius --
            # by construction it lands this lane's own far offset at EXACTLY
            # turn_lat (see its own docstring), the same distance
            # add_four_way's approach/exit lanes sit at, so entry and exit
            # here end up exactly as far apart as an intersection's own
            # innermost lanes (touching at the centerline, nothing more) --
            # not just "at least" that far apart. An earlier version also
            # capped this with right_turn_radius_base (preferring that fixed
            # radius so the bend reads as the same kind of curve as every
            # other turn in the scene, whenever it was already safe) but
            # that meant whichever of the two was smaller (offset shrinks as
            # radius grows, so a smaller capped radius overshoots the
            # target) silently won per lane -- verified directly at this
            # codebase's own alpha=24/merge_radius=16: the outer lane's own
            # exact-solve radius (7.7) was already below right_turn_radius_
            # base (9.0) so it was unaffected, but the inner lane's exact-
            # solve radius (11.7) was above it, so right_turn_radius_base
            # won there instead and overshot turn_lat by nearly 2x (3.6m
            # instead of 2.0m) -- a visibly wider gap on the inner lane than
            # the outer one, and wider than add_four_way's own equivalent
            # gap. merge_radius stays as the caller's own upper bound (still
            # well above either lane's exact-solve radius at this
            # codebase's own parameters, so it isn't the binding constraint
            # here either) for whichever geometry might someday need it.
            p_entry = point_at(a_entry, r)
            h_entry = tangent_at(a_entry)
            radius_in = min(merge_radius,
                             _solve_bend_radius(p_entry, h_entry + np.pi, -bend_angle, axis_origin, lateral_in, turn_lat))
            center_, radius_, sp, ep, cw = _bend(p_entry, h_entry + np.pi, radius_in, -bend_angle)
            p_far_entry = center_ + radius_ * np.array([np.cos(ep), np.sin(ep)])
            # Measured along direction_out, NOT direction_in, even though
            # this is the entry (in) side -- direction_in == -direction_out
            # (opposite headings of the same physical arm axis), so a value
            # measured against direction_in would need its own sign flipped
            # before it could be compared against anything measured against
            # direction_out. Using direction_out for BOTH sides throughout
            # keeps every reach on one consistent scale: bigger reach_*
            # always means "farther from the ring," full stop, for either
            # side, so a plain max() below picks the true farthest point
            # (see p_in's own reconstruction further down for why this
            # still lands on p_far_entry's own lateral_in side, not out's).
            reach_in = np.dot(p_far_entry - axis_origin, direction_out) + max(access_length - radius_in, 2.0)

            # Leaving: bend from the ring's tangent heading at exit_k to
            # heading radially outward (away from the center), same radius
            # rule as entry above.
            p_exit = point_at(a_exit, r)
            h_exit = tangent_at(a_exit)
            radius_out = min(merge_radius,
                              _solve_bend_radius(p_exit, h_exit, bend_angle, axis_origin, lateral_out, turn_lat))
            center2, radius2, sp2, ep2, cw2 = _bend(p_exit, h_exit, radius_out, bend_angle)
            p_far_exit = center2 + radius2 * np.array([np.cos(ep2), np.sin(ep2)])
            reach_out = np.dot(p_far_exit - axis_origin, direction_out) + max(access_length - radius_out, 2.0)

            lane_geom.append((turn_lat, edge_left, edge_right, center_, radius_, sp, ep, cw, p_far_entry, reach_in,
                               center2, radius2, sp2, ep2, cw2, p_far_exit, reach_out))

        # One shared reach for BOTH the in and out side of this arm, not one
        # per side -- a real road's pavement ends at the same cross-section
        # for both directions of travel, matching how add_four_way's own
        # od_far is the SAME value for a corner's entry ("start") and exit
        # ("end_e") alike, not two separate per-direction cuts. Picking the
        # single farthest reach_in/reach_out across every lane on either
        # side (mirroring od_far's own use of the outermost lane's od)
        # keeps every lane's straight portion at least as long as it
        # naturally needed, never shorter.
        reach_far = max(max(g[9] for g in lane_geom), max(g[16] for g in lane_geom))

        for lane_i, (turn_lat, edge_left, edge_right, center_, radius_, sp, ep, cw, p_far_entry, reach_in,
                     center2, radius2, sp2, ep2, cw2, p_far_exit, reach_out) in enumerate(lane_geom):
            # p_far_entry's own lateral_in component is exactly turn_lat (by
            # construction -- see radius_in above), so reconstructing it as
            # axis_origin + reach*direction_out + turn_lat*lateral_in, for
            # ANY reach, reproduces the exact same lane, just cut at a
            # different point along direction_out -- verified directly this
            # matches the original (per-lane) p_in exactly when reach ==
            # this lane's own reach_in.
            p_in = axis_origin + reach_far * direction_out + turn_lat * lateral_in
            net.add_lane(f"{prefix}farin{k}_{lane_i}", f"{prefix}bendin{k}_{lane_i}",
                         StraightLane(p_in, p_far_entry, width=lane_width, line_types=[edge_left, edge_right]))
            net.add_lane(f"{prefix}bendin{k}_{lane_i}", f"{prefix}entry{k}_{lane_i}",
                         CircularLane(center_, radius_, ep, sp, clockwise=(not cw),
                                      width=lane_width, line_types=[edge_left, edge_right]))

            p_out = axis_origin + reach_far * direction_out + turn_lat * lateral_out
            net.add_lane(f"{prefix}exit{k}_{lane_i}", f"{prefix}bendout{k}_{lane_i}",
                         CircularLane(center2, radius2, sp2, ep2, clockwise=cw2,
                                      width=lane_width, line_types=[edge_left, edge_right]))
            net.add_lane(f"{prefix}bendout{k}_{lane_i}", f"{prefix}farout{k}_{lane_i}",
                         StraightLane(p_far_exit, p_out, width=lane_width, line_types=[edge_left, edge_right]))


def connect_junctions(net, prefix_a, center_a, access_length_a, corner_a,
                       prefix_b, center_b, access_length_b, corner_b, n, lane_width=LANE_WIDTH):
    """
    Join junction A's corner_a arm to junction B's corner_b arm with a
    two-way straight road, so a vehicle can actually drive from one
    junction into the other. corner_b should be (corner_a + 2) % 4 (the
    opposite direction), so B sits in the direction A's arm points and B's
    own arm points straight back.

    Every arm's own "o" node is used for BOTH directions of travel, at
    MIRRORED lateral positions (+turn_lat for the approach into that
    junction, -turn_lat for the exit out of it) -- so A's own o-node for
    corner_a isn't one physical point, and neither is B's for corner_b.
    What actually lines up between the two junctions is: A's exit (local
    -turn_lat) feeds B's approach (local +turn_lat), and B's exit (local
    -turn_lat) feeds A's approach (local +turn_lat) -- matching how
    add_four_way/add_three_way build "end_e" and "start" themselves.
    """
    right_turn_radius_base = lane_width + RIGHT_TURN_RADIUS_EXTRA
    rotation = {c: np.array([[np.cos(np.radians(90 * c)), -np.sin(np.radians(90 * c))],
                              [np.sin(np.radians(90 * c)), np.cos(np.radians(90 * c))]]) for c in range(4)}
    rot_a, rot_b = rotation[corner_a], rotation[corner_b]
    ca, cb = np.array(center_a), np.array(center_b)

    for i in range(n):
        turn_lat = lane_width / 2 + i * lane_width
        od = right_turn_radius_base + turn_lat  # same per-lane formula as add_four_way/add_three_way
        edge_left, edge_right = S, (C if i == n - 1 else S)

        p_a_exit = ca + rot_a @ np.array([-turn_lat, access_length_a + od])
        p_b_approach = cb + rot_b @ np.array([turn_lat, access_length_b + od])
        net.add_lane(f"{prefix_a}o{corner_a}_{i}", f"{prefix_b}o{corner_b}_{i}",
                     StraightLane(p_a_exit, p_b_approach, width=lane_width, line_types=[edge_left, edge_right]))

        p_b_exit = cb + rot_b @ np.array([-turn_lat, access_length_b + od])
        p_a_approach = ca + rot_a @ np.array([turn_lat, access_length_a + od])
        net.add_lane(f"{prefix_b}o{corner_b}_{i}", f"{prefix_a}o{corner_a}_{i}",
                     StraightLane(p_b_exit, p_a_approach, width=lane_width, line_types=[edge_left, edge_right]))


def merge(net, center, heading_deg, n_lanes, before_length=150.0, taper_length=80.0,
          merge_length=80.0, after_length=150.0, lane_width=LANE_WIDTH,
          ramp_gap=2.0, amplitude=3.25, ramp_side=1, bidirectional=False, prefix=""):
    """
    A highway on-ramp merge: a direct port of highway_env's own
    MergeEnv._make_road. n_lanes main-highway lanes run straight the whole
    way, with a separate ramp lane that starts far off to one side, eases
    in with a SineLane over `taper_length`, then runs parallel right next
    to the highway for `merge_length` before simply ending -- "add a lane,
    then end it" is exactly the mechanism: nothing in the road network
    forces the merge, the ramp lane is just a dead end, so whatever drives
    it has to change into the highway before running out of road (the
    reference marks that point with an Obstacle; see the return value).

    n_lanes: number of main-highway lanes (the reference always uses 2).
    before_length: how far the highway (and, in parallel, the ramp before
    it starts curving) runs before anything about the merge is visible.
    taper_length: length of the ramp's own SineLane, easing its heading
    to match the highway's.
    merge_length: length the ramp runs parallel to the highway after the
    taper, before it ends -- this is also where the highway's own curb
    lane gets a striped (crossable) edge instead of a solid one, matching
    the reference's line_type_merge.
    after_length: how far the highway continues past the end of the merge.
    ramp_gap: clearance between the highway's curb and the ramp's own lane
    once merged in.
    amplitude: how far the ramp's SineLane shifts laterally (its own
    curve, in addition to closing ramp_gap) -- the reference's default.
    ramp_side: +1 for the ramp joining beside the highway's highest-index
    (curb) lane, -1 for beside lane 0.
    heading_deg: compass-style heading (same convention as add_four_way's
    own `angle = 90 * corner`) of the direction of travel.

    bidirectional: False (default) is the reference's own one-way highway,
    exactly as before -- every existing caller of this function keeps this
    default and is completely unaffected. True adds n_lanes MORE lanes for
    the opposite direction, mirrored across the centerline (ly=0) on the
    -lateral side, spanning the same total before+taper+merge+after length
    as one plain uninterrupted road (no ramp, no zone-based line-type
    changes -- the ramp only ever attaches to one direction, same as a real
    highway on-ramp never adds a lane to BOTH carriageways at once). This
    also makes lane 0's own inner edge (nearest the centerline) striped
    instead of solid -- a real shared divider now exists there, matching
    add_four_way's own convention (only the true outermost lane's own
    outer edge is solid) instead of the one-way reference's own "lane 0's
    inner edge is a road edge" line type.

    Returns (obstacle_position, obstacle_heading) -- the point where the
    reference places an Obstacle marking the end of the merge lane. Add it
    yourself once the Road exists: Obstacle(road, position, heading).
    """
    cx, cy = np.array(center)
    angle = np.radians(heading_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    x0 = 0.0
    x1 = before_length + taper_length
    x2 = x1 + merge_length
    x3 = x2 + after_length
    merge_edge_i = (n_lanes - 1) if ramp_side > 0 else 0

    # lane_offset: 0 when one-way (bidirectional=False, the reference's own
    # convention -- lane 0 sits exactly ON the centerline, since there's no
    # opposing lane sharing it). lane_width/2 when bidirectional=True --
    # add_four_way/round_about's own convention (turn_lat = lane_width/2 +
    # i*lane_width) instead, since lane 0 now has a REAL opposing lane on
    # the other side of the centerline and needs to sit OFFSET from it, not
    # centered on top of it. Skipping this for the bidirectional case was a
    # real bug, not just a style mismatch: bridging into add_four_way/
    # round_about's own lanes (which always use the offset convention)
    # forced every OTHER lane's own bridge into a diagonal, since only
    # (at most) one lane index could ever land on the same axis at a time
    # otherwise -- verified directly, lane 1 of a 2-lane bidirectional
    # merge landed 2m off round_about's own lane 1 this way, a real
    # 170.5-degree bridge angle instead of the needed 180.
    lane_offset = lane_width / 2 if bidirectional else 0.0
    for i in range(n_lanes):
        ly = lane_offset + i * lane_width
        edge_left = S if (i != 0 or bidirectional) else C
        edge_right = C if i == n_lanes - 1 else S
        # During the merge zone, the lane adjacent to the ramp gets a
        # crossable (striped) edge on that side instead of solid --
        # exactly the reference's line_type_merge.
        merge_left = S if (i == 0 and merge_edge_i == 0) else edge_left
        merge_right = S if (i == n_lanes - 1 and merge_edge_i == n_lanes - 1) else edge_right

        a, b, c, d = (f"{prefix}a_{i}", f"{prefix}b_{i}", f"{prefix}c_{i}", f"{prefix}d_{i}")
        net.add_lane(a, b, StraightLane(cx_cy(cx, cy, rot, x0, ly), cx_cy(cx, cy, rot, x1, ly),
                                         width=lane_width, line_types=[edge_left, edge_right]))
        net.add_lane(b, c, StraightLane(cx_cy(cx, cy, rot, x1, ly), cx_cy(cx, cy, rot, x2, ly),
                                         width=lane_width, line_types=[merge_left, merge_right]))
        net.add_lane(c, d, StraightLane(cx_cy(cx, cy, rot, x2, ly), cx_cy(cx, cy, rot, x3, ly),
                                         width=lane_width, line_types=[edge_left, edge_right]))

    if bidirectional:
        # Opposite direction: n_lanes more lanes, mirrored to the -lateral
        # side (ly = -(i+1)*lane_width -- i=0 nearest the centerline,
        # same convention as the forward lanes above), traveling from x3
        # back to x0 (StraightLane's own start/end swapped, so its own
        # heading is exactly reversed). One segment each, not three -- no
        # ramp ever attaches here, so there's no zone-based line-type
        # split to carry across.
        #
        # This reverse lane's OWN +lateral ("right", line_types index 1)
        # points toward MORE NEGATIVE ly here -- opposite the forward
        # lanes' own +lateral, because its heading is reversed -- so
        # increasing i (moving further negative) is this lane's own outer
        # side, and line_types=[left, right] below is [inner, outer] in
        # its own frame, same meaning as the forward loop's own
        # edge_left/edge_right, just re-derived for the reversed heading.
        for i in range(n_lanes):
            ly = -(lane_offset + i * lane_width)
            inner, outer = S, (C if i == n_lanes - 1 else S)
            net.add_lane(f"{prefix}ra_{i}", f"{prefix}rd_{i}",
                         StraightLane(cx_cy(cx, cy, rot, x3, ly), cx_cy(cx, cy, rot, x0, ly),
                                      width=lane_width, line_types=[inner, outer]))

    # Ramp: starts 2*amplitude beyond where it ends up (ramp_gap past the
    # highway's own curb), eases in with a SineLane, then runs parallel
    # for merge_length and stops -- same recipe as MergeEnv's ljk/lkb/lbc,
    # generalized to n_lanes and to either side (ramp_side).
    curb_ly = lane_offset + (n_lanes - 1) * lane_width if ramp_side > 0 else lane_offset
    final_ly = curb_ly + ramp_side * (lane_width / 2 + ramp_gap + lane_width / 2)
    start_ly = final_ly + ramp_side * 2 * amplitude

    p_j = cx_cy(cx, cy, rot, x0, start_ly)
    p_k = cx_cy(cx, cy, rot, before_length, start_ly)
    net.add_lane(f"{prefix}j", f"{prefix}k",
                 StraightLane(p_j, p_k, width=lane_width, line_types=[C, C], forbidden=True))

    sine_amplitude = ramp_side * amplitude
    lat_start = cx_cy(cx, cy, rot, before_length, start_ly - ramp_side * amplitude)
    lat_end = cx_cy(cx, cy, rot, x1, start_ly - ramp_side * amplitude)
    ramp_lane = SineLane(lat_start, lat_end, sine_amplitude, np.pi / taper_length, np.pi / 2,
                          width=lane_width, line_types=[C, C], forbidden=True)
    net.add_lane(f"{prefix}k", f"{prefix}b_ramp", ramp_lane)

    p_merge_start = ramp_lane.position(taper_length, 0)
    p_merge_end = ramp_lane.position(taper_length, 0) + rot @ np.array([merge_length, 0])
    net.add_lane(f"{prefix}b_ramp", f"{prefix}c_ramp",
                 StraightLane(p_merge_start, p_merge_end, width=lane_width, line_types=[N, C], forbidden=True))

    obstacle_position = p_merge_end
    obstacle_heading = angle
    return obstacle_position, obstacle_heading


def cx_cy(cx, cy, rot, lx, ly):
    """World point for local (along-travel, lateral) = (lx, ly), rotated
    by `rot` about (cx, cy) -- shared by merge()'s many straight segments."""
    return np.array([cx, cy]) + rot @ np.array([lx, ly])


def _demo_four_way():
    """Standalone add_four_way network for the __main__ grid demo below.

    access_length=47.5, not a rounder number -- highway_env's own
    LaneGraphics draws each lane's continuous/striped edges in a fixed
    4.33m grid (STRIPE_SPACING) anchored to that edge's own s=0, and
    silently drops the final partial segment if it's under 1.5m
    (draw_stripes: `if abs(start-end) > 0.5*STRIPE_LENGTH`) -- so a lane
    whose length lands just past a grid line (verified directly:
    length=40 left a 1.03m sliver, dropped) renders with a visible gap at
    its true, geometrically-correct endpoint. Every exit lane's own far
    point sits at ITS s=length (not s=0 -- s=0 is the near/stop-line end
    for exit lanes, the opposite of entry lanes, see add_four_way's own
    "start_e"/"end_e"), so it's the exit lanes' far end that's exposed to
    this. 47.5 was picked by directly searching access_length for one
    where both this corner's own lane lengths (44.0 and 40.0 become 51.5
    and 47.5) land with a safe >=1.5m remainder on the grid -- a
    highway_env rendering-only workaround for THIS demo's own viewing, not
    a change to add_four_way's geometry (which was already exact --
    verified numerically -- regardless of access_length).
    """
    net = RoadNetwork()
    add_four_way(net, center=(0.0, 0.0), n_vertical=2, n_horizontal=2, access_length=47.5, prefix="a_")
    return dict(net=net, title="add_four_way")


def _demo_three_way():
    """Standalone add_three_way network -- same access_length=47.5 as
    _demo_four_way and for the same reason (identical per-corner geometry,
    same STRIPE_SPACING grid)."""
    net = RoadNetwork()
    add_three_way(net, center=(0.0, 0.0), n_stem=2, n_cross=2, access_length=47.5,
                  prefix="a_", missing_corner=2)
    return dict(net=net, title="add_three_way")


def _demo_round_about():
    """Standalone round_about network, at this codebase's own alpha=24/
    merge_radius=16 defaults (see round_about's own docstring for why
    radius_in/out always resolve to the exact-solve radius here, landing
    every lane's entry/exit exactly turn_lat apart -- the same spacing
    add_four_way's own approach/exit lanes use)."""
    net = RoadNetwork()
    round_about(net, center=(0.0, 0.0), radius=22.0, n_lanes=2, access_length=40.0,
                alpha=24.0, merge_radius=16.0, prefix="r_")
    return dict(net=net, title="round_about")


def _demo_bend():
    """Standalone _bend network: a straight lead-in, the bend itself built
    from _bend's own returned parameters, and a straight lead-out
    continuing tangent from the bend's own end -- so what's shown is a
    straight-curve-straight road, not the bend in isolation. Both edges
    solid (line_types=[C, C]): a standalone lane with nothing adjacent on
    either side has two true curbs, not one dashed edge implying a
    crossable neighbor that doesn't exist here."""
    net = RoadNetwork()
    p_start = np.array([0.0, 0.0])
    h_start = 0.0  # heading east
    radius = 15.0
    angle_change = np.radians(90)  # a 90-degree turn, same as every other turn in this file
    lead_in, lead_out = 20.0, 20.0

    direction_in = np.array([np.cos(h_start), np.sin(h_start)])
    p_lead_in = p_start - lead_in * direction_in
    net.add_lane("b_in", "b_start", StraightLane(p_lead_in, p_start, width=LANE_WIDTH, line_types=[C, C]))

    center, r, sp, ep, cw = _bend(p_start, h_start, radius, angle_change)
    bend_lane = CircularLane(center, r, sp, ep, clockwise=cw, width=LANE_WIDTH, line_types=[C, C])
    net.add_lane("b_start", "b_end", bend_lane)

    p_end = np.array(bend_lane.position(bend_lane.length, 0))
    h_end = bend_lane.heading_at(bend_lane.length)
    direction_out = np.array([np.cos(h_end), np.sin(h_end)])
    p_lead_out = p_end + lead_out * direction_out
    net.add_lane("b_end", "b_out", StraightLane(p_end, p_lead_out, width=LANE_WIDTH, line_types=[C, C]))
    return dict(net=net, title="_bend")


def _demo_merge():
    """Standalone merge network, matching highway_env's own MergeEnv
    exactly (envs/merge_env.py -- the same environment as
    https://highway-env.farama.org/environments/merge/): ends=[150, 80, 80,
    150] (before, converging, merge, after), and ramp_gap=0 -- the
    reference doesn't have a ramp_gap parameter at all, its own ramp lands
    immediately adjacent to the highway's curb lane (MergeGenericEnv, the
    same file's own parameterized version, makes this exact:
    y_parallel = lanes*DEFAULT_WIDTH), which is exactly what this
    function's own ramp_gap=0 produces (curb_ly + lane_width/2+0+
    lane_width/2 = curb_ly+lane_width = n_lanes*lane_width, the same
    formula) -- so this IS the reference's own geometry, not a modified
    demo of it.

    On top of that reference geometry, three demo-only additions purely to
    show what the ramp is FOR (the reference's own ramp is a dead end by
    design -- see merge()'s own docstring -- nothing in the network merges
    it into the highway; in the real MergeEnv that's the controlled
    vehicle's own job, not the road's, and this demo has no driven vehicle):
    a shortened stub in place of the reference's own full 80m dead-end lane
    (same start point/heading/edge style, just enough length to read as a
    real lane a car could sit in), one illustrative SineLane lane-change
    from the end of that stub onto the highway's own outer lane (landing on
    "m_c_1", the same node the highway's own b->c segment already uses, so
    it joins the real lane graph rather than inventing a new point) cutting
    across sharply in 20m rather than lazily drifting across the whole 80m
    zone, and an explicit highlight/arrow-exclude so the shortened stub
    still reads as real (white, arrow-less) geometry while the yellow route
    traces only the path a car actually takes.
    """
    net = RoadNetwork()
    merge(net, center=(0.0, 0.0), heading_deg=0.0, n_lanes=2,
          before_length=150.0, taper_length=80.0, merge_length=80.0, after_length=150.0,
          ramp_side=1, ramp_gap=0.0, prefix="m_")

    ramp_taper = net.get_lane(("m_k", "m_b_ramp", 0))
    p_tail_start = ramp_taper.position(ramp_taper.length, 0)
    h_tail = ramp_taper.heading_at(ramp_taper.length)
    tail_length = 50.0  # independent of target_s below -- the lane's own length, not the path's
    p_tail_end = p_tail_start + tail_length * np.array([np.cos(h_tail), np.sin(h_tail)])
    del net.graph["m_b_ramp"]["m_c_ramp"]
    net.add_lane("m_b_ramp", "m_c_ramp",
                 StraightLane(p_tail_start, p_tail_end, width=LANE_WIDTH, line_types=[N, C], forbidden=True))

    p0, h0 = p_tail_start, h_tail  # same point/heading the tail stub above starts from
    highway_outer = net.get_lane(("m_b_1", "m_c_1", 0))
    target_s = 20.0  # sharp, not the full 80m merge_length -- see docstring
    target = highway_outer.position(target_s, 0)
    mg_direction = np.array([np.cos(h0), np.sin(h0)])
    mg_lateral = np.array([-mg_direction[1], mg_direction[0]])
    mg_delta = target - p0
    mg_length = float(np.dot(mg_delta, mg_direction))
    mg_shift = float(np.dot(mg_delta, mg_lateral))
    mg_amplitude = -mg_shift / 2
    mg_mid_lateral = mg_shift / 2
    mg_start = p0 + mg_mid_lateral * mg_lateral
    mg_end = p0 + mg_length * mg_direction + mg_mid_lateral * mg_lateral
    net.add_lane("m_b_ramp", "m_merged",
                 SineLane(mg_start, mg_end, mg_amplitude, np.pi / mg_length, np.pi / 2,
                          width=LANE_WIDTH, line_types=[N, N]))
    net.add_lane("m_merged", "m_c_1",
                 StraightLane(target, highway_outer.position(highway_outer.length, 0),
                              width=LANE_WIDTH, line_types=[N, N]))

    highlight = [("m_a_0", "m_b_0", 0), ("m_b_0", "m_c_0", 0), ("m_c_0", "m_d_0", 0),
                 ("m_a_1", "m_b_1", 0), ("m_b_1", "m_c_1", 0), ("m_c_1", "m_d_1", 0),
                 ("m_j", "m_k", 0), ("m_k", "m_b_ramp", 0),
                 ("m_b_ramp", "m_merged", 0), ("m_merged", "m_c_1", 0)]
    arrow_exclude = {("m_b_ramp", "m_c_ramp", 0)}

    # The reference's own full length (460m) is correct but not worth
    # showing in full at a glance -- auto-fitting the whole network into a
    # square cell stretches this into an unreadable sliver. Crop the
    # camera to just the interesting stretch (tail of "before", all of the
    # taper+merge, a little of "after" for x; the highway's own 2 lanes
    # plus how far the ramp rises to for y -- NOT the ramp's own far,
    # constant-offset lead-in at y=14.5, geometrically real but visually
    # just a flat boring line) -- the geometry itself is still the
    # reference's own full 150/80/80/150, only the view is cropped.
    view_window = (140.0, 330.0, -4.0, 12.0)
    return dict(net=net, title="merge", view_window=view_window, highlight=highlight, arrow_exclude=arrow_exclude)


def _render_cell(d, demo, cell_w, cell_h):
    """Render one _demo_*() dict's own network into its own (cell_w,
    cell_h) surface -- the same build/camera-fit/draw sequence every
    single-primitive demo in this file's own history used, generalized so
    the __main__ grid below can call it once per primitive instead of
    repeating it five times."""
    net = demo["net"]
    road = Road(network=net)
    view_window = demo.get("view_window")
    if view_window is not None:
        min_x, max_x, min_y, max_y = view_window
    else:
        min_x, max_x, min_y, max_y = d.road_bounding_box(road)
    surface = WorldSurface((cell_w, cell_h), 0, pygame.Surface((cell_w, cell_h)))
    surface.scaling = min(cell_w / ((max_x - min_x) * 1.1), cell_h / ((max_y - min_y) * 1.1))
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    surface.origin = center - np.array([cell_w / 2, cell_h / 2]) / surface.scaling
    surface.fill(surface.GREY)
    RoadGraphics.display(road, surface)
    d.draw_lane_arrows(surface, road, exclude=demo.get("arrow_exclude", frozenset()))
    # No HUMAN_ROUTE here to bias route_adjacent_lane_indexes() around
    # (these are bare networks, not scene files) -- every lane IS the
    # natural equivalent, so highlight all of them the same yellow
    # draw_bg_lanes uses everywhere else, unless a demo set its own
    # specific route instead (see _demo_merge).
    highlight = demo.get("highlight")
    if highlight is None:
        highlight = [(f, t, i) for f, tos in net.graph.items() for t, lanes in tos.items() for i in range(len(lanes))]
    d.draw_bg_lanes(surface, road, highlight, demo["title"])
    return surface


if __name__ == "__main__":

    # Grid demo: every primitive in this file, each built as its own
    # isolated network (nothing shared, no connect_junctions between them --
    # each _demo_*() above is exactly what running just that one primitive
    # alone would look like) and rendered into its own cell of one combined
    # image, so `python build_scene.py` is a single at-a-glance check of
    # every primitive instead of needing to comment/uncomment through them
    # one at a time.

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import display_all as d

    # Only forced here, in this file's own standalone demo rendering -- not
    # at import time, which would silently force headless mode on any other
    # script that imports this file and wants a real, on-screen window.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    demos = [_demo_four_way(), _demo_three_way(), _demo_round_about(), _demo_bend(), _demo_merge()]

    cols, rows = 3, 2
    cell_w, cell_h = 700, 700
    grid = pygame.Surface((cell_w * cols, cell_h * rows))
    grid.fill(WorldSurface.GREY)
    font = pygame.font.SysFont(None, 28)
    for idx, demo in enumerate(demos):
        cell = _render_cell(d, demo, cell_w, cell_h)
        row, col = divmod(idx, cols)
        grid.blit(cell, (col * cell_w, row * cell_h))
        label = font.render(demo["title"], True, (255, 255, 0))
        grid.blit(label, (col * cell_w + 10, row * cell_h + 10))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_scene_demo.png")
    pygame.image.save(grid, out)
    print("saved", out)
