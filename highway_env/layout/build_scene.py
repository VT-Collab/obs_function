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
    right_turn_radius_base = lane_width + RIGHT_TURN_RADIUS_EXTRA  # same constant add_four_way/add_three_way/turn_corner use for every turn

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
            # Radius: PREFER right_turn_radius_base -- the exact same
            # constant add_four_way/add_three_way/turn_corner use for every
            # turn in the scene, so this bend reads as the same kind of
            # curve, not a bespoke one -- capped by the caller's own
            # merge_radius if that's tighter still (merge_radius keeps its
            # original meaning as an upper bound a caller can pull in), and
            # by _solve_bend_radius's own exact-target radius as the last
            # resort. Offset shrinks as radius grows (verified directly,
            # and it's why _solve_bend_radius works at all -- see its own
            # docstring), so the SMALLEST of the three is always the safe
            # choice: right_turn_radius_base alone already reaches turn_lat
            # for every alpha actually used in this codebase (verified
            # directly -- alpha=24 clears it by nearly 2x), so that's what
            # wins here (merge_radius=16 in every one of this codebase's
            # own scenes is larger, so it isn't the binding constraint);
            # only an alpha tight enough that right_turn_radius_base
            # wouldn't be safe (verified directly: somewhere between
            # alpha=15 and alpha=20) falls through to the exact solve.
            # Either way the result is one continuous curve at ONE fixed
            # radius, never a natural-radius bend patched with a second
            # stage afterward.
            p_entry = point_at(a_entry, r)
            h_entry = tangent_at(a_entry)
            radius_in = min(merge_radius, right_turn_radius_base,
                             _solve_bend_radius(p_entry, h_entry + np.pi, -bend_angle, axis_origin, lateral_in, turn_lat))
            center_, radius_, sp, ep, cw = _bend(p_entry, h_entry + np.pi, radius_in, -bend_angle)
            p_far_entry = center_ + radius_ * np.array([np.cos(ep), np.sin(ep)])
            p_in = p_far_entry - max(access_length - radius_in, 2.0) * direction_in

            net.add_lane(f"{prefix}farin{k}_{lane_i}", f"{prefix}bendin{k}_{lane_i}",
                         StraightLane(p_in, p_far_entry, width=lane_width, line_types=[edge_left, edge_right]))
            net.add_lane(f"{prefix}bendin{k}_{lane_i}", f"{prefix}entry{k}_{lane_i}",
                         CircularLane(center_, radius_, ep, sp, clockwise=(not cw),
                                      width=lane_width, line_types=[edge_left, edge_right]))

            # Leaving: bend from the ring's tangent heading at exit_k to
            # heading radially outward (away from the center), same
            # radius rule as entry above.
            p_exit = point_at(a_exit, r)
            h_exit = tangent_at(a_exit)
            radius_out = min(merge_radius, right_turn_radius_base,
                              _solve_bend_radius(p_exit, h_exit, bend_angle, axis_origin, lateral_out, turn_lat))
            center2, radius2, sp2, ep2, cw2 = _bend(p_exit, h_exit, radius_out, bend_angle)
            p_far_exit = center2 + radius2 * np.array([np.cos(ep2), np.sin(ep2)])
            p_out = p_far_exit + max(access_length - radius_out, 2.0) * direction_out

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
          ramp_gap=2.0, amplitude=3.25, ramp_side=1, prefix=""):
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

    for i in range(n_lanes):
        ly = i * lane_width
        edge_left, edge_right = (C if i == 0 else S), (C if i == n_lanes - 1 else S)
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

    # Ramp: starts 2*amplitude beyond where it ends up (ramp_gap past the
    # highway's own curb), eases in with a SineLane, then runs parallel
    # for merge_length and stops -- same recipe as MergeEnv's ljk/lkb/lbc,
    # generalized to n_lanes and to either side (ramp_side).
    curb_ly = (n_lanes - 1) * lane_width if ramp_side > 0 else 0.0
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


if __name__ == "__main__":

    # Standalone demo: build a single junction (nothing else -- no second
    # junction, no connect_junctions) and render it to a PNG, so
    # `python build_scene.py` alone is a quick visual check that this one
    # primitive looks right, without needing a full scene file. Currently
    # showing add_three_way; the add_four_way call right above it is left
    # in place, commented out, to switch back to.
    #
    # Lanes come from add_four_way itself (this file). The direction
    # arrows do NOT -- they're display_all.draw_lane_arrows, a rendering
    # helper that lives in display_all.py and gets called on top of
    # RoadGraphics.display by every renderer in this project (play.py,
    # watch.py, scene1_background.py, and this demo too). build_scene.py
    # only ever builds the RoadNetwork; it doesn't know how to draw one.

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import display_all as d

    # Only forced here, in this file's own standalone demo rendering -- not
    # at import time, which would silently force headless mode on any other
    # script that imports this file and wants a real, on-screen window.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    net = RoadNetwork()

    # add_four_way(net, center=(0.0, 0.0), n_vertical=2, n_horizontal=2, access_length=40.0, prefix="a_")
    add_three_way(net, center=(0.0, 0.0), n_stem=2, n_cross=2, access_length=40.0, prefix="a_", missing_corner=2)

    road = Road(network=net)

    w, hh = 1000, 1000
    min_x, max_x, min_y, max_y = d.road_bounding_box(road)
    surface = WorldSurface((w, hh), 0, pygame.Surface((w, hh)))
    surface.scaling = min(w / ((max_x - min_x) * 1.1), hh / ((max_y - min_y) * 1.1))
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    surface.origin = center - np.array([w / 2, hh / 2]) / surface.scaling
    surface.fill(surface.GREY)
    RoadGraphics.display(road, surface)
    d.draw_lane_arrows(surface, road)
    # No HUMAN_ROUTE here to bias route_adjacent_lane_indexes() around (this
    # is a bare network, not a scene file) -- every lane IS the natural
    # equivalent, so highlight all of them the same yellow draw_bg_lanes
    # uses everywhere else, instead of just silently having none.
    all_lanes = [(f, t, i) for f, tos in net.graph.items() for t, lanes in tos.items() for i in range(len(lanes))]
    d.draw_bg_lanes(surface, road, all_lanes, "build_scene demo")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_scene_demo.png")
    pygame.image.save(surface, out)
    print("saved", out)
