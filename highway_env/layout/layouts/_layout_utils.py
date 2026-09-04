"""Shared helpers for build_scene.py-based layouts (mega_scene.py and
friends). Not a layout itself -- layout_names() skips files starting with
"_", and display_all.py puts this directory on sys.path so layout files can
`from _layout_utils import ...`.
"""
import numpy as np

from highway_env.road.lane import LineType, SineLane, StraightLane

_C = LineType.CONTINUOUS
_S = LineType.STRIPED
_N = LineType.NONE


def exit_far(net, prefix, corner, lane_i=0):
    """(far_point, heading_rad) of a junction's own EXIT lane
    (il{corner}_i -> o{corner}_i, add_four_way/add_three_way's convention)
    -- the point and direction a chain should continue from when leaving
    that arm."""
    lane = net.get_lane((f"{prefix}il{corner}_{lane_i}", f"{prefix}o{corner}_{lane_i}", 0))
    return np.array(lane.position(lane.length, 0)), lane.heading_at(lane.length)


def exit_far_center(net, prefix, corner, lane_i, lane_width):
    """Like exit_far, but returns the junction's own TRUE CENTERLINE far
    point/heading -- lane_i's own turn_lat lateral offset (lane_width/2 +
    lane_i*lane_width, add_four_way/add_three_way's own convention)
    undone, not that lane's own point.

    Use this, not exit_far directly, whenever the result places a
    DOWNSTREAM primitive's own center (round_about's own `center`,
    add_four_way/add_three_way's own via center_ahead_corner) -- anything
    placed straight off exit_far's own lane-specific point inherits THAT
    lane's own lateral offset as a hidden shift in the whole downstream
    structure's own axis. Verified directly: querying lane 0 vs lane 1
    from the same exit gives points differing by EXACTLY (turn_lat(1)-
    turn_lat(0))*lateral with zero along-heading component, i.e. a pure
    lateral offset -- so subtracting lane_i's own turn_lat*lateral here
    recovers the same true center point regardless of which lane_i was
    queried, and every lane bridged afterward (bridge_corners/
    bridge_roundabout with n>1) lands exactly parallel, not at a
    compounding diagonal "twist" only visible once more than one lane
    bridges in parallel (a single asymmetric bridge can silently absorb
    the same misalignment as one line's own angle, which is why this
    stayed invisible until n>1 bridging existed).
    """
    far, heading = exit_far(net, prefix, corner, lane_i)
    turn_lat = lane_width / 2 + lane_i * lane_width
    direction = np.array([np.cos(heading), np.sin(heading)])
    lateral = np.array([-direction[1], direction[0]])
    return far - turn_lat * lateral, heading


def bridge(net, from_node, to_node, p1, p2, line_types=(_C, _C)):
    """A short raw StraightLane connecting two build_scene-built junctions
    -- the same bridging mega_scene.py uses between its own primitives.

    Bridges ONE lane. When `prefix_a`'s corner and `prefix_b`'s corner both
    carry the same lane count, use bridge_corners instead -- a single-lane
    bridge() call there leaves every lane but the route's own dangling,
    unconnected at that junction's own edge (the outermost lane included),
    since nothing else in either primitive's own construction reaches past
    its own access road to join the two junctions together."""
    net.add_lane(from_node, to_node, StraightLane(p1, p2, line_types=list(line_types)))


def bridge_corners(net, prefix_a, corner_a, prefix_b, corner_b, n):
    """Bridge EVERY one of `n` lanes between prefix_a's corner_a arm and
    prefix_b's corner_b arm, in BOTH directions -- both junctions must
    already exist with at least `n` lanes at these corners (equal counts;
    see real_001_rebuilt.py's own docstring on why add_four_way/
    add_three_way need matching counts anyway).

    Use this (not repeated single bridge() calls) for a same-lane-count
    junction-to-junction hop, e.g. a 4-way's own 2 lanes continuing into
    the next T-junction's own 2 lanes -- bridging only lane 0 there leaves
    lane 1 (the outermost lane, from a driver's-side view on the +turn_lat
    side) simply dead-ending at the first junction's own edge, since
    nothing else reaches across the gap for it. A connector primitive that
    genuinely narrows (turn_corner/round_about/merge, n_lanes=1) is a
    different, intentional case -- bridge() there is correct, not a bug:
    the road really does drop a lane, same as a real one narrowing before
    a roundabout.

    Each direction is its own lane object, physically separated (mirrored
    +/-turn_lat, add_four_way/add_three_way's own approach/exit
    convention) -- never one lane object serving both directions. An
    earlier version only built the forward direction (prefix_a's exit ->
    prefix_b's approach), leaving prefix_b's own exit back toward
    prefix_a with no bridge at all: a real dead end for any background
    vehicle that took it, not "bidirectional" in the sense of one lane
    carrying opposing traffic, but a connectivity gap with the same
    symptom -- a vehicle stuck there with no forward continuation could
    end up snapping onto whatever nearby lane a fuzzy position/heading
    match found next, including one meant for the opposite direction.
    prune_to_route (see its own docstring) is what keeps this from being
    pruned right back out: it only deletes alternatives that start
    heading the same way as the route's own chosen edge there, and this
    return direction starts ~180 degrees from that, so it survives.

    Each lane i's own line_types follow add_four_way/add_three_way's own
    convention (edge_left always striped, edge_right continuous only for
    the outermost lane i==n-1) -- NOT a single fixed line_types for every
    lane, which would either mark an inner lane's own divider solid (wrong)
    or leave the true outermost edge striped (wrong the other way).
    """
    for i in range(n):
        edge_left, edge_right = _S, (_C if i == n - 1 else _S)

        far_a = net.get_lane((f"{prefix_a}il{corner_a}_{i}", f"{prefix_a}o{corner_a}_{i}", 0))
        far_a_pos = far_a.position(far_a.length, 0)
        near_b = net.get_lane((f"{prefix_b}o{corner_b}_{i}", f"{prefix_b}ir{corner_b}_{i}", 0))
        near_b_pos = near_b.position(0, 0)
        bridge(net, f"{prefix_a}o{corner_a}_{i}", f"{prefix_b}o{corner_b}_{i}", far_a_pos, near_b_pos,
               line_types=(edge_left, edge_right))

        far_b = net.get_lane((f"{prefix_b}il{corner_b}_{i}", f"{prefix_b}o{corner_b}_{i}", 0))
        far_b_pos = far_b.position(far_b.length, 0)
        near_a = net.get_lane((f"{prefix_a}o{corner_a}_{i}", f"{prefix_a}ir{corner_a}_{i}", 0))
        near_a_pos = near_a.position(0, 0)
        bridge(net, f"{prefix_b}o{corner_b}_{i}", f"{prefix_a}o{corner_a}_{i}", far_b_pos, near_a_pos,
               line_types=(edge_left, edge_right))


def bridge_roundabout(net, ra_prefix, gap_k, j_prefix, corner, j_lane_i=0, ra_lane_i=0, n=1):
    """Bridge a roundabout's gap k to a junction's corner arm, in BOTH
    directions: the junction's own exit feeding the roundabout's entry at
    that gap, and the roundabout's own exit at that SAME gap feeding back
    into the junction's own approach -- mirroring bridge_corners' own
    reasoning (see its docstring). round_about's own farin{k}/farout{k} are
    now two separate, properly lane_width-separated one-way lanes (see
    round_about's own docstring on why they weren't reliably that before),
    so leaving the return direction unbridged is a real dead end for any
    background vehicle that takes it, not merely a missing decoration.

    j_lane_i/ra_lane_i: the junction's own STARTING lane index and the
    roundabout's own STARTING lane index at this gap -- separate
    parameters (not one shared index) because a route riding a junction's
    lane 1 into a single-lane (ra_lane_i=0) roundabout is a real, common
    case, not j_lane_i == ra_lane_i.

    n: how many consecutive lane pairs to bridge -- (j_lane_i+i, ra_lane_i
    +i) for i in range(n). n=1 (default) is a single lane pair with plain
    bridge()'s own default line_types (both edges solid): the genuine-
    narrowing case (see bridge_corners' own docstring on why a plain
    bridge() there is correct, not a bug, when a connector primitive
    actually drops a lane -- the same reasoning applies to a roundabout
    with fewer lanes than the junction it meets). Pass n>1 when the
    roundabout and the junction carry the SAME lane count at this gap --
    no narrowing at all -- to bridge every lane 1:1 the way bridge_corners
    does between two junctions, using that same per-lane line_types
    convention (inner lanes striped both sides, only the outermost lane's
    own outer edge solid) instead of every lane getting a redundant solid
    edge on its inner side too.

    Only call this where the far side is a real junction with its own
    approach/exit pair. A merge's own "a" node has no such reverse lane at
    all -- a real highway on-ramp doesn't either (see merge's own
    docstring) -- so bridging a roundabout's return direction into one
    isn't meaningful; leave that connection as a single bridge() call.
    """
    for i in range(n):
        jl, ral = j_lane_i + i, ra_lane_i + i
        edge_types = ((_C, _C),) if n == 1 else ((_S, (_C if i == n - 1 else _S)),)
        edge_left, edge_right = edge_types[0]

        j_far = net.get_lane((f"{j_prefix}il{corner}_{jl}", f"{j_prefix}o{corner}_{jl}", 0))
        j_far_pos = j_far.position(j_far.length, 0)
        ra_in = net.get_lane((f"{ra_prefix}farin{gap_k}_{ral}", f"{ra_prefix}bendin{gap_k}_{ral}", 0))
        ra_in_pos = ra_in.position(0, 0)
        bridge(net, f"{j_prefix}o{corner}_{jl}", f"{ra_prefix}farin{gap_k}_{ral}", j_far_pos, ra_in_pos,
               line_types=(edge_left, edge_right))

        ra_out = net.get_lane((f"{ra_prefix}bendout{gap_k}_{ral}", f"{ra_prefix}farout{gap_k}_{ral}", 0))
        ra_out_pos = ra_out.position(ra_out.length, 0)
        j_near = net.get_lane((f"{j_prefix}o{corner}_{jl}", f"{j_prefix}ir{corner}_{jl}", 0))
        j_near_pos = j_near.position(0, 0)
        bridge(net, f"{ra_prefix}farout{gap_k}_{ral}", f"{j_prefix}o{corner}_{jl}", ra_out_pos, j_near_pos,
               line_types=(edge_left, edge_right))


def lane_change(net, from_node, to_node, p0, heading_rad, shift, length, lane_width, line_types=(_N, _N)):
    """A SineLane performing one smooth lane change: starts at `p0` (a
    point ON the current lane's own centerline, heading `heading_rad`) and
    ends `length` meters further along, shifted `shift` meters laterally
    (StraightLane's own direction_lateral sign convention -- a +90-degree
    rotation of `heading_rad`; positive lane_width = one lane to that
    side). Tangent to the straight lane at BOTH ends (zero lateral
    "velocity" at s=0 and s=length, so it reads as a real lane change, not
    a kinked jog) -- same technique build_scene.merge's own ramp taper
    uses, just algebraically inverted here to solve for a *specific*
    (p0, shift) instead of merge's own fixed ramp_gap/amplitude geometry.

    Derivation: SineLane(start, end, amplitude, pulsation=pi/length,
    phase=pi/2) evaluates its lateral offset at s=0 as +amplitude and at
    s=length as -amplitude (phase+pulsation*length = pi/2+pi -> sin=-1).
    Placing the SineLane's own (start, end) at the MIDPOINT lateral offset
    (shift/2) and setting amplitude=-shift/2 lands offset(0) = shift/2 +
    (-shift/2) = 0 (matches p0 exactly) and offset(length) = shift/2 -
    (-shift/2) = shift (verified directly, not assumed).

    line_types defaults to (NONE, NONE): a lane change is a diagonal
    detour across the middle of the road, not a real edge, so it shouldn't
    draw its own boundary line. An earlier version defaulted this to
    (STRIPED, CONTINUOUS) to paper over a missing outer-edge line, but that
    line was only missing because every call site here forks its lane
    change AWAY from a node the true straight outer lane also uses, and
    prune_to_route then deletes that true edge as an "unchosen alternative"
    -- drawing a fake solid line on the diagonal instead of the real
    straight lane. The real fix is to never fork a lane change away from
    the true outer lane's own node in the first place (see each scene
    file's own docstring: every lane change here now forks from an INNER
    lane node, so the outer lane's own pre-built straight edges -- and
    their own already-correct line types -- are never pruned at all).
    """
    _add_sine_lane_change(net, from_node, to_node, p0, heading_rad, shift, length, lane_width, line_types)


def lane_change_to(net, from_node, to_node, p0, heading_rad, target_pos, lane_width, line_types=(_N, _N)):
    """Like lane_change, but lands EXACTLY on a specific pre-existing
    point (`target_pos`, e.g. another lane index's own stop-line node)
    instead of a shift+length you supply by hand.

    Different lane indices sit at different stop-line distances from a
    junction's own center (od(i) = right_turn_radius_base + turn_lat(i),
    which grows WITH i -- see build_scene.py's add_four_way/add_three_way),
    and approach vs. exit conventions add a further +/-turn_lat offset on
    top of that (see connect_junctions' own docstring on why an "o" node
    isn't one physical point). The upshot: the lateral shift AND the
    longitudinal distance between e.g. lane 0's exit stop-line and lane 1's
    own exit stop-line are almost never a clean (LANE_WIDTH, access_length)
    pair -- verified directly (one real case worked out to an 8m lateral
    shift over 29m, not a clean 4m/25m). This solves for the shift and
    length that actually land on `target_pos`, by projecting the straight
    displacement onto (direction, lateral) at `heading_rad`, instead of
    assuming a fixed lane_width/access_length -- so a lane change can
    safely target another lane's own pre-built junction node and still
    feed cleanly into that lane's own turn geometry right after.

    line_types default -- see lane_change's own docstring on why
    (NONE, NONE) is correct once every call site forks from an inner lane.
    """
    direction = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    lateral = np.array([-direction[1], direction[0]])
    delta = np.array(target_pos) - np.array(p0)
    length = float(np.dot(delta, direction))
    shift = float(np.dot(delta, lateral))
    _add_sine_lane_change(net, from_node, to_node, p0, heading_rad, shift, length, lane_width, line_types)


def _add_sine_lane_change(net, from_node, to_node, p0, heading_rad, shift, length, lane_width, line_types):
    direction = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    lateral = np.array([-direction[1], direction[0]])
    amplitude = -shift / 2
    mid_lateral = shift / 2
    start = np.array(p0) + mid_lateral * lateral
    end = np.array(p0) + length * direction + mid_lateral * lateral
    net.add_lane(from_node, to_node,
                 SineLane(start, end, amplitude, np.pi / length, np.pi / 2,
                          width=lane_width, line_types=list(line_types)))


def center_ahead(far_point, heading_rad, lane_j, access_length, od, gap, lane_width):
    """Invert the `far = center + rot(heading) @ (turn_lat_j, access_length +
    od)` convention turn_corner's in_j and merge's own frame both use --
    both rotate their WHOLE geometry by a `heading_deg` that matches the
    incoming direction of travel directly -- to find the `center` that
    places that same lane's far point exactly `gap` meters further along
    `heading_rad` from a known point. Same helper as mega_scene.py's own
    _extend_center. NOT for add_four_way/add_three_way -- see
    center_ahead_corner, which those need instead."""
    turn_lat = lane_width / 2 + lane_j * lane_width
    rot = np.array([[np.cos(heading_rad), -np.sin(heading_rad)], [np.sin(heading_rad), np.cos(heading_rad)]])
    new_far = np.array(far_point) + gap * np.array([np.cos(heading_rad), np.sin(heading_rad)])
    return new_far - rot @ np.array([turn_lat, access_length + od])


def corner_for_heading(heading_rad):
    """Which add_four_way/add_three_way corner (0=South, 1=West, 2=North,
    3=East) has an APPROACH lane traveling `heading_rad` -- i.e. the corner
    a bridge arriving from that direction should feed into head-on. Each
    corner c's approach lane travels heading = 90*c - 90 degrees (c=0/South
    carries northbound traffic at -90 degrees, c=1/West carries eastbound at
    0, etc. -- verified directly from add_four_way's own start/end formula,
    not assumed)."""
    return int(np.round((np.degrees(heading_rad) + 90) / 90)) % 4


def ring_gap_for_heading(heading_rad):
    """Which round_about gap k (see build_scene.round_about: k=0 east,
    k=1 north, k=2 west, k=3 south) a bridge arriving from `heading_rad`
    should enter -- gap k sits at the same physical orientation as
    add_four_way corner (3-k)%4, so this is corner_for_heading composed
    with that mapping (verified directly against both real_001_rebuilt.py's
    and real_002_rebuilt.py's own hand-derived k values)."""
    return (3 - corner_for_heading(heading_rad)) % 4


def center_ahead_corner(far_point, heading_rad, lane_j, access_length, od, gap, lane_width):
    """Like center_ahead, but for placing an add_four_way/add_three_way's
    own `center` so that its receiving corner (see corner_for_heading)
    lands its far point exactly `gap` meters beyond `far_point` along
    `heading_rad`.

    add_four_way/add_three_way corners are NOT parametrized by an arbitrary
    heading like turn_corner/merge are -- each corner c's own far-point
    formula always uses the FIXED rotation rot(90*c), regardless of which
    direction a bridge happens to be arriving from. Corner c's own fixed
    angle is heading_rad + 90 degrees (see corner_for_heading's docstring:
    approach heading = corner angle - 90), NOT heading_rad itself -- using
    center_ahead's plain rot(heading_rad) here would place the target's
    center off by roughly one lane width (the turn_lat term lands on the
    wrong axis), a small but real kink at the join, same class of bug as
    add_four_way's own corner-count asymmetry (see real_001_rebuilt.py).
    """
    corner_angle = heading_rad + np.pi / 2
    turn_lat = lane_width / 2 + lane_j * lane_width
    rot = np.array([[np.cos(corner_angle), -np.sin(corner_angle)], [np.sin(corner_angle), np.cos(corner_angle)]])
    target_far = np.array(far_point) + gap * np.array([np.cos(heading_rad), np.sin(heading_rad)])
    return target_far - rot @ np.array([turn_lat, access_length + od])


def polyline(net, lane_route, step=1.0):
    """World-space points tracing `lane_route` (a list of (from, to,
    lane_id) lane indices), sampled every `step` meters along each lane's
    own centerline -- same approach as display_all.py's route_polyline().

    step=1.0 (not display_all.py's own 3.0m default): build_scene-based
    layouts chain short CircularLane arcs (a roundabout's own ring/bend
    segments are commonly 7-9m end to end -- see round_about's own access
    and merge_radius params), and this function's output is drawn as
    straight connected line segments (see display_all.py's draw_route). At
    3m spacing those short arcs get only 2-3 points, so the drawn route
    renders as a visibly faceted polygon chord-ing across what the actual
    road surface (rendered separately, at fixed fine stripe-spacing
    independent of lane length) shows as a smooth curve. This only changes
    the route's own point density, not any road geometry.

    Exposed as plain (x, y) points, not (from, to, lane_id) tuples, because
    highway_env/human/limit_vision_human.py's add_human_vehicle and
    route_aware_continuation -- built for the hand-clicked/matched-to-road
    real_*.py scenes -- only understand that format (they rank a junction's
    exit fork by progress along a dense point polyline, not graph
    traversal), and display_all.py's own route_polyline() renders points
    just as well as lane tuples, so this format works everywhere the tuple
    form did.
    """
    points = []
    for lane_index in lane_route:
        lane = net.get_lane(lane_index)
        for lon in np.linspace(0, lane.length, max(2, int(lane.length / step))):
            points.append(tuple(lane.position(lon, 0)))
    return points


def prune_to_route(net, *lane_routes, max_heading_diff_deg=100.0):
    """At every 'from' node the given routes actually pass through, delete
    only the OTHER outgoing edges that start out heading roughly the same
    way (within max_heading_diff_deg) as the route's own chosen edge there
    -- not every alternative unconditionally.

    Needed because highway_env/human/limit_vision_human.py's
    route_aware_continuation ranks fork candidates by how far each
    candidate's OWN start point sits along the route polyline -- but
    add_four_way/add_three_way/round_about's turn/exit options at one
    junction all literally start at the same shared node AND the same
    heading (a CircularLane's heading_at(0) is tangent to the incoming
    lane, same as the straight option -- verified directly: right turn,
    left turn and straight all read the identical heading at s=0 from a
    shared "ir" node, only diverging as s grows), so every same-direction
    alternative at a real fork ties EXACTLY with the intended one and the
    tie-break silently keeps whichever happened to be enumerated first --
    observed sending a route-following vehicle the wrong way at both a
    3-way's left/right fork and a roundabout's continue/leave fork.

    Restricting the deletion to heading-compatible alternatives (not
    literally everything) matters once bridge_corners builds BOTH
    directions between two junctions: the return direction shares its
    'from' node's name with the route's own forward choice there (an 'o'
    node isn't one physical point -- see bridge_corners' own docstring)
    but starts ~180 degrees from it (verified directly), so it was never
    actually a source of the tie-breaking confusion this function exists
    to prevent -- an earlier version deleted it anyway, turning a real,
    needed lane for background traffic into a dead end that looked, from
    the outside, like the same symptom as a lane genuinely carrying
    opposing traffic (a vehicle stuck there with no forward continuation
    snapping onto whatever nearby lane a fuzzy position/heading match
    found next). max_heading_diff_deg=100 sits comfortably between the
    ~0-degree ties this exists to break and the ~180-degree reverse lanes
    it must leave standing.

    Real map fragments (real_*.py) don't share start points this way,
    which is why this only shows up on a build_scene-built network. Only
    prunes the specific nodes the given routes pass through; every other
    arm/movement (background traffic's own turns elsewhere) is untouched.
    """
    keep = {}
    chosen_heading = {}
    for route in lane_routes:
        for f, t, i in route:
            keep.setdefault(f, set()).add(t)
            chosen_heading[f] = net.get_lane((f, t, i)).heading_at(0)
    for f, tos in keep.items():
        h_route = chosen_heading[f]
        for t in list(net.graph.get(f, {})):
            if t in tos:
                continue
            diffs = [abs((np.degrees(lane.heading_at(0) - h_route) + 180) % 360 - 180)
                     for lane in net.graph[f][t]]
            if all(d <= max_heading_diff_deg for d in diffs):
                del net.graph[f][t]


def route_adjacent_lane_indexes(build_network, route_points, radius=15.0):
    """Every (from, to, lane_id) in a freshly-built network whose lane
    passes within `radius` meters of some point on `route_points`.

    `build_network` is a zero-arg callable returning a fresh RoadNetwork
    (a layout's own _build_network) -- called once here so this doesn't
    share a single mutable network with whatever the caller is stepping.

    For scene1_background.add_background_traffic's own lane_indexes param:
    on a large real recorded map, spawning uniformly over every lane still
    puts plenty of traffic near any given route. A small synthetic network
    is much smaller AND a good fraction of its lanes are stubs a route
    never comes near at all (a roundabout's own unused cardinals, an
    unused four-way arm, etc.) -- uniform spawning wastes a lot of `count`
    there instead of near the route, which is largely why an early FOV
    sweep on mega_scene.py showed no measurable difference between FOV
    widths at all: the human rarely encountered anything to actually miss.
    This lets a caller spawn background traffic that actually crosses or
    shares the route's own path.
    """
    net = build_network()
    route_pts = np.array(route_points)
    result = []
    for f, tos in net.graph.items():
        for t, lanes in tos.items():
            for i, lane in enumerate(lanes):
                n_samples = max(2, int(lane.length / 5))
                close = False
                for s in np.linspace(0, lane.length, n_samples):
                    p = np.asarray(lane.position(s, 0))
                    if np.min(np.linalg.norm(route_pts - p, axis=1)) <= radius:
                        close = True
                        break
                if close:
                    result.append((f, t, i))
    return result
