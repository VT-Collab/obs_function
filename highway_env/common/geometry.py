"""Vision cone, line-of-sight occlusion. No agent logic, no task/route/goal
knowledge lives here -- every function is a pure function of positions,
headings, and vehicle extents only, and would be identical in any other
highway_env scene. Mirrors steakhouse/misha/common/geometry.py's own role
("no agent logic lives here") for a continuous 2D road domain instead of a
grid: cone/line-of-sight there, cone/rectangle-occlusion here.

Lives at highway_env/common/ (a sibling of human/ and robot/, mirroring
steakhouse/misha's own common/), not inside either -- both
highway_env/human/limit_vision_human.py (which re-exports these names
unchanged, so nothing importing from there breaks) and
highway_env/robot/filter/core/fov_posterior.py depend on this alone,
without either depending on the other.
"""
import numpy as np


def in_cone(ego_position, ego_heading, target_position, fov_deg):
    """True iff target_position lies within fov_deg of ego_heading, as seen
    from ego_position. fov_deg >= 360 is vacuously true (matches the
    Overcooked precedent's stance: no separate radius/omniscience baked in
    here either -- occlusion still applies at 360, see is_occluded)."""
    if fov_deg >= 360:
        return True
    delta = np.asarray(target_position, dtype=float) - np.asarray(ego_position, dtype=float)
    dist = np.linalg.norm(delta)
    if dist < 1e-6:
        return True
    heading_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    cos_angle = np.dot(delta, heading_vec) / dist
    # small epsilon: cos(radians(90)) is ~6e-17, not exactly 0, so an exact
    # boundary case (e.g. fov=180, target exactly perpendicular) would
    # otherwise fail this comparison by pure floating-point noise.
    return cos_angle >= np.cos(np.radians(fov_deg / 2.0)) - 1e-9


def segment_intersects_rotated_rect(a, b, center, length, width, angle):
    """True iff the line segment a->b crosses the rectangle centered at
    `center` (dimensions length x width, rotated by `angle` radians, same
    rotation convention as highway_env.utils.point_in_rotated_rectangle).

    Exact segment-vs-axis-aligned-box test (Liang-Barsky clipping) done in
    the rectangle's own unrotated frame, rather than reusing
    highway_env.utils.rotated_rectangles_intersect via a near-zero-width
    degenerate rectangle: that function is corner-sampling (does any corner
    of one rectangle land inside the other), which is a reasonable
    approximation for two comparably-sized rectangles (what it's built for
    elsewhere in highway-env) but unreliable for a thin ray THROUGH the
    middle of a blocker -- exactly the common "car hidden behind another
    car" case this module exists to detect, where neither rectangle's
    corners land inside the other at all.
    """
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])
    center = np.asarray(center, dtype=float)
    la = rot @ (np.asarray(a, dtype=float) - center)
    lb = rot @ (np.asarray(b, dtype=float) - center)
    dx, dy = lb[0] - la[0], lb[1] - la[1]
    half_l, half_w = length / 2.0, width / 2.0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, la[0] - (-half_l)), (dx, half_l - la[0]),
                 (-dy, la[1] - (-half_w)), (dy, half_w - la[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False  # parallel to this edge and outside it
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def is_occluded(ego_position, target_position, blockers):
    """True iff any vehicle in `blockers` sits on the straight line between
    ego and target. Crashed/stationary vehicles are included "for free" --
    blockers is whatever the caller passes (typically every other nearby
    vehicle, crashed or not), and wreckage should still occlude."""
    for other in blockers:
        length = getattr(other, "LENGTH", 5.0)
        width = getattr(other, "WIDTH", 2.0)
        if segment_intersects_rotated_rect(ego_position, target_position,
                                            other.position, length, width, other.heading):
            return True
    return False


def visible(ego_position, ego_heading, target, fov_deg, blockers, enable_fov=True, enable_occlusion=True):
    """The one-line combination every caller actually wants: target is
    visible iff (fov disabled, or it's in the cone) AND (occlusion
    disabled, or nothing blocks it). Kept alongside the two primitives
    above (not just inlined at each call site) so a posterior's per-
    hypothesis shadow and the real LimitedVisionHuman are guaranteed to
    evaluate visibility identically."""
    if enable_fov and not in_cone(ego_position, ego_heading, target.position, fov_deg):
        return False
    if enable_occlusion and is_occluded(ego_position, target.position, blockers):
        return False
    return True
