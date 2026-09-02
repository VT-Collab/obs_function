"""Shared helper: import one nocturne (Waymo-derived) scene's real road
geometry and two real vehicle trajectories. Not a layout itself.
"""
import json
from pathlib import Path

import numpy as np

from highway_env.road.lane import LineType, PolyLaneFixedWidth
from highway_env.road.road import Road, RoadNetwork

NOCTURNE_DATA = Path(__file__).resolve().parents[3] / "nocturne" / "data" / "formatted_json_v2_no_tl_train"
HERE_LAYOUTS = Path(__file__).resolve().parent  # this layouts/ dir
SESSIONS_DIR = HERE_LAYOUTS.parent / "pick_route_sessions"  # where pick_route.py autosaves clicks


def _load(json_path):
    with open(json_path) as f:
        return json.load(f)


def _origin(scene):
    pts = [p for r in scene["roads"] if r["type"] == "lane" for p in r["geometry"]]
    return np.array([sum(p["x"] for p in pts) / len(pts), sum(p["y"] for p in pts) / len(pts)])


def build_road_from(json_path) -> Road:
    scene = _load(json_path)
    origin = _origin(scene)
    net = RoadNetwork()

    for i, r in enumerate(scene["roads"]):
        if r["type"] not in ("lane", "road_edge"):
            continue
        raw = [(p["x"] - origin[0], p["y"] - origin[1]) for p in r["geometry"]]
        points = [raw[0]] + [p for prev, p in zip(raw, raw[1:]) if p != prev]  # drop zero-length segments
        if len(points) < 2:
            continue
        width, line_types = ((4.0, (LineType.STRIPED, LineType.STRIPED)) if r["type"] == "lane"
                              else (0.4, (LineType.CONTINUOUS, LineType.CONTINUOUS)))
        try:
            lane = PolyLaneFixedWidth(points, width=width, line_types=line_types)
        except Exception:
            continue  # a handful of real scenes have degenerate geometry; skip just that segment
        net.add_lane(f"r{i}_a", f"r{i}_b", lane)

    return Road(network=net)


def trajectories_from(json_path, human_idx: int, robot_idx: int):
    """Raw (x, y) point lists for two vehicles, in the same centered frame
    build_road_from() uses -- for display_all.py's raw-point route format."""
    scene = _load(json_path)
    origin = _origin(scene)

    def traj(idx):
        obj = scene["objects"][idx]
        return [(p["x"] - origin[0], p["y"] - origin[1])
                for p, ok in zip(obj["position"], obj["valid"]) if ok]

    return traj(human_idx), traj(robot_idx)


def _closest_point_on_polyline(point, poly):
    """(closest point, distance) on a polyline (Nx2 array) to `point`."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom < 1e-9] = 1e-9
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(proj - point, axis=1)
    i = np.argmin(d)
    return proj[i], d[i]


def densify(points, spacing: float = 2.0):
    """Linearly interpolate extra points so consecutive points are at most
    `spacing` apart. Hand-clicked points (pick_route.py) can be 10-25m
    apart -- snapping those directly to the road would cut every corner
    between clicks, since there's nothing to pull onto the curve in
    between. Densifying first gives snap_to_road something on each curve
    to actually snap."""
    pts = np.array(points, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        dist = np.linalg.norm(b - a)
        n_steps = max(1, int(np.ceil(dist / spacing)))
        for i in range(1, n_steps + 1):
            out.append(a + (b - a) * (i / n_steps))
    return [tuple(p) for p in out]


def _project_with_arclength(point, poly, cum_lengths):
    """(projected point, distance, arc-length position along poly) for point."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom[denom < 1e-9] = 1e-9
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.linalg.norm(proj - point, axis=1)
    i = np.argmin(d)
    seg_len = np.linalg.norm(b[i] - a[i])
    return proj[i], d[i], cum_lengths[i] + t[i] * seg_len


def match_to_road(json_path, points, margin: float = 25.0, max_dist: float = 6.0, switch_margin: float = 2.0,
                   min_run_len: int = 10, kink_threshold_deg: float = 15.0):
    """Map-match `points` onto the real lane graph and output, for each
    stretch assigned to one lane, that LANE'S OWN vertices over the
    matched arc-length range -- not the projected input points.

    snap_to_road() (even with lane-switch hysteresis) still independently
    projects every input point, so a rough or densified path that runs
    ambiguously between multiple real lanes near an intersection can still
    flicker: hysteresis reduces switching but doesn't guarantee a whole
    stretch resolves to one lane. This instead assigns each point to a
    lane first (sticky, same as snap_to_road), groups consecutive points
    assigned to the same lane into one run, and replaces the whole run
    with that lane's actual polyline vertices between the run's start and
    end arc-length position. Since a real lane's own geometry is already
    near-noise-free (see real_001.py's docstring), every point *within* a
    run comes out clean by construction -- there's nothing left to smooth.

    Real map data chops one continuous road into many short adjacent lane
    fragments, so lots of run transitions are normal and NOT visible kinks
    (consecutive fragments of the same physical lane point the same way).
    A visible zigzag/snag is specifically a run that jumps to a genuinely
    different, nearby lane (a parallel lane, a crossing lane) for a few
    points before jumping back. Those are found by a second pass: any run
    shorter than `min_run_len` points whose entry or exit heading breaks by
    more than `kink_threshold_deg` gets folded into its longer neighboring
    run instead (re-projected onto that neighbor's own lane, not just
    deleted), repeated until nothing short and kinked is left.
    """
    scene = _load(json_path)
    origin = _origin(scene)
    pts = np.array(points, dtype=float)
    min_xy, max_xy = pts.min(axis=0) - margin, pts.max(axis=0) + margin

    lanes = []
    for r in scene["roads"]:
        if r["type"] != "lane":
            continue
        poly = np.array([[p["x"] - origin[0], p["y"] - origin[1]] for p in r["geometry"]])
        if len(poly) < 2:
            continue
        if np.any(poly.max(axis=0) < min_xy) or np.any(poly.min(axis=0) > max_xy):
            continue
        seg_lens = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        lanes.append((poly, np.concatenate([[0.0], np.cumsum(seg_lens)])))

    if not lanes:
        return [tuple(p) for p in pts]

    lane_of = []  # lane index chosen for each input point (sticky nearest-lane)
    current = None
    for pt in pts:
        best_i, best_d = None, np.inf
        for i, (poly, cum) in enumerate(lanes):
            _, d, _ = _project_with_arclength(pt, poly, cum)
            if d < best_d:
                best_i, best_d = i, d
        if current is not None:
            _, cur_d, _ = _project_with_arclength(pt, *lanes[current])
            if cur_d <= best_d + switch_margin:
                best_i = current
        current = best_i
        lane_of.append(best_i)

    def run_output(lane_idx, i, j):
        """That lane's own vertices spanning input points i..j (re-projected
        fresh every time, so this stays correct after runs get merged)."""
        poly, cum = lanes[lane_idx]
        s_vals = [_project_with_arclength(pts[k], poly, cum)[2] for k in range(i, j + 1)]
        s_lo, s_hi = min(s_vals), max(s_vals)
        run_pts = poly[(cum >= s_lo) & (cum <= s_hi)]
        if len(run_pts) == 0:
            run_pts = np.array([poly[np.argmin(np.abs(cum - s_lo))], poly[np.argmin(np.abs(cum - s_hi))]])
        if s_vals[-1] < s_vals[0]:
            run_pts = run_pts[::-1]
        return run_pts

    def heading(a, b):
        return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))

    def angle_diff(h1, h2):
        return abs((h2 - h1 + 180) % 360 - 180)

    runs = []
    i, n = 0, len(lane_of)
    while i < n:
        j = i
        while j + 1 < n and lane_of[j + 1] == lane_of[i]:
            j += 1
        runs.append([lane_of[i], i, j])
        i = j + 1

    changed = True
    while changed and len(runs) > 2:
        changed = False
        outputs = [run_output(*r) for r in runs]
        for k in range(len(runs)):
            if runs[k][2] - runs[k][1] + 1 >= min_run_len:
                continue
            this_out = outputs[k]
            if len(this_out) < 2:
                continue
            kinked = False
            if k > 0 and len(outputs[k - 1]) >= 2:
                if angle_diff(heading(*outputs[k - 1][-2:]), heading(*this_out[:2])) > kink_threshold_deg:
                    kinked = True
            if k < len(runs) - 1 and len(outputs[k + 1]) >= 2:
                if angle_diff(heading(*this_out[-2:]), heading(*outputs[k + 1][:2])) > kink_threshold_deg:
                    kinked = True
            if not kinked:
                continue
            # Fold this run into whichever neighbor it's more aligned with (prefer the previous one, tie-break by length).
            target = k - 1 if k > 0 else k + 1
            if k > 0 and k < len(runs) - 1 and (runs[k + 1][2] - runs[k + 1][1]) > (runs[k - 1][2] - runs[k - 1][1]):
                target = k + 1
            if target < k:
                runs[target][2] = runs[k][2]
            else:
                runs[target][1] = runs[k][1]
            del runs[k]
            changed = True
            break

    out = []
    for lane_idx, i, j in runs:
        out.extend(run_output(lane_idx, i, j).tolist())

    deduped = [tuple(out[0])]
    for p in out[1:]:
        if np.linalg.norm(np.array(p) - np.array(deduped[-1])) > 0.05:
            deduped.append(tuple(p))
    return remove_zigzag_spikes(deduped, threshold_deg=kink_threshold_deg)


def remove_zigzag_spikes(points, threshold_deg: float = 25.0, max_passes: int = 6):
    """Drop points that are a real out-and-back zigzag, not a real turn.

    A genuine turn changes heading once and then continues in the new
    direction. A zigzag artifact (a run-merge boundary landing slightly
    past/before the actual junction, a leftover snapping edge-case) bends
    one way then immediately back: heading into the point and heading out
    of it both exceed `threshold_deg` from the surrounding trend, but
    connecting the point *before* it straight to the point *after* it
    tracks that trend much better than routing through it does. That
    point is cut out; genuine sharp turns pass through untouched, since
    skipping them would make things worse, not better.
    """
    def heading(a, b):
        return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))

    def ang_diff(h1, h2):
        return abs((h2 - h1 + 180) % 360 - 180)

    pts = [np.array(p) for p in points]
    for _ in range(max_passes):
        removed = False
        out = [pts[0]]
        i = 1
        while i < len(pts) - 1:
            prev, cur, nxt = out[-1], pts[i], pts[i + 1]
            h_in = heading(prev, cur)
            h_out = heading(cur, nxt)
            h_skip = heading(prev, nxt)
            kink = ang_diff(h_in, h_out)
            if kink > threshold_deg and len(out) >= 2:
                h_trend = heading(out[-2], out[-1])
                if ang_diff(h_trend, h_skip) < ang_diff(h_trend, h_in):
                    removed = True
                    i += 1
                    continue
            out.append(cur)
            i += 1
        out.append(pts[-1])
        pts = out
        if not removed:
            break
    return [tuple(p) for p in pts]


def snap_to_road(json_path, points, margin: float = 25.0, max_dist: float = 6.0, switch_margin: float = 1.5):
    """Project each point onto the nearest real mapped lane centerline
    (map-matching), instead of smoothing the noisy raw GPS trace on its own.
    The real lane geometry is already clean (near-zero curvature noise --
    see real_001.py's docstring), so snapping to it both smooths the path
    AND guarantees it actually follows the road, not just a filtered
    version of wherever the sensor said the car was.

    Sticky lane assignment: once a point snaps to a lane, later points keep
    snapping to that SAME lane unless some other lane is more than
    `switch_margin` closer. Without this, a path running roughly between
    two parallel lanes (common on a multi-lane road, and more so for a
    hand-clicked path that isn't perfectly centered) flickers back and
    forth between them frame to frame -- each point independently picks
    whichever lane is nearest *at that exact spot*, which isn't the same
    lane two points in a row. That flicker is a real, visible zigzag, not
    sensor noise; hysteresis is what actually fixes it, not more
    smoothing. A real transition (an actual turn onto a different road)
    still switches, since the new lane becomes dramatically closer there,
    well past the margin.

    Falls back to the original point where nothing mapped is within
    `max_dist`, so a vehicle briefly off the mapped area doesn't get
    snapped somewhere nonsensical.
    """
    scene = _load(json_path)
    origin = _origin(scene)
    pts = np.array(points)
    min_xy, max_xy = pts.min(axis=0) - margin, pts.max(axis=0) + margin

    polylines = []
    for r in scene["roads"]:
        if r["type"] != "lane":
            continue
        poly = np.array([[p["x"] - origin[0], p["y"] - origin[1]] for p in r["geometry"]])
        if len(poly) < 2:
            continue
        if np.any(poly.max(axis=0) < min_xy) or np.any(poly.min(axis=0) > max_xy):
            continue  # nowhere near this trajectory, skip
        polylines.append(poly)

    snapped = []
    current = None
    for pt in pts:
        best_d, best_p, best_i = np.inf, pt, None
        for i, poly in enumerate(polylines):
            p, d = _closest_point_on_polyline(pt, poly)
            if d < best_d:
                best_d, best_p, best_i = d, p, i
        if current is not None:
            cur_p, cur_d = _closest_point_on_polyline(pt, polylines[current])
            if cur_d <= best_d + switch_margin:
                best_p, best_d, best_i = cur_p, cur_d, current
        current = best_i
        snapped.append(tuple(best_p) if best_d < max_dist else tuple(pt))
    return snapped
