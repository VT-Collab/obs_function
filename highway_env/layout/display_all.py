"""Render every road layout in layouts/ at once, tiled in one window.

    python display_all.py                # all layouts, auto grid
    python display_all.py --cols 3       # force 3 columns
    python display_all.py --scene real_012   # just that one, full window
    python display_all.py --scene 12         # shorthand for real_012

Keys:  q / ESC / close window  ->  quit

For LOOKING at a whole batch of candidate road geometries in one go, so you
can pick which ones are worth building out with vehicles/FOV logic and which
aren't. No vehicles are placed here on purpose -- this is purely about
whether the road shape itself has the right structure (long-horizon, cars
can't just separate and be done).

How it works: each layouts/*.py file exposes build_road() -> Road (just a
RoadNetwork, no vehicles). We draw each Road onto its own off-screen
WorldSurface -- scaled/centered to fit that road's own bounding box, since
layouts range from a 20m roundabout to an 800m corridor -- then blit all of
them, scaled again to fit their grid cell, into one shared window. Same
two-level "draw small, then tile" trick as steakhouse's overcooked
display_all.py, just with highway_env's own WorldSurface/RoadGraphics
instead of the kitchen renderer.

A layout can optionally define HUMAN_ROUTE / ROBOT_ROUTE: a list of
(from_node, to_node, lane_id) lane indices forming one connected path
through the network, or (for an imported real scene) a plain list of (x, y)
points. When present, both are traced as colored lines on top of the road
(human = blue, robot = orange, each offset slightly off the lane centerline
so a shared lane shows as two parallel tracks instead of one occluding the
other) -- this is what actually shows whether the two cars are forced to
share road for a long stretch, or just cross paths once and separate.

ROBOT_ROUTE can also be a *list of routes* -- one human, several robot cars
active at different, non-overlapping points along the human's path (e.g.
real_001.py: the human passes close to one real vehicle for the first half
of the recording, then a different one late in it). Each sub-route is drawn
as its own disconnected polyline in its own color from ROBOT_PALETTE, cycled
in order, instead of one line jumping between unrelated encounters.

A layout that fails to build is SKIPPED with a printed reason rather than
killing the run, for the same reason as the overcooked version: this is the
tool you reach for when something in the suite is broken, so it has to
survive one broken member.
"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np
import pygame

from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.road.lane import StraightLane

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS_DIR = os.path.join(HERE, "layouts")
sys.path.insert(0, LAYOUTS_DIR)  # so layout files can `from _real_scene import ...`

TILE_SIZE = (420, 300)  # px, per-layout off-screen render before it's re-scaled into its grid cell
BG = (28, 28, 32)
LABEL = (235, 235, 235)
LABEL_H = 22
BBOX_MARGIN = 1.15  # fraction of headroom around a road's bounding box

HUMAN_COLOR = (90, 170, 255)
ROBOT_COLOR = (255, 150, 60)
ROBOT_PALETTE = [(255, 150, 60), (230, 70, 70), (190, 110, 255), (255, 215, 70), (90, 220, 150)]
ROUTE_LATERAL_OFFSET = 0.6  # m, so two routes sharing one lane render as parallel tracks
ROUTE_WIDTH = 3  # px
ROUTE_START_COLOR = (60, 230, 90)  # green marker at a route's first point
ROUTE_END_COLOR = (230, 60, 60)  # red marker at a route's last point
ROUTE_MARKER_RADIUS = 7  # px
BG_LANE_COLOR = (235, 210, 40)  # yellow overlay: every lane route_adjacent_lane_indexes() would hand to add_background_traffic as a spawn candidate
ARROW_COLOR = (220, 220, 220)  # light, road-marking grey -- reads as part of the lane, not as a route/overlay color
ARROW_LENGTH = 2.2  # m, tip to tail
ARROW_WIDTH = 1.4  # m, tail spread
ARROW_MIN_LANE_LENGTH = 4.0  # m -- skip arrows on lanes too short for one to read cleanly (tight turn-arc stubs)
BG_LANE_WIDTH = 2  # px, thin so it overlays a lane's centerline without hiding the base line-type rendering underneath


def layout_names():
    """Every layouts/*.py (excluding dunder/private files), sorted."""
    return sorted(
        f[:-len(".py")] for f in os.listdir(LAYOUTS_DIR)
        if f.endswith(".py") and not f.startswith("_")
    )


def load_layout(name):
    """Import layouts/<name>.py as a standalone module (not a package import)."""
    path = os.path.join(LAYOUTS_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"layouts.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def road_bounding_box(road):
    """(min_x, max_x, min_y, max_y) covering every lane's driving surface."""
    xs, ys = [], []
    for from_node in road.network.graph.values():
        for lanes in from_node.values():
            for lane in lanes:
                for s in np.linspace(0, lane.length, 20):
                    half_w = lane.width_at(s) / 2
                    for lat in (-half_w, half_w):
                        p = lane.position(s, lat)
                        xs.append(p[0])
                        ys.append(p[1])
    return min(xs), max(xs), min(ys), max(ys)


def _lane_sample_points(lane):
    """How many longitudinal samples to take along `lane` for drawing it as
    a polyline: a straight lane is exactly itself with just its two
    endpoints, but a curved lane (CircularLane, SineLane -- anything that
    isn't dead straight) drawn as a polyline is only as smooth as its own
    sample spacing, and a fixed "every 3m" (this file's own earlier
    convention) is coarse enough to render as visible straight-line kinks
    on this codebase's own tight turn curves (9-21m radius) -- confirmed
    directly: highway_env's own base rendering doesn't draw these curves
    at all (right/left turns are line_types=[NONE, NONE] by design, a real
    turn lane isn't painted either), so draw_bg_lanes' and this function's
    own polylines are the ONLY rendering of them, and coarse sampling was
    the entire visible cause, not the underlying geometry (verified
    directly: lane endpoints connect exactly, no gap, anywhere a turn
    meets its own approach/exit). One sample every 3m is still fine for a
    straight lane (it draws as a single segment either way), so only
    curved lanes need the denser step."""
    step = 3.0 if isinstance(lane, StraightLane) else 0.5
    return max(2, int(lane.length / step))


def route_polyline(road, route, lateral: float):
    """World-space points tracing `route`, offset `lateral` meters
    perpendicular to the local direction of travel (the same +90-degree
    "direction_lateral" convention StraightLane itself uses) so two routes
    sharing the same underlying lanes -- e.g. HUMAN_ROUTE and ROBOT_ROUTE,
    which typically enter a scene from different arms and then converge
    onto one identical shared tail -- render as two visible parallel
    tracks instead of one completely hiding the other. Two formats: a
    list of (from, to, lane_id) lane indices to sample along (offset
    directly via each lane's own position(lon, lateral)), or a plain list
    of (x, y) world points -- e.g. HUMAN_ROUTE/ROBOT_ROUTE themselves, or
    a layout with no lane graph -- offset using each point's own local
    tangent (a finite difference against its neighbors, since a plain
    point has no lane to ask for a heading)."""
    if route and len(route[0]) == 2:
        pts = np.array(route, dtype=float)
        if lateral == 0 or len(pts) < 2:
            return [tuple(p) for p in pts]
        offset_pts = np.empty_like(pts)
        for i in range(len(pts)):
            prev_i, next_i = max(i - 1, 0), min(i + 1, len(pts) - 1)
            direction = pts[next_i] - pts[prev_i]
            norm = np.linalg.norm(direction)
            perp = np.array([-direction[1], direction[0]]) / norm if norm > 1e-9 else np.zeros(2)
            offset_pts[i] = pts[i] + lateral * perp
        return [tuple(p) for p in offset_pts]
    points = []
    for lane_index in route:
        lane = road.network.get_lane(lane_index)
        for lon in np.linspace(0, lane.length, _lane_sample_points(lane)):
            points.append(lane.position(lon, lateral))
    return points


def is_multi_route(route):
    """True if `route` is a list of routes (multiple robot cars) rather than
    one route: a route's own elements are 2- or 3-tuples of plain numbers,
    a multi-route's elements are themselves lists of such tuples."""
    return bool(route) and isinstance(route[0], (list, tuple)) and bool(route[0]) \
        and isinstance(route[0][0], (list, tuple))


def draw_route(surface, road, route, color, lateral: float, name: str, label: str):
    """Draw one route's polyline, plus a green marker at its first point and
    a red marker at its last -- so a route that starts or ends somewhere
    unintended (short of a real lane, off in open space, short of the
    network another vehicle would be dropped onto) is visible at a glance
    instead of requiring a zoomed crop to spot."""
    try:
        points = route_polyline(road, route, lateral)
    except Exception as e:
        print(f"  {name}: skipping {label} route: {type(e).__name__}: {e}")
        return
    pixels = [surface.vec2pix(p) for p in points]
    if len(pixels) >= 2:
        pygame.draw.lines(surface, color, False, pixels, max(surface.pix(0.3), ROUTE_WIDTH))
        pygame.draw.circle(surface, ROUTE_START_COLOR, pixels[0], ROUTE_MARKER_RADIUS)
        pygame.draw.circle(surface, ROUTE_END_COLOR, pixels[-1], ROUTE_MARKER_RADIUS)


def draw_routes(surface, road, route, palette, lateral: float, name: str, label: str):
    """Draw `route` -- one route, or a list of routes each in its own color,
    each a disconnected polyline (no line jumping between encounters)."""
    sub_routes = route if is_multi_route(route) else [route]
    for i, sub_route in enumerate(sub_routes):
        color = palette[i % len(palette)]
        sub_label = f"{label} {i + 1}" if len(sub_routes) > 1 else label
        draw_route(surface, road, sub_route, color, lateral, name, sub_label)


def draw_bg_lanes(surface, road, lane_indexes, name: str):
    """Highlight, in yellow, every lane scene1_background.add_background_traffic
    would actually consider as a spawn candidate -- the exact
    route_adjacent_lane_indexes() set watch.py passes it as `lane_indexes`
    (see watch.py: `route_lanes = module.route_adjacent_lane_indexes(); sb.add_background_traffic(..., lane_indexes=route_lanes)`).
    Drawn as a thin centerline overlay (not replacing the base line-type
    rendering) so a lane that's structurally isolated -- present in the
    network but never actually reachable by continuing forward from
    anywhere a background vehicle would be dropped -- is still visible as
    a plain white lane with no yellow overlay, distinguishable from one
    that's a real, connected spawn candidate. Spawning here is a one-time
    starting position, not a leash: once placed, a vehicle drives forward
    via its own lane-to-lane continuation regardless of whether the next
    fragment happens to be in this set, so this highlights *where a
    vehicle can start*, not everywhere it can end up."""
    for lane_index in lane_indexes:
        try:
            lane = road.network.get_lane(lane_index)
        except KeyError as e:
            print(f"  {name}: route_adjacent_lane_indexes gave a stale lane {lane_index}: {e}")
            continue
        pixels = [surface.vec2pix(lane.position(s, 0))
                  for s in np.linspace(0, lane.length, _lane_sample_points(lane))]
        if len(pixels) >= 2:
            pygame.draw.lines(surface, BG_LANE_COLOR, False, pixels, max(surface.pix(0.15), BG_LANE_WIDTH))


def draw_lane_arrows(surface, road):
    """One small triangular arrow at the midpoint of EVERY lane in the
    network (every (from, to, lane_id) edge, not just the human/robot's
    own route or the background-eligible subset draw_bg_lanes covers),
    pointing along that lane's own heading there -- so which way traffic
    on a given lane actually flows is visible directly on the lane itself,
    not just implied by which line is dashed vs solid. A lane's direction
    is fixed by construction (see build_scene.py: every one-way movement
    is its own separate lane object, position(s) always increasing s from
    start to end), so one arrow at the midpoint is exactly this lane's own
    direction everywhere along it, not just locally true near the sample
    point -- no need for more than one per lane except on a curve tight
    enough that showing only the midpoint heading could be mistaken for
    the whole lane being straight, which for how this codebase's own
    curves are used isn't a real risk.
    """
    for from_node, tos in road.network.graph.items():
        for to_node, lanes in tos.items():
            for lane in lanes:
                if lane.length < ARROW_MIN_LANE_LENGTH:
                    continue
                s = lane.length / 2
                position = np.array(lane.position(s, 0))
                heading = lane.heading_at(s)
                direction = np.array([np.cos(heading), np.sin(heading)])
                lateral = np.array([-direction[1], direction[0]])
                tip = position + (ARROW_LENGTH / 2) * direction
                tail_center = position - (ARROW_LENGTH / 2) * direction
                left = tail_center + (ARROW_WIDTH / 2) * lateral
                right = tail_center - (ARROW_WIDTH / 2) * lateral
                pygame.draw.polygon(surface, ARROW_COLOR,
                                     [surface.vec2pix(tip), surface.vec2pix(left), surface.vec2pix(right)])


def render_offscreen(name, size=None, show_bg_lanes=True):
    """Render a layout's road (+ HUMAN_ROUTE/ROBOT_ROUTE, + every lane
    background traffic could spawn on, if defined) to its own Surface, fit
    to the road's bounding box. None if it won't build."""
    try:
        module = load_layout(name)
        road = module.build_road()
    except Exception as e:
        print(f"  skipping {name}: {type(e).__name__}: {e}")
        return None

    w, h = size or TILE_SIZE
    surface = WorldSurface((w, h), 0, pygame.Surface((w, h)))
    min_x, max_x, min_y, max_y = road_bounding_box(road)
    span_x, span_y = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
    surface.scaling = min(w / (span_x * BBOX_MARGIN), h / (span_y * BBOX_MARGIN))
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    surface.origin = center - np.array([w / 2, h / 2]) / surface.scaling

    RoadGraphics.display(road, surface)
    draw_lane_arrows(surface, road)

    if show_bg_lanes:
        route_adjacent = getattr(module, "route_adjacent_lane_indexes", None)
        if route_adjacent is not None:
            try:
                lane_indexes = route_adjacent()
            except Exception as e:
                print(f"  {name}: route_adjacent_lane_indexes() failed: {type(e).__name__}: {e}")
                lane_indexes = None
            if lane_indexes:
                draw_bg_lanes(surface, road, lane_indexes, name)

    human_route = getattr(module, "HUMAN_ROUTE", None)
    robot_route = getattr(module, "ROBOT_ROUTE", None)
    if human_route:
        draw_routes(surface, road, human_route, [HUMAN_COLOR], -ROUTE_LATERAL_OFFSET, name, "human")
    if robot_route:
        draw_routes(surface, road, robot_route, ROBOT_PALETTE, ROUTE_LATERAL_OFFSET, name, "robot")

    return surface


def draw_grid(window, tiles, cols):
    """Scale each road tile to fit its cell and blit it with a caption."""
    font = pygame.font.SysFont(None, 20)
    win_w, win_h = window.get_size()
    rows = math.ceil(len(tiles) / cols)
    cell_w, cell_h = win_w / cols, win_h / rows

    window.fill(BG)
    for i, (name, surface) in enumerate(tiles):
        col, row = i % cols, i // cols
        avail_w, avail_h = cell_w - 8, cell_h - LABEL_H - 8
        scale = min(avail_w / surface.get_width(), avail_h / surface.get_height())
        w, h = int(surface.get_width() * scale), int(surface.get_height() * scale)

        x = int(col * cell_w + (cell_w - w) / 2)
        y = int(row * cell_h + LABEL_H + (avail_h - h) / 2)
        window.blit(pygame.transform.smoothscale(surface, (w, h)), (x, y))

        caption = font.render(name, True, LABEL)
        window.blit(caption, (int(col * cell_w + (cell_w - caption.get_width()) / 2),
                              int(row * cell_h + 4)))

    draw_legend(window, font)
    pygame.display.flip()


def legend_entries(module):
    """(color, label) pairs for whatever HUMAN_ROUTE/ROBOT_ROUTE a layout defines."""
    entries = []
    if getattr(module, "HUMAN_ROUTE", None):
        entries.append((HUMAN_COLOR, "human"))
    robot_route = getattr(module, "ROBOT_ROUTE", None)
    if robot_route:
        if is_multi_route(robot_route):
            entries += [(ROBOT_PALETTE[i % len(ROBOT_PALETTE)], f"robot {i + 1}")
                        for i in range(len(robot_route))]
        else:
            entries.append((ROBOT_COLOR, "robot"))
    return entries or [(HUMAN_COLOR, "human"), (ROBOT_COLOR, "robot")]


def draw_legend(window, font, entries=None, origin=(10, None)):
    entries = entries or [(HUMAN_COLOR, "human"), (ROBOT_COLOR, "robot")]
    x, y = origin[0], origin[1] if origin[1] is not None else window.get_height() - 24
    for color, label in entries:
        pygame.draw.line(window, color, (x, y + 7), (x + 20, y + 7), ROUTE_WIDTH)
        text = font.render(label, True, LABEL)
        window.blit(text, (x + 26, y))
        x += 26 + text.get_width() + 20


def resolve_scene(arg, names):
    """Match --scene against an exact layout name, or a bare/short number
    against real_NNN (e.g. "12" or "real_12" -> "real_012")."""
    if arg in names:
        return arg
    digits = arg[len("real_"):] if arg.startswith("real_") else arg
    if digits.isdigit():
        padded = f"real_{int(digits):03d}"
        if padded in names:
            return padded
    sys.exit(f"no layout matching --scene {arg!r}. Available: {', '.join(names)}")


def show_single(name, window):
    """Render one layout full-window, with a title and route legend."""
    w, h = window.get_size()
    print(f"rendering {name} from {LAYOUTS_DIR}")
    module = load_layout(name)
    surface = render_offscreen(name, size=(w, h - LABEL_H))
    if surface is None:
        sys.exit(f"{name} failed to build (see error above)")

    window.fill(BG)
    window.blit(surface, (0, LABEL_H))
    font = pygame.font.SysFont(None, 22)
    caption = font.render(name, True, LABEL)
    window.blit(caption, ((w - caption.get_width()) // 2, 2))
    draw_legend(window, font, entries=legend_entries(module), origin=(10, 6))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cols", type=int, default=None,
                        help="columns in the grid (default: roughly square)")
    parser.add_argument("--scene", type=str, default=None,
                        help="render just one layout full-window instead of the grid "
                             "(exact name, e.g. real_012, or a bare number as shorthand for real_NNN)")
    args = parser.parse_args()

    names = layout_names()
    if not names:
        sys.exit(f"no .py layouts in {LAYOUTS_DIR}")

    pygame.init()
    desktop = pygame.display.Info()
    window = pygame.display.set_mode((int(desktop.current_w * 0.9), int(desktop.current_h * 0.85)))

    if args.scene:
        name = resolve_scene(args.scene, names)
        pygame.display.set_caption(f"highway layout: {name}")
        show_single(name, window)
    else:
        cols = args.cols or math.ceil(math.sqrt(len(names)))
        pygame.display.set_caption(f"highway layouts ({len(names)})")
        print(f"rendering {len(names)} layouts from {LAYOUTS_DIR}")
        tiles = [(n, s) for n in names for s in [render_offscreen(n)] if s is not None]
        print(f"showing {len(tiles)} in a {cols}x{math.ceil(len(tiles) / cols)} grid")
        draw_grid(window, tiles, cols)

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
