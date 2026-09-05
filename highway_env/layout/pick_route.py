"""Click out a HUMAN_ROUTE / ROBOT_ROUTE by hand, for a point-based layout
(a real_*.py scene -- one whose routes are plain (x, y) world points, not
lane-graph indices).

    python pick_route.py --scene real_001
    python pick_route.py --scene 12

Controls:
    left click     add a point to the active route, at the cursor
    right click    remove the nearest point on the active route
    h / r          switch the active route to human / robot
    z              undo the last point added to the active route
    c              clear the active route entirely
    arrow keys     pan
    -  / =         zoom out / in
    p              print both routes as Python literals, ready to paste
                    into HUMAN_ROUTE / ROBOT_ROUTE in the layout file
    q / ESC        quit (prints the routes one last time first)

The road is rendered read-only underneath (same RoadGraphics as
display_all.py) so you can see exactly what you're tracing over. Whatever
routes the layout already defines are shown first, dimmed, as a starting
point -- click 'c' then start fresh if you don't want them, or just click
more points to extend/edit what's there.

Routes are saved to pick_route_sessions/<scene>.json when you quit (q/ESC
or closing the window) -- the next time you run `pick_route.py --scene
<same>` it resumes exactly where you left off, so forgetting to paste the
printed output before quitting no longer loses the work. This does NOT
cover a hard kill (Ctrl+C, force-quit, crash) -- that still bypasses the
save, same as before. Pass --reset to discard a scene's saved session and
start clean instead.

This is for point-based routes only. A lane-graph layout's routes are
(from_node, to_node, lane_id) tuples referencing real edges in that
layout's RoadNetwork -- there's no coordinate to click. For that case, list
the layout's own available edges instead, e.g.:

    python -c "
    import sys; sys.path.insert(0, '.')
    import display_all as d
    road = d.load_layout('mega_scene').build_road()
    for f, tos in road.network.graph.items():
        for t, lanes in tos.items():
            print(f, '->', t, 'lane_ids 0..', len(lanes) - 1)
    "

and build the route by hand from edges that actually chain (each tuple's
`to` matching the next tuple's `from`).
"""
import argparse
import json
import os
import sys

import numpy as np
import pygame

import display_all as d
from highway_env.road.regulation import RegulatedRoad

POINT_RADIUS = 5
DIM_ALPHA = 90

SESSIONS_DIR = os.path.join(d.HERE, "pick_route_sessions")


def session_path(scene_name):
    return os.path.join(SESSIONS_DIR, f"{scene_name}.json")


def load_session(scene_name):
    path = session_path(scene_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {"human": [tuple(p) for p in data.get("human", [])],
            "robot": [tuple(p) for p in data.get("robot", [])]}


def save_session(scene_name, routes):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    with open(session_path(scene_name), "w") as f:
        json.dump({"human": routes["human"], "robot": routes["robot"]}, f)


def pix2world(surface, pixel):
    return np.array([pixel[0] / surface.scaling + surface.origin[0],
                      pixel[1] / surface.scaling + surface.origin[1]])


def format_route(points):
    if not points:
        return "[]"
    lines = ",\n    ".join(f"({p[0]:.3f}, {p[1]:.3f})" for p in points)
    return f"[\n    {lines},\n]"


def draw_scene(window, surface, road, starting, routes, active):
    surface.fill(surface.GREY)
    from highway_env.road.graphics import RoadGraphics
    RoadGraphics.display(road, surface)

    # Dimmed starting routes (whatever the layout file already had), for reference.
    for label, color in (("human", d.HUMAN_COLOR), ("robot", d.ROBOT_COLOR)):
        pts = starting.get(label)
        if pts and len(pts) >= 2:
            dim = tuple(int(c * 0.5) for c in color)
            pygame.draw.lines(surface, dim, False, [surface.vec2pix(p) for p in pts], 2)

    for label, color in (("human", d.HUMAN_COLOR), ("robot", d.ROBOT_COLOR)):
        pts = routes[label]
        pixels = [surface.vec2pix(p) for p in pts]
        if len(pixels) >= 2:
            pygame.draw.lines(surface, color, False, pixels, d.ROUTE_WIDTH)
        for p in pixels:
            pygame.draw.circle(surface, color, p, POINT_RADIUS, 0 if label == active else 1)

    window.blit(surface, (0, d.LABEL_H))
    font = pygame.font.SysFont(None, 22)
    status = (f"active: {active}   human pts: {len(routes['human'])}   "
              f"robot pts: {len(routes['robot'])}   "
              f"[h/r switch] [click add, right-click remove] [z undo] [c clear] [p print] [q quit]")
    window.fill(d.BG, (0, 0, window.get_width(), d.LABEL_H))
    window.blit(font.render(status, True, d.LABEL), (10, 2))
    pygame.display.flip()


def print_routes(routes):
    print("\nHUMAN_ROUTE = " + format_route(routes["human"]))
    print("\nROBOT_ROUTE = " + format_route(routes["robot"]))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="layout name or bare number, e.g. real_001 or 1")
    parser.add_argument("--reset", action="store_true", help="discard this scene's saved session and start clean")
    args = parser.parse_args()

    names = d.layout_names()
    name = d.resolve_scene(args.scene, names)
    module = d.load_layout(name)
    # road = module.build_road()  # old plain-Road path, commented out -- RegulatedRoad
    # is now the project-wide default. Inert here (this tool only lets you click points
    # on a static render, never steps the road), kept for consistency.
    road = RegulatedRoad(network=module.build_road().network)

    def as_point_list(route):
        # Only a flat list of (x, y) points can be shown/edited here -- a
        # multi-route (list of routes) or a lane-index route isn't a set of
        # clickable coordinates, so just show nothing rather than crash.
        if not route or d.is_multi_route(route) or len(route[0]) != 2:
            return []
        return list(route)

    starting = {
        "human": as_point_list(getattr(module, "HUMAN_ROUTE", None)),
        "robot": as_point_list(getattr(module, "ROBOT_ROUTE", None)),
    }

    if args.reset:
        path = session_path(name)
        if os.path.exists(path):
            os.remove(path)
        routes = {"human": [], "robot": []}
    else:
        routes = load_session(name) or {"human": [], "robot": []}
        if routes["human"] or routes["robot"]:
            print(f"resumed saved session for {name} from {session_path(name)} "
                  f"({len(routes['human'])} human pts, {len(routes['robot'])} robot pts)")

    pygame.init()
    desktop = pygame.display.Info()
    w, h = int(desktop.current_w * 0.9), int(desktop.current_h * 0.85)
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"pick_route: {name}")

    from highway_env.road.graphics import WorldSurface
    surface = WorldSurface((w, h - d.LABEL_H), 0, pygame.Surface((w, h - d.LABEL_H)))
    min_x, max_x, min_y, max_y = d.road_bounding_box(road)
    surface.scaling = min((w) / ((max_x - min_x) * 1.15), (h - d.LABEL_H) / ((max_y - min_y) * 1.15))
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    surface.origin = center - np.array([w / 2, (h - d.LABEL_H) / 2]) / surface.scaling

    active = "human"
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[1] >= d.LABEL_H:
                click_pos = (event.pos[0], event.pos[1] - d.LABEL_H)
                if event.button == 1:
                    routes[active].append(tuple(pix2world(surface, click_pos)))
                elif event.button == 3 and routes[active]:
                    world = pix2world(surface, click_pos)
                    dists = [np.linalg.norm(np.array(p) - world) for p in routes[active]]
                    routes[active].pop(int(np.argmin(dists)))
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_h:
                    active = "human"
                elif event.key == pygame.K_r:
                    active = "robot"
                elif event.key == pygame.K_z and routes[active]:
                    routes[active].pop()
                elif event.key == pygame.K_c:
                    routes[active] = []
                elif event.key == pygame.K_p:
                    print_routes(routes)
                elif event.key == pygame.K_EQUALS:
                    surface.scaling *= 1.2
                elif event.key == pygame.K_MINUS:
                    surface.scaling /= 1.2
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    step = 40 / surface.scaling
                    delta = {pygame.K_LEFT: (-step, 0), pygame.K_RIGHT: (step, 0),
                             pygame.K_UP: (0, -step), pygame.K_DOWN: (0, step)}[event.key]
                    surface.origin = surface.origin + np.array(delta)

        draw_scene(window, surface, road, starting, routes, active)
        clock.tick(30)

    save_session(name, routes)
    print_routes(routes)
    print(f"session autosaved to {session_path(name)}")
    pygame.quit()


if __name__ == "__main__":
    main()
