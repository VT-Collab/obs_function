"""WATCH the new deterministic priority_traffic rule drive a layout's own
background traffic -- PROTOTYPE, lives in highway_env/test/ on purpose so
nothing in the validated codebase (scene_background.py, watch.py, any
real_NNN layout) is touched while this is tested. See priority_traffic.py's
own module docstring for the full rule this is visualizing.

Deliberately narrower than the main highway_env/watch.py: no human/FOV
cone, no robot/FOV-posterior -- this test is specifically about whether
background traffic flows correctly under the new rule, which those don't
touch either way (they're a separate, already-validated system this
prototype has no reason to also pull in and risk). Same rendering style
(RoadGraphics.display + draw_lane_arrows + RoadGraphics.display_traffic +
a status bar), so it looks and feels the same to watch.

    python watch.py
    python watch.py --count 45          # the same stress-test density
                                         # that exposed RegulatedRoad's
                                         # own gridlock bug
    python watch.py --scene real_004_rebuilt --count 30 --seed 2
    python watch.py --no-debug-priority # turn off the on-screen labels

Keys: q / ESC / close window -> quit.
"""
import argparse
import os
import sys

# THE FIFTEENTH FIX for a real, measured bug -- not in this module's own
# code, in how deterministic "same --seed, same outcome" actually is:
# confirmed directly, running the exact same scene/seed/count repeatedly
# produced genuinely different results (0 vs 2 crashed vehicles, no code
# change at all) even after seeding both add_background_traffic's own RNG
# AND Road's own np_random (see this file's own Road(...) construction
# below). The remaining source is Python's OWN per-process hash
# randomization (PYTHONHASHSEED, on by default, different every process
# start) -- confirmed directly, forcing PYTHONHASHSEED=0 in the shell
# environment before launching made the SAME run reproduce identically
# every time. Some string-keyed set/dict iteration order somewhere below
# this file (not in this file's own code, and not something to go hunting
# for and editing in validated scene_background.py or highway_env's own
# library) is sensitive to it. PYTHONHASHSEED can only be set before the
# interpreter starts, not from inside an already-running one, so this
# re-execs the process ONCE with it fixed, before anything else (numpy,
# pygame, this project's own modules) ever imports -- pure addition, no
# existing code path is touched by this.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ENV_ROOT)
sys.path.insert(0, os.path.join(_ENV_ROOT, "layout"))
sys.path.insert(0, os.path.join(_ENV_ROOT, "layout", "layouts"))

import numpy as np
import pygame

import display_all as d  # noqa: E402 -- read-only import from the real codebase
import scene_background as sb  # noqa: E402 -- read-only import from the real codebase
import priority_traffic as pt  # noqa: E402 -- this prototype's own new module
from highway_env.road.graphics import RoadGraphics, WorldSurface  # noqa: E402
from highway_env.road.road import Road  # noqa: E402

SCENE = "real_001_rebuilt"
STATUS_COLOR = (0, 230, 255)  # cyan -- matches the main watch.py's own --debug-right-of-way color
STATUS_OFFSET_PX = 14


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=SCENE, help=f"layout to load (default {SCENE})")
    parser.add_argument("--count", type=int, default=100, help="number of background vehicles (default 18, matching the main watch.py's own default)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--dt", type=float, default=1 / 15, help="simulation timestep in seconds (default 1/15)")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many steps (default: run until closed)")
    parser.add_argument("--no-debug-priority", action="store_true",
                         help="hide the per-vehicle priority-rank / yield-target labels (on by default here, "
                              "since surfacing them is this prototype's whole point)")
    args = parser.parse_args()

    module = d.load_layout(args.scene)
    # Plain Road, not RegulatedRoad -- this prototype replaces RegulatedRoad's
    # own respect_priorities()/enforce_road_rules() entirely with priority_
    # traffic.apply_priority_driving(), it doesn't layer on top of it.
    #
    # np_random=args.seed (not the default): highway_env's own Road.__init__
    # defaults np_random to the GLOBAL, unseeded np.random module when not
    # given one explicitly -- add_background_traffic's own spawn placement
    # uses its OWN locally-seeded generator so that part is reproducible,
    # but anything a vehicle's own behavior draws from road.np_random
    # (highway_env's own IDMVehicle/ControlledVehicle machinery) was NOT,
    # confirmed directly: running this exact scene/seed/count repeatedly
    # produced a genuinely different outcome (0 vs 2 crashed vehicles) with
    # no code change at all. Seeding this explicitly makes --seed actually
    # mean "this exact run, reproducibly" end to end, not just "this exact
    # spawn layout."
    road = Road(network=module.build_road().network, np_random=np.random.RandomState(args.seed))

    route_lanes = getattr(module, "route_adjacent_lane_indexes", lambda: None)()
    # THE FOURTEENTH FIX (see priority_traffic.spawn_safe_lane_indexes'
    # own docstring): add_background_traffic itself is untouched -- this
    # just narrows the candidate-lane list it's handed, the same
    # mechanism its own docstring already documents for a different
    # reason, so a vehicle is never spawned already inside a junction's
    # interior or on a ramp that dumps straight into a roundabout/merge.
    spawn_lanes = pt.spawn_safe_lane_indexes(
        route_lanes if route_lanes is not None else sb.all_lane_indexes(road))
    # THE EIGHTEENTH FIX (see priority_traffic.SPAWN_SAFE_DISTANCE/
    # maintain_vehicle_count's own docstring): add_background_traffic's
    # own default safe_distance (10.0) only guards against spawning two
    # vehicles literally on top of each other, not against spawning an
    # already-moving vehicle too close to react to one already stopped on
    # a lane that runs close and parallel without being the same lane.
    sb.add_background_traffic(road, count=args.count, seed=args.seed, lane_indexes=spawn_lanes,
                               safe_distance=pt.SPAWN_SAFE_DISTANCE)
    lane_indexes = sb.all_lane_indexes(road)
    # THE SIXTEENTH FIX (see priority_traffic.maintain_vehicle_count's own
    # docstring): a real stress test of "N vehicles, no crash, no freeze"
    # needs N vehicles active for the WHOLE run, not just at t=0. Its own
    # generator instance, kept alive across ticks by the caller (this
    # loop) and advanced once per top-up -- not reused/re-seeded per call,
    # or every top-up would draw the exact same spawn again.
    replenish_rng = np.random.default_rng(args.seed)
    print(f"WATCH (priority_traffic prototype) {args.scene}: {len(road.vehicles)} vehicles "
          f"(magenta = yielding to a higher-priority conflict)")

    pygame.init()
    desktop = pygame.display.Info()
    w, h = int(desktop.current_w * 0.9), int(desktop.current_h * 0.85)
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"priority_traffic prototype -- {args.scene}")
    render_h = h - d.LABEL_H

    surface = WorldSurface((w, render_h), 0, pygame.Surface((w, render_h)))
    min_x, max_x, min_y, max_y = d.road_bounding_box(road)
    surface.scaling = min(w / ((max_x - min_x) * 1.15), render_h / ((max_y - min_y) * 1.15))
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
    surface.origin = center - np.array([w / 2, render_h / 2]) / surface.scaling

    human_route = getattr(module, "HUMAN_ROUTE", None)
    robot_route = getattr(module, "ROBOT_ROUTE", None)
    font = pygame.font.SysFont(None, 22)
    status_font = pygame.font.SysFont(None, 16, bold=True)

    running = True
    step_count = 0
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False

        road.act()
        pt.apply_priority_driving(road, lane_indexes, args.dt)
        road.step(args.dt)
        pt.advance_vehicles_keep_crashed(road, lane_indexes)
        pt.maintain_vehicle_count(road, args.count, spawn_lanes, replenish_rng)
        step_count += 1
        if args.steps is not None and step_count >= args.steps:
            running = False

        surface.fill(surface.GREY)
        RoadGraphics.display(road, surface)
        d.draw_lane_arrows(surface, road)
        RoadGraphics.display_traffic(road, surface, simulation_frequency=round(1 / args.dt), offscreen=True)
        if human_route:
            d.draw_routes(surface, road, human_route, [d.HUMAN_COLOR], -d.ROUTE_LATERAL_OFFSET, args.scene, "human")
        if robot_route:
            d.draw_routes(surface, road, robot_route, d.ROBOT_PALETTE, d.ROUTE_LATERAL_OFFSET, args.scene, "robot")

        if not args.no_debug_priority:
            for v in road.vehicles:
                label = pt.priority_debug_label(v)
                if label is None:
                    continue
                sx, sy = surface.vec2pix(v.position)
                text = status_font.render(label, True, STATUS_COLOR)
                surface.blit(text, (sx - text.get_width() / 2.0, sy - STATUS_OFFSET_PX))

        n_yielding = sum(1 for v in road.vehicles if getattr(v, "is_priority_yielding", False))
        avg_speed = sum(v.speed for v in road.vehicles) / max(len(road.vehicles), 1)
        window.fill(d.BG)
        window.blit(surface, (0, d.LABEL_H))
        window.blit(font.render(
            f"priority_traffic -- {args.scene}  step {step_count}  ({len(road.vehicles)} vehicles, "
            f"{n_yielding} yielding)  avg speed={avg_speed:.1f}m/s  [q/ESC quit]",
            True, d.LABEL), (10, 2))
        pygame.display.flip()
        clock.tick(round(1 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
