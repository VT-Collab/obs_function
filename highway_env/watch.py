"""WATCH LimitedVisionHuman drive scene 1 -- you see the FULL scene; the
human's blind area (out of cone, or occluded by another vehicle) is
GREYED, not hidden, so you can see both what's really there and what the
human can/can't see of it at the same time. Mirrors steakhouse/misha's own
watch.py: full state for the observer, grey over the human's blind spot.

    python watch.py
    python watch.py --fov 45
    python watch.py --chase                      # observer camera in the human's seat, still grey not black
    python watch.py --fov 360 --no-occlusion      # ablation: no tint at all
    python watch.py --count 15 --seed 0 --steps 900

For the actually-blacked-out, "you are the one who can't see it" version,
use play.py instead -- same simulation, same flags, different tint.

A robot vehicle also drives ROBOT_ROUTE here (full visibility, no FOV
limit), running a baseline policy that -- for --robot-policy fov_aware,
the default -- reads a live FOVPosterior belief about what FOV the human
currently appears to have, inferred purely from how it's driving. Its
current best guess is shown on the label bar. Disable with --no-robot.

Keys: q / ESC / close window -> quit.
"""
import argparse
import os
import sys
import types

import numpy as np
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))  # highway_env/
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "human"))
sys.path.insert(0, os.path.join(_HERE, "layout"))
import display_all as d  # noqa: E402
import scene1_background as sb  # noqa: E402
import limit_vision_human as h  # noqa: E402
import fov_render as fr  # noqa: E402
from highway_env.road.graphics import RoadGraphics, WorldSurface  # noqa: E402
from robot.methods import make_baseline, combo_names, resolve_combo, describe_robots  # noqa: E402
from robot.nominal_policy.vehicles import make_vehicle  # noqa: E402
from robot.filter.core.fov_posterior import FOVPosterior  # noqa: E402

SCENE = "real_001_rebuilt"
BLIND_GREY = 130            # alpha -- moderate, matches steakhouse/misha's BLIND_GREY=120 (dim, not hidden)
BLIND_COLOR = (70, 70, 82)  # same bluish-grey tint steakhouse/misha uses for the human's blind area
FOV_CANDIDATES = (30.0, 60.0, 90.0, 180.0, 360.0)
STATUS_COLOR = (255, 230, 0)  # bright yellow -- readable over both road grey and the blind-area tint
STATUS_OFFSET_PX = 14  # how far above each vehicle's own center the --debug-status label sits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=SCENE, help=f"layout to load (default {SCENE}; e.g. real_005_rebuilt for a different junction chain -- see layout/layouts/ for the full set, or layout/other_layouts/ for mega_scene, crossing_turn, and the real_NNN map-derived scenes)")
    parser.add_argument("--count", type=int, default=18, help="number of background vehicles (default 18 -- these compact chained-junction layouts have much smaller capacity than a real recorded map; mega_scene measured 40+ as a permanent gridlock, not just a slowdown, and these are similarly small)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--dt", type=float, default=1 / 15, help="simulation timestep in seconds (default 1/15)")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many steps (default: run until closed)")
    parser.add_argument("--fov", type=float, default=h.FOV_DEG_DEFAULT, help=f"human's FOV cone width in degrees (default {h.FOV_DEG_DEFAULT})")
    parser.add_argument("--no-occlusion", action="store_true", help="disable occlusion (cone-only perception)")
    parser.add_argument("--no-fov", action="store_true", help="disable the cone too (omnidirectional -- with --no-occlusion, a true ablation baseline)")
    parser.add_argument("--grey-alpha", type=int, default=BLIND_GREY, help=f"darkness of the grey overlay, 0-255 (default {BLIND_GREY})")
    parser.add_argument("--debug-lines", action="store_true", help="also draw thin lines to nearby vehicles (green=visible, red=hidden)")
    parser.add_argument("--debug-status", action="store_true",
                         help="label every stopped vehicle with WHY it's stopped (END OF ROAD / SAME LANE <gap> "
                              "-- an ordinary leader ahead / CROSS TRAFFIC -- a different-heading vehicle whose "
                              "own path crosses this one, e.g. at a junction / STOPPED -- no specific cause "
                              "found), re-derived from the exact same checks that vehicle's own driving logic "
                              "runs -- see scene1_background.vehicle_status_label")
    parser.add_argument("--chase", action="store_true", help="ego/chase camera centered on the human, rotated so its heading always points up (still grey, not black -- you're still the observer)")
    parser.add_argument("--scale", type=float, default=10.0, help="pixels per meter in --chase mode (default 10)")
    parser.add_argument("--no-robot", action="store_true", help="skip spawning the robot + live FOV posterior readout")
    parser.add_argument("--robot-policy", default="fov_aware", choices=["nominal", "cautious", "fov_aware"],
                         help="policy WRAPPER the robot drives with (default fov_aware) -- whether/how extra caution is blended in")
    parser.add_argument("--robot-vehicle", default="idm", choices=["idm", "linear", "aggressive", "defensive"],
                         help="underlying vehicle DYNAMICS the robot uses (default idm) -- orthogonal to "
                              "--robot-policy, e.g. --robot-vehicle defensive --robot-policy fov_aware")
    parser.add_argument("--robot", default=None,
                         help="shorthand for --robot-vehicle/--robot-policy TOGETHER, as one "
                              "\"{vehicle}_{policy}\" name (e.g. --robot defensive_fov_aware) -- overrides both "
                              "flags above when given. --list-robots prints every valid name.")
    parser.add_argument("--list-robots", action="store_true", help="print every valid --robot combo name and exit")
    args = parser.parse_args()

    if args.list_robots:
        print(describe_robots())
        return
    if args.robot is not None:
        combo = resolve_combo(args.robot)
        if combo is None:
            raise SystemExit(f"unknown --robot {args.robot!r}. Choose from:\n  " +
                              "\n  ".join(combo_names()))
        args.robot_vehicle, args.robot_policy = combo

    module = d.load_layout(args.scene)
    road = module.build_road()
    human = h.add_human_vehicle(road, module.HUMAN_ROUTE, fov_deg=args.fov,
                                 enable_fov=not args.no_fov, enable_occlusion=not args.no_occlusion)

    robot_route = getattr(module, "ROBOT_ROUTE", None)
    robot = None
    if robot_route and not args.no_robot:
        robot = make_vehicle(args.robot_vehicle, road, np.asarray(robot_route[0], dtype=float),
                              heading=0.0, speed=9.0)
        robot.route_points = robot_route
        lane_index = road.network.get_closest_lane_index(robot.position)
        lane = road.network.get_lane(lane_index)
        lon, _ = lane.local_coordinates(robot.position)
        robot.position, robot.heading = lane.position(max(lon, 0.0), 0), lane.heading_at(max(lon, 0.0))
        robot.lane_index, robot.lane, robot.target_lane_index = lane_index, lane, lane_index
        robot.color = (255, 165, 0)  # orange -- VehicleGraphics.get_color() checks .color before its type defaults
        road.vehicles.append(robot)
    robot_baseline = make_baseline(args.robot_policy) if robot is not None else None
    posterior = FOVPosterior(FOV_CANDIDATES) if robot is not None else None

    route_lanes = getattr(module, "route_adjacent_lane_indexes", lambda: None)()
    sb.add_background_traffic(road, count=args.count, seed=args.seed, lane_indexes=route_lanes)
    lane_indexes = sb.all_lane_indexes(road)
    print(f"WATCH {args.scene}: {len(road.vehicles)} vehicles (1 LimitedVisionHuman, fov={args.fov:.0f} "
          f"occlusion={'on' if not args.no_occlusion else 'off'}"
          f"{f', robot={args.robot_vehicle}/{args.robot_policy}' if robot is not None else ''}"
          f"{', chase cam' if args.chase else ''}) -- grey is what the HUMAN cannot see, you see it all")

    pygame.init()
    desktop = pygame.display.Info()
    w, hh = int(desktop.current_w * 0.9), int(desktop.current_h * 0.85)
    window = pygame.display.set_mode((w, hh))
    pygame.display.set_caption(f"WATCH -- LimitedVisionHuman on {args.scene} (fov={args.fov:.0f})")
    render_size = (w, hh - d.LABEL_H)

    if not args.chase:
        surface = WorldSurface(render_size, 0, pygame.Surface(render_size))
        min_x, max_x, min_y, max_y = d.road_bounding_box(road)
        surface.scaling = min(w / ((max_x - min_x) * 1.15), render_size[1] / ((max_y - min_y) * 1.15))
        center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        surface.origin = center - np.array([w / 2, render_size[1] / 2]) / surface.scaling
    chase_anchor = (render_size[0] / 2.0, render_size[1] * 0.72)

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
        h.apply_human_aware_car_following(road, lane_indexes, args.dt)

        if robot is not None and not robot.crashed:
            candidates_r = sb.nearby_vehicles(road, robot, 35.0)
            front_r = sb.find_front_vehicle(road, robot, lane_indexes, candidates_r)
            # See play.py's identical block for why this is recomputed and
            # threaded through ctx.crossing_conflict_accel instead of
            # min()-ed against the filter's result afterward.
            heads_r = {id(v) for v in candidates_r
                       if sb.find_front_vehicle(road, v, lane_indexes, sb.nearby_vehicles(road, v, 35.0)) is None}
            if front_r is None:
                heads_r.add(id(robot))
            conflict_r = sb.crossing_conflict_brake(robot, candidates_r, heads=heads_r)
            belief = posterior.beliefs() if args.robot_policy == "fov_aware" else None
            ctx = types.SimpleNamespace(robot=robot, front_vehicle=front_r,
                                         human=human if not human.crashed else None, belief=belief, dt=args.dt,
                                         crossing_conflict_accel=conflict_r)
            final = robot_baseline.action(ctx)
            if conflict_r is not None:
                final = min(final, conflict_r)
            robot.action["acceleration"] = max(final, -robot.speed / args.dt)

        road.step(args.dt)
        h.advance_vehicles_with_route(road, lane_indexes)

        if posterior is not None and not human.crashed:
            candidates_h = sb.nearby_vehicles(road, human, 35.0)
            posterior.update(human, candidates_h, human.action["acceleration"],
                              front_vehicle_fn=lambda vis: sb.find_front_vehicle(road, human, lane_indexes, vis))

        step_count += 1
        if args.steps is not None and step_count >= args.steps:
            running = False

        candidates = sb.nearby_vehicles(road, human, fr.CONE_RADIUS)
        visible = human.visible_candidates(candidates)
        visible_ids = {id(v) for v in visible}
        progress, _ = h._route_progress(module.HUMAN_ROUTE, human.position)
        total_len = h.route_total_length(module.HUMAN_ROUTE)

        if args.chase:
            buf = fr.chase_camera_buffer(human, args.scale, render_size, chase_anchor)
            buf.fill(buf.GREY)
            RoadGraphics.display(road, buf)
            d.draw_lane_arrows(buf, road)
            RoadGraphics.display_traffic(road, buf, simulation_frequency=round(1 / args.dt), offscreen=True)
            d.draw_routes(buf, road, module.HUMAN_ROUTE, [d.HUMAN_COLOR], -d.ROUTE_LATERAL_OFFSET, args.scene, "human")
            if robot_route:
                d.draw_routes(buf, road, robot_route, d.ROBOT_PALETTE, d.ROUTE_LATERAL_OFFSET, args.scene, "robot")
            fr.draw_fov_mask(buf, human, candidates, args.fov, not args.no_fov, not args.no_occlusion,
                              args.grey_alpha, color=BLIND_COLOR)
            fr.redraw_ego(buf, human)
            fr.redraw_partial(buf, human, candidates, args.fov, not args.no_fov, not args.no_occlusion)
            if args.debug_lines:
                fr.draw_debug_lines(buf, human, visible_ids, candidates)
            frame = fr.render_chase_frame(buf, human, render_size, chase_anchor, d.BG)
        else:
            surface.fill(surface.GREY)
            RoadGraphics.display(road, surface)
            d.draw_lane_arrows(surface, road)
            RoadGraphics.display_traffic(road, surface, simulation_frequency=round(1 / args.dt), offscreen=True)
            d.draw_routes(surface, road, module.HUMAN_ROUTE, [d.HUMAN_COLOR], -d.ROUTE_LATERAL_OFFSET, args.scene, "human")
            if robot_route:
                d.draw_routes(surface, road, robot_route, d.ROBOT_PALETTE, d.ROUTE_LATERAL_OFFSET, args.scene, "robot")
            fr.draw_fov_mask(surface, human, candidates, args.fov, not args.no_fov, not args.no_occlusion,
                              args.grey_alpha, color=BLIND_COLOR)
            fr.redraw_ego(surface, human)
            fr.redraw_partial(surface, human, candidates, args.fov, not args.no_fov, not args.no_occlusion)
            if args.debug_lines:
                fr.draw_debug_lines(surface, human, visible_ids, candidates)
            frame = surface

        if args.debug_status:
            for v in road.vehicles:
                label = sb.vehicle_status_label(road, v, lane_indexes)
                if label is None:
                    continue
                if args.chase:
                    sx, sy = fr.chase_screen_pos(human, v.position, args.scale, chase_anchor)
                else:
                    sx, sy = surface.vec2pix(v.position)
                text = status_font.render(label, True, STATUS_COLOR)
                frame.blit(text, (sx - text.get_width() / 2.0, sy - STATUS_OFFSET_PX))

        belief_str = ""
        if posterior is not None:
            belief_str = f"  robot believes human fov={posterior.map_fov():.0f} ({posterior.belief_for(posterior.map_fov())*100:.0f}%)"

        window.fill(d.BG)
        window.blit(frame, (0, d.LABEL_H))
        window.blit(font.render(
            f"WATCH {args.scene}  step {step_count}  ({len(road.vehicles)} vehicles)  "
            f"human: speed={human.speed:.1f} progress={progress/total_len*100:.0f}% "
            f"sees {len(visible)}/{len(candidates)} nearby{belief_str}  [q/ESC quit]",
            True, d.LABEL), (10, 2))
        pygame.display.flip()
        clock.tick(round(1 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
