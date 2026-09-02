"""PLAY -- you're in the human's seat. Limited field of view: everything
outside your cone, or hidden behind another vehicle, is BLACK, not
dimmed -- genuinely gone, matching "does not sense and know and act on
anything outside FOV." Mirrors steakhouse/misha's play.py human seat
(`surf.fill((0,0,0))`, only the visible part drawn back in on top).

    python play.py
    python play.py --fov 45
    python play.py --no-chase                    # whole-map view instead of the ego camera, still black
    python play.py --fov 360 --no-occlusion       # ablation: no mask at all -- the true unrestricted baseline
    python play.py --count 15 --seed 0 --steps 900

The human still drives itself (LimitedVisionHuman/IDM) -- this is the
"look through its eyes while it drives" view, not manual control yet.
For the full-visibility observer version (grey, not black, over the same
simulation), use watch.py instead.

Keys: q / ESC / close window -> quit.
"""
import argparse
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # highway_env/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layout"))
import display_all as d  # noqa: E402
import scene1_background as sb  # noqa: E402
import limit_vision_human as h  # noqa: E402
import fov_render as fr  # noqa: E402
from highway_env.road.graphics import RoadGraphics, WorldSurface  # noqa: E402

SCENE = "mega_scene"
BLACK_ALPHA = 255       # fully opaque -- genuinely hidden, not dimmed (see module docstring)
BLACK_COLOR = (0, 0, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=18, help="number of background vehicles (default 18 -- mega_scene's own capacity is much smaller than a real recorded map; 40+ was measured to gridlock permanently, not just slow down)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--dt", type=float, default=1 / 15, help="simulation timestep in seconds (default 1/15)")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many steps (default: run until closed)")
    parser.add_argument("--fov", type=float, default=h.FOV_DEG_DEFAULT, help=f"human's FOV cone width in degrees (default {h.FOV_DEG_DEFAULT})")
    parser.add_argument("--no-occlusion", action="store_true", help="disable occlusion (cone-only perception)")
    parser.add_argument("--no-fov", action="store_true", help="disable the cone too (omnidirectional -- with --no-occlusion, a true ablation baseline)")
    parser.add_argument("--debug-lines", action="store_true", help="also draw thin lines to nearby vehicles (green=visible, red=hidden)")
    parser.add_argument("--no-chase", action="store_true", help="whole-map camera instead of the ego/chase view (still black, not grey)")
    parser.add_argument("--scale", type=float, default=10.0, help="pixels per meter in chase mode (default 10)")
    args = parser.parse_args()

    module = d.load_layout(SCENE)
    road = module.build_road()
    human = h.add_human_vehicle(road, module.HUMAN_ROUTE, fov_deg=args.fov,
                                 enable_fov=not args.no_fov, enable_occlusion=not args.no_occlusion)
    sb.add_background_traffic(road, count=args.count, seed=args.seed)
    lane_indexes = sb.all_lane_indexes(road)
    print(f"PLAY {SCENE}: {len(road.vehicles)} vehicles (1 LimitedVisionHuman, fov={args.fov:.0f} "
          f"occlusion={'on' if not args.no_occlusion else 'off'}) -- everything outside your cone is BLACK")

    pygame.init()
    desktop = pygame.display.Info()
    w, hh = int(desktop.current_w * 0.9), int(desktop.current_h * 0.85)
    window = pygame.display.set_mode((w, hh))
    pygame.display.set_caption(f"PLAY -- human seat on {SCENE} (fov={args.fov:.0f})")
    render_size = (w, hh - d.LABEL_H)
    chase = not args.no_chase

    if not chase:
        surface = WorldSurface(render_size, 0, pygame.Surface(render_size))
        min_x, max_x, min_y, max_y = d.road_bounding_box(road)
        surface.scaling = min(w / ((max_x - min_x) * 1.15), render_size[1] / ((max_y - min_y) * 1.15))
        center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        surface.origin = center - np.array([w / 2, render_size[1] / 2]) / surface.scaling
    chase_anchor = (render_size[0] / 2.0, render_size[1] * 0.72)

    robot_route = getattr(module, "ROBOT_ROUTE", None)
    font = pygame.font.SysFont(None, 22)

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
        road.step(args.dt)
        h.advance_vehicles_with_route(road, lane_indexes)
        step_count += 1
        if args.steps is not None and step_count >= args.steps:
            running = False

        candidates = sb.nearby_vehicles(road, human, fr.CONE_RADIUS)
        visible = human.visible_candidates(candidates)
        visible_ids = {id(v) for v in visible}
        progress, _ = h._route_progress(module.HUMAN_ROUTE, human.position)
        total_len = h.route_total_length(module.HUMAN_ROUTE)

        if chase:
            buf = fr.chase_camera_buffer(human, args.scale, render_size, chase_anchor)
            buf.fill(buf.GREY)
            RoadGraphics.display(road, buf)
            RoadGraphics.display_traffic(road, buf, simulation_frequency=round(1 / args.dt), offscreen=True)
            d.draw_routes(buf, road, module.HUMAN_ROUTE, [d.HUMAN_COLOR], -d.ROUTE_LATERAL_OFFSET, SCENE, "human")
            if robot_route:
                d.draw_routes(buf, road, robot_route, d.ROBOT_PALETTE, d.ROUTE_LATERAL_OFFSET, SCENE, "robot")
            fr.draw_fov_mask(buf, human, candidates, args.fov, not args.no_fov, not args.no_occlusion,
                              BLACK_ALPHA, color=BLACK_COLOR)
            fr.redraw_ego(buf, human)
            if args.debug_lines:
                fr.draw_debug_lines(buf, human, visible_ids, candidates)
            frame = fr.render_chase_frame(buf, human, render_size, chase_anchor, d.BG)
        else:
            surface.fill(surface.GREY)
            RoadGraphics.display(road, surface)
            RoadGraphics.display_traffic(road, surface, simulation_frequency=round(1 / args.dt), offscreen=True)
            d.draw_routes(surface, road, module.HUMAN_ROUTE, [d.HUMAN_COLOR], -d.ROUTE_LATERAL_OFFSET, SCENE, "human")
            if robot_route:
                d.draw_routes(surface, road, robot_route, d.ROBOT_PALETTE, d.ROUTE_LATERAL_OFFSET, SCENE, "robot")
            fr.draw_fov_mask(surface, human, candidates, args.fov, not args.no_fov, not args.no_occlusion,
                              BLACK_ALPHA, color=BLACK_COLOR)
            fr.redraw_ego(surface, human)
            if args.debug_lines:
                fr.draw_debug_lines(surface, human, visible_ids, candidates)
            frame = surface

        window.fill(d.BG)
        window.blit(frame, (0, d.LABEL_H))
        window.blit(font.render(
            f"PLAY {SCENE}  step {step_count}  ({len(road.vehicles)} vehicles)  "
            f"human: speed={human.speed:.1f} progress={progress/total_len*100:.0f}% "
            f"sees {len(visible)}/{len(candidates)} nearby  [q/ESC quit]",
            True, d.LABEL), (10, 2))
        pygame.display.flip()
        clock.tick(round(1 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
