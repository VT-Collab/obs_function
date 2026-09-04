"""Playable SUBTASK-ONLY interface for the highway user study.

    python interface.py --layout real_001_rebuilt --fov 90 --robot-vehicle idm --robot-policy fov_aware
    python interface.py --list                                the layouts you can name

SAME simulation/rendering as play.py -- you're in the human's seat, cone
and all, everything outside it BLACK -- see play.py's own header for what
that means. THE ONE DIFFERENCE: you never hold UP/DOWN. The car drives
itself along an ORDINARY forward stretch automatically, no click needed;
you click a MANEUVER -- turn, change lane, merge into roundabout, exit
roundabout -- to authorize the one specific maneuver the fixed path needs
next, or click WAIT to hold it back (click WAIT again to resume automatic
driving). ApproximateLimitVisionHuman executes whichever maneuver was
authorized to completion, exactly as built into the path -- it never
deviates from HUMAN_ROUTE, and once a maneuver is under way nothing
(including WAIT) can interrupt it partway. What's different from play.py
is entirely upstream of that driving: WHICH maneuver is authorized right
now is your click, not a continuous optimizer, and -- unlike play.py's own
LimitedVisionHuman -- there is NO automatic collision avoidance once
authorized: a maneuver clicked (or automatic driving left running) into a
hazard this car's own FOV didn't show you is a real crash, not something
IDM quietly brakes for. See approximate_limit_vision_human.py's own module
docstring for why that's the point, not a bug.

WHY A BUTTON IS GRAYED OUT, EXACTLY. Every maneuver button always appears
(the participant is briefed on the FULL set of possible maneuvers in
advance, same as knowing the road rules) -- but a maneuver button is only
clickable when it is BOTH (a) the one subtask actually required next by
the fixed path (see user_study/subtasks.py -- "only if the car needs to
merge, then [merge] could even be available") and (b) not gated by
ApproximateLimitVisionHuman.gated(), which checks ONLY what THIS car's own
FOV cone/occlusion can currently see -- a narrower cone can genuinely miss
a real hazard and leave a dangerous button clickable, by design. WAIT is
always clickable EXCEPT while a maneuver is already committed (nothing is
clickable then at all -- see the module docstring on why a committed
maneuver can't be interrupted).

THE ROBOT reads a LIVE belief about your cone from
user_study.fov_posterior_user_study.SubtaskFOVPosterior -- built purely
from watching whether ApproximateLimitVisionHuman is observed proceeding or
held at each maneuver (pure geometry: human.position/speed and the exact
same gated() check every hypothesis re-runs), NEVER from which button you
clicked -- see that module's own docstring for why this is a meaningfully
different (and in testing, much more discriminating) observation model
than robot/filter/core/fov_posterior.py's own continuous-IDM-residual one.

Controls
    click a maneuver button   authorize it once the path requires it
    click WAIT                hold the car back; click again to resume
    q / ESC / close window    quit
"""
import argparse
import os
import sys
import types

import numpy as np
import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))          # highway_env/user_study
_HIGHWAY_ENV_DIR = os.path.dirname(_HERE)                    # highway_env
sys.path.insert(0, _HERE)
sys.path.insert(0, _HIGHWAY_ENV_DIR)
sys.path.insert(0, os.path.join(_HIGHWAY_ENV_DIR, "human"))
sys.path.insert(0, os.path.join(_HIGHWAY_ENV_DIR, "layout"))
import display_all as d  # noqa: E402
import scene1_background as sb  # noqa: E402
import limit_vision_human as h  # noqa: E402
import fov_render as fr  # noqa: E402
from highway_env.road.graphics import RoadGraphics, WorldSurface  # noqa: E402
from robot.methods import make_baseline, combo_names, resolve_combo, describe_robots  # noqa: E402
from robot.nominal_policy.vehicles import make_vehicle  # noqa: E402

from subtasks import build_subtasks, seed_maneuver_traffic, DISPLAY_NAME  # noqa: E402
from approximate_limit_vision_human import ApproximateLimitVisionHuman, available_choices  # noqa: E402
from fov_posterior_user_study import SubtaskFOVPosterior  # noqa: E402

BLACK_ALPHA = 255
BLACK_COLOR = (0, 0, 0)
FOV_CANDIDATES = (30.0, 60.0, 90.0, 180.0, 360.0)
ROBOT_PALETTE = [(255, 165, 0), (255, 105, 180), (0, 220, 220), (200, 255, 60)]


def _point_at_progress(route_points, target_progress):
    """(position, heading) at `target_progress` meters of arc length along
    `route_points` -- the reverse of limit_vision_human._route_progress
    (position -> progress). Used to stagger multiple robots' own spawn
    points along ROBOT_ROUTE without hand-picking world coordinates per
    layout. Clamped to the route's own last point/heading past its end."""
    pts = np.asarray(route_points, dtype=float)
    acc = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = b - a
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            continue
        if acc + seg_len >= target_progress:
            t = (target_progress - acc) / seg_len
            return a + t * seg, np.arctan2(seg[1], seg[0])
        acc += seg_len
    a, b = pts[-2], pts[-1]
    return pts[-1], np.arctan2((b - a)[1], (b - a)[0])


def _spawn_robot_on_route(road, robot_route, progress, kind, speed, color):
    """A robot vehicle, snapped onto the nearest real lane to the route
    point at `progress` meters in -- same spawn convention play.py's own
    single-robot block uses, just at an arbitrary point along the route
    instead of always its very start."""
    approx_pos, _ = _point_at_progress(robot_route, progress)
    robot = make_vehicle(kind, road, approx_pos, heading=0.0, speed=speed)
    robot.route_points = robot_route
    lane_index = road.network.get_closest_lane_index(robot.position)
    lane = road.network.get_lane(lane_index)
    lon, _ = lane.local_coordinates(robot.position)
    robot.position, robot.heading = lane.position(max(lon, 0.0), 0), lane.heading_at(max(lon, 0.0))
    robot.lane_index, robot.lane, robot.target_lane_index = lane_index, lane, lane_index
    robot.color = color
    road.vehicles.append(robot)
    return robot

# Fixed display order and layout for the button row -- every button always
# shows (the participant is briefed on the full maneuver vocabulary in
# advance), only which ones are ENABLED changes tick to tick. "wait"/"go"
# are the two always-clickable modes; the rest are maneuver kinds from
# subtasks.py's own DISPLAY_NAME, gated per approximate_limit_vision_human's
# own rules.
BUTTON_ORDER = ["wait", "turn", "lane_change", "merge_in", "merge_out"]
BUTTON_LABEL = {"wait": "WAIT"}
BUTTON_LABEL.update({k: DISPLAY_NAME[k].upper() for k in ("turn", "lane_change", "merge_in", "merge_out")})

BTN_W, BTN_H, BTN_GAP = 150, 40, 10
ENABLED_COLOR = (70, 130, 180)
ENABLED_TEXT = (255, 255, 255)
DISABLED_COLOR = (60, 60, 60)
DISABLED_TEXT = (120, 120, 120)
SELECTED_BORDER = (255, 200, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", "--scene", dest="scene", default="real_001_rebuilt",
                         help="layout in layout/layouts/ (default real_001_rebuilt)")
    parser.add_argument("--list", action="store_true", help="print the layouts in layout/layouts/ and exit")
    parser.add_argument("--fov", type=float, default=90.0, choices=list(FOV_CANDIDATES),
                         help="your vision cone, in degrees (default 90)")
    parser.add_argument("--no-occlusion", action="store_true", help="disable occlusion (cone-only perception)")
    parser.add_argument("--count", type=int, default=35,
                         help="number of background vehicles (default 35 -- deliberately dense: at the old "
                              "default of 18, random traffic essentially never landed close enough to a "
                              "maneuver's own lane at the moment the human reached it, so no subtask button "
                              "was ever actually FOV-gated in practice and the posterior never saw an "
                              "informative tick -- confirmed directly by instrumenting a full run)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--dt", type=float, default=1 / 15, help="simulation timestep in seconds (default 1/15)")
    parser.add_argument("--scale", type=float, default=10.0, help="pixels per meter in chase mode (default 10)")
    parser.add_argument("--no-chase", action="store_true", help="whole-map camera instead of the ego/chase view")
    parser.add_argument("--no-robot", action="store_true", help="skip spawning the robot partner(s) + live posterior")
    parser.add_argument("--n-robots", type=int, default=2,
                         help="how many robot partners to spawn along ROBOT_ROUTE, staggered so they don't "
                              "immediately collide with each other (default 2 -- more robots means more "
                              "chances for a maneuver's own gating check to actually see one)")
    parser.add_argument("--robot-policy", default="fov_aware", choices=["nominal", "cautious", "fov_aware"],
                         help="every robot partner's policy wrapper (default fov_aware)")
    parser.add_argument("--robot-vehicle", default="idm", choices=["idm", "linear", "aggressive", "defensive"],
                         help="every robot partner's underlying vehicle dynamics (default idm)")
    parser.add_argument("--robot-weight-seen", type=float, default=1.0,
                         help="FOVFilter's own weight_seen for every robot partner (default 1.0: stay visible "
                              "to you; negative: try to stay OUT of your view; 0: disables the filter, plain "
                              "baseline behavior) -- only used when --robot-policy fov_aware")
    parser.add_argument("--robot-stagger", type=float, default=18.0,
                         help="meters of ROBOT_ROUTE progress between each robot's own spawn point (default 18)")
    parser.add_argument("--robot", default=None,
                         help="shorthand for --robot-vehicle/--robot-policy TOGETHER, as one "
                              "\"{vehicle}_{policy}\" name (e.g. --robot defensive_fov_aware) -- overrides both "
                              "flags above when given. --list-robots prints every valid name.")
    parser.add_argument("--list-robots", action="store_true", help="print every valid --robot combo name and exit")
    args = parser.parse_args()

    if args.list:
        print("\n".join(sorted(n for n in os.listdir(os.path.join(_HIGHWAY_ENV_DIR, "layout", "layouts"))
                                if n.endswith(".py") and not n.startswith("_"))))
        return
    if args.list_robots:
        print(describe_robots())
        return
    if args.robot is not None:
        combo = resolve_combo(args.robot)
        if combo is None:
            raise SystemExit(f"unknown --robot {args.robot!r}. Choose from:\n  " + "\n  ".join(combo_names()))
        args.robot_vehicle, args.robot_policy = combo

    module = d.load_layout(args.scene)
    road = module.build_road()
    subtasks = build_subtasks(module, "human")

    start = np.asarray(module.HUMAN_ROUTE[0], dtype=float)
    lane_index = road.network.get_closest_lane_index(start)
    lane = road.network.get_lane(lane_index)
    human = ApproximateLimitVisionHuman(
        # speed=0.0: the car must never move before the user clicks anything
        # at all -- selected starts at "wait", but a nonzero spawn speed
        # would still visibly coast forward before the hard clamp catches
        # it, which is exactly the "it went by itself" bug this is fixing.
        # target_speed=8.0 SEPARATELY: ControlledVehicle.__init__ defaults
        # target_speed to whatever `speed` was passed (target_speed =
        # target_speed or self.speed) -- leaving it implicit here would
        # silently lock this car's own IDM cruising target at 0 too, so
        # once authorized it brakes against its own free-flow term instead
        # of actually driving (confirmed: oscillated 0 <-> 0.2 m/s forever
        # once "go" was clicked, never reaching a real cruising speed).
        road, lane.position(0, 0), heading=lane.heading_at(0), speed=0.0, target_speed=8.0,
        route_points=module.HUMAN_ROUTE, subtasks=subtasks,
        fov_deg=args.fov, enable_occlusion=not args.no_occlusion)
    human.lane_index, human.lane, human.target_lane_index = lane_index, lane, lane_index
    road.vehicles.append(human)

    robot_route = getattr(module, "ROBOT_ROUTE", None)
    robots = []
    if robot_route and not args.no_robot:
        # Staggered along ROBOT_ROUTE (not all piled at its own start) so
        # multiple robots don't immediately collide with each other, and so
        # they cover DIFFERENT points along the human's own drive -- more
        # distinct chances for a maneuver's own gating check to actually
        # have a real vehicle nearby (see --count's own comment: a single
        # robot plus modest background traffic essentially never produced
        # one in testing).
        for i in range(max(0, args.n_robots)):
            robots.append(_spawn_robot_on_route(
                road, robot_route, progress=i * args.robot_stagger, kind=args.robot_vehicle,
                speed=9.0, color=ROBOT_PALETTE[i % len(ROBOT_PALETTE)]))
    robot_baseline = (make_baseline(args.robot_policy, weight_seen=args.robot_weight_seen)
                       if robots and args.robot_policy == "fov_aware" else
                       make_baseline(args.robot_policy) if robots else None)
    posterior = SubtaskFOVPosterior(FOV_CANDIDATES) if robots else None

    # Seeded BEFORE the general random traffic below, and specifically onto
    # the human's own maneuver lanes -- see seed_maneuver_traffic's own
    # docstring: uniformly-random background traffic essentially never
    # lands on a turn/merge's own short target lane at the right moment,
    # which is exactly why gating never differed by FOV in testing.
    seed_maneuver_traffic(road, subtasks, seed=args.seed)
    route_lanes = getattr(module, "route_adjacent_lane_indexes", lambda: None)()
    sb.add_background_traffic(road, count=args.count, seed=args.seed, lane_indexes=route_lanes)
    lane_indexes = sb.all_lane_indexes(road)
    print(f"USER STUDY {args.scene}: fov={args.fov:.0f} occlusion={'on' if not args.no_occlusion else 'off'}"
          f"{f', {len(robots)} robot(s)={args.robot_vehicle}/{args.robot_policy}' if robots else ''} "
          f"-- click a subtask button to drive")

    pygame.init()
    desktop = pygame.display.Info()
    w, hh = int(desktop.current_w * 0.9), int(desktop.current_h * 0.85)
    window = pygame.display.set_mode((w, hh))
    pygame.display.set_caption(f"USER STUDY -- {args.scene} (fov={args.fov:.0f})")
    button_bar_h = BTN_H + 2 * BTN_GAP
    render_size = (w, hh - d.LABEL_H - button_bar_h)
    chase = not args.no_chase

    if not chase:
        surface = WorldSurface(render_size, 0, pygame.Surface(render_size))
        min_x, max_x, min_y, max_y = d.road_bounding_box(road)
        surface.scaling = min(w / ((max_x - min_x) * 1.15), render_size[1] / ((max_y - min_y) * 1.15))
        center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        surface.origin = center - np.array([w / 2, render_size[1] / 2]) / surface.scaling
    chase_anchor = (render_size[0] / 2.0, render_size[1] * 0.72)

    font = pygame.font.SysFont(None, 22)
    btn_font = pygame.font.SysFont(None, 20, bold=True)

    button_rects = _button_rects(w, hh, button_bar_h)

    running = True
    step_count = 0
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                _handle_click(human, button_rects, event.pos)

        road.act()
        sb.apply_better_car_following(road, lane_indexes, args.dt)
        human.advance(road, lane_indexes, args.dt)

        belief = posterior.beliefs() if (posterior is not None and args.robot_policy == "fov_aware") else None
        for robot in robots:
            if robot.crashed:
                continue
            candidates_r = sb.nearby_vehicles(road, robot, 35.0)
            front_r = sb.find_front_vehicle(road, robot, lane_indexes, candidates_r)
            # Recomputed here (not reused from apply_better_car_following's
            # own earlier pass) specifically so it can be threaded into the
            # filter's OWN candidate generation via ctx.crossing_conflict_accel
            # (see FOVFilter.action's own docstring) instead of min()-ed
            # against the filter's result afterward -- that used to silently
            # override the filter's own choice on every tick a conflict was
            # active, which in dense traffic was often, defeating the whole
            # point of asking it to reason about your FOV at all.
            heads_r = {id(v) for v in candidates_r
                       if sb.find_front_vehicle(road, v, lane_indexes, sb.nearby_vehicles(road, v, 35.0)) is None}
            if front_r is None:
                heads_r.add(id(robot))
            conflict_r = sb.crossing_conflict_brake(road, robot, candidates_r, heads=heads_r)
            ctx = types.SimpleNamespace(robot=robot, front_vehicle=front_r,
                                         human=human if not human.crashed else None, belief=belief, dt=args.dt,
                                         crossing_conflict_accel=conflict_r)
            final = robot_baseline.action(ctx)
            # Still min()-ed against conflict_r directly here -- a no-op for
            # fov_aware (its own candidates already respect it) but the only
            # thing enforcing it at all for nominal/cautious, which don't
            # read ctx.crossing_conflict_accel themselves.
            if conflict_r is not None:
                final = min(final, conflict_r)
            robot.action["acceleration"] = max(final, -robot.speed / args.dt)

        road.step(args.dt)
        h.advance_vehicles_with_route(road, lane_indexes)
        h._unstick_frozen_background(road, args.dt)
        # Without this, a genuine route-vs-route deadlock (e.g. two robots,
        # or a robot and the human, both stopped right at a shared junction
        # point) never resolves -- _unstick_frozen_background deliberately
        # never removes a route-following vehicle (see its own docstring),
        # so that class of standoff needs this call specifically. Confirmed
        # missing here was a real, live bug, not theoretical: a robot parked
        # motionless for 300+ simulated seconds directly on top of the
        # human's own turn-lane conflict zone, permanently blocking that
        # maneuver from ever becoming legal, in a run that otherwise had no
        # other robot/background traffic anomaly.
        h._resolve_stuck_route_pair(road, args.dt)

        if posterior is not None and not human.crashed:
            candidates_h = sb.nearby_vehicles(road, human, 35.0)
            posterior.update(human, candidates_h)

        step_count += 1

        candidates = sb.nearby_vehicles(road, human, fr.CONE_RADIUS)
        visible = human.visible_candidates(candidates)
        progress, _ = h._route_progress(module.HUMAN_ROUTE, human.position)
        total_len = h.route_total_length(module.HUMAN_ROUTE)

        if chase:
            buf = fr.chase_camera_buffer(human, args.scale, render_size, chase_anchor)
            buf.fill(buf.GREY)
            RoadGraphics.display(road, buf)
            RoadGraphics.display_traffic(road, buf, simulation_frequency=round(1 / args.dt), offscreen=True)
            d.draw_routes(buf, road, module.HUMAN_ROUTE, [d.HUMAN_COLOR], -d.ROUTE_LATERAL_OFFSET, args.scene, "human")
            if robot_route:
                d.draw_routes(buf, road, robot_route, d.ROBOT_PALETTE, d.ROUTE_LATERAL_OFFSET, args.scene, "robot")
            fr.draw_fov_mask(buf, human, candidates, args.fov, True, not args.no_occlusion,
                              BLACK_ALPHA, color=BLACK_COLOR)
            fr.redraw_ego(buf, human)
            fr.redraw_partial(buf, human, candidates, args.fov, True, not args.no_occlusion)
            frame = fr.render_chase_frame(buf, human, render_size, chase_anchor, d.BG)
        else:
            surface.fill(surface.GREY)
            RoadGraphics.display(road, surface)
            RoadGraphics.display_traffic(road, surface, simulation_frequency=round(1 / args.dt), offscreen=True)
            d.draw_routes(surface, road, module.HUMAN_ROUTE, [d.HUMAN_COLOR], -d.ROUTE_LATERAL_OFFSET, args.scene, "human")
            if robot_route:
                d.draw_routes(surface, road, robot_route, d.ROBOT_PALETTE, d.ROUTE_LATERAL_OFFSET, args.scene, "robot")
            fr.draw_fov_mask(surface, human, candidates, args.fov, True, not args.no_occlusion,
                              BLACK_ALPHA, color=BLACK_COLOR)
            fr.redraw_ego(surface, human)
            fr.redraw_partial(surface, human, candidates, args.fov, True, not args.no_occlusion)
            frame = surface

        window.fill(d.BG)
        window.blit(frame, (0, d.LABEL_H))

        belief_str = ""
        if posterior is not None:
            belief_str = f"  robot believes your fov={posterior.map_fov():.0f} ({posterior.belief_for(posterior.map_fov())*100:.0f}%)"
        span = human.active_span or human.current_span()
        span_str = DISPLAY_NAME[span[0]] if span is not None else "?"
        window.blit(font.render(
            f"USER STUDY {args.scene}  step {step_count}  speed={human.speed:.1f} "
            f"progress={progress/total_len*100:.0f}%  needed now: {span_str}  "
            f"selected: {human.selected}{belief_str}  [click a button, q/ESC quit]",
            True, d.LABEL), (10, 2))

        _draw_buttons(window, button_rects, human, visible, btn_font)

        pygame.display.flip()
        clock.tick(round(1 / args.dt))

        if human.crashed:
            pygame.time.wait(1500)
            running = False

    pygame.quit()


def _button_rects(w, hh, button_bar_h):
    n = len(BUTTON_ORDER)
    total_w = n * BTN_W + (n - 1) * BTN_GAP
    x0 = (w - total_w) // 2
    y0 = hh - button_bar_h + BTN_GAP
    return {kind: pygame.Rect(x0 + i * (BTN_W + BTN_GAP), y0, BTN_W, BTN_H)
            for i, kind in enumerate(BUTTON_ORDER)}


def _is_enabled(human, kind, visible):
    """Path-legal (available_choices -- the human never deviates from
    HUMAN_ROUTE, so a maneuver kind is legal only when the path itself
    requires it right now; "wait" is always legal, EXCEPT while a
    maneuver is already committed -- see approximate_limit_vision_human.py's
    own module docstring on why that can't be interrupted) AND, for a
    maneuver kind specifically, not gated by what this car's own FOV can
    currently see. "go" is never a button (see module docstring -- it's
    the automatic default, toggled back to by clicking WAIT again)."""
    if kind not in available_choices(human):
        return False
    if kind == "wait":
        return True
    return not human.gated(human.current_span(), visible)


def _handle_click(human, button_rects, pos):
    for kind, rect in button_rects.items():
        if rect.collidepoint(pos):
            candidates = sb.nearby_vehicles(human.road, human, 35.0)
            visible = human.visible_candidates(candidates)
            if not _is_enabled(human, kind, visible):
                return
            if kind == "wait":
                # toggle: WAIT while driving holds it back, WAIT again
                # while already held resumes automatic driving -- there is
                # no separate GO button (see module docstring).
                human.select("go" if human.selected == "wait" else "wait")
            else:
                human.select(kind)
            return


def _draw_buttons(window, button_rects, human, visible, btn_font):
    for kind, rect in button_rects.items():
        enabled = _is_enabled(human, kind, visible)
        color = ENABLED_COLOR if enabled else DISABLED_COLOR
        text_color = ENABLED_TEXT if enabled else DISABLED_TEXT
        pygame.draw.rect(window, color, rect, border_radius=6)
        if human.selected == kind:
            pygame.draw.rect(window, SELECTED_BORDER, rect, width=3, border_radius=6)
        text = "RESUME" if kind == "wait" and human.selected == "wait" else BUTTON_LABEL[kind]
        label = btn_font.render(text, True, text_color)
        window.blit(label, label.get_rect(center=rect.center))


if __name__ == "__main__":
    main()
