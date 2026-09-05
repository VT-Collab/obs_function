import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'layout'))
sys.path.insert(0, os.path.join(os.getcwd(), 'human'))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'user_study'))
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import numpy as np
import types
import display_all as d
import scene_background as sb
import limit_vision_human as h
from subtasks import build_subtasks, seed_maneuver_traffic, DISPLAY_NAME
from approximate_limit_vision_human import ApproximateLimitVisionHuman, available_choices
from fov_posterior_user_study import SubtaskFOVPosterior
from robot.methods import make_baseline
from robot.nominal_policy.vehicles import make_vehicle
import interface as iface

DT = 1 / 15
FOV_CANDIDATES = (30.0, 60.0, 90.0, 180.0, 360.0)


def run(scene, true_fov, seed, n_robots=3, background_count=35, robot_weight_seen=1.0, steps=6000):
    module = d.load_layout(scene)
    # road = module.build_road()  # old plain-Road path, commented out -- RegulatedRoad
    # is now the project-wide default (see scene_background.VisibleRegulatedRoad); render
    # is provably unaffected (_compare_regulated_render.py, all 10 layouts byte-for-byte
    # identical either way), this only adds its own priority-based yielding on top,
    # alongside this project's own crossing_conflict_brake.
    road = sb.VisibleRegulatedRoad(network=module.build_road().network)
    subtasks = build_subtasks(module, 'human')

    start = np.asarray(module.HUMAN_ROUTE[0], dtype=float)
    lane_index = road.network.get_closest_lane_index(start)
    lane = road.network.get_lane(lane_index)
    human = ApproximateLimitVisionHuman(road, lane.position(0, 0), heading=lane.heading_at(0), speed=0.0,
                                         target_speed=8.0, route_points=module.HUMAN_ROUTE, subtasks=subtasks,
                                         fov_deg=true_fov)
    human.lane_index, human.lane, human.target_lane_index = lane_index, lane, lane_index
    road.vehicles.append(human)

    robot_route = getattr(module, 'ROBOT_ROUTE', None)
    robots = []
    for i in range(n_robots):
        robots.append(iface._spawn_robot_on_route(road, robot_route, i * 18.0, 'idm', 9.0, (255, 165, 0)))
    robot_baseline = make_baseline('fov_aware', weight_seen=robot_weight_seen)
    posterior = SubtaskFOVPosterior(FOV_CANDIDATES)

    seed_maneuver_traffic(road, subtasks, seed=seed)
    route_lanes = getattr(module, 'route_adjacent_lane_indexes', lambda: None)()
    sb.add_background_traffic(road, count=background_count, seed=seed, lane_indexes=route_lanes)
    lane_indexes = sb.all_lane_indexes(road)

    n_updates = 0
    max_dist = 0.0
    robot_speeds_near_human = []  # (dist_to_human, speed) samples, to check FOVFilter's own effect
    for step in range(steps):
        road.act()
        sb.apply_better_car_following(road, lane_indexes, DT)
        human.advance(road, lane_indexes, DT)
        span = human.active_span
        if span is not None and span[0] != 'forward' and human.selected != span[0] and human.speed < 0.3 \
                and human._committed_span is None:
            candidates = sb.nearby_vehicles(road, human, 35.0)
            visible = human.visible_candidates(candidates)
            if not human.gated(span, visible):
                human.select(span[0])

        belief = posterior.beliefs()
        for robot in robots:
            if robot.crashed:
                continue
            candidates_r = sb.nearby_vehicles(road, robot, 35.0)
            front_r = sb.find_front_vehicle(road, robot, lane_indexes, candidates_r)
            heads_r = {id(v) for v in candidates_r
                       if sb.find_front_vehicle(road, v, lane_indexes, sb.nearby_vehicles(road, v, 35.0)) is None}
            if front_r is None:
                heads_r.add(id(robot))
            conflict_r = sb.crossing_conflict_brake(road, robot, candidates_r, heads=heads_r)
            ctx = types.SimpleNamespace(robot=robot, front_vehicle=front_r,
                                         human=human if not human.crashed else None, belief=belief, dt=DT,
                                         crossing_conflict_accel=conflict_r)
            final = robot_baseline.action(ctx)
            if conflict_r is not None:
                final = min(final, conflict_r)
            robot.action['acceleration'] = max(final, -robot.speed / DT)
            d_h = float(np.linalg.norm(np.asarray(robot.position) - np.asarray(human.position)))
            robot_speeds_near_human.append((d_h, robot.speed))

        road.step(DT)
        h.advance_vehicles_with_route(road, lane_indexes)
        h._unstick_frozen_background(road, DT)
        h._resolve_stuck_route_pair(road, DT)

        if not human.crashed:
            all_candidates = sb.nearby_vehicles(road, human, 35.0)
            before = dict(posterior.beliefs())
            posterior.update(human, all_candidates)
            if dict(posterior.beliefs()) != before:
                n_updates += 1

        _, dist = h._route_progress(module.HUMAN_ROUTE, human.position)
        max_dist = max(max_dist, dist)

        if human.crashed:
            break

    prog, _ = h._route_progress(module.HUMAN_ROUTE, human.position)
    total = h.route_total_length(module.HUMAN_ROUTE)
    n_robot_crashes = sum(1 for r in robots if r.crashed)
    return dict(progress=prog / total, crashed=human.crashed, n_robot_crashes=n_robot_crashes,
                n_updates=n_updates, max_dist=max_dist, map_fov=posterior.map_fov(),
                belief=posterior.beliefs(), robot_speeds_near_human=robot_speeds_near_human)


if __name__ == "__main__":
    print("=== dense-scene sweep: does the posterior now get informative ticks? ===")
    for true_fov in FOV_CANDIDATES:
        r = run('real_001_rebuilt', true_fov, seed=0)
        print(f"fov={true_fov:>5.0f}  progress={r['progress']*100:.0f}%  human_crashed={r['crashed']}  "
              f"n_robot_crashes={r['n_robot_crashes']}  n_updates={r['n_updates']}  "
              f"map_guess={r['map_fov']:.0f}  max_dist_from_route={r['max_dist']:.3f}")
