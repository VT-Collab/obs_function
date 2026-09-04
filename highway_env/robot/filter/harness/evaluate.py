"""Headless, pygame-free episode runner + CLI grid sweep -> JSONL. Mirrors
steakhouse/no_larping (and misha)'s own robot/filter/harness/evaluate.py:
one run_episode(...) -> dict per episode, one JSON line per episode written
by the CLI, aggregation/comparison left entirely to analysis/report.py
(this file never compares anything itself).

Order of operations each tick matters and is fixed to match the
steakhouse harness's own documented choice: the robot's baseline decides
its action FIRST, using whatever belief the posterior already held BEFORE
this tick's human action is folded in -- exactly what a real robot has
available (it cannot see the future tick's human action before deciding
its own).

    python evaluate.py --scene real_001_rebuilt --fovs 30,90,360 --seeds 0,1,2 \\
        --methods nominal,cautious,fov_aware --steps 1800 --out runs.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))              # highway_env/robot/filter/harness
_HIGHWAY_ENV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # highway_env
_HUMAN_DIR = os.path.join(_HIGHWAY_ENV_DIR, "human")
sys.path.insert(0, _HIGHWAY_ENV_DIR)
sys.path.insert(0, os.path.join(_HIGHWAY_ENV_DIR, "layout"))
sys.path.insert(0, _HUMAN_DIR)
import display_all as d  # noqa: E402
import scene1_background as sb  # noqa: E402
import limit_vision_human as h  # noqa: E402
from robot.methods import make_baseline  # noqa: E402
from robot.nominal_policy.vehicles import make_vehicle  # noqa: E402
from robot.filter.core.fov_posterior import FOVPosterior  # noqa: E402

DT = 1 / 15
FOV_CANDIDATES = (30.0, 60.0, 90.0, 180.0, 360.0)


def _layout_sha(scene):
    path = os.path.join(_HIGHWAY_ENV_DIR, "layout", "layouts", f"{scene}.py")
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def run_episode(scene, fov, seed, method, steps=1800, background_count=28, robot_speed=9.0, trace=False,
                 robot_vehicle="idm"):
    """method: the POLICY WRAPPER (nominal_policy.baselines.REGISTRY plus
    filter.core.fov_aware_baseline's "fov_aware") -- decides WHETHER extra
    caution is blended in, and how.
    robot_vehicle: the underlying VEHICLE DYNAMICS (nominal_policy.
    vehicles.REGISTRY: idm/linear/aggressive/defensive) -- decides HOW the
    robot accelerates when nothing overrides it. Orthogonal to `method`:
    any combination is valid, e.g. robot_vehicle="defensive",
    method="fov_aware".
    """
    module = d.load_layout(scene)
    road = module.build_road()

    human = h.add_human_vehicle(road, module.HUMAN_ROUTE, fov_deg=fov)

    robot_route = getattr(module, "ROBOT_ROUTE", None)
    robot = None
    if robot_route:
        robot = make_vehicle(robot_vehicle, road, np.asarray(robot_route[0], dtype=float),
                              heading=0.0, speed=robot_speed)
        robot.route_points = robot_route  # opts it into route_aware_continuation, same as the human
        # heading corrected to the nearest real lane, matching add_human_vehicle's own spawn convention
        lane_index = road.network.get_closest_lane_index(robot.position)
        lane = road.network.get_lane(lane_index)
        lon, _ = lane.local_coordinates(robot.position)
        robot.position = lane.position(max(lon, 0.0), 0)
        robot.heading = lane.heading_at(max(lon, 0.0))
        robot.lane_index, robot.lane, robot.target_lane_index = lane_index, lane, lane_index
        road.vehicles.append(robot)

    route_lanes = getattr(module, "route_adjacent_lane_indexes", lambda: None)()
    sb.add_background_traffic(road, count=background_count, seed=seed, lane_indexes=route_lanes)
    lane_indexes = sb.all_lane_indexes(road)

    baseline = make_baseline(method) if method != "full_info" else make_baseline("nominal")
    posterior = FOVPosterior(FOV_CANDIDATES) if robot is not None else None

    human_total = h.route_total_length(module.HUMAN_ROUTE)
    robot_total = h.route_total_length(robot_route) if robot_route else 0.0

    min_gap_hr = float("inf")
    human_crashed_at = None
    robot_crashed_at = None
    map_fov_correct_ticks = 0
    ticks_with_posterior = 0
    trace_rows = []
    t0 = time.time()

    for step in range(steps):
        road.act()
        h.apply_human_aware_car_following(road, lane_indexes, DT)

        if robot is not None and not robot.crashed:
            candidates_r = sb.nearby_vehicles(road, robot, 35.0)
            front_r = sb.find_front_vehicle(road, robot, lane_indexes, candidates_r)
            belief = posterior.beliefs() if method == "fov_aware" else None
            ctx = types.SimpleNamespace(robot=robot, front_vehicle=front_r,
                                         human=human if not human.crashed else None, belief=belief, dt=DT)
            # Floored -- baseline.action() calls ctx.robot.acceleration() raw (no
            # crossing_conflict_brake floor of its own), so min()-ing it in here can undo the
            # floor apply_human_aware_car_following already applied a few lines up and drive
            # speed hugely negative over a sustained brake (observed: -38 m/s after a long
            # merge-point standoff). See play.py's identical line.
            robot.action["acceleration"] = max(min(robot.action["acceleration"], baseline.action(ctx)), -robot.speed / DT)

        road.step(DT)
        h.advance_vehicles_with_route(road, lane_indexes)

        if robot is not None and posterior is not None and not human.crashed:
            candidates_h = sb.nearby_vehicles(road, human, 35.0)
            posterior.update(human, candidates_h, human.action["acceleration"],
                              front_vehicle_fn=lambda vis: sb.find_front_vehicle(road, human, lane_indexes, vis))
            ticks_with_posterior += 1
            if posterior.map_fov() == min(FOV_CANDIDATES, key=lambda c: abs(c - fov)):
                map_fov_correct_ticks += 1

        if human.crashed and human_crashed_at is None:
            human_crashed_at = step
        if robot is not None and robot.crashed and robot_crashed_at is None:
            robot_crashed_at = step
        if robot is not None and not human.crashed and not robot.crashed:
            min_gap_hr = min(min_gap_hr, float(np.linalg.norm(
                np.asarray(human.position) - np.asarray(robot.position))))

        if trace and step % 15 == 0:
            trace_rows.append(dict(step=step, human_pos=list(map(float, human.position)),
                                    robot_pos=list(map(float, robot.position)) if robot is not None else None))

    human_progress, _ = h._route_progress(module.HUMAN_ROUTE, human.position)
    robot_progress = 0.0
    if robot is not None:
        robot_progress, _ = h._route_progress(robot_route, robot.position)

    result = {
        "scene": scene, "layout_sha": _layout_sha(scene), "fov": fov, "seed": seed, "method": method,
        "robot_vehicle": robot_vehicle,
        "steps": steps, "human_crashed": human_crashed_at is not None, "human_crashed_at": human_crashed_at,
        "robot_crashed": robot_crashed_at is not None, "robot_crashed_at": robot_crashed_at,
        "scene_crash_count": sum(1 for v in road.vehicles if v.crashed),
        "human_progress_frac": float(min(human_progress / human_total, 1.0)) if human_total else None,
        "robot_progress_frac": float(min(robot_progress / robot_total, 1.0)) if robot_total else None,
        "min_gap_human_robot": None if min_gap_hr == float("inf") else min_gap_hr,
        "map_fov_accuracy": (map_fov_correct_ticks / ticks_with_posterior) if ticks_with_posterior else None,
        "ms_per_tick": (time.time() - t0) * 1000.0 / steps,
        "wall_s": time.time() - t0,
    }
    if trace:
        result["trace"] = trace_rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="real_001_rebuilt")
    parser.add_argument("--fovs", default="30,60,90,180,360")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default="nominal,cautious,fov_aware",
                         help="policy wrapper(s) -- see robot.methods.REGISTRY")
    parser.add_argument("--robot-vehicles", default="idm",
                         help="comma-separated vehicle dynamics -- see robot.nominal_policy.vehicles.REGISTRY "
                              "(idm,linear,aggressive,defensive). Orthogonal to --methods: every combination runs.")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--background-count", type=int, default=28)
    parser.add_argument("--out", default=None, help="JSONL path (default: stdout)")
    args = parser.parse_args()

    fovs = [float(x) for x in args.fovs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    methods = args.methods.split(",")
    robot_vehicles = args.robot_vehicles.split(",")

    out = open(args.out, "w") if args.out else sys.stdout
    n_total = len(fovs) * len(seeds) * len(methods) * len(robot_vehicles)
    n_done = 0
    t0 = time.time()
    for robot_vehicle in robot_vehicles:
        for method in methods:
            for fov in fovs:
                for seed in seeds:
                    row = run_episode(args.scene, fov, seed, method, steps=args.steps,
                                       background_count=args.background_count, robot_vehicle=robot_vehicle)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    n_done += 1
                    print(f"[{n_done}/{n_total}] scene={args.scene} robot_vehicle={robot_vehicle:>10} "
                          f"method={method:>10} fov={fov:>5.0f} seed={seed}: "
                          f"human_crashed={row['human_crashed']} robot_crashed={row['robot_crashed']} "
                          f"human_progress={row['human_progress_frac']:.0%} "
                          f"robot_progress={(row['robot_progress_frac'] or 0):.0%} "
                          f"min_gap={row['min_gap_human_robot']}", file=sys.stderr)
    if args.out:
        out.close()
    print(f"done: {n_total} episodes in {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
