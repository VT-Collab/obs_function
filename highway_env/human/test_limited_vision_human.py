"""Validation for LimitedVisionHuman. Two things this has to demonstrate,
MEASURED rather than assumed (see limit_vision_human.py's module docstring):
  (A) behavior genuinely differs by FOV width
  (B) the human stays functional (nonzero speed, no worse crash rate) at
      every FOV width tested

Run:
    python test_limited_vision_human.py            # unit checks + smoke test
    python test_limited_vision_human.py --sweep     # + the full FOV sweep
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # highway_env/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layout"))
import display_all as d  # noqa: E402
import scene1_background as sb  # noqa: E402
import limit_vision_human as h  # noqa: E402

SCENE = "real_001_rebuilt"
DT = 1 / 15
HARD_BRAKE_THRESHOLD = -4.0  # m/s^2; also always true when crossing_conflict_brake fires (-8.0)


# ============================================================================
# UNIT CHECKS -- pure geometry, no simulation
# ============================================================================

def check_in_cone():
    # directly ahead, narrow cone: visible
    assert h.in_cone((0, 0), 0.0, (10, 0), 60.0)
    # directly behind, narrow cone: not visible
    assert not h.in_cone((0, 0), 0.0, (-10, 0), 60.0)
    # 90 degrees off-axis, narrow cone: not visible
    assert not h.in_cone((0, 0), 0.0, (0, 10), 60.0)
    # 90 degrees off-axis, wide (180) cone: visible
    assert h.in_cone((0, 0), 0.0, (0, 10), 180.0)
    # directly behind, 360 cone: visible (cone test vacuous at 360)
    assert h.in_cone((0, 0), 0.0, (-10, 0), 360.0)
    print("check_in_cone: OK")


def check_segment_intersects_rotated_rect():
    # segment straight through the center of an axis-aligned rect
    assert h.segment_intersects_rotated_rect((-10, 0), (10, 0), (0, 0), 5.0, 2.0, 0.0)
    # segment that clearly misses (well outside lateral extent)
    assert not h.segment_intersects_rotated_rect((-10, 10), (10, 10), (0, 0), 5.0, 2.0, 0.0)
    # rect rotated 90deg: a segment that would miss it unrotated now blocks
    # (rect long axis now spans y, so a vertical-offset segment threading
    # through x=0 at y=10 misses either way -- use a segment along y instead)
    assert h.segment_intersects_rotated_rect((0, -10), (0, 10), (0, 0), 5.0, 2.0, np.pi / 2)
    # segment entirely behind the rect (doesn't reach it): t stays in [0,1]
    # but geometry doesn't overlap -- short segment stopping before the rect
    assert not h.segment_intersects_rotated_rect((-10, 0), (-6, 0), (0, 0), 5.0, 2.0, 0.0)
    print("check_segment_intersects_rotated_rect: OK")


class _FakeVehicle:
    def __init__(self, position, heading=0.0, length=5.0, width=2.0):
        self.position = np.array(position, dtype=float)
        self.heading = heading
        self.LENGTH = length
        self.WIDTH = width


def check_is_occluded():
    # B sits directly between A and C -> C is occluded from A's viewpoint
    blocker = _FakeVehicle((5, 0))
    assert h.is_occluded((0, 0), (10, 0), [blocker])
    # B sits well off to the side -> does not occlude
    blocker2 = _FakeVehicle((5, 10))
    assert not h.is_occluded((0, 0), (10, 0), [blocker2])
    print("check_is_occluded: OK")


def check_route_progress_monotonic():
    route = [(0, 0), (10, 0), (20, 0), (20, 10)]
    p0, _ = h._route_progress(route, (0, 0))
    p1, _ = h._route_progress(route, (10, 0))
    p2, _ = h._route_progress(route, (20, 5))
    assert p0 < p1 < p2, (p0, p1, p2)
    total = h.route_total_length(route)
    assert abs(total - 30.0) < 1e-6
    print("check_route_progress_monotonic: OK")


def check_visible_candidates_identity_when_disabled():
    """The ablation baseline: with both flags off, visible_candidates() must
    be the identity function -- the whole point of the flags being on ONE
    class rather than a subclass."""
    module = d.load_layout(SCENE)
    road = module.build_road()
    human = h.add_human_vehicle(road, module.HUMAN_ROUTE, enable_fov=False, enable_occlusion=False)
    sb.add_background_traffic(road, count=80, seed=0)  # seed/count chosen to put real traffic near spawn
    candidates = sb.nearby_vehicles(road, human, 35.0)
    assert len(candidates) > 0, "test is vacuous with no nearby candidates -- pick a different seed/count"
    assert human.visible_candidates(candidates) == candidates
    print(f"check_visible_candidates_identity_when_disabled: OK ({len(candidates)} candidates)")


def check_ablation_matches_unfiltered_logic():
    """With both flags off, visible_candidates() is the identity, so the
    override's own find_front_vehicle()/crossing_conflict_brake() calls see
    exactly the same (unfiltered) candidate set as if computed directly.
    This checks the override reproduces THAT combination exactly.

    Deliberately NOT compared against apply_better_car_following()'s own
    output: that function's result also folds in whatever road.act()'s
    stock IDMVehicle.act() already set via the UNFILTERED road.neighbour_
    vehicles() -- a check that doesn't go through nearby_vehicles() at all,
    so it can never be made FOV-aware for the human without reimplementing
    it too. The human's action deliberately does NOT include that
    contribution (see apply_human_aware_car_following's docstring): find_
    front_vehicle() already strictly supersedes it here (same-lane AND
    fragment-continuation, vs. exact-lane-index-only), so dropping it loses
    no real coverage while keeping the human's decision entirely inside the
    perception it's supposed to be limited to.
    """
    module = d.load_layout(SCENE)
    road = module.build_road()
    human = h.add_human_vehicle(road, module.HUMAN_ROUTE, enable_fov=False, enable_occlusion=False)
    sb.add_background_traffic(road, count=80, seed=0)  # seed/count chosen to put real traffic near spawn
    lane_indexes = sb.all_lane_indexes(road)

    road.act()
    candidates = sb.nearby_vehicles(road, human, 35.0)
    assert len(candidates) > 0, "test is vacuous with no nearby candidates -- pick a different seed/count"
    assert human.visible_candidates(candidates) == candidates  # the ablation precondition

    front = sb.find_front_vehicle(road, human, lane_indexes, candidates)
    expected = human.acceleration(ego_vehicle=human, front_vehicle=front, rear_vehicle=None)
    heads = {id(v) for v in candidates
              if sb.find_front_vehicle(road, v, lane_indexes, sb.nearby_vehicles(road, v, 35.0)) is None}
    if front is None:
        heads.add(id(human))
    conflict = sb.crossing_conflict_brake(human, candidates, heads=heads)
    if conflict is not None:
        expected = min(expected, conflict)
    expected = max(expected, -human.speed / DT)

    h.apply_human_aware_car_following(road, lane_indexes, DT)
    actual = human.action["acceleration"]

    assert abs(expected - actual) < 1e-9, (expected, actual)
    print(f"check_ablation_matches_unfiltered_logic: OK ({expected:.4f} == {actual:.4f})")


def run_unit_checks():
    check_in_cone()
    check_segment_intersects_rotated_rect()
    check_is_occluded()
    check_route_progress_monotonic()
    check_visible_candidates_identity_when_disabled()
    check_ablation_matches_unfiltered_logic()
    print("\nALL UNIT CHECKS PASSED\n")


# ============================================================================
# SIMULATION HARNESS
# ============================================================================

def run_one(fov_deg, seed, background_count=28, steps=1800, enable_occlusion=True, log_prefix=""):
    module = d.load_layout(SCENE)
    road = module.build_road()
    human = h.add_human_vehicle(road, module.HUMAN_ROUTE, fov_deg=fov_deg,
                                 enable_occlusion=enable_occlusion)
    # Bias spawn density toward lanes the human's own route actually comes
    # near, when the layout exposes that (mega_scene does; real_*.py scenes
    # don't and fall back to every lane, unchanged from before). A uniform
    # spawn over a small synthetic network like mega_scene puts a lot of
    # `count` on lanes the human never visits (round_about's own unused
    # cardinals, etc.), which is why an earlier sweep here showed literally
    # identical numbers at every FOV width -- the human rarely had anything
    # nearby to actually miss.
    route_lanes = getattr(module, "route_adjacent_lane_indexes", lambda: None)()
    sb.add_background_traffic(road, count=background_count, seed=seed, lane_indexes=route_lanes)
    lane_indexes = sb.all_lane_indexes(road)
    total_len = h.route_total_length(module.HUMAN_ROUTE)

    speeds = []
    crashed_ids = set()
    hard_brakes = []  # (step, distance_to_trigger) -- one per DISTINCT episode, not per step
    was_hard_braking = False
    min_gap = float("inf")
    human_crashed_at = None

    for step in range(steps):
        road.act()
        h.apply_human_aware_car_following(road, lane_indexes, DT)
        accel_before = human.action["acceleration"]
        road.step(DT)
        h.advance_vehicles_with_route(road, lane_indexes)

        crashed_ids.update(id(v) for v in road.vehicles if v.crashed)
        if human.crashed and human_crashed_at is None:
            human_crashed_at = step

        if not human.crashed:
            speeds.append(human.speed)
            is_hard_braking = accel_before <= HARD_BRAKE_THRESHOLD
            if is_hard_braking and not was_hard_braking:
                # rising edge only: the moment braking STARTS is the actual
                # reaction distance; logging every step of one sustained
                # brake would count one episode dozens of times and confound
                # "hard_brake_count" with "how long each brake lasted".
                candidates = sb.nearby_vehicles(road, human, 35.0)
                visible = human.visible_candidates(candidates)
                if visible:
                    dists = [np.linalg.norm(np.array(human.position) - np.array(v.position)) for v in visible]
                    hard_brakes.append((step, min(dists)))
                else:
                    hard_brakes.append((step, float("nan")))
            was_hard_braking = is_hard_braking
            for v in road.vehicles:
                if v is human or v.crashed:
                    continue
                gap = np.linalg.norm(np.array(human.position) - np.array(v.position))
                min_gap = min(min_gap, gap)

        # advance_vehicles_with_route() never despawns a route-following
        # vehicle (see its own docstring) -- this assert is a tripwire, not
        # a normal code path, so a regression here fails loudly instead of
        # silently reporting a misleadingly-zeroed progress number.
        assert human in road.vehicles, f"human was despawned at step {step} -- this should be impossible"

    progress, _ = h._route_progress(module.HUMAN_ROUTE, human.position)
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    result = {
        "fov_deg": fov_deg, "seed": seed,
        "human_crashed": human_crashed_at is not None,
        "human_crashed_at": human_crashed_at,
        "scene_crash_count": len(crashed_ids),
        "avg_speed": avg_speed,
        "route_progress_frac": min(progress / total_len, 1.0),
        "hard_brake_count": len(hard_brakes),
        "hard_brake_dists": [d for _, d in hard_brakes if not np.isnan(d)],
        "min_gap": min_gap if min_gap != float("inf") else None,
    }
    reaction_dist = np.mean(result["hard_brake_dists"]) if result["hard_brake_dists"] else float("nan")
    min_gap_str = f"{result['min_gap']:.2f}" if result["min_gap"] is not None else "None"
    print(f"{log_prefix}fov={fov_deg:>5.0f} seed={seed}: "
          f"crashed={result['human_crashed']} scene_crashes={result['scene_crash_count']} "
          f"avg_speed={avg_speed:.2f} progress={result['route_progress_frac']*100:.0f}% "
          f"hard_brakes={result['hard_brake_count']} reaction_dist={reaction_dist:.1f} "
          f"min_gap={min_gap_str}")
    return result


def run_sweep(fov_widths=(360, 90, 60, 30), seeds=(0, 1), background_count=28, steps=1800):
    print(f"\n=== FOV SWEEP: widths={fov_widths} seeds={seeds} "
          f"background_count={background_count} steps={steps} ({steps*DT:.0f}s sim) ===\n")
    t0 = time.time()
    results = []
    for seed in seeds:
        for fov in fov_widths:
            results.append(run_one(fov, seed, background_count=background_count, steps=steps))
    print(f"\nsweep wall time: {time.time()-t0:.1f}s\n")

    print("=== FUNCTIONAL-AT-EVERY-FOV CHECK (property B) ===")
    ok_b = True
    for r in results:
        functional = (not r["human_crashed"]) and r["avg_speed"] > 0.5 and r["route_progress_frac"] > 0.1
        if not functional:
            ok_b = False
        print(f"  fov={r['fov_deg']:>5.0f} seed={r['seed']}: "
              f"{'OK' if functional else 'FAIL'} "
              f"(crashed={r['human_crashed']}, avg_speed={r['avg_speed']:.2f}, "
              f"progress={r['route_progress_frac']*100:.0f}%)")
    print(f"Property B (functional at every FOV): {'PASS' if ok_b else 'FAIL'}\n")

    print("=== BEHAVIOR-DIFFERS-BY-FOV CHECK (property A) ===")
    by_fov = {}
    for r in results:
        by_fov.setdefault(r["fov_deg"], []).append(r)
    for fov in sorted(by_fov, reverse=True):
        rs = by_fov[fov]
        avg_reaction = np.mean([d for r in rs for d in r["hard_brake_dists"]]) if any(r["hard_brake_dists"] for r in rs) else float("nan")
        avg_brakes = np.mean([r["hard_brake_count"] for r in rs])
        avg_progress = np.mean([r["route_progress_frac"] for r in rs])
        avg_speed = np.mean([r["avg_speed"] for r in rs])
        print(f"  fov={fov:>5.0f}: avg_reaction_dist={avg_reaction:.2f}  avg_hard_brakes={avg_brakes:.1f}  "
              f"avg_progress={avg_progress*100:.0f}%  avg_speed={avg_speed:.2f}")
    print("(inspect the trend above across fov -- narrower should show shorter reaction distance "
          "and/or more hard brakes and/or less progress; this is the property A check)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="also run the full FOV sweep (slow)")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--background-count", type=int, default=28)
    args = parser.parse_args()

    run_unit_checks()
    if args.sweep:
        run_sweep(background_count=args.background_count, steps=args.steps)
