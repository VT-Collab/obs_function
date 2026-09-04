"""Scratch tests for the rewritten FOVFilter (robot/filter/core/fov_filter.py).
Standalone: builds plain duck-typed stand-ins for ctx.robot/ctx.human rather
than a real Road/Vehicle, since FOVFilter.action() only ever touches
.position/.heading/.speed/.action/.crashed and one .acceleration() call --
see fov_filter.py's own module docstring on why the rollout is pure
kinematics with no lane/route dependency at all.

Run:
    python test_fov_filter.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # highway_env/
from robot.filter.core.fov_filter import FOVFilter  # noqa: E402


class _Mock:
    """A robot or human stand-in: fixed .acceleration() return value (the
    "baseline"), everything else a plain attribute FOVFilter reads."""

    def __init__(self, position, heading, speed, accel=0.0, crashed=False):
        self.position = np.array(position, dtype=float)
        self.heading = heading
        self.speed = speed
        self.action = {"acceleration": accel}
        self.crashed = crashed
        self._base_accel = accel

    def acceleration(self, ego_vehicle, front_vehicle, rear_vehicle):
        return self._base_accel


def _ctx(robot, human, belief):
    return types.SimpleNamespace(robot=robot, front_vehicle=None, human=human, belief=belief, dt=1 / 15)


# ============================================================================
# 1. weight_seen == 0 -- exact no-op (never even a candidate search)
# ============================================================================

def check_zero_weight_is_noop():
    robot = _Mock((3, 1), np.radians(90), 5.0, accel=0.5)
    human = _Mock((0, 0), 0.0, 8.0, accel=0.0)
    f = FOVFilter(weight_seen=0.0)
    a = f.action(_ctx(robot, human, {90.0: 1.0}))
    assert a == robot._base_accel, f"expected exact baseline {robot._base_accel}, got {a}"
    print("check_zero_weight_is_noop: OK")


# ============================================================================
# 2. weight_seen > 0, robot on a path that would exit the cone if it speeds
#    up -- should brake below baseline to stay seen longer.
# ============================================================================

def check_positive_weight_seen_slows_to_stay_visible():
    # Human at the origin, facing +x (heading 0), a fixed 60 deg cone (30
    # deg half-angle). Robot sits at (10, 1) heading +y (90 deg) at a slow
    # 1 m/s, comfortably inside the cone (atan2(1,10) = 5.7 deg). Baseline
    # wants to speed up (accel=+0.5): over the 4s rollout its y-speed ramps
    # 1 -> 3 m/s, covering 8m, landing at y=9 -- atan2(9,10) = 42 deg,
    # OUTSIDE the cone well before the horizon ends. Braking instead (any
    # accel <~ -0.25) keeps y-speed low enough that it never crosses the
    # 30 deg edge across the whole 4s -- verified by hand (avg speed *4s
    # keeps y under 10*tan(30) = 5.77) before asserting on the filter's own
    # choice. This is the worked example from FOVFilter's own class
    # docstring: a robot whose OWN motion would carry it out of view
    # should trade a little baseline progress for more ticks seen when
    # weight_seen > 0.
    robot = _Mock((10.0, 1.0), np.radians(90.0), 1.0, accel=0.5)
    human = _Mock((0.0, 0.0), 0.0, 0.0, accel=0.0)  # human stationary: isolates the robot's own motion
    belief = {60.0: 1.0}

    f = FOVFilter(weight_seen=1.0, depth_s=4.0, max_extra_accel=1.5, max_extra_decel=3.0, n_candidates=13)
    a = f.action(_ctx(robot, human, belief))
    assert a < robot._base_accel - 1e-6, (
        f"expected a brake below baseline ({robot._base_accel}) to stay in-cone longer, got {a}")

    # And the bounded-deviation constraint actually bounds it, on both ends.
    assert robot._base_accel - f.max_extra_decel - 1e-6 <= a <= robot._base_accel + f.max_extra_accel + 1e-6
    print(f"check_positive_weight_seen_slows_to_stay_visible: OK (baseline={robot._base_accel}, chosen={a:.3f})")


# ============================================================================
# 3. weight_seen < 0 -- sign flip: same geometry, now rewarded for LEAVING
#    the cone, so it should accelerate (at least as much as baseline, if
#    not more, up to the bound).
# ============================================================================

def check_negative_weight_seen_prefers_leaving():
    robot = _Mock((10.0, 1.0), np.radians(90.0), 1.0, accel=0.5)
    human = _Mock((0.0, 0.0), 0.0, 0.0, accel=0.0)
    belief = {60.0: 1.0}

    f = FOVFilter(weight_seen=-1.0, depth_s=4.0, max_extra_accel=1.5, max_extra_decel=3.0, n_candidates=13)
    a = f.action(_ctx(robot, human, belief))
    assert a > robot._base_accel + 1e-6, (
        f"expected acceleration above baseline ({robot._base_accel}) to leave the cone sooner, got {a}")
    print(f"check_negative_weight_seen_prefers_leaving: OK (baseline={robot._base_accel}, chosen={a:.3f})")


# ============================================================================
# 4. Robot far enough away that NO candidate accel (over one rollout
#    horizon) changes whether it's seen -- every candidate ties on the
#    visibility term, so the tie-break picks baseline back exactly.
# ============================================================================

def check_far_away_robot_matches_baseline():
    # 500m north of the human, well outside a forward-facing 90 deg cone
    # (perpendicular, in fact) -- a few seconds of any candidate accel
    # moves it by tens of meters at most, nowhere near enough to swing
    # whether a point 500m away is angularly inside a 45 deg half-cone.
    robot = _Mock((0.0, 500.0), 0.0, 10.0, accel=0.7)
    human = _Mock((0.0, 0.0), 0.0, 10.0, accel=0.0)
    belief = {90.0: 1.0}

    f = FOVFilter(weight_seen=1.0, depth_s=4.0, max_extra_accel=1.5, max_extra_decel=3.0, n_candidates=13)
    a = f.action(_ctx(robot, human, belief))
    assert abs(a - robot._base_accel) < 1e-6, f"expected baseline {robot._base_accel} unchanged, got {a}"
    print(f"check_far_away_robot_matches_baseline: OK (baseline={robot._base_accel}, chosen={a:.3f})")


# ============================================================================
# 5. No human / no belief -- graceful fallback to plain baseline.
# ============================================================================

def check_missing_human_or_belief_falls_back():
    robot = _Mock((3.0, 1.0), np.radians(90.0), 3.0, accel=1.0)
    human = _Mock((0.0, 0.0), 0.0, 0.0, accel=0.0)
    f = FOVFilter(weight_seen=1.0)

    a1 = f.action(_ctx(robot, None, {60.0: 1.0}))
    a2 = f.action(_ctx(robot, human, None))
    assert a1 == robot._base_accel and a2 == robot._base_accel
    print("check_missing_human_or_belief_falls_back: OK")


if __name__ == "__main__":
    check_zero_weight_is_noop()
    check_positive_weight_seen_slows_to_stay_visible()
    check_negative_weight_seen_prefers_leaving()
    check_far_away_robot_matches_baseline()
    check_missing_human_or_belief_falls_back()
    print("\nALL FOV FILTER CHECKS PASSED")
