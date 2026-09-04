"""Which VEHICLE DYNAMICS the robot drives with -- idm/linear/aggressive/
defensive -- kept separate from the POLICY WRAPPER (nominal_policy/
baselines.py's NominalBaseline/CautiousBaseline, filter/core/
fov_aware_baseline.py's FOVAwareBaseline) that decides whether to blend in
extra caution. NominalBaseline/FOVAwareBaseline both just call
ctx.robot.acceleration(...) -- they never care which of these four classes
ctx.robot actually is -- so any vehicle here composes with any wrapper:
"defensive, fov-aware" and "aggressive, nominal" are both just
(make_vehicle("defensive"), make_baseline("fov_aware")) or
(make_vehicle("aggressive"), make_baseline("nominal")).

highway_env.vehicle.behavior ships exactly these four IDMVehicle-family
classes (verified against the installed package, not assumed):
IDMVehicle (nonlinear IDM formula), LinearVehicle (IDMVehicle subclass, a
linear-in-features formula instead), AggressiveVehicle/DefensiveVehicle
(LinearVehicle subclasses, two preset parameter tunings). All four need
scene1_background.NoResnapIDMVehicle's own on_state_update() override
(disables the default per-step "resnap to closest lane in the whole
network", which is both expensive and actively wrong once lane
transitions are managed manually by advance_vehicles_with_route) -- the
_NoResnapMixin below is the exact same one-method override, reused via
multiple inheritance instead of copied four times.

highway_env.vehicle.controller.ControlledVehicle/MDPVehicle are
deliberately NOT included here: verified directly in the installed
source that ControlledVehicle.act() only does speed_control(target_speed)
+ lane-target steering, with NO car-following or collision avoidance at
all -- it's the actuation layer an RL policy sits on top of, not a
complete driving policy by itself, so it isn't a "vehicle" option in this
registry (see the "count the baselines" conversation this file follows
from).
"""
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle, AggressiveVehicle, DefensiveVehicle

import scene1_background as sb


class _NoResnapMixin:
    """See scene1_background.NoResnapIDMVehicle's own docstring for why
    this is needed at all -- identical override, just mixed into the
    other three IDMVehicle-family classes instead of copied."""

    def on_state_update(self) -> None:
        pass


class NoResnapLinearVehicle(_NoResnapMixin, LinearVehicle):
    pass


class NoResnapAggressiveVehicle(_NoResnapMixin, AggressiveVehicle):
    pass


class NoResnapDefensiveVehicle(_NoResnapMixin, DefensiveVehicle):
    pass


REGISTRY = {
    "idm": sb.NoResnapIDMVehicle,
    "linear": NoResnapLinearVehicle,
    "aggressive": NoResnapAggressiveVehicle,
    "defensive": NoResnapDefensiveVehicle,
}


def make_vehicle(kind, road, position, heading=0.0, speed=9.0):
    if kind not in REGISTRY:
        raise KeyError(f"unknown robot vehicle {kind!r}, choose from {sorted(REGISTRY)}")
    return REGISTRY[kind](road, position, heading=heading, speed=speed)
