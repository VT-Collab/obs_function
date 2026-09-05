"""Robot baseline policies -- what the robot vehicle does each tick, with
NO FOV reasoning at all. Mirrors steakhouse's own _BaseRobot / SoloRobot /
HandoffRobot split (nominal_policy/baselines.py): one common
`.action(ctx) -> acceleration` interface, several concrete policies.

The FOV-AWARE policy (which wraps one of these with a live belief about
the human's FOV) lives one level up in filter/, not here -- same split as
steakhouse's own nominal_policy/ (plain baselines) vs. filter/ (the
FOV-aware layer wrapped around one), so a baseline here can never import
anything from filter/ and stays usable completely on its own.

`ctx` is a plain namespace carrying only what a policy might need --
never a task/route identity, so a policy can't accidentally special-case
"this junction" the way this whole FOV-filter effort is explicitly meant
to avoid:
    ctx.robot           the robot IDMVehicle
    ctx.front_vehicle   whatever's directly ahead of the robot in its own
                        lane (scene_background.find_front_vehicle's
                        result) -- ordinary car-following target
    ctx.human           the LimitedVisionHuman, or None if out of range
    ctx.belief          FOVPosterior.beliefs() dict, or None if this
                        policy isn't given one (never populated for
                        anything defined here -- only filter/'s own
                        FOVAwareBaseline reads it)
    ctx.dt              seconds since the last tick
"""
import numpy as np


class NominalBaseline:
    """Ignores the human entirely: ordinary IDM car-following against
    ctx.front_vehicle, ego's own acceleration() (full information, no FOV
    reasoning of the robot's own). The reference point every other policy
    is compared against."""
    name = "nominal"

    def action(self, ctx):
        return ctx.robot.acceleration(ego_vehicle=ctx.robot, front_vehicle=ctx.front_vehicle, rear_vehicle=None)


class CautiousBaseline(NominalBaseline):
    """Always brakes an extra fixed amount once the human is within
    trigger_dist, regardless of any belief about whether the human can see
    it -- the "always safe, never efficient" reference point an FOV-aware
    policy should beat on throughput while matching (or nearly matching)
    on safety. Unconditional caution is the whole point: this is what a
    policy with no perception of the human's FOV at all, but told to be
    careful near people, looks like."""
    name = "cautious"

    def __init__(self, extra_decel=2.0, trigger_dist=20.0):
        self.extra_decel = extra_decel
        self.trigger_dist = trigger_dist

    def action(self, ctx):
        accel = super().action(ctx)
        if ctx.human is not None:
            dist = float(np.linalg.norm(np.asarray(ctx.robot.position) - np.asarray(ctx.human.position)))
            if dist < self.trigger_dist:
                accel -= self.extra_decel
        return accel


REGISTRY = {cls.name: cls for cls in (NominalBaseline, CautiousBaseline)}
