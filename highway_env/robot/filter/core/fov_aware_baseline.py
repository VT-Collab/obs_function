"""The one policy that actually uses the FOV posterior + filter: reads a
LIVE belief about the human's FOV width and rolls a short horizon forward
under a handful of candidate accels to see which one best manages the
robot's own visibility to the human -- see fov_filter.FOVFilter's own
docstring for the mechanism (mirrors steakhouse/misha/robot/filter/core/
my_fov_filter.py's rollout-and-score structure, adapted to a continuous
scalar accel instead of a discrete grid-cell search).

Lives in filter/ (not nominal_policy/), same split steakhouse itself uses:
nominal_policy/ never imports anything from filter/, but filter/ freely
builds on top of a nominal_policy baseline plus its own posterior -- this
class is that composition for the one baseline (Nominal) worth wrapping
here, mirroring steakhouse/misha/robot/methods.py's own "a baseline column
and the same filter layer wrapped over each entry" registry pattern.

Question under test: with weight_seen > 0, does the robot manage its own
visibility to the human (slow down/hold back a bit while still ahead in the
human's cone, no different at all once far enough away that no candidate
accel changes whether it's seen) -- see FOVFilter's own class docstring for
the exact worked example this is checked against.
"""
from robot.nominal_policy.baselines import NominalBaseline
from robot.filter.core.fov_filter import FOVFilter


class FOVAwareBaseline(NominalBaseline):
    name = "fov_aware"

    def __init__(self, **filter_kwargs):
        self.filter = FOVFilter(**filter_kwargs)

    def action(self, ctx):
        return self.filter.action(ctx)
