"""Exact discrete Bayes filter over a small set of candidate FOV cone
widths for another vehicle (typically the LimitedVisionHuman), inferred
purely from how that vehicle actually drives.

Mirrors steakhouse/no_larping (and misha)'s own robot/filter/core/
fov_posterior.py: one "shadow" evaluation per candidate FOV, likelihood =
how well that hypothesis explains the OBSERVED behavior this tick, belief
updated by plain Bayes' rule and renormalized -- no particle filter, no
learned model, `len(fov_candidates)` numbers in a dict. `alpha`/`sigma`
below is calibrated trust in the underlying acceleration model, not a
claimed property of the real driver, same as that module's own `alpha`.

CONTINUOUS-CONTROL ADAPTATION: Overcooked's shadow predicts one DISCRETE
action per hypothesis and scores a hit/miss against the human's observed
discrete action. There is no discrete action space here, so each
hypothesis's shadow instead predicts a continuous acceleration (by asking
the observed vehicle's OWN `.acceleration()` method what it would compute
if only IT were fed that hypothesis's visible-front-vehicle, rather than
building a whole separate simulated copy of it), and the likelihood is
Gaussian in the residual between predicted and observed acceleration. Same
exact-Bayes-over-a-handful-of-hypotheses structure, same "no task
knowledge" inputs (positions/headings/extents of nearby traffic and the
observed vehicle's own kinematics only -- no route, no destination, no
junction/task identity), just a continuous observation model instead of a
discrete one.
"""
import numpy as np

from common.geometry import visible


class FOVPosterior:
    def __init__(self, fov_candidates=(30.0, 60.0, 90.0, 180.0, 360.0), sigma=1.5, prior=None):
        self.fov_candidates = tuple(fov_candidates)
        self.sigma = sigma
        n = len(self.fov_candidates)
        prior = np.full(n, 1.0 / n) if prior is None else np.asarray(prior, dtype=float)
        self.belief = prior / prior.sum()

    def update(self, observed_vehicle, candidates, observed_acceleration, front_vehicle_fn, enable_occlusion=True):
        """One Bayes update from a single tick's observation.

        observed_vehicle: the vehicle whose FOV width is being inferred --
        needs only .position, .heading, and .acceleration(ego_vehicle,
        front_vehicle, rear_vehicle) (any highway_env IDMVehicle already
        has this).
        candidates: every OTHER nearby vehicle, UNFILTERED -- each
        hypothesis applies its own cone/occlusion test to this same set,
        so the caller does no FOV reasoning of its own.
        observed_acceleration: what `observed_vehicle` actually did this
        tick (its true, already-computed action -- ground truth for the
        likelihood, not something this class computes).
        front_vehicle_fn(visible_list) -> front_vehicle: caller-supplied
        (e.g. scene1_background.find_front_vehicle bound to the current
        road/lane_indexes), so this module never imports anything
        road/lane-specific -- it only ever sees whatever list of vehicle-
        like objects the caller hands it.

        Returns the updated belief as {fov: probability}.
        """
        likelihoods = np.empty(len(self.fov_candidates))
        for i, fov in enumerate(self.fov_candidates):
            vis = [c for c in candidates
                   if visible(observed_vehicle.position, observed_vehicle.heading, c, fov,
                              [o for o in candidates if o is not c], enable_occlusion=enable_occlusion)]
            front = front_vehicle_fn(vis)
            predicted = observed_vehicle.acceleration(ego_vehicle=observed_vehicle, front_vehicle=front,
                                                        rear_vehicle=None)
            residual = observed_acceleration - predicted
            likelihoods[i] = np.exp(-(residual ** 2) / (2 * self.sigma ** 2))

        posterior = self.belief * likelihoods
        total = posterior.sum()
        self.belief = posterior / total if total > 1e-12 else np.full(len(self.belief), 1.0 / len(self.belief))
        return self.beliefs()

    def beliefs(self):
        return dict(zip(self.fov_candidates, self.belief))

    def map_fov(self):
        return self.fov_candidates[int(np.argmax(self.belief))]

    def belief_for(self, fov):
        return float(self.belief[self.fov_candidates.index(fov)])
