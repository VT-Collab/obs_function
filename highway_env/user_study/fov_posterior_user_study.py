"""FOVPosterior for the discrete-subtask user study. Infers the human's
FOV purely from whether ApproximateLimitVisionHuman is observed PROCEEDING
or HELD at a maneuver boundary -- interface.py's own UI state (which
button the user clicked) is never an input here, only the human vehicle's
own position/speed and the SAME geometric gated() check every FOV
hypothesis would independently run against the same nearby traffic.

WHY A SEPARATE CLASS from robot.filter.core.fov_posterior.FOVPosterior:
that one's likelihood is a Gaussian residual between the human's OBSERVED
acceleration and each hypothesis's own IDM "shadow" prediction -- a
continuous-control observation model that fits LimitedVisionHuman's own
continuous, always-reacting driving. ApproximateLimitVisionHuman's own
"reaction" IS the discrete gate itself (held vs proceeding), which most of
the time carries FAR MORE signal than a subtle accel residual would: a
384-vs-30-degree cone either agrees or disagrees on whether a specific,
already-visible-or-not hazard exists at THIS maneuver, a binary, robust
distinction -- not a fuzzy continuous one. This is also why the original
class's own belief was found (see robot/filter/harness's own sweep
results) to collapse into an exact tie among every FOV >= 60 degrees on a
real layout: once a forward cone is wide enough to see whatever's directly
ahead, continuous IDM behavior stops differing at all between wider cones.
A held/proceeding decision at a genuine side/behind hazard (this domain's
merge_in/merge_out/turn spans in particular) keeps discriminating wider
cones from each other long after a plain forward-facing IDM residual would
have stopped.

NO SIGNAL on a "forward" span (gated() is always False there, for every
hypothesis, by construction -- see subtasks.py) -- update() is a no-op
then, exactly mirroring FOVPosterior's own "nothing to learn from this
tick" cases (e.g. no visible front vehicle at all).
"""
import numpy as np

from common.geometry import visible


class SubtaskFOVPosterior:
    def __init__(self, fov_candidates=(30.0, 60.0, 90.0, 180.0, 360.0), prior=None,
                 moving_eps=0.3, likelihood_floor=0.05, enable_occlusion=True):
        self.fov_candidates = tuple(fov_candidates)
        self.moving_eps = moving_eps
        self.likelihood_floor = likelihood_floor
        self.enable_occlusion = enable_occlusion
        n = len(self.fov_candidates)
        prior = np.full(n, 1.0 / n) if prior is None else np.asarray(prior, dtype=float)
        self.belief = prior / prior.sum()

    def update(self, human, candidates):
        """One Bayes update from a single tick's observation of `human` (an
        ApproximateLimitVisionHuman). `candidates`: every OTHER nearby
        vehicle, UNFILTERED -- each hypothesis applies its own cone/
        occlusion test to this same set, same convention as the other
        FOVPosterior. Returns the updated belief dict; a no-op (belief
        unchanged) whenever the human isn't currently at a maneuver span
        at all (see module docstring).
        """
        span = human.active_span if human.active_span is not None else human.current_span()
        if span is None or span[0] == "forward":
            return self.beliefs()

        moving = human.speed > self.moving_eps
        likelihoods = np.empty(len(self.fov_candidates))
        for i, fov in enumerate(self.fov_candidates):
            vis = [c for c in candidates
                   if visible(human.position, human.heading, c, fov,
                              [o for o in candidates if o is not c],
                              enable_occlusion=self.enable_occlusion)]
            would_gate = human.gated(span, vis)
            # A driver who always proceeds the instant it's legal (the
            # task-free stand-in for "the user is trying to make
            # progress", not a claim about any specific real user) is
            # consistent with hypothesis fov iff observed motion matches
            # what THAT hypothesis's own gate would allow.
            consistent = (not would_gate) if moving else would_gate
            likelihoods[i] = 1.0 if consistent else self.likelihood_floor

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
