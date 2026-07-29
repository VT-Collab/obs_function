"""
EXACT Bayesian inference of the human's FOV against the SAMPLING human
(fov/human/agent/limited_vision_human.py).

    P(theta | tau_{1:t}) ~ P(theta) * PROD_i P(tau_i | o_{1:i}, theta)

Unlike bayes_fov.py (which faces the DETERMINISTIC human and therefore needs an
epsilon "exploration mass" so a single mismatch does not zero a hypothesis),
this filter faces a human whose policy P(tau | o, theta) is a KNOWN categorical -
uniform over the currently available advancing subtasks. So the per-step
likelihood is read off EXACTLY:

    P(tau_obs | o, theta) = shadow_theta.subtask_distribution(state, committed)[tau_obs]

One shadow LimitedVisionSteakHuman per candidate FOV maintains that FOV's own
FOV-gated, decaying beliefs (via observe()), so shadow theta's distribution is
what a human with FOV=theta would actually use at this instant. A tiny FLOOR
keeps a zero-probability step finite (large but recoverable), exactly like the
minigrid design's 1e-9 guard.

The discrimination is automatic and interpretable:
  * At a FRESH sample the true theta assigns 1/|A_theta| to what was drawn;
    a theta whose belief makes that subtask unavailable assigns 0 -> floored.
  * During a COMMITTED errand every theta that still deems it helpful assigns
    1.0 (no information); a theta that (through a wider cone) has SEEN the errand
    become useless would have re-sampled, so it assigns 0 to the continued
    subtask -> that observation is exactly the FOV signal.
"""
import math

from fov.human.agent.limited_vision_human import (
    LimitedVisionSteakHuman, SAMPLING_SUBTASKS)
from fov.human.planning.steak_planner import SteakMotionPlanner

FLOOR = 1e-6      # finite floor on a single-step likelihood (recoverable hit)


def fov_prior(candidate_fovs):
    return {f: 1.0 / len(candidate_fovs) for f in candidate_fovs}


class SamplingBayesFOVInference:
    """Log-space posterior over candidate FOVs, updated online from the human's
    observed SUBTASK (not raw action) using the known sampling likelihood."""

    def __init__(self, mdp, mlp, candidate_fovs, human_agent_index=1,
                 prior=None, floor=FLOOR):
        self.candidate_fovs = list(candidate_fovs)
        self.floor = floor
        self.prior = fov_prior(self.candidate_fovs) if prior is None else prior
        self.log_post = {f: math.log(self.prior[f]) for f in self.candidate_fovs}

        planner = SteakMotionPlanner(mdp, mlp)
        self.shadows = {
            f: LimitedVisionSteakHuman(mdp, f, planner, agent_index=human_agent_index)
            for f in self.candidate_fovs
        }
        self.committed = None        # human's current advancing commitment (observed)
        self.n_informative = 0       # steps where hypotheses disagreed (read by eval)

    def update(self, state, observed_subtask):
        """Observe one (state, subtask) pair and update the posterior. Call
        BEFORE stepping the env, so every shadow scores the state the human
        actually acted on."""
        probs = {}
        for f, sh in self.shadows.items():
            try:
                sh.observe(state)
                dist = sh.subtask_distribution(state, self.committed)
                p = dist.get(observed_subtask, 0.0)
            except Exception:
                p = 0.0
            probs[f] = p

        if len({round(p, 6) for p in probs.values()}) > 1:
            self.n_informative += 1
        for f in self.candidate_fovs:
            self.log_post[f] += math.log(max(probs[f], self.floor))
        self._normalise()

        # carry the human's advancing commitment forward (looks/drops reset it)
        self.committed = observed_subtask if observed_subtask in SAMPLING_SUBTASKS else None
        return self.posterior()

    def posterior(self):
        return {f: math.exp(lp) for f, lp in self.log_post.items()}

    def map_fov(self):
        return max(self.log_post, key=self.log_post.get)

    def p_true(self, true_fov):
        return self.posterior().get(true_fov, 0.0)

    def entropy(self):
        p = self.posterior()
        return -sum(pf * math.log(pf) for pf in p.values() if pf > 0)

    def _normalise(self):
        m = max(self.log_post.values())
        log_z = m + math.log(sum(math.exp(lp - m) for lp in self.log_post.values()))
        self.log_post = {f: lp - log_z for f, lp in self.log_post.items()}
