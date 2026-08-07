"""
MISHA NEW CHANGE - THE ONLY PLACE FOV INFORMATION EXISTS IN THE ROBOT'S POLICY.

Everything to do with inferring the human's field of view is confined to this
package. baseline/ imports nothing from here and nothing from
fov/robot/inference/ - verified by deleting both directories and confirming the
baseline still trains. That is what makes the comparison a real control.

WHAT THIS MODULE CONSUMES
    map_fov   argmax of the Bayesian posterior - the current best estimate
    entropy   how uncertain that estimate still is
and nothing else. Notably NOT the posterior mass: we measured the filter
reporting P=1.0 even when the true FOV is absent from the candidate set, so the
mass is not a usable confidence. Entropy relative to uniform is.

WHY THE STEAKHOUSE MAPPING DIFFERS FROM MINIGRID
Minigrid's robot has no body: its actions are "wait" + reveal types, so the
module's job there is deciding WHAT TO SAY. Here the robot is a physical cook
with no communication channel at all, so the module decides WHAT TO DO, and the
mapping is driven by two effects measured in this domain:

  1. A blind human cannot FIND stations. At fov=30 the human's entire subtask
     trace is ['explore'] and it delivers 0; at fov=180 it runs a clean delivery
     cycle. So the narrower the estimate, the more the robot should simply do
     the work itself rather than leave jobs the human will never discover.

  2. A human only skips redundant work it can SEE. Measured redundant tasks
     avoided: fov=30 -> 13, fov=90 -> 27, fov=360 -> 69. So against a WIDE-FOV
     human the robot can deliberately pick up an item inside their cone to steer
     them off that task - a coordination lever that simply does not exist
     against a blind partner, because they will never look.

Effect 1 pushes toward taking over as vision narrows; effect 2 pushes toward
signalling-by-doing as vision widens. The bias below encodes exactly that, which
is why it is not a monotone function of fov.
"""
import math

import torch

from steakhouse.fov.robot.policy.old.baseline.features import ACTIONS
# The EXACT filter for the finalized SAMPLING human (validated ~0.87 final-correct
# in evaluate_sampling_inference). The old bayes_fov.BayesFOVInference faced the
# deterministic human and scored the raw ACTION with an epsilon fudge - wrong
# likelihood model for this human, so the module would be misinformed.
from steakhouse.fov.robot.policy.old.inference.bayes_fov_sampling import SamplingBayesFOVInference

TAKE_IDX = [i for i, a in enumerate(ACTIONS) if a.startswith("take_")]
STAGE_IDX = [i for i, a in enumerate(ACTIONS) if a == "stage_visible"]
WORK_IDX = [i for i, a in enumerate(ACTIONS) if a == "work"]


def blindness(map_fov):
    """0 = sees everything, 1 = sees almost nothing."""
    return max(0.0, min(1.0, (360.0 - float(map_fov)) / 330.0))


def confidence(entropy, n_candidates):
    """1 = posterior collapsed, 0 = still uniform."""
    return max(0.0, 1.0 - entropy / math.log(max(2, n_candidates)))


def logit_bias(map_fov, entropy, n_candidates, strength=1.5,
               kw=1.0, ks=1.0, kt=0.0, device=None):
    """FOV-conditioned bias over ACTIONS. Directions corrected from the measured
    failure of the first design (which thrashed take_* when blind and lost):

      BLIND human (b->1): contributes ~nothing and cannot find stations, so the
        fastest team outcome is the robot cooking the whole pipeline COHERENTLY.
        'work' = the greedy full-pipeline action, so push THAT (kw), not the five
        take_* (which the policy flails between). kt keeps a small take_* option.

      SIGHTED human (b->0): competent AND reactive - it yields / stops fetching a
        task it SEES the robot doing. So acting where the human can see it
        (stage_visible = the VISIBILITY lever) offloads work onto the human and
        cuts double-cooking. Push stage (ks).

    Scaled by confidence(entropy): an unsure filter cannot steer anything."""
    bias = torch.zeros(len(ACTIONS), device=device)
    if map_fov is None:
        return bias
    c = confidence(entropy, n_candidates)
    if c <= 0.0:
        return bias
    b = blindness(map_fov)
    w = strength * c
    for i in WORK_IDX:
        bias[i] += w * b * kw                    # blind -> cook it yourself, fast
    for i in TAKE_IDX:
        bias[i] += w * b * kt                    # (optional) mild take-over when blind
    for i in STAGE_IDX:
        bias[i] += w * (1.0 - b) * ks            # sighted -> act IN the human's view
    return bias


class FOVModule:
    """Wraps a trained FOV-BLIND policy. The network is never retrained - only
    its logits are biased - so any measured difference is attributable to FOV
    information rather than to extra capacity or a luckier optimisation run."""

    def __init__(self, mdp, mlp, candidate_fovs, human_index=1, strength=1.5,
                 kw=1.0, ks=1.0, kt=0.0):
        self.mdp, self.mlp = mdp, mlp
        self.candidate_fovs = list(candidate_fovs)
        self.human_index = human_index
        self.strength = strength
        self.kw, self.ks, self.kt = kw, ks, kt   # blind->work, sighted->stage, blind->take
        self.reset()

    def reset(self):
        self.inf = SamplingBayesFOVInference(self.mdp, self.mlp, self.candidate_fovs,
                                             human_agent_index=self.human_index)

    def observe(self, state, human_subtask):
        """Feed the human's observed SUBTASK (from env info['human_subtask']) and
        the PRE-step state it acted on (env info['obs_state']) into the filter."""
        if human_subtask is not None:
            try:
                self.inf.update(state, human_subtask)
            except Exception:
                pass

    def bias(self, device=None):
        # ablation hook: force_fov overrides the inference (oracle = true fov;
        # wrong = a fixed fov) with full confidence, to prove the win is driven by
        # the INFERRED fov and not by a generic bias. None (default) = real filter.
        if getattr(self, "force_fov", None) is not None:
            return logit_bias(self.force_fov, 0.0, len(self.candidate_fovs),
                              self.strength, self.kw, self.ks, self.kt, device)
        return logit_bias(self.inf.map_fov(), self.inf.entropy(),
                          len(self.candidate_fovs), self.strength,
                          self.kw, self.ks, self.kt, device)

    def estimate(self):
        return self.inf.map_fov(), self.inf.entropy()
