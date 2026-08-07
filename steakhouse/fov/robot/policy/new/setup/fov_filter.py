"""THE FILTER.  Joint posterior over (theta, tau) from the human's actions alone.

Subclasses the validated kernel in filter/inference.py rather than copying it.
That file's Bayes machinery -- the transition half, the P(action | tau, theta)
likelihood, the pruning -- is the one that measured 69-100% MAP accuracy with
n_skipped = 0, and a copy would silently drift from it the first time either
changed. Only ONE thing is overridden: how the shadows are built.

===========================================================================
THE SHADOWS MUST BE THE PARTNER YOU ACTUALLY HAVE
===========================================================================
A shadow is a hypothesis-human: "if their cone were 60 degrees, what would they
believe, and what would they do?" The filter scores each hypothesis by how well
its predicted action matches the one the human really emitted.

That only works if the shadows make the same assumptions as the real human. Two
of ours differ from the stock constructor and BOTH have to be mirrored:

    occlude=True    the real human's sight is blocked by walls. A shadow without
                    it believes its hypothesis-human can see straight through
                    the island, so it predicts they know things they cannot, and
                    it will reject the true cone for behaving "wrongly".
    familiar        the real human knows WHERE the stations are. A shadow
                    without it models a partner who is still lost -- it predicts
                    exploration while the real human walks confidently to the
                    grill, and every hypothesis takes the same likelihood hit.

Get either wrong and the filter is not merely less accurate, it is biased in a
direction nobody would notice: all hypotheses mispredict together, the posterior
stays near the prior, and the module's belief-weighted average quietly becomes
an unweighted one.

NO ORACLE. The true cone and the human's internal subtask label are never read
anywhere in here. The only inputs are the world state -- which the robot fully
observes by construction -- and the human's emitted primitive action, which is a
physical event anyone in the kitchen can watch.
"""

import importlib.util
import os

import _paths  # noqa: F401   MUST be first


def _load_kernel():
    """Load filter/inference.py BY FILE PATH, not by import name.

    filter/ must never go on sys.path -- it holds qmdp.py, cost_function.py and
    play_episode.py, and this package has files with all three names, so any path
    entry for it silently shadows ours. Loading one file directly gets us the
    validated kernel with no namespace risk at all.
    """
    path = os.path.join(_paths.FILTER_DIR, "inference.py")
    spec = importlib.util.spec_from_file_location("_filter_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SamplingBayesFOVInference


_Kernel = _load_kernel()

from env import seed_familiar                                           # noqa: E402
from fov.human.agent.limited_vision_human import (                      # noqa: E402
    LimitedVisionSteakHuman)
from fov.human.planning.steak_planner import SteakMotionPlanner         # noqa: E402

#the hypothesis set. Six cones, the same ones the human library was validated on.
#NOTE an identifiability limit worth stating in any writeup: on the old layouts
#fov 90 and fov 120 produced byte-identical behaviour in the ZSC eval, so the
#posterior cannot separate them from actions alone. Keeping both is honest --
#the mass simply splits -- but do not read a 50/50 between them as uncertainty
#about anything that matters.
CANDIDATE_FOVS = [30, 60, 90, 120, 180, 360]


class FOVFilter(_Kernel):
    """The kernel, with shadows that match our human."""

    def __init__(self, mdp, candidate_fovs=CANDIDATE_FOVS, human_agent_index=1,
                 prior=None, occlude=True, familiar=True, pin=None):
        #mlp=None is CORRECT, not a placeholder: the shadows route with their own
        #BFS over seen floor. Handing them a MediumLevelPlanner would give them a
        #map of the whole kitchen and destroy the experiment.
        super().__init__(mdp, None, candidate_fovs, human_agent_index, prior)

        planner = SteakMotionPlanner(mdp, None)
        self.shadows = {}
        for fov in self.candidate_fovs:
            sh = LimitedVisionSteakHuman(mdp, fov, planner,
                                         agent_index=human_agent_index,
                                         occlude=occlude)
            if familiar:
                seed_familiar(sh, mdp)
            self.shadows[fov] = sh

        #`pin` drives the ablation arms. None = the real inferred posterior.
        #An int = collapse all mass onto that cone every tick, which is how the
        #fov360 control and the oracle arm are built without a second code path.
        self.pin = pin
        if pin is not None:
            self._collapse()

    def _collapse(self):
        """Force every unit of mass onto the pinned cone, preserving the tau
        split within it. Called after each update so the pin survives evidence."""
        if self.pin is None:
            return
        keep = {k: v for k, v in self.b.items() if k[0] == self.pin}
        z = sum(keep.values())
        if z <= 0:
            #the pinned cone has been ruled out by evidence -- which is exactly
            #what an arm that ASSUMES a wrong cone should experience. Re-seed it
            #flat rather than dividing by zero, so the arm keeps asserting its
            #assumption instead of silently reverting to the real posterior.
            taus = {k[1] for k in self.b} or {(None, None)}
            keep = {(self.pin, t): 1.0 / len(taus) for t in taus}
            z = 1.0
        self.b = {k: v / z for k, v in keep.items()}

    def update(self, state, observed_action):
        out = super().update(state, observed_action)
        self._collapse()
        return out
