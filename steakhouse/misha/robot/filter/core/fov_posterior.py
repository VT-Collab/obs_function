"""P(theta | the human's actions) -- inference over the human's vision cone.

This is the INFERENCE half of the FOV work and nothing else. The control half
lives in my_fov_filter.py, which reads the posterior this file maintains and
searches over how to execute its baseline's job against it.

The two used to share a module (qmdp_fov.py), whose other half was a filter that
re-ranked whole SUB-TASKS and then walked the one shortest path to the winner.
That filter is gone (as is its successor, qmdp.py, also deleted -- see
robot/methods.py's module docstring): with the target cell living inside a
sub-task it could only choose among cells its baseline had already shortlisted,
and it had no way to say
"same job, longer way round" or "same counter, two ticks later" -- see
robot/filter/RESULTS.md for what replaced it and what that bought.

THE ROBOT NEVER READS THE HUMAN'S BELIEFS. Everything it knows about what the
human can see comes from having watched them act, which is what makes the cone a
hidden variable worth inferring rather than a field to look up.
"""
import copy
import os
import sys

_NL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.environ.get("STEAK_ROOT", os.path.dirname(_NL)))
sys.path.insert(0, _NL)

from human.limited_vision_human import LimitedVisionHuman         # noqa: E402

FOVS = (30, 60, 90, 180, 360)
N_ACTIONS = 6


class FOVPosterior:
    """P(theta | a_H_{0:t}) by soft-deterministic likelihood.

    The human is a deterministic ladder, so a hypothesised cone yields ONE
    predicted action. We do not match it hard: a single modelling error would
    zero a hypothesis forever and the posterior could never recover. alpha is
    how much we trust our human model, not a property of the human.

    One shadow LimitedVisionHuman per cone does the predicting. They are the real
    agent, not an approximation of it, which is why this is exact rather than
    fitted -- and it is also why the whole thing is only as good as that human
    model: a change to the ladder changes what every hypothesis predicts.
    """

    def __init__(self, mdp, fovs=FOVS, alpha=0.9, human_index=1, seed=0,
                 human_kw=None):
        """`human_kw` IS HOW THE AGENT IN THE SEAT WAS BUILT, and it matters twice.

        The shadows are the model: they predict, they get reweighted by whether
        they were right, and the filter's whole forecast is one of them rolled
        forward. Build them differently from the human actually playing and both
        jobs are done against the wrong agent -- the posterior scores hypotheses
        on actions the real human would not take, and the forecast predicts a
        cone belonging to nobody.

        THERE USED TO BE A `human_cls` HERE TOO, and it was the sharp edge. The
        look-for human was a subclass, so every construction site had to be told
        which class the seat held, and the default was the base class: a harness
        that forgot got a plain ladder shadow against a look-for human and never
        heard about it. One merged LimitedVisionHuman makes that unrepresentable
        -- there is no other class to build -- so only the KWARGS can still
        disagree, and those are exactly what human_kw carries: forget_horizon,
        and enable_look_for, without which --no-look-for would be predicted by a
        look-for shadow. Same bug, flag instead of class.

        The docstring above is not weakened by this: the shadow is still the REAL
        agent rather than an approximation of it. This parameter is what keeps
        that sentence true when the seat is configured.
        """
        self.mdp, self.fovs, self.alpha = mdp, tuple(fovs), alpha
        self.human_index = human_index
        self.p = {f: 1.0 / len(self.fovs) for f in self.fovs}
        kw = dict(human_kw or {})
        self.human_kw = kw
        self.shadows = {f: LimitedVisionHuman(mdp, f, agent_index=human_index,
                                              seed=seed, **kw)
                        for f in self.fovs}
        self.predicted = {}

    def update(self, state, human_action):
        """Advance every shadow on the TRUE state, then reweight by agreement.

        Each shadow observes the real world through its own cone, so its beliefs
        stay honest; position and orientation are read from the true state each
        tick, so no shadow can drift away from where the human actually is.

        Call this ONCE per tick and after the robot has decided. `sh.action` is
        what advances a shadow's perception, so a second call in the same tick
        would observe it twice and inflate seen_count, which feeds the human's
        explore rule.
        """
        self.predicted = {}
        for f, sh in self.shadows.items():
            #one action
            act, _ = sh.action(state)
            self.predicted[f] = act

        #soft gate
        miss = (1.0 - self.alpha) / (N_ACTIONS - 1)
        total = 0.0
        new = {}
        for f in self.fovs:
            lik = self.alpha if self.predicted[f] == human_action else miss
            new[f] = self.p[f] * lik
            total += new[f]
        if total <= 0:
            new = {f: 1.0 / len(self.fovs) for f in self.fovs}
            total = 1.0
        self.p = {f: v / total for f, v in new.items()}
        return self.p

    def map_fov(self):
        return max(self.p, key=self.p.get)

    def beliefs_for(self, fov):
        """A copy of what a theta-human would currently believe.

        my_fov_filter.py clones the WHOLE shadow rather than calling this, because the
        ladder carries a held sub-task and a clock across ticks and a human
        rebuilt around a bare view would not reproduce its own next move. Kept
        because it is the honest read-only answer to "what does a theta-human
        believe right now", which is a question worth being able to ask.
        """
        return copy.deepcopy(self.shadows[fov].view)
