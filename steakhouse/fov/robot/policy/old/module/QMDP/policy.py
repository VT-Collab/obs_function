"""The robot: baseline distribution, blended with the module's, argmax.

===========================================================================
THE BLEND
===========================================================================
Two distributions over the same 6 primitives arrive each tick:

    p_base   the frozen self-play policy. Knows the TASK -- how to cook a
             steak in this kitchen -- and nothing about the partner, because
             its partner during training was a copy of itself.
    p_mod    the QMDP module. Knows the PARTNER -- what they can see, what
             they are doing, what your action would do to that -- and nothing
             about the task, by construction (cost.py reads no reward).

They are pooled logarithmically, in the UNNORMALISED (bias) form:

    log p  =  log p_base  +  beta . log p_mod          (+ const)      "bias"

`beta` is the single tuning factor: 0 is the baseline exactly, large beta is
the module alone. Log-linear rather than a linear mixture because it is
conservative in the right direction -- an action the baseline is confident is
wrong keeps a small probability no matter what the module says, so the module
re-ranks among sensible actions rather than inventing a bad one.

WHY THE BIAS FORM AND NOT THE CONVEX ONE. The convex pool
(1-lam) log p_base + lam log p_mod raises p_base to the power (1-lam), which is
a TEMPERATURE change. When the robot samples its action -- and the sampled
robot is the strong baseline here, see RESULTS.md -- that alone alters
behaviour even if the module says nothing, so any measured win is confounded
with "the baseline got more exploratory". The bias form has the property the
experiment needs:

    p_mod uniform  ==>  p == p_base  exactly, at every beta,
                        under argmax AND under sampling.

so the "module with all cost weights zero" ablation is a provable no-op and any
difference is attributable to the cost function. `--blend loglinear` (convex)
and `--blend linear` are kept for the ablation table.

The action is the argmax by default, as specified. `--sample` draws from the
pooled distribution instead; both arms always use the same rule.

===========================================================================
ABLATION MODES -- how we know the win is the FOV inference
===========================================================================
`module_mode` replaces the module's output while leaving everything else
(including the wall-clock structure and the filter) identical:

  real        the module
  uniform     p_mod = 1/6. Provably identical to the baseline -> pins that the
              blend machinery contributes nothing on its own.
  noise       p_mod = a fresh random distribution each tick, matched in
              entropy. Tests "is the win just perturbing a deterministic
              policy?" -- the single most likely artefact here.
  shuffle     the REAL scores, randomly permuted over the 6 actions. Same
              magnitudes, wrong assignment: isolates the mapping from the
              scale.
  fixed:<f>   the module runs on a posterior pinned to cone f instead of the
              inferred one. Tests that the win needs the INFERENCE, not just
              some human model. (Same idea as fov_module.py's `force_fov`.)

===========================================================================
WHAT IS AND IS NOT AVAILABLE TO THIS CLASS
===========================================================================
IN    the world state, the human's emitted action from the PREVIOUS tick, the
      trained weights, and the human MODEL (the shadow agent's own code).
OUT   the human's true FOV, the human's subtask label, any belief of the real
      human, sparse or shaped reward, delivery counts, remaining step budget as
      an objective. None of these is passed to __init__ or to act().

test_no_cheating.py enforces this by construction: it runs an episode with the
true FOV replaced by a decoy after the human is built, and asserts the robot's
action sequence is bit-identical.
"""
import numpy as np

import _paths  # noqa: F401

from overcooked_ai_py.mdp.overcooked_mdp import Action

from human_model import PredictiveHumanModel
from qmdp import QMDPFOVModule, N_ACTIONS

_EPS = 1e-12


def softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def bias_pool(p_base, p_mod, beta):
    """log p_base + beta * log p_mod, renormalised. Uniform p_mod -> p_base."""
    lb = np.log(np.clip(p_base, _EPS, None))
    lm = np.log(np.clip(p_mod, _EPS, None))
    z = lb + beta * lm
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def log_linear_pool(p_base, p_mod, lam):
    """(1-lam) log p_base + lam log p_mod, renormalised (convex pool)."""
    lb = np.log(np.clip(p_base, _EPS, None))
    lm = np.log(np.clip(p_mod, _EPS, None))
    z = (1.0 - lam) * lb + lam * lm
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def linear_pool(p_base, p_mod, lam):
    p = (1.0 - lam) * np.asarray(p_base) + lam * np.asarray(p_mod)
    s = p.sum()
    return p / s if s > 0 else np.full(N_ACTIONS, 1.0 / N_ACTIONS)


_POOLS = {"bias": bias_pool, "loglinear": log_linear_pool, "linear": linear_pool}


class BlendedRobotPolicy:
    """baseline + QMDP module. beta=0 reproduces the baseline bit for bit."""

    def __init__(self, actor, mdp, candidate_fovs, lam=0.5, weights=None,
                 module_temperature=1.0, blend="bias",
                 human_action_topk=2, human_index=1, robot_index=0,
                 rng=None, sample=False, module_mode="real",
                 base_temperature=1.0, normalize_q=False):
        self.actor = actor
        self.mdp = mdp
        self.lam = float(lam)
        self.blend = blend
        self.pool = _POOLS[blend]
        self.sample = bool(sample)
        self.rng = rng or np.random.RandomState(0)
        self.hi = human_index
        self.ri = robot_index
        self.module_mode = module_mode
        #FAIRNESS KNOB, applied to the BASELINE arm too. A sampled recurrent
        #policy's behaviour depends on how sharp its distribution is, and the
        #trained softmax is not automatically the best operating point. Sweep
        #this at beta=0 to find the baseline's own optimum, so the module is
        #compared against the strongest version of what it is modifying, not
        #against an arbitrary one. 1.0 = the network's own distribution.
        self.base_temperature = float(base_temperature)
        #even at beta=0 the filter is constructed and run, so both conditions
        #see the identical human and identical wall-clock structure; only the
        #pooling weight differs. That keeps the comparison a real control.
        self.hm = PredictiveHumanModel(mdp, candidate_fovs, human_index,
                                       robot_index)
        self.module = QMDPFOVModule(mdp, self.hm, weights, module_temperature,
                                    human_action_topk, human_index=human_index,
                                    robot_index=robot_index,
                                    normalize=normalize_q)
        self.trace = []

    # -- ablations -----------------------------------------------------------

    def _apply_mode(self, p_mod, q):
        """Replace the module output per `module_mode`. See the header."""
        mode = self.module_mode
        if mode == "real":
            return p_mod, q
        if mode == "uniform":
            return np.full(N_ACTIONS, 1.0 / N_ACTIONS), np.zeros(N_ACTIONS)
        if mode == "noise":
            #matched in scale to the real scores so the perturbation is the
            #same SIZE, only meaningless
            scale = float(np.std(q)) if q is not None else 1.0
            fake = self.rng.normal(0.0, max(scale, 1e-6), N_ACTIONS)
            return softmax_np(fake), fake
        if mode == "shuffle":
            perm = self.rng.permutation(N_ACTIONS)
            return np.asarray(p_mod)[perm], np.asarray(q)[perm]
        if mode.startswith("fixed:"):
            return p_mod, q            # handled upstream, in act()
        raise ValueError("unknown module_mode %r" % mode)

    # -- the tick ------------------------------------------------------------

    def act(self, env):
        """Choose the robot's primitive action for env.state.

        Called BEFORE the human has moved this tick. Everything it reads is
        either the current state or evidence from strictly earlier ticks.
        """
        state = env.state

        #1. shadows perceive s_t. Looking is not evidence about the cone, so
        #   this consumes nothing from the posterior.
        self.hm.sync_observe(state)

        #2. baseline distribution over the 6 primitives (advances the GRU)
        p_base = self.actor.probs(env.robot_obs())
        if abs(self.base_temperature - 1.0) > 1e-9:
            lp = np.log(np.clip(p_base, _EPS, None)) / self.base_temperature
            lp -= lp.max()
            e = np.exp(lp)
            p_base = e / e.sum()

        if self.lam <= 0.0:
            p = p_base
            p_mod = None
            q = None
        else:
            #3. predictive posterior: what is the partner about to do, under
            #   every cone still alive
            hyps, h_probs = self.hm.predict(state)
            if self.module_mode.startswith("fixed:"):
                #ablation: pin the cone to a fixed hypothesis instead of the
                #inferred posterior. Everything else -- the shadow, its
                #beliefs, the subtask kernel -- is unchanged, so this isolates
                #the contribution of INFERRING the cone from the contribution
                #of having a human model at all.
                want = float(self.module_mode.split(":", 1)[1])
                keep = [h for h in hyps if float(h["fov"]) == want]
                if keep:
                    z = sum(h["prob"] for h in keep)
                    hyps = [dict(h, prob=h["prob"] / z) for h in keep]
                    h_probs = {}
                    for h in hyps:
                        h_probs[h["action"]] = h_probs.get(h["action"], 0.0) + h["prob"]
            #4. QMDP: roll each candidate forward, score the divergence it
            #   would cause in the partner's model
            p_mod, q = self.module.distribution(state, hyps, h_probs)
            p_mod, q = self._apply_mode(p_mod, q)
            #5. pool
            p = self.pool(p_base, p_mod, self.lam)

        idx = (int(self.rng.choice(N_ACTIONS, p=p)) if self.sample
               else int(np.argmax(p)))
        self.trace.append({"p_base": p_base, "p_mod": p_mod, "q": q,
                           "action": idx, "defer": self.module.last.get("defer")})
        return Action.INDEX_TO_ACTION[idx], idx

    def observe_human(self, state, human_action):
        """Feed the tick's evidence. Called AFTER the human has moved."""
        self.hm.update(state, human_action)

    def reset(self):
        """New episode: wipe the GRU AND rebuild the filter.

        The shadows carry per-episode knowledge (discovered cells, beliefs,
        decay clock). Reusing them across episodes would hand the human's model
        a map it earned in a different episode -- the exact "starts omniscient"
        failure limited_vision_human.py was written to avoid.
        """
        self.actor.reset()
        self.trace = []
        self.hm = PredictiveHumanModel(self.mdp, self.hm.candidate_fovs,
                                       self.hi, self.ri)
        self.module.hm = self.hm
