"""
NOT USED FOR NOW




MISHA NEW CHANGE - online Bayesian inference of the human's FOV, against the
from-scratch limited-vision human (fov/human/agent/limited_vision_human.py).

    P(theta | a_{1:t}) ~ P(theta) * PROD_i pi(a_i | s_{1:i}, theta)

One SHADOW agent per candidate FOV. Each shadow is a full
LimitedVisionSteakHuman with that FOV, watching the same real state stream and
maintaining its OWN FOV-gated, decaying beliefs - so shadow theta knows exactly
what a human with FOV=theta would know, and predicts what they would do. The
likelihood of the action the human actually took is then read off directly.

Ported from the validated minigrid design
(minigrid-ai/robot/estimation/bayesian_posterior/bayes_fov.py): log-space
posterior with log-sum-exp normalisation. NOT linear-space with a belief floor -
a floor resurrects every losing hypothesis by floor/n on EVERY step, capping
confidence and letting one lucky step flip the argmax.

WHY THIS CAN WORK NOW. The previous filter was pointed at the stock
SteakLimitVisionHumanModel, whose subtask choice is nearly FOV-invariant (it
never forgets, starts omniscient, and sees adjacent tiles regardless of FOV). Its
hypotheses therefore chose the same subtasks and the posterior had nothing to
separate: 642 of 681 searched layouts showed exactly zero FOV effect. With the
rewritten human, 12/12 validated layouts show substantial subtask divergence, so
the likelihood finally carries signal.
"""
import math

from overcooked_ai_py.mdp.overcooked_mdp import Action
from fov.human.agent.limited_vision_human import LimitedVisionSteakHuman
from fov.human.planning.steak_planner import SteakMotionPlanner

# Floor on any single-step likelihood. Mirrors minigrid's 1e-9 guard: a
# hypothesis that finds the observed action impossible takes a large but FINITE
# hit, so later evidence can still revive it.
MIN_P = 1e-9


def fov_prior(candidate_fovs):
    return {f: 1.0 / len(candidate_fovs) for f in candidate_fovs}


class BayesFOVInference:
    """Posterior over candidate FOVs, updated online from observed actions."""

    def __init__(self, mdp, mlp, candidate_fovs, human_agent_index=1,
                 prior=None, epsilon=0.10):
        """
        epsilon: exploration mass. The human is deterministic given its beliefs,
            so without a noise term a single mismatched step would zero a
            hypothesis outright. epsilon spreads that mass over the other
            actions, making the update informative but not brittle.
        """
        self.mdp = mdp
        self.candidate_fovs = list(candidate_fovs)
        self.human_agent_index = human_agent_index
        self.epsilon = epsilon
        self.n_actions = len(Action.ALL_ACTIONS)

        self.prior = fov_prior(self.candidate_fovs) if prior is None else prior
        self.log_posterior = {f: math.log(self.prior[f]) for f in self.candidate_fovs}

        planner = SteakMotionPlanner(mdp, mlp)
        self.shadows = {
            f: LimitedVisionSteakHuman(mdp, f, planner, agent_index=human_agent_index)
            for f in self.candidate_fovs
        }
        self.n_steps = 0
        self.n_divergent_steps = 0     # steps where shadows predicted differently
        self.n_crash_steps = 0
        # per-hypothesis running agreement with the observed action, recorded
        # inside update() so diagnostics never need to re-run a shadow
        self.n_agree = {f: 0 for f in self.candidate_fovs}

    def update(self, state, observed_action):
        """Observe one (state, action) pair and update the posterior.

        Call BEFORE stepping the environment, so every shadow processes the same
        state the human actually acted on.
        """
        preds, probs = {}, {}
        for f, shadow in self.shadows.items():
            try:
                a_pred, _ = shadow.action(state)
            except Exception:
                a_pred = None
            preds[f] = a_pred
            if a_pred is None:
                probs[f] = None
            elif a_pred == observed_action:
                probs[f] = 1.0 - self.epsilon
            else:
                probs[f] = self.epsilon / max(1, self.n_actions - 1)

        crashed = [f for f, p in probs.items() if p is None]
        # If any shadow failed to produce a prediction, skip the STEP entirely
        # rather than only skipping that hypothesis. Because _log_normalise
        # applies a common shift, skipping one hypothesis is arithmetically
        # identical to awarding it likelihood 1.0 - a systematic reward for
        # crashing rather than a neutral abstention.
        if crashed:
            self.n_crash_steps += 1
        else:
            for f, p in probs.items():
                self.log_posterior[f] += math.log(max(p, MIN_P))
            self._log_normalise()

        for f, a in preds.items():
            if a == observed_action:
                self.n_agree[f] += 1
        self.n_steps += 1
        live = [a for a in preds.values() if a is not None]
        if len(set(live)) > 1:
            self.n_divergent_steps += 1
        return self.posterior()

    def posterior(self):
        return {f: math.exp(lp) for f, lp in self.log_posterior.items()}

    def map_fov(self):
        return max(self.log_posterior, key=self.log_posterior.get)

    def entropy(self):
        return -sum(math.exp(lp) * lp for lp in self.log_posterior.values())

    def _log_normalise(self):
        m = max(self.log_posterior.values())
        log_z = m + math.log(sum(math.exp(lp - m) for lp in self.log_posterior.values()))
        self.log_posterior = {f: lp - log_z for f, lp in self.log_posterior.items()}
