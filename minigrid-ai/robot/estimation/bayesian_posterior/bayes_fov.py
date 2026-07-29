"""
Online Bayesian inference over the agent's field of view (FOV) parameter θ,
given a trajectory of observed actions a^H_{1:t} and environment states s_{1:t}.

Formula:
    P(θ | a^H_{1:t}) ∝ P(θ) · Π_{i=1}^t π^H(a^H_i | s_{1:i}, θ, τ_i(θ))

τ_i(θ) is DETERMINISTIC given (θ, s_{1:i}): each hypothesis agent calls
_update_kb(state) → _decide(state) to get τ without touching the true agent.

π^H is computed by BayesianPlanner.action_probs: runs next_action (real BFS)
on the hypothesis KB, then wraps the predicted action in epsilon-greedy noise.

Numerics: log-space posterior, log-sum-exp normalisation.
"""

from __future__ import annotations
import math
import os, sys
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from human.agents.bayes_agent import BayesHumanAgent, fov_prior
from human.planning.bayes_planner import BayesianPlanner

CANDIDATE_FOVS = [60, 120, 180]


class BayesFOVInference:
    """
    Posterior distribution P(θ) over candidate FOV values.

    One BayesHumanAgent (hypothesis agent) runs per candidate θ. Each agent
    maintains its own FOV-specific KB so each hypothesis sees only what an
    agent with that FOV would see.
    """

    def __init__(
        self,
        candidate_fovs: List[int] = None,
        prior: Dict[int, float] = None,
    ):
        """
        Parameters
        ----------
        candidate_fovs : list of ints
            The finite set of FOV values θ can take.
        prior : dict {fov: probability}, optional
            P(θ) prior. Defaults to uniform.
        """
        
        self.candidate_fovs = candidate_fovs or CANDIDATE_FOVS
        
        #TODO 2 — Compute prior using fov_prior() if none is given.
        #          Convert each prior probability to log-space and store in
        #          self.log_posterior: Dict[int, float].
        #          Hint: math.log(prior[fov]) for each fov.
        self.prior = fov_prior(self.candidate_fovs) if prior is None else prior
        self.log_posterior = {fov: math.log(self.prior[fov]) for fov in self.candidate_fovs}
        
        # TODO 3 — Create one BayesHumanAgent per candidate FOV and store
        #          in self.hypothesis_agents: Dict[int, BayesHumanAgent].
        #          Each agent must be constructed with its own fov value.
        self.hypothesis_agents = {
            fov: BayesHumanAgent(fov=fov) for fov in self.candidate_fovs
        }
        
        
        # TODO 4 — Create a single shared BayesianPlanner and store it as
        #          self.planner (stateless — safe to share across hypotheses).
        self.planner = BayesianPlanner()

    # ── Episode management ────────────────────────────────────────────────────

    def reset(self, state) -> None:
        
        # Call at the START of each episode.

        # TODO 5 — Call agent.init_knowledge_base(state) on every hypothesis agent
        #          so their KBs are fresh for the new episode.
        for agent in self.hypothesis_agents.values():
            agent.init_knowledge_base(state)
        
        # TODO 6 — Reset self.log_posterior back to the log-prior (same as __init__).
        #          Use fov_prior() again so the reset is self-contained.
        prior = fov_prior(self.candidate_fovs)
        self.log_posterior = {
            fov: math.log(prior[fov]) for fov in self.candidate_fovs
        }
        

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, state, observed_action: int) -> None:
        """
        Observe one (state_t, a^H_t) pair and update the posterior.

        For each hypothesis θ:
          1. _update_kb(state)      — let the hypothesis "see" through FOV=θ
          2. τ = _decide(state)     — deterministic subtask for this θ
          3. p = π^H(a^H_t | s_t, θ, τ)   — likelihood of the observed action
          4. log P(θ) += log p      — Bayes update in log-space

        Then normalise so the distribution sums to 1 in probability space.
        """
        
        for fov, agent in self.hypothesis_agents.items():
            agent._update_kb(state)
            tau = agent._decide(state)
            action_dist = self.planner.action_probs(tau, state, agent.knowledge_base)
            p = action_dist.get(observed_action, 1e-9)
            self.log_posterior[fov] += math.log(max(p, 1e-9))
        
        # TODO 9 — Call self._log_normalise().
        self._log_normalise()
        

    # ── Posterior queries ─────────────────────────────────────────────────────

    def posterior(self) -> Dict[int, float]:
        """
        Return current P(θ) as a probability dict (NOT log-space).

        TODO 10 — Return {fov: math.exp(lp) for fov, lp in self.log_posterior.items()}.
        """
        #note that math.exp undoes the log and you put the for loop after it
        return {fov: math.exp(lp) for fov, lp in self.log_posterior.items()}
        

    def map_fov(self) -> int:
        """
        MAP estimate: argmax_θ P(θ | a^H_{1:t}).

        TODO 11 — Return the fov key with the highest value in self.log_posterior.
                  Hint: max(dict, key=dict.__getitem__)
        """
        return max(self.log_posterior, key=self.log_posterior.get)

    def entropy(self) -> float:
        """
        Shannon entropy H = -Σ P(θ) log P(θ)  (in nats).
        High entropy = uncertain. Low entropy = confident about the fov estimation

        TODO 12 — Iterate over self.log_posterior.values().
                  For each lp:
                    p = math.exp(lp)
                    H -= p * lp    (because log P = lp, so p * log(p) = p * lp)
                  Return H.
        """
        H = 0.0
        for lp in self.log_posterior.values():
            p = math.exp(lp)
            H -= p*lp
        return H
        
        

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_normalise(self) -> None:
        """
        After a Bayes update, the log-posteriors are un-normalised scores.
        This rescales them so they represent real probabilities that sum to 1,
        without letting numbers overflow or underflow.

        TODO 13 — The three lines below are the implementation. Read the
                  comments, make sure you understand each one, then keep them.
        """

        # ── Step 1: find the max or biggest log-probability in the dict ──────────
        # e.g. log_posterior = {60: -1.2,  120: -0.5,  180: -3.1}
        #      max_lp = -0.5   ← the least-negative value
        # We need this only as a numerical safety trick (see step 2).
        max_lp = max(self.log_posterior.values())

        # ── Step 2: subtact max value for safe calculation, aka compute log(Z), where Z = sum of all raw probabilities ─
        #
        # The naive way:  log_Z = math.log( sum(math.exp(lp) for lp ...) )
        # Problem: after 190 steps lp can be -600.  exp(-600) = 0 in Python
        # (underflow), so the whole sum becomes 0 and log(0) crashes.
        #
        # Fix — log-sum-exp trick:
        #   Subtract max_lp INSIDE the exp  →  (lp - max_lp) is always ≤ 0
        #   so exp(lp - max_lp) is always in [0, 1]  →  safe, no underflow.
        #   Add max_lp back OUTSIDE to cancel the subtraction mathematically.
        #
        # Concrete example:
        #   lp values:  -1.2,  -0.5,  -3.1       max_lp = -0.5
        #   shifted:    -0.7,   0.0,  -2.6        (each lp minus max_lp)
        #   exp(shift):  0.497, 1.000,  0.074      (all safely in [0,1])
        #   sum:         1.571
        #   log(1.571):  0.452
        #   log_Z = -0.5 + 0.452 = -0.048         (add max_lp back)
        log_Z = max_lp + math.log(
            sum(math.exp(lp - max_lp) for lp in self.log_posterior.values())
        )

        # ── Step 3: divide by total aka subtract log_Z from every entry ────────────────────────
        # Normal space:  P_normalised = P / Z          (divide by the total)
        # Log space:     log(P / Z)   = log(P) - log(Z)  →  lp - log_Z
        #
        # After this line, sum(exp(lp) for all lp) == 1.0 — valid probabilities.
        self.log_posterior = {fov: lp - log_Z for fov, lp in self.log_posterior.items()}
        


# ── Quick sanity-check ─────────────────────────────────────────────────────────

def run_inference_on_episode(seed: int = 0, true_fov: int = 120,
                              render: bool = False):
    """
    Run one full episode with a known true_fov, feed observed actions to the
    inference engine, and print the posterior at each step.

    TODO 14 (optional) — Fill in the loop below, or just read it to understand
    how the inference engine connects to the environment.

    The ordering matters:
      - inf.update(state, action)  BEFORE  env.step(action)
      so that each hypothesis agent processes the same state the true agent saw.
    """
    import gymnasium as gym
    from human.planning.bayes_planner import BayesianPlanner as Planner

    env = gym.make("MiniGrid-LockedRoom-v0",
                   render_mode="human" if render else None)
    env.reset(seed=seed)
    state = env.unwrapped

    true_agent   = BayesHumanAgent(fov=true_fov)
    true_planner = Planner()
    true_agent.init_knowledge_base(state)

    inf = BayesFOVInference(candidate_fovs=CANDIDATE_FOVS)
    inf.reset(state)

    done = False
    t = 0
    while not done:
        state = env.unwrapped

        subtask = true_agent.select_subtask(state)
        action  = true_planner.next_action(subtask, state, true_agent.knowledge_base)
        if action is None:
            action = 2

        # TODO 14a — call inf.update(state, action)
        inf.update(state, action)
        # TODO 14b — print the posterior, MAP estimate, and entropy
        post = inf.posterior()
        print(f"t={t:3d}  subtask={subtask:<20}  "
              f"P(60)={post[60]:.3f}  P(120)={post[120]:.3f}  "
              f"P(180)={post[180]:.3f}  MAP={inf.map_fov()}  "
              f"H={inf.entropy():.3f}  true={true_fov}")
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        t += 1

    env.close()
    print(f"\nFinal MAP: {inf.map_fov()}  (true: {true_fov})  "
          f"correct={inf.map_fov() == true_fov}")
    return inf.posterior()


if __name__ == "__main__":
    run_inference_on_episode(seed=0, true_fov=120)
