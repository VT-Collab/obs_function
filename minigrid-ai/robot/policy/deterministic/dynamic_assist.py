"""
dynamic_assist.py — Proposed policy: assistance driven by Bayesian FOV inference.

The robot maintains a posterior P(θ | a^H_{1:t}) over the human's FOV and uses
the MAP estimate θ̂_t to drive the shadow agent. As the posterior concentrates
on the true FOV (typically within ~20-30 steps), the shadow becomes an accurate
model of what the human actually sees — making the patience-threshold intervention
more precise than the static baseline.

Adaptive model design follows the motivation of Javdani et al. (2015) "Shared
Autonomy via Hindsight Optimization" (IJRR): the robot integrates uncertainty
over the human's intent/parameters and acts to minimise expected task cost.
Here the uncertain parameter is θ (FOV) rather than the goal, and the robot
acts by revealing information rather than taking physical actions.

Timing convention (important):
    for each step t:
        robot.step(state, human_kb)     ← may modify KB; uses θ̂_{t-1}
        subtask = human.select_subtask(state)
        action  = planner.next_action(...)
        robot.observe(state, action)    ← updates posterior → θ̂_t
        env.step(action)
"""

from __future__ import annotations
import math, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from robot.policy.deterministic._base import AssistBase
from robot.estimation.bayesian_posterior.bayes_fov import BayesFOVInference, CANDIDATE_FOVS


class DynamicAssist(AssistBase):
    """
    Assistance policy whose assumed FOV tracks the MAP of P(θ | a^H_{1:t}).

    Parameters
    ----------
    patience : int
        Intervention patience K.
    candidate_fovs : list[int], optional
        FOV hypothesis set; defaults to CANDIDATE_FOVS = [60, 120, 180].
    """

    def __init__(self, patience: int = 5, candidate_fovs=None):
        super().__init__(patience=patience)
        self.inf = BayesFOVInference(candidate_fovs=candidate_fovs or CANDIDATE_FOVS)

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def reset(self, state) -> None:
        super().reset(state)
        self.inf.reset(state)

    # ── Observation hook ──────────────────────────────────────────────────────

    def observe(self, state, action: int) -> None:
        """
        Update the FOV posterior with the human's chosen action.
        Call AFTER the human decides their action but BEFORE env.step().
        """
        self.inf.update(state, action)

    # ── Patience ─────────────────────────────────────────────────────────────

    def _effective_patience(self) -> int:
        """
        MISHA NEW CHANGE — scales the base class's follow-up patience by how
        confident the FOV posterior currently is, via entropy() — something
        StaticAssist structurally cannot do, since it has no notion of
        confidence, only a fixed guess. Low entropy (confident, typically
        after the MAP estimate converges ~20-30 steps in) shrinks the wait
        further, since the shadow is more likely well-calibrated and worth
        trusting quickly. High entropy (uncertain, typically early in the
        episode) falls back to the base class's value, same as static gets,
        since a shadow built on an unreliable guess shouldn't be rushed.
        """
        base = super()._effective_patience()
        if self.n_assists == 0:
            return base  # first intervention: full patience, same as everyone
        max_entropy = math.log(len(self.inf.candidate_fovs))
        confidence = 1.0 - (self.inf.entropy() / max_entropy) if max_entropy > 0 else 1.0
        confidence = min(max(confidence, 0.0), 1.0)
        return max(1, round(base * (1.0 - 0.5 * confidence)))

    # ── Step override ─────────────────────────────────────────────────────────

    def step(self, state, human_kb: dict):
        """
        Use the MAP hypothesis agent directly instead of a separate shadow.

        All 3 hypothesis agents are advanced every observe() call, so the MAP
        agent already has a complete, accurate KB — no rebuild needed when the
        MAP FOV changes.

        human_kb is only ever a WRITE target (the reveal destination) — what's
        missing and whether to intervene is judged entirely from skb, the MAP
        hypothesis agent's own belief KB.
        """
        #the highest confidence fov currently
        map_fov = self.inf.map_fov()
        shadow  = self.inf.hypothesis_agents[map_fov]
        shadow.select_subtask(state)
        skb = shadow.knowledge_base

        target = self._next_needed(state, skb)
        if target is None:
            self.timer = 0
            return None
        self.timer += 1
        if self.timer >= self._effective_patience():  # MISHA NEW CHANGE
            self._reveal(human_kb, target, state)
            self.n_assists += 1
            self.timer = 0
            return target
        return None

    # ── Knowledge injection ─────────────────────────────────────────────────────

    def _reveal(self, human_kb: dict, target, state) -> None:
        """
        Mirror the reveal into human_kb and all 3 hypothesis agents' KBs —
        every candidate-FOV shadow should treat a given hint as known, not
        just the one whose FOV happens to be MAP this step. Overrides the
        base implementation, which mirrors into self.shadow instead; Dynamic
        never advances self.shadow, so writing there would be a no-op.
        """
        self._write_reveal(human_kb, target, state)
        for agent in self.inf.hypothesis_agents.values():
            self._write_reveal(agent.knowledge_base, target, state)

    # ── FOV accessor (used by base reset only) ────────────────────────────────

    def _get_assumed_fov(self) -> int:
        return self.inf.map_fov()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def posterior(self) -> dict:
        return self.inf.posterior()

    def entropy(self) -> float:
        return self.inf.entropy()
