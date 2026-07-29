"""
Exact Bayesian inference of a human player's field of view (FOV).

The trick: we already have the *exact* policy a human uses to decide their
next move - SteakLimitVisionHumanModel. It's the very same class that
generates human behavior in the simulator. So instead of approximating
"what would a human with FOV=f do here" by looking up similar past
observations (nearest-neighbor / KNN), we just build a real instance of that
policy for every candidate FOV and ask it directly.

Each candidate FOV gets its own "shadow" agent that watches the real game
and keeps its own FOV-limited knowledge base up to date, exactly like a real
human with that FOV would. At every step, each shadow tells us how likely
IT would have been to take the action the human actually just took. That
probability is the Bayes likelihood term - computed directly from the known
policy, not approximated from data.

    posterior(fov) is proportional to prior(fov) * P(human's actual action | state, FOV=fov)
"""
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import Action
from overcooked_ai_py.agents.agent import SteakLimitVisionHumanModel


class FOVBayesFilter:
    """Tracks a belief (probability distribution) over a fixed set of candidate FOVs."""

    def __init__(self, mlam, start_state, fov_candidates=(60, 120, 180),
                 human_agent_index=1, action_noise=0.05):
        """
        mlam: the medium-level planner shared with the real game (same one the real human uses)
        start_state: the game's starting OvercookedState
        fov_candidates: FOV hypotheses in degrees, e.g. (60, 120, 180)
        human_agent_index: which player index is the human we're watching
        action_noise: tiny uniform floor mixed into each shadow's predicted action
            probability. The policy is otherwise a deterministic greedy planner, so a
            single action it didn't predict would otherwise get exactly 0 probability
            and permanently zero out that hypothesis forever. This floor just means
            "assume the human has a small chance of doing anything" - it doesn't change
            which hypothesis explains the data best, it just keeps the filter alive.
        """
        self.fov_candidates = list(fov_candidates)
        self.action_noise = action_noise
        self.belief = np.ones(len(self.fov_candidates)) / len(self.fov_candidates)

        # one shadow "copy" of the human policy per FOV hypothesis
        self.shadows = []
        for fov in self.fov_candidates:
            shadow = SteakLimitVisionHumanModel(
                mlam, start_state, vision_limit=True, vision_bound=fov, debug=False,
            )
            shadow.set_agent_index(human_agent_index)
            shadow.init_knowledge_base(start_state)
            self.shadows.append(shadow)

    def step(self, state, human_action):
        """Feed in the real game state and the action the human actually took.

        Returns the updated belief (posterior) over fov_candidates.
        """
        action_idx = Action.ACTION_TO_INDEX[human_action]
        n_actions = len(Action.ALL_ACTIONS)

        # ask each shadow: "how likely were you to take this exact action here?"
        likelihood = np.array([
            shadow.action(state)[1]["action_probs"][action_idx]
            for shadow in self.shadows
        ])
        likelihood = (1 - self.action_noise) * likelihood + self.action_noise / n_actions

        posterior = self.belief * likelihood
        self.belief = posterior / posterior.sum()
        return self.belief

    def estimate(self):
        """Current best guess: the FOV hypothesis with the highest posterior probability."""
        return self.fov_candidates[int(np.argmax(self.belief))]
