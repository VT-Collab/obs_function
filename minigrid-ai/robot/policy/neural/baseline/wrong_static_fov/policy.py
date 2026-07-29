# ═══════════════════════════════════════════════════════════════════════════
# robot/policy/neural/ - the robot's reveal decision, LEARNED instead of
# hand-coded, plus a Bayesian-FOV module bolted on top of it.
#
#   baseline/    FOV-BLIND policy (vocabulary, network, training env, PPO)
#   module/      the ONLY place FOV information is ever used
#   end_to_end/  adapter into eval_three_way.py + comparison scripts
#
# THIS FILE - the network itself. Deliberately tiny:
#
#   obs (13) -> Linear 64 -> Tanh -> Linear 64 -> Tanh -> { actor: 6 logits
#                                                         { critic: 1 value
#
# Small because the decision it has to make is small - given "which reveals are
# valid, how long each has been pending, is the human holding a key", pick one
# of 6.
# The hard part of this project is the experimental design around it, not
# capacity. Non-recurrent v0: the pending-timers in the observation carry the
# only history the policy needs, so there is no hidden state to maintain.
#
# The one piece of structure baked in is action masking: invalid reveals are
# driven to -1e9 before sampling, so the policy can never propose revealing
# something that does not exist. That is bookkeeping, not a learned preference.
#
# Trained by train.py against env_wrapper.py; the SAME class is instantiated at
# eval time by end_to_end/neural_assist.py, where module/fov_module.py adjusts
# the logits this forward() returns. The network is never retrained per filter -
# one checkpoint, many filters - which is what makes the filters ablatable.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch_ac

from robot.policy.neural.baseline.static_fov.features import N_ACTIONS, OBS_DIM


class CommPolicy(nn.Module, torch_ac.ACModel):
    """Non-recurrent v0. forward(obs) -> (dist, value) - torch_ac calls it with
    exactly one argument when acmodel.recurrent is False (verified against
    torch_ac/algos/base.py's collect_experiences)."""
    recurrent = False

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 64):
        super().__init__()
        self.n_actions = n_actions
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        # obs[:, :n_actions] is the availability mask from features.py. Column 0
        # (WAIT) is always 1, so this only ever masks out reveal actions that
        # aren't currently well-defined - action masking, not a policy decision.
        mask = obs[:, :self.n_actions] > 0.5
        x = self.body(obs)
        logits = self.actor(x).masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        value = self.critic(x).squeeze(-1)
        return dist, value
