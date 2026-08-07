"""
MISHA NEW CHANGE - the FOV-BLIND baseline policy network.

Deliberately small: the observation is 17 dims and the action space is 7, so
capacity is not the bottleneck - the information available is, which is the
whole experiment. The module variant reuses this exact network and only biases
its logits, so any performance difference is attributable to FOV information
rather than to a bigger model.
"""
import torch
import torch.nn as nn
from steakhouse.fov.robot.policy.old.baseline.features import OBS_DIM, N_ACTIONS


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, x, logit_bias=None):
        """logit_bias is where module/fov_module.py injects FOV information.
        None for the baseline - that absence IS the control condition."""
        logits, v = self.forward(x)
        if logit_bias is not None:
            logits = logits + logit_bias
        d = torch.distributions.Categorical(logits=logits)
        a = d.sample()
        return a, d.log_prob(a), d.entropy(), v
