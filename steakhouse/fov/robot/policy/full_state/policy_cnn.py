"""
Small CNN ActorCritic for the grid full-state observation. AdaptiveAvgPool makes
it size-agnostic, so ONE architecture trains on any layout's H x W. Same act()
signature as the MLP ActorCritic (accepts an optional logit_bias) so the FOV
module can bolt on later unchanged.
"""
import torch
import torch.nn as nn

from fov.robot.policy.baseline.features import N_ACTIONS
from fov.robot.policy.full_state.features_grid import N_CHANNELS


class CNNActorCritic(nn.Module):
    def __init__(self, in_ch=N_CHANNELS, n_actions=N_ACTIONS, pool=4, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((pool, pool))     # -> (64, pool, pool)
        self.body = nn.Sequential(nn.Linear(64 * pool * pool, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self.pool(self.conv(x)).flatten(1)
        h = self.body(h)
        return self.pi(h), self.v(h).squeeze(-1)

    def act(self, x, logit_bias=None):
        logits, v = self.forward(x)
        if logit_bias is not None:
            logits = logits + logit_bias
        d = torch.distributions.Categorical(logits=logits)
        a = d.sample()
        return a, d.log_prob(a), d.entropy(), v
