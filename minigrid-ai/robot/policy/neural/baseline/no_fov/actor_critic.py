"""
Baseline robot policy that directly maps state to action without
any FOV consideration - completely fov invariant

Need to
1. feature the state -> image or latent variable      <- ConvEncoder
2. Q value for each robot action                      <- see q_net.py (SAC/DQN)
"""

# ═══════════════════════════════════════════════════════════════════════════
# baseline/no_fov/ - the RAW-STATE baseline. See features.py for the design.
#
# THIS FILE - the networks. One shared conv encoder, two policy-gradient heads:
#
#   NoFovAC     encoder -> MLP -> {actor 6 logits, critic 1}      feed-forward
#   NoFovRecAC  encoder -> GRU -> {actor 6 logits, critic 1}      recurrent
#
# WHY A CONV ENCODER. The observation is a (16,19,19) spatial stack, not a flat
# vector. Convolution is the right inductive bias: "is there a key two cells
# from the human" is the same computation wherever on the grid it happens, and a
# conv gets that translation-equivariance for free, where an MLP over 5776
# flattened inputs would have to learn it again at every position. Follows the
# Nature-DQN encoder (Mnih et al. 2015), resized for a 19x19 input.
#
# The channel count is read from features.N_CHANNELS, never hardcoded - it was
# 22 before the six object kinds LockedRoom never contains were dropped. Any
# checkpoint trained at 22 input channels will not load here.
#
# WHY A RECURRENT VARIANT. This is a POMDP. A single frame cannot say what the
# human has already seen with their own eyes, only what is physically there.
# static_fov papers over that with hand-crafted "how long has this been pending"
# timers; here the network must build its own memory. Ni, Eysenbach &
# Salakhutdinov (ICML 2022), "Recurrent Model-Free RL is a Strong Baseline for
# Many POMDPs", is the reference result - a plain GRU on a standard algorithm is
# competitive with far more elaborate POMDP machinery. torch_ac supports this
# natively: set recurrent=True, take (obs, memory), return (dist, value, memory),
# declare memory_size, and PPO does truncated BPTT over `recurrence` steps.
#
# NO ACTION MASKING. static_fov drives illegal reveals to -1e9 before sampling;
# that mask is hand-written domain logic and does a real share of the work. Here
# the actor emits 6 logits over 6 always-available actions. An unresolvable or
# repeated utterance costs comm_cost and changes nothing, so the policy has to
# learn what is worth saying from reward alone. Removing the mask also removes
# the entropy pathology it caused: with a mask, 94% of states permitted exactly
# one action, so achievable entropy was log(1)=0 and any entropy-regularised
# method chased an impossible target.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch_ac

from robot.policy.neural.baseline.no_fov.features import N_ACTIONS, N_CHANNELS, GRID


class ConvEncoder(nn.Module):
    """(B, C, 19, 19) -> (B, feat). Nature-DQN style, sized for a 19x19 grid."""

    def __init__(self, in_ch: int = N_CHANNELS, feat: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),   # 19 -> 10
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),      # 10 -> 5
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),      #  5 -> 3
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, in_ch, GRID, GRID)).shape[1]
        self.out = nn.Sequential(nn.Linear(n, feat), nn.ReLU())
        self.feat = feat

    def forward(self, x):
        return self.out(self.net(x))


# NO action masking anywhere in this folder. All 6 actions are always available;
# an unresolvable or repeated utterance costs comm_cost and does nothing, and the
# policy learns that from reward. See features.py's header for why.
#
# The told-flags remain in the OBSERVATION - what the robot has already said is
# its own memory, i.e. state - but they no longer gate anything. That is the
# line this baseline draws: state in, no rules.


def _bias_toward_wait(actor: nn.Linear, wait_bias: float):
    """Start the policy biased toward WAIT (action 0), not uniform.

    Without this the policy begins uniform over 6 actions and so SPEAKS ~5/6 of
    every step - measured: 124-157 words/episode - and the weak comm_cost signal
    could not pull it back within 300k frames (entropy stayed ~1.7/1.79). A big
    positive bias on the WAIT logit makes silence the default the policy must be
    given a reason to break, which matches the task: speaking is the rare event.
    Only the initial bias - the network is free to learn any logits from here.
    """
    with torch.no_grad():
        actor.bias.zero_()
        actor.bias[0] = wait_bias


class NoFovAC(nn.Module, torch_ac.ACModel):
    """Feed-forward actor-critic - the straight PPO baseline."""
    recurrent = False

    def __init__(self, feat: int = 256, hidden: int = 128, wait_bias: float = 2.5):
        super().__init__()
        self.encoder = ConvEncoder(feat=feat)
        self.body = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        _bias_toward_wait(self.actor, wait_bias)

    def forward(self, obs):
        x = self.body(self.encoder(obs))
        return (Categorical(logits=self.actor(x)),
                self.critic(x).squeeze(-1))


class NoFovRecAC(nn.Module, torch_ac.ACModel):
    """Recurrent actor-critic (GRU). Ni et al. 2022.

    torch_ac zeroes memory on episode boundaries by multiplying it with the
    done-mask before each call, so nothing here tracks episode ends.
    """
    recurrent = True

    def __init__(self, feat: int = 256, hidden: int = 128, wait_bias: float = 2.5):
        super().__init__()
        self.encoder = ConvEncoder(feat=feat)
        self.rnn = nn.GRUCell(feat, hidden)
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        self.hidden = hidden
        _bias_toward_wait(self.actor, wait_bias)

    @property
    def memory_size(self):
        return self.hidden      # GRU carries one vector; an LSTM would need 2x

    def forward(self, obs, memory):
        h = self.rnn(self.encoder(obs), memory)
        return (Categorical(logits=self.actor(h)),
                self.critic(h).squeeze(-1),
                h)
