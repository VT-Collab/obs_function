"""
ValueNorm -- the one algorithmic gap between this stack and ZSC-Eval.

Ported to match zsceval/utils/valuenorm.py line for line (same beta, same
epsilon, same debiasing, same 1e-2 variance floor) so that turning it on makes
this an apples-to-apples ZSC-Eval run rather than "our own thing".

===========================================================================
WHAT PROBLEM IT SOLVES
===========================================================================
The critic is a plain regression onto returns, and in this task the MAGNITUDE
of a return grows by more than an order of magnitude over a single run:

    steak_mid_1 specialist   vloss  0.68  ->  9.35
    winning specialists      vloss  13 - 26
    generalist (failed)      vloss  0.25 - 0.63, flat

Early on the policy delivers nothing and collects a trickle of shaped reward,
so returns live near 0. Late on it delivers three steaks plus the whole shaped
chain, so returns live near 150. The critic's loss is quadratic in that error,
so the gradient it produces at episode 490 is ~400x the one at episode 10 for
the same RELATIVE mistake. One fixed learning rate cannot be right for both.

ValueNorm keeps a running mean/variance of the return targets and trains the
critic in NORMALIZED space, so the loss has a stable scale for the whole run.
The critic's output is then de-normalized wherever a real value is needed
(GAE, the bootstrap), so nothing downstream has to know this exists.

===========================================================================
HOW IT IS WIRED IN HERE  (deliberately, without touching buffer.py)
===========================================================================
The buffer's GAE assumes buffer.values are REAL values. Rather than teach it
about normalization, the runner de-normalizes at the point of STORAGE:

    collect / bootstrap   store  denormalize(critic_out)  ->  GAE unchanged
    value loss            compare raw critic_out against normalize(returns)

Mathematically identical to ZSC-Eval's arrangement, and buffer.py stays byte
for byte the file that produced the validated seed-1 baselines.

===========================================================================
THE DEBIASING TERM
===========================================================================
running_mean is an exponential moving average initialized at 0, so for the
first few hundred updates it is biased toward 0 simply because it started
there. debiasing_term is the same EMA applied to the constant 1, which is
exactly the accumulated weight -- dividing by it removes that startup bias.
Same trick Adam uses on its moment estimates.

var is floored at 1e-2 so that a rollout where every return is identical
(very common in the first episodes: nothing is delivered, returns are all
near-zero) cannot divide by ~0 and blow the loss up.
"""

import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """Running mean/std normalizer for value targets. Not trained by gradient."""

    def __init__(self, input_shape=1, norm_axes=1, beta=0.99999, epsilon=1e-5,
                 device=torch.device("cpu")):
        super().__init__()
        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.tpdv = dict(dtype=torch.float32, device=device)

        #buffers, not parameters: they are statistics, no optimizer touches them.
        #registered so they ride along in state_dict() and land in the checkpoint
        #-- a critic restored without its normalizer predicts in the wrong units.
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))
        self.to(**self.tpdv)

    def running_mean_var(self):
        d = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / d
        mean_sq = self.running_mean_sq / d
        #E[x^2] - E[x]^2. floored so an all-identical batch cannot divide by ~0
        var = (mean_sq - mean ** 2).clamp(min=1e-2)
        return mean, var

    @torch.no_grad()
    def update(self, input_vector):
        """Fold one batch of RETURNS into the running statistics."""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector ** 2).mean(dim=tuple(range(self.norm_axes)))

        w = self.beta
        self.running_mean.mul_(w).add_(batch_mean * (1.0 - w))
        self.running_mean_sq.mul_(w).add_(batch_sq_mean * (1.0 - w))
        self.debiasing_term.mul_(w).add_(1.0 - w)

    def normalize(self, input_vector):
        """real value -> normalized. Used on the RETURN before the value loss."""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.running_mean_var()
        return (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]

    def denormalize(self, input_vector):
        """normalized -> real value. Used on the CRITIC OUTPUT before GAE."""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.running_mean_var()
        return input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]


class IdentityValueNorm:
    """The --use_valuenorm off path. Same interface, does nothing.

    Exists so the runner has no `if self.value_norm is not None` branches --
    the frozen (ZSC-gap) configuration and the normalized one run through
    literally the same lines, which is what makes them comparable.
    """

    def update(self, x):
        pass

    def normalize(self, x):
        return x

    def denormalize(self, x):
        return x

    def state_dict(self):
        return {}
