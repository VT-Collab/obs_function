"""The frozen self-play baseline, as a distribution over the 6 primitive actions.

This is the ONLY place the trained network is touched, and it is never
retrained, fine-tuned, or otherwise modified. Both conditions in the
experiment -- "baseline" and "baseline + QMDP module" -- run these exact
weights; the only difference downstream is whether the module's scores are
mixed into the logits. That is what makes the comparison attributable to the
module rather than to extra capacity.

`action_probs` is the hook SP/self_play.py documents for exactly this purpose:
it walks base -> rnn -> ACTLayer.distri and stops one step before the sample,
so the 6 numbers it returns are precisely the ones policy.act() would have
sampled from. We import it rather than re-deriving it, so a future change to
the network's forward pass cannot silently desynchronise the two.
"""
import types

import numpy as np
import torch

import _paths  # noqa: F401

from algorithm.rMAPPOPolicy import R_MAPPOPolicy
from SP.self_play import action_probs as _sp_action_probs


def load_policy(layout, obs_shape, ckpt_path, device=None, hidden_size=64,
                recurrent_N=1):
    """Rebuild the SP policy for `layout` and load its checkpoint.

    obs_shape MUST be (23, W, H) for THIS layout. The specialists trained
    unpadded, so CNNBase's Linear(32*W*H, 64) is sized by the grid and a
    checkpoint from another kitchen will not load. CARC_RUNS.md section 0.
    """
    device = device or torch.device("cpu")
    ck = torch.load(ckpt_path, map_location="cpu")
    #ck["args"] holds every hyperparameter the run used, so nothing is guessed
    saved = ck.get("args", {}) or {}
    args = types.SimpleNamespace(
        hidden_size=int(saved.get("hidden_size", hidden_size)),
        recurrent_N=int(saved.get("recurrent_N", recurrent_N)),
        lr=5e-4, critic_lr=5e-4, opti_eps=1e-5)
    policy = R_MAPPOPolicy(args, obs_shape, obs_shape, act_dim=6, device=device)
    policy.actor.load_state_dict(ck["actor"])
    if "critic" in ck:
        policy.critic.load_state_dict(ck["critic"])
    policy.actor.eval()
    policy.critic.eval()
    return policy, args


class BaselineActor:
    """Stateful wrapper: obs -> probs (6,), carrying the GRU memory forward.

    The recurrent state is the easy thing to get wrong. `action_probs` advances
    the GRU exactly like any other forward pass, so it must be threaded through
    the episode or the policy is memoryless and is not the policy that was
    trained. masks=0 on the first tick wipes the memory, matching how
    self_play.py starts an episode.
    """

    def __init__(self, policy, args, device=None):
        self.policy = policy
        self.device = device or torch.device("cpu")
        self.hidden = int(args.hidden_size)
        self.layers = int(args.recurrent_N)
        self.reset()

    def reset(self):
        self.rnn = torch.zeros(1, self.layers, self.hidden, device=self.device)
        self._first = True

    def probs(self, obs_np):
        """obs_np (1, 23, W, H) -> np.ndarray (6,), sums to 1."""
        obs = torch.from_numpy(obs_np).to(self.device)
        #mask 0 on the very first tick = fresh episode, wipe the GRU
        masks = torch.zeros(1, 1, device=self.device) if self._first \
            else torch.ones(1, 1, device=self.device)
        self._first = False
        p, self.rnn = _sp_action_probs(self.policy, obs, self.rnn, masks)
        out = p.detach().cpu().numpy().reshape(-1).astype(np.float64)
        s = out.sum()
        return out / s if s > 0 else np.full(6, 1.0 / 6.0)
