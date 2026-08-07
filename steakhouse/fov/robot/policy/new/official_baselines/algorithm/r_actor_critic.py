"""

Actor critic file used by ppo, and it itself calls cnn, rnn, and act

===========================================================================
THE TWO HALVES
===========================================================================
R_Actor    obs -> which action to take
           CNNBase -> RNNLayer -> ACTLayer(Categorical)

R_Critic   obs -> ONE number: "how much total reward do I expect from here"
           CNNBase -> RNNLayer -> Linear(hidden, 1)

Same trunk shape, different head. The actor head picks among 6 choices,
the critic head predicts a single scalar.

They do NOT share weights. Each builds its own CNN and its own GRU. That is
what MAPPO does, and it means a bad critic can never corrupt the actor's
features.

===========================================================================
HOW THE CRITIC ACTUALLY LEARNS  (it is NOT policy gradient)
===========================================================================
The actor learns by policy gradient -- ratio * advantage, the PPO thing.
The critic does something completely different and much more ordinary:
plain supervised regression.

  1. you play an episode and OBSERVE the real rewards
  2. you add them up (discounted) -> that is the "return". a number you
     MEASURED. it is not a prediction, it already happened.
  3. the critic's job is to have guessed that number ahead of time:

         value_loss = ( V(s) - return_you_actually_got )^2

  4. backprop that through v_out -> rnn -> base. Same as any regression.

That is the whole thing. No distribution, no ratio, no advantage inside
the critic's own loss. The label comes from experience.

WHY HAVE A CRITIC AT ALL
    the actor needs a baseline to know if an action was good:

        advantage = return - V(s)

    "was this better or worse than what I normally expect from this state?"
    That advantage is what multiplies the policy ratio in the actor's loss.

THE PART THAT CONFUSES EVERYONE
    V(s) is .detach()ed when the advantage is computed. So the critic's
    output FEEDS the actor's loss, but no gradient ever flows backwards
    from the actor's loss into the critic. The two are trained by two
    separate losses:

        actor   <- ratio * advantage       (policy gradient)
        critic  <- MSE(V, returns)         (supervised)

    MAPPO even gives them separate optimizers. Neither one trains the other.

===========================================================================
SHAPES  (N kitchens, T ticks, L gru layers, H hidden -- same as the utils)
===========================================================================
obs            (N, 23, W, H)     while playing
               (T*N, 23, W, H)   while training
rnn_states     (N, L, H)         always
masks          (N, 1) / (T*N, 1) 0 = episode just restarted, wipe memory

R_Actor.forward           -> actions (N,1), action_log_probs (N,1), rnn_states (N,L,H)
R_Actor.evaluate_actions  -> action_log_probs (T*N,1), dist_entropy (scalar)
R_Critic.forward          -> values (N,1), rnn_states (N,L,H)
"""


import torch
import torch.nn as nn

from utils.cnn import CNNBase
from utils.rnn import RNNLayer
from utils.act import ACTLayer


class R_Actor(nn.Module):
    """
    obs -> action.   CNN sees the frame, GRU remembers the past,
    ACTLayer turns that into a choice among the 6 actions.
    """

    def __init__(self, args, obs_shape, action_dim=6):
        super().__init__()
        self.base = CNNBase(args, obs_shape)
        self.rnn  = RNNLayer(self.base.output_size, args.hidden_size, args.recurrent_N)
        self.act  = ACTLayer(args.hidden_size, action_dim)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        #runs while COLLECTING data -- one tick, every kitchen at once
        x = self.base(obs)                                  # (N, 23, W, H) -> (N, 64)
        x, rnn_states = self.rnn(x, rnn_states, masks)      # (N, 64), (N, L, 64)
        actions, action_log_probs = self.act(x, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks):
        #runs during the PPO UPDATE, on a (T*N, ...) replay batch.
        #does NOT sample -- it re-scores the actions already taken, under the
        #weights as they are right now, so ppo can form the ratio.
        x = self.base(obs)                                  # (T*N, 64)
        x, _ = self.rnn(x, rnn_states, masks)
        #rnn_states is thrown away here on purpose: during the update we only
        #care about the outputs, the buffer already holds the states we need
        return self.act.evaluate_actions(x, action)         # log_probs, entropy


class R_Critic(nn.Module):
    """
    simply map hidden state to 1 value number.

    Same trunk as the actor but a Linear(hidden, 1) head instead of a
    Categorical -- it is predicting "expected total future reward from this
    state", not choosing anything.
    """

    def __init__(self, args, share_obs_shape):
        super().__init__()
        self.base = CNNBase(args, share_obs_shape)
        self.rnn  = RNNLayer(self.base.output_size, args.hidden_size, args.recurrent_N)

        self.v_out = nn.Linear(args.hidden_size, 1)
        #gain=1.0 here, NOT the 0.01 the actor uses. 0.01 exists to make a
        #POLICY start undecided between choices. the critic is regressing a
        #scalar, so it wants a normal-strength init.
        nn.init.orthogonal_(self.v_out.weight, gain=1.0)
        nn.init.constant_(self.v_out.bias, 0)

    def forward(self, share_obs, rnn_states, masks):
        #share_obs is the critic's input. for self-play with full state you can
        #hand it the same obs as the actor. it is kept separate so that later
        #you can give the critic MORE than the actor sees (centralized
        #training, decentralized execution) without touching this signature.
        x = self.base(share_obs)
        x, rnn_states = self.rnn(x, rnn_states, masks)
        #(N, 1): one predicted value per kitchen, as a column so it stacks
        #with actions/rewards/masks in the buffer
        return self.v_out(x), rnn_states
