"""
ACT layer -- turns features into an actual action
structure ref: https://github.com/sjtu-marl/ZSC-Eval/tree/master
(structure only -- everything here is stock PyTorch)

===========================================================================
NOTATIONS
===========================================================================
N = how many kitchens running side by side (same N as cnn.py / rnn.py)
T = ticks in a replay chunk (training only)
F = input feature width = whatever RNNLayer emits = 64
A = how many actions exist = 6
    confirmed against the mdp: Action.ALL_ACTIONS is
       [(0,-1), (0,1), (1,0), (-1,0), (0,0), 'interact']
       = north, south, east, west, stay, interact

===========================================================================
INPUT / OUTPUT
===========================================================================
x       (N, F)     while playing   -- one row per kitchen
        (T*N, F)   while training  -- flat, same as the other layers
        this is RNNLayer's output: "what's going on right now, memory included"

forward returns
    actions           (N, 1)   which action to take, an int in [0, 6)
    action_log_probs  (N, 1)   log of how likely that action was

evaluate_actions returns
    action_log_probs  (T*N, 1) the SAME stored actions, re-scored by the
                               weights as they are RIGHT NOW
    dist_entropy      scalar   how undecided the policy is on average

===========================================================================
WHAT A "LOGIT" IS, AND WHY THERE IS NOTHING TO EXTRACT
===========================================================================
nn.Linear(64, 6) emits 6 raw scores -- one per action. Example:

    [ 2.1, -0.4,  0.0,  1.7, -3.2,  0.9 ]
       N     S     E     W   stay  interact

Those 6 numbers ARE the logits. They are not probabilities -- they can be
negative, and they do not sum to 1. "Logits" is just the name for
"unnormalized scores that are about to be softmaxed".

So there is no .logit attribute to pull off the tensor. The Linear output
already IS the thing.
(Careful: torch.Tensor.logit DOES exist, but it is a totally unrelated
elementwise function -- inverse sigmoid, log(p/(1-p)). On logits it returns
NaN, because logits are not in the range (0,1). And writing `x.logit`
without calling it just hands you a bound method object. Neither errors.)

===========================================================================
WHAT Categorical DOES
===========================================================================
torch.distributions.Categorical(logits=...) softmaxes those 6 scores into
6 probabilities that sum to 1:

    logits  [ 2.1, -0.4,  0.0,  1.7, -3.2,  0.9 ]
              |
              v  softmax
    probs   [0.42, 0.03, 0.05, 0.28, 0.00, 0.13]      sums to 1.0

Then it gives you three things:

  .sample()      roll a weighted die. 42% of the time you get action 0,
                 28% action 3, and so on. This is how the agent EXPLORES --
                 it does not always take the best-looking action.
                 NOTE it is a METHOD: distr.sample() with parentheses.
                 distr.sample without them is just the function object.

  .log_prob(a)   log of the probability of the action you actually took.
                 if you sampled action 0, that is log(0.42) = -0.87.
                 always <= 0, because probabilities are always <= 1.

  .entropy()     how spread out the distribution is, as one number.
                 max is ln(6) = 1.7918, meaning "perfectly undecided, all 6
                 equally likely". near 0 means "totally committed to one
                 action". PPO adds this to the loss as a bonus so the policy
                 does not collapse onto one action too early.

===========================================================================
WHY LOG PROBABILITIES AND NOT PLAIN PROBABILITIES
===========================================================================
PPO's whole update is built on the ratio between the new policy and the old:

    ratio = pi_new(a) / pi_old(a)

Dividing two tiny numbers is numerically nasty. In log space division
becomes subtraction, which is stable:

    ratio = exp(logp_new - logp_old)

That is the only reason everything is carried around as logs.

===========================================================================
"wth is gain"
===========================================================================
gain is just a scale factor that orthogonal_ multiplies the weights by.
Small gain -> small weights -> all 6 logits start near zero -> softmax gives
about 1/6 to every action -> the policy starts genuinely undecided and
explores.

measured on a random batch, max deviation from uniform (1/6):
    gain=0.01  ->  0.0035      basically uniform. good.
    gain=1.0   ->  0.2871      already strongly opinionated at init, purely
                               from random numbers

that second one means training burns its early steps unlearning a
preference that never meant anything. Standard PPO practice is a small gain
on the FINAL action layer specifically (0.01), while the hidden layers use
the normal relu gain (~1.41) like cnn.py and rnn.py do.

===========================================================================
"what is this bias why set to 0 huh"
===========================================================================
nn.Linear computes  y = xW + b. The bias b is the 6 numbers added on at the
end -- a constant nudge per action, independent of what the agent sees.

A nonzero bias at init means "prefer action 3 no matter what the kitchen
looks like". That is a preference invented by the random number generator,
not learned from anything. Zeroing it means the only thing steering the
policy at step 0 is the observation.

Same reasoning as the biases in cnn.py and rnn.py -- zero the biases,
shape the weights. The bias is still a trainable parameter; it just starts
at zero instead of starting at a random opinion.

===========================================================================
"how is an action ever a COLUMN?"
===========================================================================
(N, 1) does NOT mean one action is made of several numbers. It is still one
integer per kitchen. (N, 1) means N rows tall, 1 number wide -- the 1 is the
WIDTH, not extra data.

    shape (N,)          shape (N, 1)
    [3, 0, 5, 1]        [ [3],
                          [0],
                          [5],
                          [1] ]

Same four numbers. The second is just standing up as a column.

Why bother? Because the rollout buffer keeps a whole row of per-kitchen
quantities side by side, and they all have to be the same shape to line up:

               action   logprob   reward   value    mask
    kitchen 0 [   3  ] [ -0.87 ] [  0.0 ] [ 2.1 ] [  1  ]
    kitchen 1 [   0  ] [ -1.20 ] [ 20.0 ] [ 5.4 ] [  1  ]
    kitchen 2 [   5  ] [ -0.44 ] [  0.0 ] [ 1.8 ] [  0  ]
    kitchen 3 [   1  ] [ -2.01 ] [  0.0 ] [ 0.9 ] [  1  ]
      ^ (N,1)   (N,1)    (N,1)     (N,1)   (N,1)

If actions were (N,) while rewards were (N, 1), any arithmetic mixing them
would broadcast to (N, N) instead of erroring -- the exact same silent trap
as `hxs * masks` in rnn.py. Keeping every column (N, 1) makes that
impossible.

It also generalizes: an env with a multi-part action would use (N, k) with
k > 1. Steakhouse has one discrete action, so k is 1 -- but the slot is
there.

This is a CONVENTION, not a requirement. Use flat (N,) everywhere if you
prefer; just never mix the two.

===========================================================================
WHY TWO METHODS INSTEAD OF ONE
===========================================================================
They run at different times, on different weights.

  forward()           runs while COLLECTING data. "what should I do right
                      now?" -> sample an action, store it and its log-prob
                      in the buffer, step the env.

  evaluate_actions()  runs during the UPDATE, later. By then the weights
                      have already been changed by earlier gradient steps.
                      You feed it the actions you ALREADY took and ask "how
                      likely would I be to take those now?" Comparing that
                      against the stored old log-prob is exactly the ratio
                      above. You cannot reuse forward() for this, because
                      forward() samples a FRESH action -- you need the
                      log-prob of the specific action that was actually
                      taken.

===========================================================================
WHERE THIS SITS IN THE STACK
===========================================================================
──────────── CNNBase  (cnn.py)  ← DONE, you built this
Conv2d(23, 32, k=3,s=1,p=1) → ReLU
Conv2d(32, 64, k=3,s=1,p=1) → ReLU
Conv2d(64, 32, k=3,s=1,p=1) → ReLU
Flatten
Linear(32*W*H, 64) → ReLU → LayerNorm(64)
Linear(64, output_size) → ReLU → LayerNorm(output_size)
                                        emits (N, 64)

──────────── RNNLayer  (rnn.py)  ← DONE
GRU(64, 64, num_layers=1)
LayerNorm(64)
                                        emits (N, 64) + new hidden state

──────────── MLPLayer  (mlp.py)  ← SKIP THIS
(not used: layer_after_N defaults to 0)

──────────── ACTLayer  (act.py)  ← THIS FILE
Linear(64, 6) → Categorical
                                        emits action, log_prob
"""


import torch
import torch.nn as nn


class ACTLayer(nn.Module):

    def __init__(self, input_dim=64, output_dim=6, gain=0.01):
        super().__init__()

        #the whole layer: 64 features in, one raw score per action out.
        #no activation and no LayerNorm after this -- logits are meant to be
        #raw and unbounded. squashing them would cap how confident the policy
        #is ever allowed to get.
        self.action_out = nn.Linear(input_dim, output_dim)

        #gain = the scale factor orthogonal_ multiplies the weights by.
        #0.01 keeps the logits near zero, so softmax starts at ~1/6 per action
        #and the policy explores instead of arriving with a random opinion.
        #  gain=0.01 -> 0.0035 off uniform      gain=1.0 -> 0.2871 off uniform
        nn.init.orthogonal_(self.action_out.weight, gain=gain)
        #bias is the per-action constant in y = xW + b. a nonzero one at init
        #says "prefer action 3 regardless of what the kitchen looks like",
        #which is an opinion invented by the RNG. zero it so only the
        #observation steers the policy at step 0. it still trains normally.
        nn.init.constant_(self.action_out.bias, 0)

    def distri(self, x):
        #turns the 6 raw scores into a probability distribution over actions.
        #shared by forward() and evaluate_actions() so the two can never drift.
        #logits= (not probs=) lets torch softmax internally, which is the
        #numerically stable way to do it.
        return torch.distributions.Categorical(logits=self.action_out(x))

    def forward(self, x, deterministic=False):
        #runs while COLLECTING data. x is (N, 64) straight out of RNNLayer.
        distr = self.distri(x)

        #deterministic=False while training   -> sample, so the agent explores
        #deterministic=True  while evaluating -> always take the best action,
        #                                        so runs are reproducible
        #NOTE the parentheses. distr.sample without them is the method object,
        #not a sampled action.
        if deterministic:
            actions = distr.probs.argmax(dim=-1)        # (N,)
        else:
            actions = distr.sample()                    # (N,)

        #log-prob of the action we actually picked -- not of all 6.
        #this gets stored in the buffer and becomes logp_old at update time.
        action_log_probs = distr.log_prob(actions)      # (N,)

        #stand both up as columns so they stack with reward/value/mask in the
        #buffer. see the "how is an action ever a COLUMN?" note above.
        return actions.unsqueeze(-1), action_log_probs.unsqueeze(-1)

    def evaluate_actions(self, x, actions):
        #runs during the PPO UPDATE, on a (T*N, 64) batch replayed from the
        #buffer. note it does NOT sample -- the actions are handed to us.
        distr = self.distri(x)

        #drop the column shape, and force int64: log_prob uses these to index
        #into the distribution, so a float tensor throws.
        a = actions.squeeze(-1).long()                  # (T*N,)

        #log-prob under the CURRENT weights. the caller compares this against
        #the stored logp_old to build the PPO ratio.
        action_log_probs = distr.log_prob(a).unsqueeze(-1)      # (T*N, 1)

        #.mean() collapses to one scalar because it goes into the loss as a
        #single entropy-bonus term. keep it unmeaned only if you want
        #per-sample entropy for logging.
        dist_entropy = distr.entropy().mean()                   # scalar

        return action_log_probs, dist_entropy