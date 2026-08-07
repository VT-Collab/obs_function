"""
The policy wrapper PPO talks to. Owns the actor, the critic, and their two
optimizers, and exposes the 5 entry points the training loop calls.

NOTE this is deliberately NOT an nn.Module. It does not compute anything
itself and it has no forward(). It is a container. The actual networks are
R_Actor and R_Critic in r_actor_critic.py.

===========================================================================
WHEN EACH METHOD FIRES
===========================================================================
    for episode in range(episodes):
        policy.lr_decay(episode, episodes)                    # 5

        for t in range(episode_length):            # ---- COLLECT ----
            v, a, logp, rs_a, rs_c = policy.get_actions(...)  # 1
            obs, rew, done = envs.step(a.cpu().numpy())
            buffer.insert(...)

        next_v = policy.get_values(...)            # 2  bootstrap the tail
        buffer.compute_returns(next_v)                # GAE lives in the buffer

        for epoch in range(ppo_epoch):             # ---- UPDATE ----
            for batch in buffer.sample():
                v, logp, ent = policy.evaluate_actions(...)   # 3
                # ratio = exp(logp - logp_old), then the two losses

    policy.act(...)                                # 4  eval / rendering only

get_actions and evaluate_actions look redundant but are not:
    get_actions       SAMPLES a fresh action        -> gives you logp_old
    evaluate_actions  SCORES an action already taken -> gives you logp_new
that pair is the entire PPO ratio.

===========================================================================
SELF-PLAY: ONE POLICY, BOTH AGENTS
===========================================================================
You do NOT build two policies. Both agents share these exact weights --
that is what makes it self-play. The two agents are just extra ROWS in the
batch:

    n_rollout_threads = 32,  num_agents = 2   ->   N = 64 rows

Every shape in cnn.py / rnn.py / act.py already treats N as "however many
rows there are", so both agents ride through the same weights for free.

What stops the two agents from behaving identically is the ego-centric
channel ordering in features.py: agent_index decides who lands in channels
[0,1] and who lands in [2,3], so each agent sees itself first. Same weights,
different input, different behaviour.

===========================================================================
TWO OPTIMIZERS, NOT ONE
===========================================================================
The actor and the critic learn from completely different losses:

    actor   <- ratio * advantage        policy gradient
    critic  <- MSE(V, returns)          plain supervised regression

and the advantage is .detach()ed, so no gradient ever crosses between them.
Since the updates are independent, they get independent optimizers and
independent learning rates (critic_lr is often larger -- regression
tolerates a bigger step than a policy does).

===========================================================================
GPU / CARC
===========================================================================
device is threaded through here and .to(device) is applied to both networks.
Two things this file cannot do for you:

  1. the INPUTS must be moved too. features.py hands back numpy, so the
     runner has to do  torch.from_numpy(obs).float().to(device)  for obs,
     rnn_states and masks. A CPU tensor into a CUDA model raises loudly.

  2. actions must come BACK to cpu before touching the env --
     envs.step(actions.cpu().numpy()) -- the mdp is pure python.

Honest note: for overcooked-scale MAPPO the GPU usually buys very little.
The bottleneck is envs.step(), which is python and CPU-bound and cannot be
batched onto a GPU. The conv stack on a 15x10 grid is microseconds either
way. More parallel envs (cpus) is the real throughput lever -- which is what
the existing fov/*.sbatch scripts already do with --partition=main and
--cpus-per-task=32. Profile before paying the GPU queue.
"""

import torch

from algorithm.r_actor_critic import R_Actor, R_Critic


class R_MAPPOPolicy:

    def __init__(self, args, obs_shape, share_obs_shape, act_dim=6,
                 device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr

        #actor decides WHAT TO DO, critic guesses HOW GOOD THIS SPOT IS.
        #separate networks, no shared weights -- a bad critic can then never
        #corrupt the features the actor is learning from.
        self.actor = R_Actor(args, obs_shape, act_dim).to(device)
        self.critic = R_Critic(args, share_obs_shape).to(device)

        #one optimizer each, because one loss each. see the note above.
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.lr, 
            eps=args.opti_eps
            )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=self.critic_lr, 
            eps=args.opti_eps
            )

    # ------------------------------------------------------------------ 1
    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic,
                    masks, deterministic=False):
        """
        ROLLOUT. Called once per tick while collecting data.
        "what do I do, and how good is this spot?"

        Both networks get run because the buffer needs BOTH: the action to
        step the env with, and the value to compute advantages with later.

        deterministic=False here -- you WANT sampling while collecting, so
        the agent explores instead of repeating itself.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, deterministic)

        values, rnn_states_critic = self.critic(
            share_obs, rnn_states_critic, masks)

        #every one of these gets stored in the buffer for this tick.
        #action_log_probs is the logp_old that the update compares against.
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    # ------------------------------------------------------------------ 2
    def get_values(self, share_obs, rnn_states_critic, masks):
        """
        BOOTSTRAP. Called ONCE, after the rollout ends.

        The return at tick t is "rewards from t onward". At the last tick you
        have not seen the rest -- the episode was cut off, not finished. So you
        ask the critic to guess the value of the state you stopped on, and use
        that as the stand-in tail. Without it every rollout would pretend the
        world ends at episode_length and undervalue everything near the end.
        """
        values, _ = self.critic(share_obs, rnn_states_critic, masks)
        return values

    # ------------------------------------------------------------------ 3
    def evaluate_actions(self, share_obs, obs, rnn_states_actor,
                         rnn_states_critic, action, masks):
        """
        UPDATE. Called on a (T*N, ...) batch replayed out of the buffer.

        Does NOT sample. You hand it the actions that were ALREADY taken and
        ask "how likely am I to take those NOW?" -- because the weights have
        moved since the rollout. That gives logp_new; the buffer holds
        logp_old; ratio = exp(logp_new - logp_old).

        Returns values too, because the critic loss needs a fresh V(s) to
        regress against the stored returns.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks)

        values, _ = self.critic(share_obs, rnn_states_critic, masks)

        #dist_entropy is a scalar: the exploration bonus in the actor loss
        return values, action_log_probs, dist_entropy

    # ------------------------------------------------------------------ 4
    def act(self, obs, rnn_states_actor, masks, deterministic=True):
        """
        EVAL / RENDER only. Never called in the training loop.
        Just an action, no value and no log-prob, because nothing is learning.

        deterministic=True by default here -- when measuring performance you
        want the policy's actual best answer, repeatable across runs, not a
        different dice roll every time.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    # ------------------------------------------------------------------ 5
    def lr_decay(self, episode, episodes):
        """
        Linear anneal to 0 over training. Optional but standard in MAPPO:
        big steps early while everything is wrong, tiny steps late so the
        policy settles instead of bouncing around its own optimum.
        """
        for optimizer, base_lr in ((self.actor_optimizer, self.lr),
                                   (self.critic_optimizer, self.critic_lr)):
            lr = base_lr * (1.0 - episode / float(episodes))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
