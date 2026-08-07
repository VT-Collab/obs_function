"""
Self Play, Gerald Tesauro. Td-gammon, a self-teaching backgammon program, achieves master-level play.
Neural computation, 6(2):215-219, 1994

An reinforcement learning policy approach where agents only learn through playing against themselves

In this case, it is a PPO where both chefs are driven by the same network
For that to be coherent, the observation must be written from the acting agent's own perspective

In this domain, train the robot with itself to collaboratively complete the steakhouse task

Calls r_actor_critic.py, which does in order:
1. CNNLayer
2. RNNLayer
3. ACTLayer
4. Categorical head over 6 actions
5. collect actions, action_log_probs, rnn_states

===========================================================================
THE LOOP
===========================================================================
    for each episode (= one rollout of T ticks):
        COLLECT    T x  policy.get_actions -> envs.step -> buffer.insert
        BOOTSTRAP  1 x  policy.get_values  -> buffer.compute_returns (GAE)
        UPDATE     ppo_epoch x minibatches -> two losses, two optimizers
        buffer.after_update()

===========================================================================
SHAPES, ONCE
===========================================================================
E = n_rollout_threads    kitchens running in parallel
A = 2                    chefs per kitchen
N = E * A                ROWS -- both chefs flattened into the batch
T = episode_length       ticks per rollout

envs hand back (E, 2, 23, W, H); everything downstream sees (N, 23, W, H).
The flatten order is thread-major:
    [env0-chef0, env0-chef1, env1-chef0, env1-chef1, ...]
and it has to stay that way end to end or the two chefs get scrambled.

===========================================================================
THE TWO LOSSES
===========================================================================
actor -- clipped surrogate (policy gradient):

    ratio = exp(logp_new - logp_old)      how far the policy has moved
    loss  = -min( ratio * A ,  clip(ratio, 1-eps, 1+eps) * A )
            - entropy_coef * entropy

    the clip IS ppo. the data was collected by an older version of the
    policy. if the current one has drifted far from it, the ratio gets
    capped so one update cannot take a huge step on stale data.
    min() makes it pessimistic: it refuses to reward drifting further
    even when the advantage says the drift was good.

critic -- plain regression:

    loss = (V - returns)^2

    NOT policy gradient. The label is the return actually measured during
    the rollout. advantage is .detach()ed in the buffer, so no gradient
    ever crosses from the actor loss into the critic.

===========================================================================
REWARD SHAPING ANNEAL
===========================================================================
    reward = sparse + shaping_coef * shaped
    shaping_coef goes 1 -> 0 across --shaping_anneal_eps episodes.

Sparse delivery reward is far too rare to bootstrap from -- an untrained
policy will basically never complete a steak by accident. The shaped terms
(meat in pan, dish pickup, ...) give early gradient, then anneal away so the
final policy is optimizing the real objective and not the breadcrumbs.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

#so `python SP/self_play.py` finds algorithm/ and utils/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.rMAPPOPolicy import R_MAPPOPolicy
from utils.buffer import SelfPlayBuffer
from utils.env_wrapper import VecSteakEnv


def parse_args(argv=None):
    p = argparse.ArgumentParser("steakhouse self-play PPO")
    # env
    p.add_argument("--layouts", type=str, default="steak_side_2",
                   help="comma-separated layout names. >1 trains ONE policy across "
                        "all of them, every grid padded onto a shared canvas.")
    p.add_argument("--run_name", type=str, default=None,
                   help="checkpoint filename stem. defaults to the layout when "
                        "there is one, else the layout count.")
    p.add_argument("--n_rollout_threads", type=int, default=16)
    p.add_argument("--episode_length", type=int, default=400)
    p.add_argument("--n_orders", type=int, default=4)
    p.add_argument("--horizon", type=int, default=260)
    p.add_argument("--num_env_steps", type=int, default=int(5e6))
    # net
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--recurrent_N", type=int, default=1)
    # ppo
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--critic_lr", type=float, default=5e-4)
    p.add_argument("--opti_eps", type=float, default=1e-5)
    p.add_argument("--ppo_epoch", type=int, default=15)
    p.add_argument("--num_mini_batch", type=int, default=1)
    p.add_argument("--data_chunk_length", type=int, default=10)
    p.add_argument("--clip_param", type=float, default=0.2)
    p.add_argument("--value_loss_coef", type=float, default=1.0)
    #ZSC-Eval anneals entropy 0.2 -> 0.05 -> 0.01 over 0 -> 5e6 -> 1e7 steps
    #(train_sp.sh). Starting at the FINAL value gives the policy no reason to
    #explore, which on a reward this sparse means it commits to noise.
    p.add_argument("--entropy_coef", type=float, default=0.01,
                   help="final entropy coefficient")
    p.add_argument("--entropy_coef_start", type=float, default=0.2,
                   help="initial entropy coefficient, annealed to --entropy_coef")
    p.add_argument("--entropy_coef_horizon", type=float, default=1e7,
                   help="env steps over which entropy anneals start -> final")
    p.add_argument("--use_huber_loss", action="store_true",
                   help="huber instead of mse on the value loss (ZSC-Eval default)")
    p.add_argument("--huber_delta", type=float, default=10.0)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--use_linear_lr_decay", action="store_true")
    #ZSC-Eval uses reward_shaping_horizon=1e8 with num_env_steps=1e7 -- the
    #horizon is 10x the whole run, so shaping decays to only ~0.9 by the end.
    #They effectively NEVER turn it off. Annealing to 0 early kills the only
    #dense signal this task has.
    p.add_argument("--reward_shaping_horizon", type=float, default=1e8)
    # misc
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=1)
    return p.parse_args(argv)


def action_probs(policy, obs, rnn_states, masks):
    """
    THE HOOK FOR FILTERS / INFERENCE.

    Returns the full action distribution for a state, without sampling and
    without touching gradients:

        probs, rnn_states = action_probs(policy, obs, rnn_states, masks)
        #   probs       (N, 6)   sums to 1 along dim 1
        #   rnn_states  (N, L, H)  feed this back in on the next tick

    Use this instead of policy.act() whenever you need the DISTRIBUTION rather
    than a single choice -- e.g. a filter/posterior on top of the policy, a KL
    against a partner, or logging which actions were live.

    Why it exists: policy.act() and policy.get_actions() both collapse the
    distribution to one sampled or argmaxed integer and throw the other 5
    numbers away. This walks the same three layers and stops one step earlier,
    at ACTLayer.distri(), which is the Categorical itself.

    It is the identical forward pass -- base -> rnn -> action_out -- so the
    probabilities here are exactly the ones policy.act() would have sampled
    from. Nothing is recomputed differently.

    NOTE it advances the GRU memory, same as any other forward. Thread
    rnn_states through your eval loop or the agent has no memory.
    """
    with torch.no_grad():
        x = policy.actor.base(obs)                              # (N, F)
        x, rnn_states = policy.actor.rnn(x, rnn_states, masks)  # (N, H)
        distri = policy.actor.act.distri(x)                     # Categorical

    #.probs is (N, 6). .logits is there too if you want them unnormalized.
    return distri.probs, rnn_states


def ppo_update(policy, buffer, args, device, entropy_coef):
    """One full pass: ppo_epoch sweeps over num_mini_batch minibatches."""
    stats = dict(value_loss=0.0, policy_loss=0.0, entropy=0.0, clip_frac=0.0, n=0)

    for _ in range(args.ppo_epoch):
        for batch in buffer.recurrent_generator(args.num_mini_batch, device,
                                               args.data_chunk_length):

            values, log_probs, entropy = policy.evaluate_actions(
                batch["obs"], batch["obs"],          # share_obs == obs for now
                batch["rnn_actor"], batch["rnn_critic"],
                batch["actions"], batch["masks"])

            # ------------------------------------------- actor
            #exp(difference of logs) = a division, done stably.
            #ratio > 1 -> the policy now likes this action MORE than it did
            #when the action was collected.
            ratio = torch.exp(log_probs - batch["old_log_probs"])
            adv = batch["advantages"]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param,
                                1.0 + args.clip_param) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            actor_loss = policy_loss - entropy_coef * entropy

            policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            #clip the gradient NORM, not individual values. one freak
            #transition can otherwise blow the policy up in a single step.
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),
                                           args.max_grad_norm)
            policy.actor_optimizer.step()

            # ------------------------------------------- critic
            #CLIPPED value loss -- the critic gets the same trust region the
            #actor does. value_preds is V(s) as it was at collection time; the
            #new V is only allowed to move clip_param away from it before the
            #loss stops rewarding the move. max() of the two makes it
            #pessimistic, same trick as min() on the actor side.
            #without this the critic can lurch far from the predictions the
            #advantages were computed against, and the two halves disagree
            #about what state was worth what.
            value_pred_clipped = batch["value_preds"] + torch.clamp(
                values - batch["value_preds"], -args.clip_param, args.clip_param)
            if args.use_huber_loss:
                #huber is linear past huber_delta instead of quadratic, so one
                #freak return cannot dominate the batch. ZSC-Eval default.
                e_o = batch["returns"] - values
                e_c = batch["returns"] - value_pred_clipped
                hub = lambda e: torch.where(e.abs() <= args.huber_delta,
                                            0.5 * e ** 2,
                                            args.huber_delta * (e.abs() - 0.5 * args.huber_delta))
                value_loss = torch.max(hub(e_o), hub(e_c)).mean()
            else:
                value_loss = torch.max((values - batch["returns"]) ** 2,
                                       (value_pred_clipped - batch["returns"]) ** 2).mean()

            policy.critic_optimizer.zero_grad()
            (value_loss * args.value_loss_coef).backward()
            torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),
                                           args.max_grad_norm)
            policy.critic_optimizer.step()

            with torch.no_grad():
                #how often the clip actually bit. climbing toward 1 means the
                #policy is moving too fast -- drop lr or ppo_epoch.
                clipped = (torch.abs(ratio - 1.0) > args.clip_param).float().mean()
            stats["value_loss"] += value_loss.item()
            stats["policy_loss"] += policy_loss.item()
            stats["entropy"] += entropy.item()
            stats["clip_frac"] += clipped.item()
            stats["n"] += 1

    n = max(stats.pop("n"), 1)
    return {k: v / n for k, v in stats.items()}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print(f"[setup] device={device}  cuda_available={torch.cuda.is_available()}", flush=True)

    envs = VecSteakEnv(args.layouts, args.n_rollout_threads,
                       args.n_orders, args.horizon, args.seed)

    E, A = args.n_rollout_threads, envs.num_agents
    N = E * A
    T = args.episode_length
    run_name = args.run_name or (envs.layout_names[0] if len(envs.layout_names) == 1
                                 else f"multi{len(envs.layout_names)}")
    print(f"[setup] layouts={len(envs.layout_names)} canvas={envs.pad_shape} "
          f"obs={envs.obs_shape} threads={E} agents={A} rows={N} T={T} "
          f"run={run_name}", flush=True)
    #padding waste: how much of the canvas each kitchen actually fills
    canvas = envs.pad_shape[0] * envs.pad_shape[1]
    fill = {n: round(w * h / canvas, 2) for n, (w, h) in envs.layout_sizes.items()}
    print(f"[setup] canvas fill fraction per layout: {fill}", flush=True)

    policy = R_MAPPOPolicy(args, envs.obs_shape, envs.obs_shape,
                           envs.n_actions, device)
    buffer = SelfPlayBuffer(T, N, envs.obs_shape, args.recurrent_N,
                            args.hidden_size, args.gamma, args.gae_lambda)

    obs = envs.reset()                                   # (E, 2, 23, W, H)
    buffer.reset(obs.reshape(N, *envs.obs_shape))

    episodes = max(args.num_env_steps // (T * E), 1)
    recent_returns, per_layout, start = [], {}, time.time()
    os.makedirs(args.save_dir, exist_ok=True)

    for episode in range(episodes):
        if args.use_linear_lr_decay:
            policy.lr_decay(episode, episodes)

        env_steps_done = episode * T * E
        #decays over reward_shaping_horizon ENV STEPS (not episodes), matching
        #ZSC-Eval. With horizon 10x the run it barely moves off 1.0.
        shaping_coef = max(0.0, 1.0 - env_steps_done / args.reward_shaping_horizon)
        #entropy: linear 0.2 -> 0.01 over entropy_coef_horizon steps
        ec_frac = min(1.0, env_steps_done / max(args.entropy_coef_horizon, 1.0))
        entropy_coef = (args.entropy_coef_start
                        + (args.entropy_coef - args.entropy_coef_start) * ec_frac)

        # ------------------------------------------------------- COLLECT
        for step in range(T):
            with torch.no_grad():
                o = torch.from_numpy(buffer.obs[step]).to(device)
                ra = torch.from_numpy(buffer.rnn_actor[step]).to(device)
                rc = torch.from_numpy(buffer.rnn_critic[step]).to(device)
                mk = torch.from_numpy(buffer.masks[step]).to(device)

                values, actions, log_probs, ra2, rc2 = policy.get_actions(
                    o, o, ra, rc, mk)

            #back to cpu numpy: the mdp is pure python, it knows nothing
            #about tensors or devices
            act_np = actions.cpu().numpy().reshape(E, A)

            obs, sparse, shaped, dones, truncs, infos = envs.step(act_np)

            for info in infos:
                if "episode_sparse" in info:
                    recent_returns.append(info["episode_sparse"])
                    per_layout.setdefault(info["layout"], []).append(
                        info["episode_sparse"])

            #SHARED team reward -- the same number goes to both chefs in a
            #kitchen, because a delivery is not attributable to one of them
            reward = np.repeat(sparse + shaping_coef * shaped, A).reshape(N, 1)

            #masks: 0 where the episode just ended. this is exactly what wipes
            #the GRU memory in rnn.py so a new episode starts clean.
            masks = np.repeat(1.0 - dones.astype(np.float32), A).reshape(N, 1)

            #bad_masks: 0 only where the episode was CUT OFF by the horizon.
            #GAE then falls back to the critic instead of pretending the world
            #ended. Fires on nearly every boundary -- the only true terminal is
            #running the order list out, which an untrained policy never does.
            bad_masks = np.repeat(1.0 - truncs.astype(np.float32), A).reshape(N, 1)

            buffer.insert(
                obs.reshape(N, *envs.obs_shape),
                ra2.cpu().numpy(), rc2.cpu().numpy(),
                actions.cpu().numpy(), log_probs.cpu().numpy(),
                values.cpu().numpy(), reward, masks, bad_masks)

        # ----------------------------------------------------- BOOTSTRAP
        with torch.no_grad():
            next_values = policy.get_values(
                torch.from_numpy(buffer.obs[-1]).to(device),
                torch.from_numpy(buffer.rnn_critic[-1]).to(device),
                torch.from_numpy(buffer.masks[-1]).to(device))
        buffer.compute_returns(next_values.cpu().numpy())

        # -------------------------------------------------------- UPDATE
        stats = ppo_update(policy, buffer, args, device, entropy_coef)
        buffer.after_update()

        # ---------------------------------------------------------- LOGS
        if episode % args.log_interval == 0:
            steps = (episode + 1) * T * E
            avg = float(np.mean(recent_returns[-50:])) if recent_returns else float("nan")
            fps = int(steps / max(time.time() - start, 1e-6))
            print(f"ep {episode:5d}/{episodes} | steps {steps:>9,} | fps {fps:>5} "
                  f"| sparse_ret {avg:7.2f} | vloss {stats['value_loss']:8.3f} "
                  f"| ploss {stats['policy_loss']:7.4f} | ent {stats['entropy']:.3f} "
                  f"| clip {stats['clip_frac']:.3f} | shape_c {shaping_coef:.2f} "
                  f"| ent_c {entropy_coef:.3f}",
                  flush=True)
            #with a mixed pool the aggregate hides which kitchens are failing
            if len(envs.layout_names) > 1 and per_layout:
                brk = " ".join(f"{k.replace('steak_',''):s}:{np.mean(v[-20:]):.1f}"
                               for k, v in sorted(per_layout.items()))
                print(f"        per-layout: {brk}", flush=True)

        if episode % args.save_interval == 0 or episode == episodes - 1:
            torch.save({"actor": policy.actor.state_dict(),
                        "critic": policy.critic.state_dict(),
                        "episode": episode, "args": vars(args)},
                       os.path.join(args.save_dir, f"sp_{run_name}.pt"))

    print(f"[done] {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    train(parse_args())
