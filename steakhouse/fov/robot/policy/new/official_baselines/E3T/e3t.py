"""
E3T -- Efficient End-to-End Training for zero-shot coordination.

Zhao et al., "E3T: Efficient End-to-End Training for Zero-Shot Coordination".
Single stage. No population, no pre-training, no policy pool -- which is why it
is the first thing after SP that can actually be launched here.

===========================================================================
THE IDEA IN ONE PARAGRAPH
===========================================================================
Self-play fails at zero-shot coordination because the partner it trains against
is ITSELF: perfectly predictable, perfectly synchronized, converging on one
private convention. Pair that policy with anything else -- another seed, a
scripted agent, a human with a 90-degree cone -- and the convention is gone and
so is the performance.

E3T keeps self-play's single-stage cheapness and breaks the symmetry two ways:

    LAGGED   the partner is a slow-moving copy of the ego, updated
             theta_p <- (1-tau) * theta_p + tau * theta_ego  each episode.
             So the ego trains against its own recent past, not its exact
             self -- a moving target it cannot perfectly anticipate.

    NOISY    the partner takes a uniformly random action with probability
             epsilon. This is the part that actually matters for us: it is a
             cheap stand-in for "my partner did something I did not predict",
             which is the entire failure mode a limited-FOV human creates.

===========================================================================
WHAT THIS FILE IMPLEMENTS, AND WHAT IT DELIBERATELY DOES NOT
===========================================================================
This follows ZSC-Eval's E3T, not the paper's, because ZSC-Eval is what the
baseline numbers in the literature are reported against:

    zsceval/algorithms/r_mappo/algorithm/rMAPPOPolicy_epsilon.py   the epsilon mix
    zsceval/algorithms/r_mappo/r_mappo_target.py                   the soft copy
    scripts/overcooked/shell/train_e3t.sh    epsilon 0.25, weights_copy_factor 0.1

The paper ALSO has a partner-modelling module: a context encoder that predicts
the partner's next action from recent history and concatenates that prediction
into the ego's observation. ZSC-Eval does not implement it -- there is no such
head anywhere in their tree -- so neither does this. Worth knowing, because it
is the component most obviously relevant to a partner whose behaviour depends
on a hidden FOV, and it is the natural extension if E3T underperforms here.

===========================================================================
SP-EPSILON: THE FREE ABLATION
===========================================================================
    --weights_copy_factor 1.0

At tau = 1.0 the soft copy becomes an exact copy, so the partner IS the ego
policy and the only thing left is the epsilon noise. That is plain self-play
with a randomly-perturbed partner. Running 0.1 and 1.0 side by side answers
"is E3T's gain the lagged partner, or just the noise?" for the cost of a
different flag -- no second implementation, no risk the two arms differ
anywhere else.

===========================================================================
SHAPES -- DIFFERENT FROM SELF-PLAY, READ THIS
===========================================================================
self_play.py trains on N = E * 2 rows, because both chefs share the weights and
both are learning. Here only the EGO learns:

    N = E                    one row per kitchen, the ego's
    ego obs      (E, 23, W, H)     from the ego's own index
    partner obs  (E, 23, W, H)     from the partner's index
    buffer       N = E             the partner's transitions are NEVER stored

Storing the partner's rows would be a straightforward disaster: they are
off-policy with respect to the ego (different weights, plus epsilon noise), so
their log-probs do not correspond to the network being updated, and PPO's ratio
would be meaningless.

===========================================================================
RANDOM INDEX
===========================================================================
--random_index resamples which seat the ego occupies at every episode reset
(ZSC-Eval passes it for E3T). The observation is ego-centric so the two seats
look symmetric to the network, but the LAYOUT is not: the two start positions
differ, and on an asymmetric kitchen a fixed seat teaches a fixed half of the
job. Since the FOV-human evaluation puts the robot in one specific seat, being
competent in both is exactly what we want to measure.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

#so `python E3T/e3t.py` finds algorithm/ and utils/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.rMAPPOPolicy import R_MAPPOPolicy
from utils.buffer import SelfPlayBuffer
from utils.ckpt import RunWriter
from utils.env_wrapper import VecSteakEnv
from utils.schedules import add_schedule_args, piecewise
from utils.valuenorm import IdentityValueNorm, ValueNorm


def parse_args(argv=None):
    p = argparse.ArgumentParser("steakhouse E3T")
    # env
    p.add_argument("--layouts", type=str, default="steak_gc00")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--n_rollout_threads", type=int, default=50)
    p.add_argument("--episode_length", type=int, default=400)
    p.add_argument("--n_orders", type=int, default=4)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--num_env_steps", type=int, default=int(1e7))
    # net
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--recurrent_N", type=int, default=1)
    # ---------------------------------------------------------------- e3t
    p.add_argument("--epsilon", type=float, default=0.25,
                   help="partner takes a uniformly random action with this "
                        "probability. ZSC-Eval train_e3t.sh uses 0.25.")
    p.add_argument("--weights_copy_factor", type=float, default=0.1,
                   help="tau in theta_p <- (1-tau)*theta_p + tau*theta_ego, "
                        "applied once per episode. 0.1 = E3T. "
                        "1.0 = the SP-epsilon ablation (partner IS the ego).")
    p.add_argument("--random_index", action="store_true", default=True,
                   help="resample the ego's seat each episode (ZSC default)")
    p.add_argument("--fixed_index", dest="random_index", action="store_false")
    p.add_argument("--ego_index", type=int, default=0,
                   help="seat used when --fixed_index")
    # ppo
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--critic_lr", type=float, default=5e-4)
    p.add_argument("--opti_eps", type=float, default=1e-5)
    p.add_argument("--ppo_epoch", type=int, default=15)
    p.add_argument("--num_mini_batch", type=int, default=1)
    p.add_argument("--data_chunk_length", type=int, default=10)
    p.add_argument("--clip_param", type=float, default=0.2)
    p.add_argument("--value_loss_coef", type=float, default=1.0)
    add_schedule_args(p)
    p.add_argument("--use_huber_loss", action="store_true")
    p.add_argument("--huber_delta", type=float, default=10.0)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--use_linear_lr_decay", action="store_true")
    p.add_argument("--reward_shaping_horizon", type=float, default=1e8)
    p.add_argument("--use_valuenorm", action="store_true", default=True)
    p.add_argument("--no_valuenorm", dest="use_valuenorm", action="store_false")
    # misc
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args(argv)


# =========================================================================
# THE PARTNER
# =========================================================================
@torch.no_grad()
def soft_copy(source, target, tau):
    """theta_target <- (1 - tau) * theta_target + tau * theta_source.

    Mirrors zsceval r_mappo_target._copy_params. tau = 1.0 is an exact copy,
    which is what makes --weights_copy_factor 1.0 the SP-epsilon ablation.
    """
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1.0 - tau) * tp.data + tau * sp.data)


class Partner:
    """A lagged, noisy copy of the ego. Never trained, never in the buffer."""

    def __init__(self, args, obs_shape, act_dim, device, ego):
        self.policy = R_MAPPOPolicy(args, obs_shape, obs_shape, act_dim, device)
        self.epsilon = args.epsilon
        self.tau = args.weights_copy_factor
        self.act_dim = act_dim
        self.device = device
        #start AS the ego rather than as an unrelated random net. E3T's premise
        #is "the partner is a past version of me", and at episode 0 the only
        #past version there is is the ego itself.
        soft_copy(ego.actor, self.policy.actor, 1.0)
        soft_copy(ego.critic, self.policy.critic, 1.0)
        self.policy.actor.eval()
        self.policy.critic.eval()

    def sync(self, ego):
        soft_copy(ego.actor, self.policy.actor, self.tau)
        soft_copy(ego.critic, self.policy.critic, self.tau)

    @torch.no_grad()
    def act(self, obs, rnn_states, masks):
        """-> (actions (E,1) int64, rnn_states). Sampled, then epsilon-mixed."""
        actions, _, rnn_states = self.policy.actor(obs, rnn_states, masks, False)
        if self.epsilon > 0:
            #per-ELEMENT coin flip, exactly as rMAPPOPolicy_epsilon does it --
            #every kitchen decides independently every tick
            flip = torch.rand(actions.shape, device=actions.device)
            rand_a = torch.randint(0, self.act_dim, actions.shape,
                                   device=actions.device, dtype=actions.dtype)
            actions = torch.where(flip < self.epsilon, rand_a, actions)
        return actions, rnn_states


# =========================================================================
# PPO UPDATE  -- ego only. identical arithmetic to SP/self_play_pop.py
# =========================================================================
def ppo_update(policy, buffer, args, device, entropy_coef, value_norm):
    stats = dict(value_loss=0.0, policy_loss=0.0, entropy=0.0, clip_frac=0.0, n=0)

    for _ in range(args.ppo_epoch):
        for batch in buffer.recurrent_generator(args.num_mini_batch, device,
                                                args.data_chunk_length):

            values, log_probs, entropy = policy.evaluate_actions(
                batch["obs"], batch["obs"],
                batch["rnn_actor"], batch["rnn_critic"],
                batch["actions"], batch["masks"])

            ratio = torch.exp(log_probs - batch["old_log_probs"])
            adv = batch["advantages"]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param,
                                1.0 + args.clip_param) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            actor_loss = policy_loss - entropy_coef * entropy

            policy.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),
                                           args.max_grad_norm)
            policy.actor_optimizer.step()

            value_norm.update(batch["returns"])
            returns_n = value_norm.normalize(batch["returns"])
            value_preds_n = value_norm.normalize(batch["value_preds"])

            value_pred_clipped = value_preds_n + torch.clamp(
                values - value_preds_n, -args.clip_param, args.clip_param)
            if args.use_huber_loss:
                e_o = returns_n - values
                e_c = returns_n - value_pred_clipped
                hub = lambda e: torch.where(e.abs() <= args.huber_delta,
                                            0.5 * e ** 2,
                                            args.huber_delta * (e.abs() - 0.5 * args.huber_delta))
                value_loss = torch.max(hub(e_o), hub(e_c)).mean()
            else:
                value_loss = torch.max((values - returns_n) ** 2,
                                       (value_pred_clipped - returns_n) ** 2).mean()

            policy.critic_optimizer.zero_grad()
            (value_loss * args.value_loss_coef).backward()
            torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),
                                           args.max_grad_norm)
            policy.critic_optimizer.step()

            with torch.no_grad():
                clipped = (torch.abs(ratio - 1.0) > args.clip_param).float().mean()
            stats["value_loss"] += value_loss.item()
            stats["policy_loss"] += policy_loss.item()
            stats["entropy"] += entropy.item()
            stats["clip_frac"] += clipped.item()
            stats["n"] += 1

    n = max(stats.pop("n"), 1)
    return {k: v / n for k, v in stats.items()}


# =========================================================================
# TRAIN
# =========================================================================
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed + 7919)

    device = torch.device("cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu")
    envs = VecSteakEnv(args.layouts, args.n_rollout_threads,
                       args.n_orders, args.horizon, args.seed)

    E = args.n_rollout_threads
    T = args.episode_length
    N = E                                   # EGO ROWS ONLY -- see the docstring
    obs_shape = envs.obs_shape
    run_name = args.run_name or envs.layout_names[0]

    mode = "SP-epsilon" if args.weights_copy_factor >= 1.0 else "E3T"
    print(f"[setup] {mode} device={device} layout={run_name} seed={args.seed} "
          f"obs={obs_shape} threads={E} ego_rows={N} T={T} "
          f"eps={args.epsilon} tau={args.weights_copy_factor} "
          f"random_index={args.random_index} valuenorm={args.use_valuenorm}",
          flush=True)

    ego = R_MAPPOPolicy(args, obs_shape, obs_shape, envs.n_actions, device)
    partner = Partner(args, obs_shape, envs.n_actions, device, ego)

    buffer = SelfPlayBuffer(T, N, obs_shape, args.recurrent_N,
                            args.hidden_size, args.gamma, args.gae_lambda)
    value_norm = ValueNorm(1, device=device) if args.use_valuenorm else IdentityValueNorm()
    writer = RunWriter(args.save_dir, run_name, args)

    #which seat the ego holds, per kitchen. resampled wherever an episode ends.
    if args.random_index:
        ego_idx = rng.randint(0, 2, size=E)
    else:
        ego_idx = np.full(E, args.ego_index, dtype=int)
    rows = np.arange(E)

    obs = envs.reset()                                   # (E, 2, 23, W, H)
    buffer.reset(obs[rows, ego_idx])
    partner_obs = obs[rows, 1 - ego_idx]
    partner_rnn = torch.zeros(E, args.recurrent_N, args.hidden_size, device=device)
    partner_masks = torch.ones(E, 1, device=device)

    episodes = max(args.num_env_steps // (T * E), 1)
    recent_returns, start = [], time.time()

    for episode in range(episodes):
        if args.use_linear_lr_decay:
            ego.lr_decay(episode, episodes)

        env_steps_done = episode * T * E
        shaping_coef = max(0.0, 1.0 - env_steps_done / args.reward_shaping_horizon)
        entropy_coef = piecewise(env_steps_done, args.entropy_coefs,
                                 args.entropy_coef_horizons)

        # ------------------------------------------------------- COLLECT
        for step in range(T):
            with torch.no_grad():
                o = torch.from_numpy(buffer.obs[step]).to(device)
                ra = torch.from_numpy(buffer.rnn_actor[step]).to(device)
                rc = torch.from_numpy(buffer.rnn_critic[step]).to(device)
                mk = torch.from_numpy(buffer.masks[step]).to(device)

                values, actions, log_probs, ra2, rc2 = ego.get_actions(o, o, ra, rc, mk)
                values = value_norm.denormalize(values)

                p_obs = torch.from_numpy(partner_obs).to(device)
                p_actions, partner_rnn = partner.act(p_obs, partner_rnn, partner_masks)

            #reassemble the joint action in SEAT order, not ego/partner order.
            #getting this backwards silently swaps the two chefs and the run
            #still "works" -- it just trains on the wrong seat.
            joint = np.empty((E, 2), dtype=np.int64)
            joint[rows, ego_idx] = actions.cpu().numpy().reshape(E)
            joint[rows, 1 - ego_idx] = p_actions.cpu().numpy().reshape(E)

            obs, sparse, shaped, dones, truncs, infos = envs.step(joint)

            for info in infos:
                if "episode_sparse" in info:
                    recent_returns.append(info["episode_sparse"])

            #the wrapper has ALREADY auto-reset the finished kitchens, so `obs`
            #for those rows is the first frame of a NEW episode -- pick the new
            #seat before slicing it, or the first tick of every episode is read
            #from the wrong chef
            if args.random_index and dones.any():
                ego_idx = np.where(dones, rng.randint(0, 2, size=E), ego_idx)

            #shared team reward: the ego's row is the whole team's number
            reward = (sparse + shaping_coef * shaped).astype(np.float32).reshape(N, 1)
            masks = (1.0 - dones.astype(np.float32)).reshape(N, 1)
            bad_masks = (1.0 - truncs.astype(np.float32)).reshape(N, 1)

            #same masks wipe the partner's GRU at an episode boundary
            partner_masks = torch.from_numpy(masks).to(device)
            partner_obs = obs[rows, 1 - ego_idx]

            buffer.insert(
                obs[rows, ego_idx],
                ra2.cpu().numpy(), rc2.cpu().numpy(),
                actions.cpu().numpy(), log_probs.cpu().numpy(),
                values.cpu().numpy(), reward, masks, bad_masks)

        # ----------------------------------------------------- BOOTSTRAP
        with torch.no_grad():
            next_values = ego.get_values(
                torch.from_numpy(buffer.obs[-1]).to(device),
                torch.from_numpy(buffer.rnn_critic[-1]).to(device),
                torch.from_numpy(buffer.masks[-1]).to(device))
            next_values = value_norm.denormalize(next_values)
        buffer.compute_returns(next_values.cpu().numpy())

        # -------------------------------------------------------- UPDATE
        stats = ppo_update(ego, buffer, args, device, entropy_coef, value_norm)
        buffer.after_update()

        #the partner follows the ego, one soft step per episode. at tau=0.1 it
        #trails by roughly ten episodes; at tau=1.0 it simply becomes the ego.
        partner.sync(ego)

        # ---------------------------------------------------------- LOGS
        steps = (episode + 1) * T * E
        avg = float(np.mean(recent_returns[-50:])) if recent_returns else float("nan")
        writer.log({"episode": episode, "env_steps": steps,
                    "sparse_ret": None if np.isnan(avg) else round(avg, 4),
                    "value_loss": round(stats["value_loss"], 5),
                    "policy_loss": round(stats["policy_loss"], 6),
                    "entropy": round(stats["entropy"], 5),
                    "clip_frac": round(stats["clip_frac"], 5),
                    "shaping_coef": round(shaping_coef, 4),
                    "entropy_coef": round(entropy_coef, 5)})

        if episode % args.log_interval == 0:
            fps = int(steps / max(time.time() - start, 1e-6))
            print(f"ep {episode:5d}/{episodes} | steps {steps:>9,} | fps {fps:>5} "
                  f"| sparse_ret {avg:7.2f} | vloss {stats['value_loss']:8.3f} "
                  f"| ploss {stats['policy_loss']:7.4f} | ent {stats['entropy']:.3f} "
                  f"| clip {stats['clip_frac']:.3f} | shape_c {shaping_coef:.2f} "
                  f"| ent_c {entropy_coef:.3f}", flush=True)

        if episode % args.save_interval == 0 or episode == episodes - 1:
            writer.save_periodic(ego.actor, episode)

    writer.save_final(ego.actor, ego.critic, episodes - 1, value_norm)
    writer.close()
    print(f"[done] {time.time() - start:.0f}s  -> {args.save_dir}", flush=True)


if __name__ == "__main__":
    train(parse_args())
