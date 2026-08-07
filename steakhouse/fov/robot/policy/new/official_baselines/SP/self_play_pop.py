"""
SELF-PLAY, POPULATION EDITION.  The ZSC-Eval-parity re-run of self_play.py.

`self_play.py` is untouched and still produces the validated seed-1 baselines.
This file is what the multi-seed baseline table and the FCP stage-1 pool are
built from. Four differences, and nothing else:

    1. CHECKPOINT HISTORY.  actor_periodic_<ep>.pt every save_interval, a NEW
       file each time. self_play.py rewrote one filename, so a finished run
       left only final weights -- and FCP's population is init/mid/final per
       seed, which cannot be recovered after the fact. utils/ckpt.py.

    2. PROGRESS LOG.  progress.jsonl, one line per episode. ZSC picks the "mid"
       checkpoint as the one SCORING half the final score, not the one halfway
       through the clock -- and the learning curves here inflect anywhere from
       episode 80 (gc00) to episode 460 (none_3), so the two rules disagree
       wildly. Needs a recorded score per save. utils/ckpt.py.

    3. VALUENORM, ON BY DEFAULT.  The last algorithmic gap vs ZSC-Eval. Return
       magnitude grows ~20x over a run here (vloss 0.68 -> 9.35 on mid_1, 13-26
       on the winners), which is exactly the regime it exists for.
       --no_valuenorm reproduces the old behaviour exactly. utils/valuenorm.py.

    4. SEEDS ARE FIRST-CLASS.  --seed is now part of the run identity and the
       save path, because eleven of them get run per layout.

Everything else -- the PPO update, GAE, the entropy and shaping anneals, the
observation, the dense shaper, the network -- is the same code on the same
defaults. That is deliberate: the point of this run is variance and a pool, not
a new algorithm.

    python -m fov.robot.policy.new.official_baselines.SP.self_play_pop \\
        --layouts steak_gc00 --seed 5 --save_dir /scratch1/$USER/...

===========================================================================
SEEDS DO NOT COMBINE
===========================================================================
Eleven seeds is eleven independent policies, never one merged policy. Weights
from separately-initialized runs are not averageable -- there is no reason the
two networks put the same concept in the same unit. What eleven seeds buys is a
DISTRIBUTION: "gc00 scores 54.2 +/- 6.1" instead of "gc00 scored 60 once".
ZSC-Eval runs seeds 5..15 for exactly this and reports mean +/- std.

The one place separate seeds are used together is a POPULATION (FCP, MEP,
TrajeDi), and even there they stay distinct frozen networks that an ego agent
is paired with one at a time. Still no merging.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

#so `python SP/self_play_pop.py` finds algorithm/ and utils/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.rMAPPOPolicy import R_MAPPOPolicy
from utils.buffer import SelfPlayBuffer
from utils.ckpt import RunWriter
from utils.env_wrapper import VecSteakEnv
from utils.schedules import add_schedule_args, piecewise
from utils.valuenorm import IdentityValueNorm, ValueNorm

#the distribution hook lives in self_play.py and stays there -- one definition,
#so a filter built on it cannot drift from what this trains
from SP.self_play import action_probs  # noqa: F401


def parse_args(argv=None):
    p = argparse.ArgumentParser("steakhouse self-play PPO (population edition)")
    # env
    p.add_argument("--layouts", type=str, default="steak_gc00",
                   help="comma-separated layout names. >1 trains ONE policy "
                        "across all of them on a padded canvas -- known to fail, "
                        "see CARC_RUNS.md section 8. Pass one layout.")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--n_rollout_threads", type=int, default=50)
    p.add_argument("--episode_length", type=int, default=400)
    p.add_argument("--n_orders", type=int, default=4)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--num_env_steps", type=int, default=int(1e7))
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
    #piecewise entropy anneal. defaults reproduce self_play.py's single line
    #(0.2 -> 0.01 over 1e7). ZSC-Eval's two-segment version is one flag away:
    #    --entropy_coefs 0.2 0.05 0.01 --entropy_coef_horizons 0 5e6 1e7
    add_schedule_args(p)
    p.add_argument("--use_huber_loss", action="store_true")
    p.add_argument("--huber_delta", type=float, default=10.0)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--use_linear_lr_decay", action="store_true")
    p.add_argument("--reward_shaping_horizon", type=float, default=1e8)
    #ON by default -- this is the ZSC-Eval default and the gap this run closes.
    #--no_valuenorm restores self_play.py's exact behaviour for the ablation.
    p.add_argument("--use_valuenorm", action="store_true", default=True)
    p.add_argument("--no_valuenorm", dest="use_valuenorm", action="store_false")
    # misc
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--save_interval", type=int, default=25)
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args(argv)


def ppo_update(policy, buffer, args, device, entropy_coef, value_norm):
    """One full pass: ppo_epoch sweeps over num_mini_batch minibatches.

    Identical to self_play.py's, plus the value-normalization arithmetic.

    WHERE VALUENORM SITS.  The critic's raw output lives in NORMALIZED space.
    The buffer holds real-scale values and returns (the runner de-normalizes
    before storing, so buffer.py's GAE never has to know), so the value loss
    has to bring the target back into the critic's space:

        value_norm.update(returns)            fold this batch into the stats
        target      = normalize(returns)      real -> normalized
        pred_old    = normalize(value_preds)  real -> normalized
        loss = clipped_huber(values, target, around=pred_old)

    With --no_valuenorm all three calls are the identity and this is literally
    the old arithmetic, which is what keeps the two configurations comparable.
    """
    stats = dict(value_loss=0.0, policy_loss=0.0, entropy=0.0, clip_frac=0.0, n=0)

    for _ in range(args.ppo_epoch):
        for batch in buffer.recurrent_generator(args.num_mini_batch, device,
                                                args.data_chunk_length):

            values, log_probs, entropy = policy.evaluate_actions(
                batch["obs"], batch["obs"],          # share_obs == obs for now
                batch["rnn_actor"], batch["rnn_critic"],
                batch["actions"], batch["masks"])

            # ------------------------------------------- actor
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

            # ------------------------------------------- critic
            #fold the real returns into the running stats BEFORE normalizing,
            #matching zsceval r_mappo.cal_value_loss
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
          f"run={run_name} seed={args.seed} valuenorm={args.use_valuenorm}", flush=True)

    policy = R_MAPPOPolicy(args, envs.obs_shape, envs.obs_shape,
                           envs.n_actions, device)
    buffer = SelfPlayBuffer(T, N, envs.obs_shape, args.recurrent_N,
                            args.hidden_size, args.gamma, args.gae_lambda)
    value_norm = ValueNorm(1, device=device) if args.use_valuenorm else IdentityValueNorm()

    writer = RunWriter(args.save_dir, run_name, args)

    obs = envs.reset()                                   # (E, 2, 23, W, H)
    buffer.reset(obs.reshape(N, *envs.obs_shape))

    episodes = max(args.num_env_steps // (T * E), 1)
    recent_returns, per_layout, start = [], {}, time.time()

    for episode in range(episodes):
        if args.use_linear_lr_decay:
            policy.lr_decay(episode, episodes)

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

                values, actions, log_probs, ra2, rc2 = policy.get_actions(
                    o, o, ra, rc, mk)
                #the critic speaks in normalized units; the buffer and its GAE
                #speak in real ones. convert here, once, so buffer.py is
                #untouched and identical to the validated runs
                values = value_norm.denormalize(values)

            act_np = actions.cpu().numpy().reshape(E, A)

            obs, sparse, shaped, dones, truncs, infos = envs.step(act_np)

            for info in infos:
                if "episode_sparse" in info:
                    recent_returns.append(info["episode_sparse"])
                    per_layout.setdefault(info["layout"], []).append(
                        info["episode_sparse"])

            reward = np.repeat(sparse + shaping_coef * shaped, A).reshape(N, 1)
            masks = np.repeat(1.0 - dones.astype(np.float32), A).reshape(N, 1)
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
            next_values = value_norm.denormalize(next_values)
        buffer.compute_returns(next_values.cpu().numpy())

        # -------------------------------------------------------- UPDATE
        stats = ppo_update(policy, buffer, args, device, entropy_coef, value_norm)
        buffer.after_update()

        # ---------------------------------------------------------- LOGS
        steps = (episode + 1) * T * E
        avg = float(np.mean(recent_returns[-50:])) if recent_returns else float("nan")

        #EVERY episode, not every log_interval. the mid-checkpoint rule matches
        #on score, so the finer this curve is the better that match lands --
        #and 500 json lines cost nothing
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
                  f"| ent_c {entropy_coef:.3f}",
                  flush=True)
            if len(envs.layout_names) > 1 and per_layout:
                brk = " ".join(f"{k.replace('steak_',''):s}:{np.mean(v[-20:]):.1f}"
                               for k, v in sorted(per_layout.items()))
                print(f"        per-layout: {brk}", flush=True)

        # ---------------------------------------------------- CHECKPOINTS
        #a NEW file every time. this is the whole difference from self_play.py
        if episode % args.save_interval == 0 or episode == episodes - 1:
            writer.save_periodic(policy.actor, episode)

    writer.save_final(policy.actor, policy.critic, episodes - 1, value_norm)
    writer.close()
    print(f"[done] {time.time() - start:.0f}s  -> {args.save_dir}", flush=True)


if __name__ == "__main__":
    train(parse_args())
