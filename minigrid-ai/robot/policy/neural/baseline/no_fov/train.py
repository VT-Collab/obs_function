# ═══════════════════════════════════════════════════════════════════════════
# baseline/no_fov/ - the RAW-STATE baseline. See features.py for the design.
#
# THIS FILE - one entry point for all three methods.
#
#   --method ppo        NoFovAC    feed-forward conv actor-critic, torch_ac PPO
#   --method rec_ppo    NoFovRecAC conv + GRU, torch_ac PPO with BPTT
#   --method sac        DiscreteSAC off-policy, own loop + replay
#
#   python -m robot.policy.neural.baseline.no_fov.train --method rec_ppo --seed 1
#
# WHY A --seed FLAG, unlike static_fov/train.py. That script seeds nothing, so
# its "multiseed" checkpoints are not seeds at all - re-running the same command
# produces different weights and a specific checkpoint can never be regenerated.
# Measured consequence: retraining 6 checkpoints against a provably identical env
# moved every evaluation number by 1-4 points. Seed everything here.
#
# COMPARING THE THREE FAIRLY. PPO's natural budget is frames; SAC's is also
# frames but it takes many more gradient steps per frame. Hold ENVIRONMENT STEPS
# equal (--frames) and let each method use them as it likes - that is the honest
# axis, since env steps are the expensive resource and the thing a sample-
# efficiency claim is about.
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import argparse, os, random, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

import numpy as np
import torch
import torch_ac

from robot.policy.neural.baseline.no_fov.my_env_wrapper import NoFovAssistEnv
from robot.policy.neural.baseline.no_fov.actor_critic import NoFovAC, NoFovRecAC

# sac.py was removed from the tree. Import it lazily so --method ppo/rec_ppo
# still run: a hard import here made EVERY method die at import time with
# ModuleNotFoundError, including the two that never touch SAC. Restore sac.py
# and --method sac starts working again with no edit here.
try:
    from robot.policy.neural.baseline.no_fov.sac import DiscreteSAC, Replay
    HAVE_SAC = True
except ImportError:
    DiscreteSAC = Replay = None
    HAVE_SAC = False

CKPT_DIR = "robot/policy/neural/baseline/no_fov/checkpoints"


try:
    import wandb
    HAVE_WANDB = True
except Exception:
    HAVE_WANDB = False


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def make_env(args):
    """Build the env AND pin its RNG.

    seed_all() covers python/numpy/torch but NOT the env: gymnasium only reseeds
    self._np_random when reset() is called with seed != None, and otherwise draws
    from OS entropy. torch_ac calls env.reset() with no argument, so without this
    first seeded reset the layout/FOV stream differs every run and two runs with
    the same --seed diverge. Verified: without it, identical seeds gave different
    weights.
    """
    # env_wrapper
    # env = NoFovAssistEnv(comm_cost=args.comm_cost,
    #                      seeds=range(args.seed_start, args.seed_start + args.seed_count),
    #                      charge_effective_only=not args.charge_all,
    #                      shaping=args.shaping,
    #                      reveal_bonus=args.reveal_bonus)
    env = NoFovAssistEnv(comm_cost=args.comm_cost,
                         seeds=range(args.seed_start, args.seed_start + args.seed_count),
                         time_cost=args.time_cost,
                         key_bonus=args.key_bonus)
    env.reset(seed=args.seed)
    return env


def wb_init(args):
    if not (HAVE_WANDB and not args.no_wandb):
        return False
    wandb.init(project=args.project,
               name=args.run_name or f"nofov_{args.method}_s{args.seed}",
               config=vars(args), tags=["no_fov", args.method])
    return True


def wb_log(on, d, step):
    if on:
        wandb.log(d, step=step)


def train_ppo(args, recurrent: bool, on_wb: bool):
    env = make_env(args)
    model = NoFovRecAC() if recurrent else NoFovAC()
    algo = torch_ac.PPOAlgo(
        envs=[env], acmodel=model,
        num_frames_per_proc=args.frames_per_update,
        # BPTT window. Must be 1 for a non-recurrent model (torch_ac asserts it);
        # 8 gives the GRU eight steps of gradient without the cost of full-episode
        # backprop. batch_size must stay divisible by it.
        recurrence=8 if recurrent else 1,
        batch_size=64,
        entropy_coef=args.entropy_coef,
        # Feed-forward PPO diverged at the torch_ac default lr=1e-3: it found
        # positive advantage for speaking, eroded the WAIT bias, and ran away to
        # spamming (entropy 0.29 -> 1.09, success -> 0%). A lower lr keeps the
        # policy from over-committing to that early over-speaking signal. The GRU
        # variant is stable at the default, so this mainly matters for --ppo.
        lr=args.ppo_lr,
    )
    updates = args.frames // args.frames_per_update
    t0 = time.perf_counter()
    for u in range(updates):
        # NEW: two training schedules. Both only move dials - the env and the
        # reward function are untouched, and neither reads anything about the human.
        prog = u / max(1, updates - 1)          # 0.0 -> 1.0 across training

        # Speaking starts FREE so the policy can find out WHEN it helps, then the
        # bill ramps in. Without this, the certain -comm_cost per word beats the
        # delayed +1 and it goes mute by 15k frames.
        if args.comm_warmup_frac > 0:
            env.comm_cost = args.comm_cost * min(1.0, prog / args.comm_warmup_frac)

        # Start curious, end committed. Fixed 0.0 collapsed to mute; fixed 0.01
        # never left uniform.
        if args.entropy_start is not None:
            algo.entropy_coef = (args.entropy_start
                                 + (args.entropy_end - args.entropy_start) * prog)

        exps, logs = algo.collect_experiences()
        ul = algo.update_parameters(exps)
        r = logs["return_per_episode"]
        avg_return = sum(r) / len(r) if r else float("nan")
        frames = (u + 1) * args.frames_per_update
        wb_log(on_wb, {"frames": frames, "avg_return": avg_return,
                       "entropy": ul["entropy"], "policy_loss": ul["policy_loss"],
                       "value_loss": ul["value_loss"], "grad_norm": ul["grad_norm"]}, frames)
        if u % 10 == 0:
            print(f"update {u:4d}/{updates}  frames {frames:7d}  return {avg_return:+.3f}  "
                  f"entropy {ul['entropy']:.3f}  ploss {ul['policy_loss']:+.4f}  "
                  f"cc {env.comm_cost:.4f}  ec {algo.entropy_coef:.4f}  "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
    return model.state_dict()


def train_sac(args, on_wb: bool):
    env = make_env(args)
    agent = DiscreteSAC(lr=args.lr, target_entropy_ratio=args.sac_target_entropy,
                        alpha_init=args.sac_alpha)
    buf = Replay(args.buffer)
    obs, _ = env.reset()
    ep_ret, rets, logs = 0.0, [], {}
    t0 = time.perf_counter()
    for step in range(args.frames):
        # Uniform random until the buffer has something to learn from - standard
        # SAC warm-up; acting on an untrained actor just wastes early transitions.
        if step < args.start_steps:
            a = int(np.random.randint(env.action_space.n))   # no mask - all 6 always
        else:
            a = agent.act(obs)
        nobs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, nobs, float(term))   # `term` only: a timeout is not a
        obs = nobs                              # real terminal state for bootstrapping
        ep_ret += r
        if term or trunc:
            rets.append(ep_ret); ep_ret = 0.0
            obs, _ = env.reset()
        # One gradient update every update_every env steps. The conv encoder
        # makes each update ~70ms on CPU, so 1-per-4-steps keeps an 80k-frame run
        # near ~25 min instead of ~90; still ample replay reuse for off-policy.
        if step >= args.start_steps and step % args.update_every == 0:
            for _ in range(args.updates_per_step):
                logs = agent.update(buf, batch=args.batch)
        if step % args.log_every == 0 and logs:
            avg_return = float(np.mean(rets[-30:])) if rets else float("nan")
            wb_log(on_wb, {"frames": step, "avg_return": avg_return,
                           "episodes": len(rets), **logs}, step)
            print(f"step {step:7d}/{args.frames}  return {avg_return:+.3f}  "
                  f"alpha {logs['alpha']:.3f}  H {logs['entropy']:.3f}  "
                  f"lossQ {logs['loss_q']:.3f}  [{time.perf_counter()-t0:.0f}s]", flush=True)
    return agent.state_dict()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["ppo", "rec_ppo", "sac"], required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--frames", type=int, default=256_000, help="ENV STEPS - held equal across methods")
    # comm_cost and time_cost are deliberately EQUAL by default, which gives them
    # a single meaning: one word costs exactly one step of the human's time, so a
    # reveal pays for itself iff it saves them at least one step.
    #
    # Measured over 80 episodes, mute robot vs the deterministic StaticAssist:
    #     MUTE    success 0.662   mean steps 133.9 (sd 55.9)   words 0.0
    #     ASSIST  success 0.812   mean steps 115.8             words 6.3
    # so good assistance buys +0.150 success and saves 18.1 steps for 6.3 words.
    # signal(assist - mute) = 0.150 + 11.8*c, POSITIVE FOR EVERY c: once time is
    # charged there is no break-even and silence is never optimal. (The old
    # c~0.028 break-even applied when words cost and time was free. Ignore it.)
    #
    # 0.005 is chosen because three things line up there:
    #   - signal/noise peaks (0.380, flat over 0.005-0.007, falling both sides)
    #   - a mute robot scores 0.662 - 0.005*133.9 = -0.008, i.e. ~0.000, so the
    #     SIGN of the return tells you whether the policy beat silence
    #   - 1/c = 200 > max_steps 190, so succeeding always outweighs being fast
    # Corollary worth knowing: +11.8 = 18.1 steps saved - 6.3 words spent, so the
    # policy has a budget of ~18 words before speaking stops paying for itself.
    p.add_argument("--comm-cost", type=float, default=0.005,
                   help="cost per spoken word. Keep equal to --time-cost.")
    p.add_argument("--time-cost", type=float, default=0.005,
                   help="cost per env step. Keep equal to --comm-cost; see the note above "
                        "for why 0.005 and not the 0.015 that was in use before.")
    p.add_argument("--key-bonus", type=float, default=0.15,
                   help="one-shot reward when the human picks up the key that opens the "
                        "goal room. Pure grid geometry, no FOV. 0 disables.")
    # p.add_argument("--shaping", type=float, default=0.0,
    #                help="potential-based shaping scale. NOTE: PHI tracks the HUMAN's "
    #                     "progress, which the robot barely controls, so large values inject "
    #                     "variance that drowns the speaking signal - 1.0 left PPO at entropy "
    #                     "1.69/1.79 (near-uniform) after 149k frames. 0 or ~0.2.")
    # p.add_argument("--reveal-bonus", type=float, default=0.15,
    #                help="immediate reward when an informative reveal unblocks a stuck human. "
    #                     "Credits the speech act directly - the fix for PPO staying flat 1M "
    #                     "frames under the distant +1 goal reward. 0 disables.")
    # p.add_argument("--charge-all", action="store_true",
    #                help="bill every utterance, not just informative ones (the old behaviour)")
    p.add_argument("--entropy-coef", type=float, default=0.0)   # ppo/rec_ppo
    # NEW: training schedules. Both default to OFF, so omitting them reproduces
    # the old behaviour exactly.
    p.add_argument("--comm-warmup-frac", type=float, default=0.0,
                   help="ramp comm_cost 0 -> --comm-cost over this fraction of training. "
                        "0.3 works; 0 disables.")
    p.add_argument("--entropy-start", type=float, default=None,
                   help="anneal policy entropy bonus from here to --entropy-end. "
                        "Overrides --entropy-coef. Try 0.01. None disables.")
    p.add_argument("--entropy-end", type=float, default=0.0005)
    p.add_argument("--ppo-lr", type=float, default=1e-3,
                   help="torch_ac PPO lr. Lower (3e-4) stabilises feed-forward --ppo, "
                        "which diverges at the default.")
    p.add_argument("--sac-target-entropy", type=float, default=0.0,
                   help="SAC target entropy as a fraction of log|A|. <=0 => FIXED alpha "
                        "(near-greedy on Q), which is what actually works here: auto-tuning "
                        "drives the policy to uniform because per-step action-values are "
                        "nearly flat. >0 re-enables Haarnoja auto-tuning.")
    p.add_argument("--sac-alpha", type=float, default=0.03,
                   help="fixed SAC temperature when --sac-target-entropy<=0. Small = "
                        "near-greedy with a whisper of exploration.")
    p.add_argument("--frames-per-update", type=int, default=512)   # ppo
    p.add_argument("--lr", type=float, default=3e-4)               # sac
    p.add_argument("--buffer", type=int, default=100_000)          # sac
    p.add_argument("--batch", type=int, default=64)               # sac
    p.add_argument("--start-steps", type=int, default=5_000)       # sac
    p.add_argument("--update-every", type=int, default=4)          # sac
    p.add_argument("--updates-per-step", type=int, default=1)      # sac
    p.add_argument("--seed-start", type=int, default=1000,
                   help="training layouts - MUST stay clear of eval seeds 0-49")
    p.add_argument("--seed-count", type=int, default=20_000)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--log-every", type=int, default=2000, help="sac: env steps between logs")
    p.add_argument("--project", type=str, default="minigrid_no_fov_baselines")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    if args.method == "sac" and not HAVE_SAC:
        p.error("--method sac needs robot/policy/neural/baseline/no_fov/sac.py, which is not "
                "in the tree. Restore that file, or use --method ppo / rec_ppo.")

    seed_all(args.seed)
    on_wb = wb_init(args)
    sd = (train_sac(args, on_wb) if args.method == "sac"
          else train_ppo(args, recurrent=args.method == "rec_ppo", on_wb=on_wb))

    out = args.save or os.path.join(CKPT_DIR, args.method, f"model_s{args.seed}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(sd, out)
    print(f"saved -> {out}")
    if on_wb:
        wandb.save(out); wandb.finish()


if __name__ == "__main__":
    main()
