"""
collect_bayes_traj.py

Run episodes with BayesHumanAgent + BayesianPlanner.

The true FOV is sampled from the prior P(θ) = pick_fov(CANDIDATE_FOVS) each
episode — consistent with the generative model:

    θ      ~ P(θ)                              (sampled here via pick_fov)
    τ_t    ~ P_τ(τ_t | τ_{t-1}, s_{1:t}, θ)   (bayes_agent.py)
    a^H_t  ~ π^H(a^H_t | s_{1:t}, θ, τ_t)     (bayes_planner.py)

Usage:
    python scripts/collect_bayes_traj.py --seed 0         # 1 episode, FOV drawn from prior
    python scripts/collect_bayes_traj.py --seeds 0 1 2    # 3 episodes
    python scripts/collect_bayes_traj.py --no-render --seeds 0 1 2 3 4
"""

import argparse
import time
from pathlib import Path
import sys

import gymnasium as gym

sys.path.append(str(Path(__file__).resolve().parents[1]))

from human.agents.bayes_agent import BayesHumanAgent, pick_fov, initial_subtask_prior
from human.planning.bayes_planner import BayesianPlanner
from scripts.collect_data import build_step_record, write_episode_csv

ENV_ID          = "MiniGrid-LockedRoom-v0"
CANDIDATE_FOVS  = [60, 120, 180]   # the enumerable set P(θ) is defined over
ACTION_NAMES    = {0:"LEFT", 1:"RIGHT", 2:"FWD", 3:"PICKUP", 4:"DROP", 5:"TOGGLE", 6:"DONE"}


def run_episode(seed: int, render: bool = True, delay: float = 0.08, fixed_fov: int = None):
    # θ ~ P(θ): sample the true FOV from the prior, just like the generative model
    true_fov = fixed_fov if fixed_fov is not None else pick_fov(CANDIDATE_FOVS)

    env = gym.make(ENV_ID, render_mode="human" if render else None)
    env.reset(seed=seed)
    env.unwrapped.agent_fov = true_fov   # set on env so cone renders correctly

    state = env.unwrapped

    agent   = BayesHumanAgent(fov=true_fov)
    planner = BayesianPlanner()
    agent.init_knowledge_base(state)

    print(f"\n=== seed={seed}  true_fov={true_fov} (sampled from prior) ===")
    print(f"Mission: {state.mission}")

    # Show the initial subtask distribution P(τ_1) as a sanity check
    isp = initial_subtask_prior()
    top = sorted(isp.items(), key=lambda x: -x[1])[:3]
    print(f"P(τ_1) top-3: {[(t, round(p,3)) for t,p in top]}")
    print(f"{'t':>4}  {'subtask':<22}  {'action':<8}  note")
    print("─" * 58)

    done       = False
    t          = 0
    rew        = 0.0
    records    = []
    prev_sub   = agent.current_subtask

    while not done:
        state = env.unwrapped

        subtask = agent.select_subtask(state)
        action  = planner.next_action(subtask, state, agent.knowledge_base)
        if action is None:
            action = 2  # FWD fallback

        note = "← changed" if subtask != prev_sub else ""
        print(f"{t:>4}  {subtask:<22}  {ACTION_NAMES.get(action, str(action)):<8}  {note}")
        prev_sub = subtask

        rec = build_step_record(
            state,
            episode_seed=seed,
            timestep=t,
            agent_subtask=subtask,
            agent_action=action,
        )
        records.append(rec)

        _, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        t += 1

        if render:
            time.sleep(delay)

    env.close()

    succeeded = rew != 0.0 or bool((info or {}).get("success", False))
    print("─" * 58)
    print(f"steps={t}  reward={rew:.3f}  success={succeeded}  fov={true_fov}")
    return rew, succeeded, t, true_fov, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--seeds",     type=int,   nargs="+", default=None,
                        help="Run multiple seeds (overrides --seed)")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument("--delay",     type=float, default=0.08,
                        help="Seconds between rendered frames")
    parser.add_argument("--out",       type=str,   default=None,
                        help="Optional CSV output path")
    parser.add_argument("--fov",       type=int,   default=None,
                        help="Fix FOV instead of sampling from prior (for testing)")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]

    total, successes = 0, 0
    fovs_seen = []
    all_records = []

    for seed in seeds:
        rew, ok, _, fov, recs = run_episode(
            seed=seed, render=args.render, delay=args.delay, fixed_fov=args.fov,
        )
        total += 1
        successes += int(ok)
        fovs_seen.append(fov)
        all_records.extend(recs)

    print(f"\n=== Batch summary: {successes}/{total} succeeded ===")
    print(f"FOVs drawn from prior: {fovs_seen}")

    if args.out:
        write_episode_csv(all_records, args.out)
        print(f"Wrote {len(all_records)} rows → {args.out}")


if __name__ == "__main__":
    main()