"""
    collect_traj.py
    
    Run seeds between 0-100 and Only record successfull episodes.
    Then turn episodes of minigrid playing into csv data

"""


# scripts/100mini.py
# Usage: python scripts/100mini.py

import time
from pathlib import Path
import sys
import gymnasium as gym

# Ensure project root is on path (so imports below work when run as a script)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from human.agents.mini_agent import limitVisionHumanModel
from human.planning.mini_planning import MotionPlanner
from scripts.collect_data import build_step_record, write_episode_csv

ENV_ID = "MiniGrid-LockedRoom-v0"
FOV_DEG = 179
SEED_START = 0
SEED_END = 100
OUT_CSV = "runs/lockedroom_success_all.csv"
RENDER = True   # set True to watch a single episode in real time


def run_one_episode(seed: int):
    """
    Runs ONE episode. Returns (reward, succeeded_bool, steps, episode_records).
    """
    env = gym.make(ENV_ID, render_mode="human" if RENDER else None)
    env.reset(seed=seed)
    env.unwrapped.agent_fov = FOV_DEG

    human = limitVisionHumanModel(fov=FOV_DEG)
    human.init_knowledge_base(env.unwrapped)

    planner = MotionPlanner()

    done = False
    t = 0
    rew = 0.0
    terminated = False
    truncated = False
    last_info = {}

    episode_records = []

    try:
        while not done:
            state = env.unwrapped

            subtask = human.ml_action(state)

            action = planner.next_action(subtask, state, human.knowledge_base)
            if action is None:
                action = env.action_space.sample()

            obs, rew, terminated, truncated, info = env.step(action)
            last_info = info if isinstance(info, dict) else {}
            done = terminated or truncated

            # collect per-step dict (ints except 'agent_subtask')
            step_dict = build_step_record(
                env.unwrapped,
                episode_seed=seed,
                timestep=t,
                agent_subtask=subtask,
                agent_action=action,
            )
            episode_records.append(step_dict)

            t += 1

    finally:
        env.close()

    succeeded = (rew != 0.0) or bool(last_info.get("success", False))
    return rew, succeeded, t, episode_records


def run_batch():
    total = 0
    successes = 0
    rewards = []
    steps_list = []

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    for s in range(SEED_START, SEED_END + 1):
        rew, ok, steps, recs = run_one_episode(seed=s)
        total += 1
        successes += int(ok)
        rewards.append(rew)
        steps_list.append(steps)

        # per-episode line
        print(f"[seed={s:03d}] steps={steps:4d}  reward={rew:.3f}  success={bool(ok)}")

        # append only successful episodes
        if ok and recs:
            write_episode_csv(recs, OUT_CSV, append=True)

    success_rate = successes / total if total else 0.0
    avg_rew = sum(rewards) / total if total else 0.0
    avg_steps = sum(steps_list) / total if total else 0.0

    print(f"\n=== Batch Summary (seeds {SEED_START}-{SEED_END}, FOV={FOV_DEG}) ===")
    print(f"Total episodes:   {total}")
    print(f"Successes:        {successes}")
    print(f"Success rate:     {success_rate:.3f}")
    print(f"Avg reward:       {avg_rew:.3f}")
    print(f"Avg steps:        {avg_steps:.1f}")
    print(f"Wrote successful episodes to: {OUT_CSV}")


if __name__ == "__main__":
    run_batch()
