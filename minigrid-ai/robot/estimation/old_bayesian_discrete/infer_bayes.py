# infer_bayes.py
"""
Run 3 MiniGrid episodes. At each step:
  - vectorize current state (state-only)
  - compute Bayes likelihood via kNN over collected CSVs (with 'fov' labels)
  - update prior with bayes_inference (same as Overcooked)
  - set estimated_fov = argmax(prior)
  - call human.estimated_ml_action(state) after human.set_estimated_fov(estimated_fov)
  - log extras into CSV: agent_action, estimated_subtask, estimated_fov, bayes_90, bayes_120, bayes_179
Simple and direct.
"""

import sys, time
from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd

# project imports
sys.path.append(str(Path(__file__).resolve().parents[2]))
from human.agents.mini_agent import limitVisionHumanModel
from human.planning.mini_planning import MotionPlanner
from robot.methods.old_bayesian_discrete.collect_data import build_step_record  # we will write our own simple writer here
from robot.methods.old_bayesian_discrete.dataset import flatten_record
from robot.methods.old_bayesian_discrete.bayes_inference import bayes_inference, process_bayes_knn

# ---- config ----
ENV_ID = "MiniGrid-LockedRoom-v0"
TRUE_FOV = 90     # the ground-truth FOV for running these 3 episodes
SEEDS = [100, 101, 102]


REPO_ROOT = Path(__file__).resolve().parents[2]   # .../minigrid-ai
def P(*parts): return REPO_ROOT.joinpath(*parts)

# collected datasets per FOV (must exist and include all base columns)
# Add a 'fov' column after load. Keep paths simple; edit to yours.
# collected datasets per FOV
COLLECTED_CSVS = {
    90:  P("scripts", "runs", "lockedroom_success_all_90.csv"),
    120: P("scripts", "runs", "lockedroom_success_all_120.csv"),
    179: P("scripts", "runs", "lockedroom_success_all_179.csv"),
}

# where to save your output
OUT_CSV = P("runs", f"fov{TRUE_FOV}_minigrid_bayes_outputs.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---- helpers ----
def load_collected_with_fov() -> pd.DataFrame:
    frames = []
    for fov, path in COLLECTED_CSVS.items():
        df = pd.read_csv(path)
        df["fov"] = int(fov)
        # build obs_vec from row using dataset.flatten_record
        df["obs_vec"] = df.apply(lambda r: flatten_record(r.to_dict()), axis=1)
        frames.append(df[["fov", "obs_vec"]])
    return pd.concat(frames, ignore_index=True)

def write_episode_csv_with_bayes(records, filepath: str, append: bool = False):
    """
    Records are dicts from build_step_record(...) with extra keys:
      - agent_action (int)
      - estimated_subtask (str)
      - estimated_fov (int)
      - bayes_90, bayes_120, bayes_179 (floats)
    Write a header once; then append rows.
    """
    import csv, os
    # build header = original + extras (exact order)
    base_header = [
        "episode_seed","timestep",
        "agent_pos_x","agent_pos_y",
        "agent_orientation_x","agent_orientation_y",
    ]
    # agent carry one-hot
    color_names = ["blue","green","grey","purple","red","yellow"]
    base_header += [f"agent_carry_color_{c}" for c in color_names]
    base_header += ["agent_subtask"]

    # doors
    for i in range(1, 6+1):
        base_header += [f"door_{i}_x", f"door_{i}_y", f"door_{i}_state"]
        base_header += [f"door_{i}_color_{c}" for c in color_names]
    # keys
    for i in range(1, 3+1):
        base_header += [f"key_{i}_x", f"key_{i}_y"]
        base_header += [f"key_{i}_color_{c}" for c in color_names]
    # goal
    base_header += ["goal_x","goal_y"]

    # extras
    extras = ["agent_action", "estimated_subtask", "estimated_fov", "bayes_90", "bayes_120", "bayes_179"]
    header = base_header + extras

    parent = Path(filepath).parent
    parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    write_header = True
    if append and Path(filepath).exists():
        write_header = False

    with open(filepath, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for rec in records:
            row = [rec[k] for k in base_header] + [rec[e] for e in extras]
            w.writerow(row)

# ---- main run loop ----
def run_one_episode(seed: int, collected_df: pd.DataFrame):
    env = gym.make(ENV_ID, render_mode=None)
    env.reset(seed=seed)
    env.unwrapped.agent_fov = TRUE_FOV

    human = limitVisionHumanModel(fov=TRUE_FOV)
    human.init_knowledge_base(env.unwrapped)

    planner = MotionPlanner()

    done = False
    t = 0
    rew = 0.0
    last_info = {}
    episode_records = []

    # uniform prior over [90,120,179]
    prior = np.array([1/3, 1/3, 1/3], dtype=float)

    try:
        while not done:
            state = env.unwrapped

            # actual subtask (with true fov in the model)
            subtask = human.ml_action(state)

            # ---- Build record BEFORE stepping ----
            step_dict = build_step_record(
                state,                   # same env.unwrapped
                episode_seed=seed,
                timestep=t,
                agent_subtask=subtask,    # matches current state
            )

            # ---- Bayes update (state-only) ----
            obs_vec_now = flatten_record(step_dict)
            evidence, likelihood = process_bayes_knn(
                obs_world_vec=obs_vec_now,
                collected_data=collected_df,
                prior_len=3,
                k=50,
                use_cosine=True,
                eps=1e-6
            )
            prior = bayes_inference(prior, evidence, likelihood)

            # MAP fov and estimated subtask (same state)
            fov_values = (90, 120, 179)
            est_fov = fov_values[int(np.argmax(prior))]
            human.set_estimated_fov(est_fov)
            est_subtask = human.estimated_ml_action(state)

            # ---- Plan and take the next action ----
            action = planner.next_action(subtask, state, human.knowledge_base)
            if action is None:
                action = env.action_space.sample()

            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_info = info if isinstance(info, dict) else {}

            # ---- Attach extras & store ----
            step_dict["agent_action"] = int(action)
            step_dict["estimated_subtask"] = est_subtask
            step_dict["estimated_fov"] = int(est_fov)
            step_dict["bayes_90"], step_dict["bayes_120"], step_dict["bayes_179"] = [float(x) for x in prior.tolist()]

            episode_records.append(step_dict)
            t += 1


    finally:
        env.close()

    succeeded = (rew != 0.0) or bool(last_info.get("success", False))
    return rew, succeeded, t, episode_records


def main():
    collected_df = load_collected_with_fov()

    # fresh output
    Path(OUT_CSV).unlink(missing_ok=True)

    for i, seed in enumerate(SEEDS):
        rew, ok, steps, recs = run_one_episode(seed, collected_df)
        print(f"[seed={seed}] steps={steps} reward={rew:.3f} success={bool(ok)}")
        if recs:
            write_episode_csv_with_bayes(recs, OUT_CSV, append=(i > 0))
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
