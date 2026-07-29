# dataset.py
"""
MiniGrid dataset helpers: build state-only 1D vectors from CSV rows.
Columns included (state-only):
  - agent_pos_x, agent_pos_y
  - agent_orientation_x, agent_orientation_y
  - agent_carry_color_{blue,green,grey,purple,red,yellow}
  - door_i_x, door_i_y, door_i_state, door_i_color_{...}  for i=1..num_doors
  - key_i_x, key_i_y, key_i_color_{...}                   for i=1..num_keys
  - goal_x, goal_y
"""

from typing import Dict, List, Sequence
import numpy as np

COLOR_NAMES = ["blue", "green", "grey", "purple", "red", "yellow"]

def header_state_only(num_doors: int = 6, num_keys: int = 3) -> List[str]:
    cols = [
        "agent_pos_x", "agent_pos_y",
        "agent_orientation_x", "agent_orientation_y",
    ]
    cols += [f"agent_carry_color_{c}" for c in COLOR_NAMES]

    for i in range(1, num_doors + 1):
        cols += [f"door_{i}_x", f"door_{i}_y", f"door_{i}_state"]
        cols += [f"door_{i}_color_{c}" for c in COLOR_NAMES]

    for i in range(1, num_keys + 1):
        cols += [f"key_{i}_x", f"key_{i}_y"]
        cols += [f"key_{i}_color_{c}" for c in COLOR_NAMES]

    cols += ["goal_x", "goal_y"]
    return cols

def flatten_record(rec: Dict, num_doors: int = 6, num_keys: int = 3) -> np.ndarray:
    """
    rec: ONE CSV row as a dict-like (e.g., df.iloc[i].to_dict()).
    Returns a 1D numpy array in the fixed order given by header_state_only(...).
    """
    cols = header_state_only(num_doors=num_doors, num_keys=num_keys)
    vec = [float(rec[c]) for c in cols]
    return np.asarray(vec, dtype=float)

def flatten_episode(records: Sequence[Dict], num_doors: int = 6, num_keys: int = 3) -> np.ndarray:
    """
    records: list of per-step dicts (already in MiniGrid CSV format).
    Returns (T, D) array where each row is flatten_record(rec).
    """
    if not records:
        return np.zeros((0, len(header_state_only(num_doors, num_keys))), dtype=float)
    rows = [flatten_record(rec, num_doors=num_doors, num_keys=num_keys) for rec in records]
    return np.stack(rows, axis=0)
