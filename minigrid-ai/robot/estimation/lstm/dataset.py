"""
generate 1D vector from the given data
"""

# utils/vectorize_data.py
from __future__ import annotations
from typing import Dict, List, Sequence
import numpy as np
from minigrid.core.constants import COLOR_NAMES

def _door_block(rec: Dict, i: int) -> List[int]:
    """door i (1-based): x, y, state, color one-hot(6) -> 9 ints."""
    out = [
        int(rec[f"door_{i}_x"]),
        int(rec[f"door_{i}_y"]),
        int(rec[f"door_{i}_state"]),
    ]
    out += [int(rec[f"door_{i}_color_{name}"]) for name in COLOR_NAMES]
    return out

def _key_block(rec: Dict, i: int) -> List[int]:
    """key i (1-based): x, y, color one-hot(6) -> 8 ints."""
    out = [
        int(rec[f"key_{i}_x"]),
        int(rec[f"key_{i}_y"]),
    ]
    out += [int(rec[f"key_{i}_color_{name}"]) for name in COLOR_NAMES]
    return out

def expected_dim(
    *,
    num_doors: int = 6,
    num_keys: int = 3,
) -> int:
    """
    Length of the flattened vector (ints only), with NO subtask included.
    """
    d = 0
    d += 2  # episode_seed, timestep
    d += 4  # agent_pos_x, agent_pos_y, agent_orientation_x, agent_orientation_y
    d += len(COLOR_NAMES)  # agent_carry one-hot (6)
    d += num_doors * (3 + len(COLOR_NAMES))  # each door: x,y,state + 6-color one-hot = 9
    d += num_keys * (2 + len(COLOR_NAMES))   # each key : x,y + 6-color one-hot = 8
    d += 2  # goal_x, goal_y
    return d  # default 92

def flatten_record(
    rec: Dict,
    *,
    num_doors: int = 6,
    num_keys: int = 3,
) -> np.ndarray:
    """
    Turn ONE timestep dict into a 1D int vector with fixed order, skipping agent_subtask.
    Order mirrors your CSV except the subtask column is omitted.
    """
    vec: List[int] = [
        int(rec["episode_seed"]),
        int(rec["timestep"]),
        int(rec["agent_pos_x"]),
        int(rec["agent_pos_y"]),
        int(rec["agent_orientation_x"]),
        int(rec["agent_orientation_y"]),
        #int(rec["agent_action"]),
    ]

    # agent carrying color one-hot (6)
    vec += [int(rec[f"agent_carry_color_{name}"]) for name in COLOR_NAMES]

    # doors (exactly num_doors)
    for i in range(1, num_doors + 1):
        vec += _door_block(rec, i)

    # keys (exactly num_keys)
    for i in range(1, num_keys + 1):
        vec += _key_block(rec, i)

    # goal
    vec += [int(rec["goal_x"]), int(rec["goal_y"])]

    arr = np.asarray(vec, dtype=int)
    exp = expected_dim(num_doors=num_doors, num_keys=num_keys)
    if arr.size != exp:
        raise ValueError(f"flatten_record length {arr.size} != expected {exp}")
    return arr

def flatten_episode(
    records: Sequence[Dict],
    *,
    num_doors: int = 6,
    num_keys: int = 3,
) -> np.ndarray:
    """
    Vectorize a whole episode (list of step dicts) -> shape (T, D) int array.
    """
    if not records:
        return np.zeros((0, expected_dim(num_doors=num_doors, num_keys=num_keys)), dtype=int)
    rows = [flatten_record(rec, num_doors=num_doors, num_keys=num_keys) for rec in records]
    return np.stack(rows, axis=0)

def header_for_numpy(
    *,
    num_doors: int = 6,
    num_keys: int = 3,
) -> List[str]:
    """
    Optional: return names matching the flattening order (useful for debugging).
    This omits agent_subtask on purpose.
    """
    cols = [
        "episode_seed", "timestep",
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
