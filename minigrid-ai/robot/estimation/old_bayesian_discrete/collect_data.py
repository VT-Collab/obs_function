# collect_data.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from minigrid.core.constants import COLOR_NAMES
import csv
import os

# Door state coding
_DOOR_OPEN, _DOOR_CLOSED, _DOOR_LOCKED = 0, 1, 2

# 0=right,1=down,2=left,3=up
DIR_TO_VEC = [(1,0), (0,1), (-1,0), (0,-1)]


def _color_to_idx(c) -> int:
    """MiniGrid may store color as int or str; return an index into COLOR_NAMES."""
    if isinstance(c, int):
        return int(c)
    return int(COLOR_NAMES.index(c))


def _one_hot_color(c) -> List[int]:
    """One-hot over COLOR_NAMES (length 6)."""
    idx = _color_to_idx(c)
    out = [0] * len(COLOR_NAMES)
    if 0 <= idx < len(COLOR_NAMES):
        out[idx] = 1
    return out


def _door_state_int(door) -> int:
    """Map MiniGrid Door flags to 0/1/2."""
    if getattr(door, "is_open", False):
        return _DOOR_OPEN
    if getattr(door, "is_locked", False):
        return _DOOR_LOCKED
    return _DOOR_CLOSED


def _scan_doors(env) -> List[Tuple[int, int, Any]]:
    """Return [(x, y, door_obj)] sorted by (x, y)."""
    doors = []
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if obj is not None and getattr(obj, "type", None) == "door":
                doors.append((x, y, obj))
    doors.sort(key=lambda t: (t[0], t[1]))
    return doors


def _scan_keys_and_goal(env) -> Tuple[List[Tuple[int, int, Any]], Optional[Tuple[int, int]]]:
    """Return (keys_on_grid_sorted, goal_xy_or_None)."""
    keys = []
    goal_xy = None
    for x in range(env.width):
        for y in range(env.height):
            obj = env.grid.get(x, y)
            if obj is None:
                continue
            t = getattr(obj, "type", None)
            if t == "key":
                keys.append((x, y, obj))
            elif t == "goal":
                goal_xy = (x, y)
    keys.sort(key=lambda t: (t[0], t[1]))
    return keys, goal_xy


def build_step_record(
    env,
    *,
    episode_seed: int,
    timestep: int,
    agent_subtask: str,
    num_doors: int = 6,
    num_keys: int = 3,
) -> Dict[str, Any]:
    """
    Build a single-step dict with ints everywhere except 'agent_subtask'.
    """
    d: Dict[str, Any] = {}

    # Episode meta
    d["episode_seed"] = int(episode_seed)
    d["timestep"] = int(timestep)

    # Agent pose + facing
    ax, ay = env.agent_pos
    d["agent_pos_x"] = int(ax)
    d["agent_pos_y"] = int(ay)
    ox, oy = DIR_TO_VEC[int(env.agent_dir)]
    d["agent_orientation_x"] = int(ox)
    d["agent_orientation_y"] = int(oy)

    # High-level label (string)
    d["agent_subtask"] = agent_subtask

    # Agent carrying (key color one-hot)
    car = getattr(env, "carrying", None)
    if car is not None and getattr(car, "type", None) == "key":
        carry_oh = _one_hot_color(getattr(car, "color", 0))
    else:
        carry_oh = [0] * len(COLOR_NAMES)
    for ci, name in enumerate(COLOR_NAMES):
        d[f"agent_carry_color_{name}"] = int(carry_oh[ci])

    # Doors (pad/truncate to exactly num_doors)
    doors = _scan_doors(env)
    if len(doors) < num_doors:
        doors += [(-1, -1, None)] * (num_doors - len(doors))
    else:
        doors = doors[:num_doors]

    for i, (dx, dy, door) in enumerate(doors, start=1):
        d[f"door_{i}_x"] = int(dx)
        d[f"door_{i}_y"] = int(dy)
        if door is None:
            d[f"door_{i}_state"] = int(_DOOR_CLOSED)
            for name in COLOR_NAMES:
                d[f"door_{i}_color_{name}"] = 0
        else:
            d[f"door_{i}_state"] = int(_door_state_int(door))
            oh = _one_hot_color(getattr(door, "color", 0))
            for ci, name in enumerate(COLOR_NAMES):
                d[f"door_{i}_color_{name}"] = int(oh[ci])

    # Keys on grid
    keys_on_grid, goal_xy = _scan_keys_and_goal(env)
    if len(keys_on_grid) < num_keys:
        keys_on_grid += [(-1, -1, None)] * (num_keys - len(keys_on_grid))
    else:
        keys_on_grid = keys_on_grid[:num_keys]

    for i, (kx, ky, key_obj) in enumerate(keys_on_grid, start=1):
        if key_obj is None:
            d[f"key_{i}_x"] = -1
            d[f"key_{i}_y"] = -1
            for name in COLOR_NAMES:
                d[f"key_{i}_color_{name}"] = 0
        else:
            d[f"key_{i}_x"] = int(kx)
            d[f"key_{i}_y"] = int(ky)
            oh = _one_hot_color(getattr(key_obj, "color", 0))
            for ci, name in enumerate(COLOR_NAMES):
                d[f"key_{i}_color_{name}"] = int(oh[ci])

    # Goal
    if goal_xy is None:
        d["goal_x"] = -1
        d["goal_y"] = -1
    else:
        gx, gy = goal_xy
        d["goal_x"] = int(gx)
        d["goal_y"] = int(gy)

    return d


def csv_header(num_doors: int = 6, num_keys: int = 3):
    cols = [
        "episode_seed", "timestep",
        "agent_pos_x", "agent_pos_y",
        "agent_orientation_x", "agent_orientation_y",
    ]
    cols += [f"agent_carry_color_{name}" for name in COLOR_NAMES]
    cols += ["agent_subtask"]
    for i in range(1, num_doors + 1):
        cols += [f"door_{i}_x", f"door_{i}_y", f"door_{i}_state"]
        cols += [f"door_{i}_color_{name}" for name in COLOR_NAMES]
    for i in range(1, num_keys + 1):
        cols += [f"key_{i}_x", f"key_{i}_y"]
        cols += [f"key_{i}_color_{name}" for name in COLOR_NAMES]
    cols += ["goal_x", "goal_y"]
    return cols


def record_to_row(rec: dict, num_doors: int = 6, num_keys: int = 3):
    row = [
        rec["episode_seed"], rec["timestep"],
        rec["agent_pos_x"], rec["agent_pos_y"],
        rec["agent_orientation_x"], rec["agent_orientation_y"],
    ]
    row += [rec[f"agent_carry_color_{name}"] for name in COLOR_NAMES]
    row += [rec["agent_subtask"]]
    for i in range(1, num_doors + 1):
        row += [rec[f"door_{i}_x"], rec[f"door_{i}_y"], rec[f"door_{i}_state"]]
        row += [rec[f"door_{i}_color_{name}"] for name in COLOR_NAMES]
    for i in range(1, num_keys + 1):
        row += [rec[f"key_{i}_x"], rec[f"key_{i}_y"]]
        row += [rec[f"key_{i}_color_{name}"] for name in COLOR_NAMES]
    row += [rec["goal_x"], rec["goal_y"]]
    return row


def write_episode_csv(records, filepath: str, num_doors: int = 6, num_keys: int = 3, append: bool = False):
    header = csv_header(num_doors=num_doors, num_keys=num_keys)
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    mode = "a" if append else "w"
    write_header = True
    if append and os.path.exists(filepath):
        write_header = False
    with open(filepath, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for rec in records:
            w.writerow(record_to_row(rec, num_doors=num_doors, num_keys=num_keys))
