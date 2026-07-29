""" 
collect_data.py

contains helper functions collect_traj.py imports from to pack everything into a flat dict
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from minigrid.core.constants import COLOR_NAMES
import csv
import os


# Door state coding you wanted
# 0=open, 1=closed, 2=locked
_DOOR_OPEN, _DOOR_CLOSED, _DOOR_LOCKED = 0, 1, 2
# NEW (top-level, near imports)
DIR_TO_VEC = [(1,0), (0,1), (-1,0), (0,-1)]  # 0=right,1=down,2=left,3=up

#color string to color int
def _color_to_idx(c) -> int:
    """MiniGrid may store color as int or str; return an index into COLOR_NAMES."""
    if isinstance(c, int):
        return int(c)
    return int(COLOR_NAMES.index(c))

#returns the one-hot encoder; if index is 2 then ur encoder would be 0 0 1 0 0 0 
def _one_hot_color(c) -> List[int]:
    """One-hot over COLOR_NAMES (length 6)."""
    idx = _color_to_idx(c)
    out = [0] * len(COLOR_NAMES)
    if 0 <= idx < len(COLOR_NAMES):
        out[idx] = 1
    return out

#Converts a door’s state into an integer code: 0 for open, 1 for closed, 2 for locked.
def _door_state_int(door) -> int:
    """Map MiniGrid Door flags to 0/1/2."""
    if getattr(door, "is_open", False):
        return _DOOR_OPEN
    if getattr(door, "is_locked", False):
        return _DOOR_LOCKED
    return _DOOR_CLOSED

#Iterates through the grid to find all door objects and returns a sorted list of their coordinates and references.
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

#Finds all key objects and the goal tile in the grid, returning a sorted list of keys and the goal’s (x, y) position if present.
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

#Extracts all relevant environment information for one timestep—including agent position, orientation, carried key, door/key states, and goal—and packages it into a flat dictionary used for CSV logging.

def build_step_record(
    env,
    *,
    episode_seed: int,
    timestep: int,
    agent_subtask: str,
    
    agent_action: str,          # NEW

    
    
    num_doors: int = 6,
    num_keys: int = 3,
    
) -> Dict[str, Any]:
    """
    Build a single-step dict with ints everywhere except 'agent_subtask'.
    Fields:
      episode_seed, timestep
      agent_pos_x, agent_pos_y, agent_dir
      agent_subtask (string)
      agent_carry_color_<name> (one-hot over 6 colors)
      door_i_x, door_i_y, door_i_state, door_i_color_<name>  for i=1..6
      key_i_x, key_i_y, key_i_color_<name>                   for i=1..3
      goal_x, goal_y
    NOTE: If a key is carried by the agent, we DO NOT place it into any key slot.
          Its key slot will be all zeros (x=-1,y=-1, color one-hot=0), and the
          carried color is encoded ONLY via agent_carry_color_*.
    """
    d: Dict[str, Any] = {}

    # Episode meta
    d["episode_seed"] = int(episode_seed)
    d["timestep"] = int(timestep)

    # Agent
    ax, ay = env.agent_pos
    d["agent_pos_x"] = int(ax)
    d["agent_pos_y"] = int(ay)
    
    
    # NEW: inside build_step_record(...) right after setting agent_dir / agent_subtask
    ox, oy = DIR_TO_VEC[int(env.agent_dir)]
    d["agent_orientation_x"] = int(ox)
    d["agent_orientation_y"] = int(oy)

    
    d["agent_subtask"] = agent_subtask  # the only non-int field
    
    d["agent_action"] = agent_action               # NEW


    # Agent carrying (color one-hot; zeros if nothing or non-key)
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
        doors = doors + [(-1, -1, None)] * (num_doors - len(doors))
    else:
        doors = doors[:num_doors]

    for i, (dx, dy, door) in enumerate(doors, start=1):
        d[f"door_{i}_x"] = int(dx)
        d[f"door_{i}_y"] = int(dy)
        if door is None:
            d[f"door_{i}_state"] = int(_DOOR_CLOSED)  # neutral default
            for ci, name in enumerate(COLOR_NAMES):
                d[f"door_{i}_color_{name}"] = 0
        else:
            d[f"door_{i}_state"] = int(_door_state_int(door))
            oh = _one_hot_color(getattr(door, "color", 0))
            for ci, name in enumerate(COLOR_NAMES):
                d[f"door_{i}_color_{name}"] = int(oh[ci])

    # Keys on grid; if agent is carrying a key, it will NOT appear on the grid
    keys_on_grid, goal_xy = _scan_keys_and_goal(env)
    # Do not inject a synthetic carried-key entry; per your spec we zero the slot instead
    if len(keys_on_grid) < num_keys:
        keys_on_grid = keys_on_grid + [(-1, -1, None)] * (num_keys - len(keys_on_grid))
    else:
        keys_on_grid = keys_on_grid[:num_keys]

    for i, (kx, ky, key_obj) in enumerate(keys_on_grid, start=1):
        # If key_obj is None OR (rarely) a carried key was removed from grid,
        # we want zeros for this slot.
        if key_obj is None:
            d[f"key_{i}_x"] = -1
            d[f"key_{i}_y"] = -1
            for ci, name in enumerate(COLOR_NAMES):
                d[f"key_{i}_color_{name}"] = 0
            continue

        # Normal on-grid key
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



#from dictionary, write to csv
"""agent_subtask (the only string) and the door would be like (x,y), (state), episode, timestep, agent_pos_x, agent_pos_y, agent_orientation_x, agent_orientation_y, agent_carrying_key_color, agent_subtask, door1, door2, door3, door4, door 5, door 6, and key1, key2, key3, and finally goal_pos.
    door would be like (x,y), (state), then one hot encoding of colors
"""
# NEW: helper to build the CSV header in the requested order
#Builds and returns the ordered list of column names expected in the dataset CSV, matching the data layout produced by build_step_record().
def csv_header(num_doors: int = 6, num_keys: int = 3):
    cols = [
        "episode_seed", "timestep",
        "agent_pos_x", "agent_pos_y",
        "agent_orientation_x", "agent_orientation_y",
    ]
    # agent carrying one-hot
    cols += [f"agent_carry_color_{name}" for name in COLOR_NAMES]

    
    cols += [
        "agent_subtask", "agent_action"
    ]
    # doors
    for i in range(1, num_doors + 1):
        cols += [f"door_{i}_x", f"door_{i}_y", f"door_{i}_state"]
        cols += [f"door_{i}_color_{name}" for name in COLOR_NAMES]

    # keys
    for i in range(1, num_keys + 1):
        cols += [f"key_{i}_x", f"key_{i}_y"]
        cols += [f"key_{i}_color_{name}" for name in COLOR_NAMES]

    # goal
    cols += ["goal_x", "goal_y"]
    return cols



# NEW: flatten a single step dict to a CSV row matching csv_header(...)
def record_to_row(rec: dict, num_doors: int = 6, num_keys: int = 3):
    row = [
        rec["episode_seed"], rec["timestep"],
        rec["agent_pos_x"], rec["agent_pos_y"],
        rec["agent_orientation_x"], rec["agent_orientation_y"],
    ]
    # agent carry one-hot
    row += [rec[f"agent_carry_color_{name}"] for name in COLOR_NAMES]

    row += [
        rec["agent_subtask"],
        rec.get("agent_action", ""),
        
    ]

    # doors
    for i in range(1, num_doors + 1):
        row += [rec[f"door_{i}_x"], rec[f"door_{i}_y"], rec[f"door_{i}_state"]]
        row += [rec[f"door_{i}_color_{name}"] for name in COLOR_NAMES]

    # keys
    for i in range(1, num_keys + 1):
        row += [rec[f"key_{i}_x"], rec[f"key_{i}_y"]]
        row += [rec[f"key_{i}_color_{name}"] for name in COLOR_NAMES]

    # goal
    row += [rec["goal_x"], rec["goal_y"]]
    return row



#    """Write a list of step dicts to CSV with fixed columns/order."""
def write_episode_csv(records, filepath: str, num_doors: int = 6, num_keys: int = 3, append: bool = False):
    """Write a list of step dicts to CSV with fixed columns/order."""
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
