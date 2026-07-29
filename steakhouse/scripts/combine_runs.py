#Misha new file that works similar to cimbine_json except we read from collect_soa_data's location, 
# aka user_study and then create new csv/pkl at
# CSV: /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/data/fov120_Overcooked2_2-4/120.csv
# PKL: /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/data/fov120_Ove
#how to run:
# cd Steakhouse-AI/src/scripts

#first (done)
# /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python combine_runs.py \
#   --fov 179 \
#   --layout Overcooked2_2-4 \
#   --logs_dir ../../user_study/log \
#   --out_root ../../data

#second (done)
# /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python combine_runs.py \
#   --fov 179 \
#   --layout Overcooked2_1-2 \
#   --logs_dir ../../user_study/log \
#   --out_root ../../data

#third (done)
# /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python combine_runs.py \
#   --fov 179 \
#   --layout Overcooked2_2-5 \
#   --logs_dir ../../user_study/log \
#   --out_root ../../data



#!/usr/bin/env python3

#To-do: 
# Csv results
# add episodes number column, change intent to string instead of id, 
# after getting the good csv:
# handle non-65 vector case -- theres no pots no more

"""
Combine Steakhouse/Overcooked JSON logs -> CSV/PKL, idempotently.

- Scans JSON logs under user_study/log/.
- Builds a flat DataFrame from each step in finished episodes (last step has done=true).
- Writes to: data/fov{FOV}_{LAYOUT}/{FOV}.csv and {FOV}.pkl
  e.g., data/fov120_Overcooked2_2-4/120.csv

Re-running is safe: we dedupe by (source_json, timestep) and sort rows deterministically.
"""


import os
import sys
import json
import pickle
import argparse
from typing import Any, Dict, List

import pandas as pd

# Make src/ importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils import flatten_obs_data  # dict -> str

# Try to map intent ids to strings (falls back gracefully)
try:
    from agents.steak_agent import ML_ACTION_LIST
except Exception:
    ML_ACTION_LIST = None

DEFAULT_LOGS_DIR = os.path.join(os.path.dirname(__file__), "../../user_study/log")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fov", type=int, required=True, help="FOV number, e.g., 120")
    ap.add_argument("--layout", type=str, required=True, help="Layout name, e.g., Overcooked2_2-4")
    ap.add_argument("--logs_dir", type=str, default=DEFAULT_LOGS_DIR, help="Root folder with JSON logs")
    ap.add_argument("--out_root", type=str, default=DEFAULT_DATA_DIR, help="Root folder for outputs")
    return ap.parse_args()

def dataset_dir(out_root: str, fov: int, layout: str) -> str:
    return os.path.abspath(os.path.join(out_root, f"fov{fov}_{layout}"))

def list_jsons(root: str) -> List[str]:
    root = os.path.abspath(root)
    out = []
    for subdir, _, files in os.walk(root):
        for f in files:
            if f.endswith(".json"):
                out.append(os.path.join(subdir, f))
    return sorted(out)

def safe_get(d: dict, k: str, default=None):
    return d.get(k, default)

def episode_from_path(path: str) -> str:
    """
    Expect logs like .../user_study/log/<episode>/<file>.json
    Returns that <episode> (string). If not numeric, still returns the folder name.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent  # keep as string; downstream CSV shows '0','1',...

def intent_id_to_str(i: Any) -> Any:
    """
    Map integer id -> intent string using ML_ACTION_LIST if available.
    If missing/invalid, return None or the original.
    """
    if i is None:
        return None
    try:
        idx = int(i)
    except Exception:
        return i
    if ML_ACTION_LIST is not None and 0 <= idx < len(ML_ACTION_LIST):
        return ML_ACTION_LIST[idx]
    return str(idx)

def rows_from_json(path: str, layout_fallback: str) -> List[Dict[str, Any]]:
    """
    Convert one log JSON to rows. Accepts unfinished episodes.
    Adds 'episode' from folder name. Includes intent strings.
    """
    with open(path, "r") as f:
        data = json.load(f)

    steps = data.get("episode", [])
    if not steps:
        return []

    rows = []
    rel = os.path.relpath(os.path.abspath(path))
    layout_name = data.get("layout_name", layout_fallback)
    ep = episode_from_path(path)

    for s in steps:
        r: Dict[str, Any] = {
            # keep for dedupe; not exported in final columns
            "source_json": rel,

            # required
            "episode": ep,
            "timestep": safe_get(s, "timestep"),

            # player state + actions
            "p0_held_object": safe_get(s, "p0_held_object"),
            "p0_action": safe_get(s, "p0_action"),
            "p1_held_object": safe_get(s, "p1_held_object"),
            "p1_action": safe_get(s, "p1_action"),

            # kitchen objects
            "chopping_board_state": safe_get(s, "chopping_board_state"),
            "grill_states_state": safe_get(s, "grill_states_state"),
            "sink_state": safe_get(s, "sink_state"),
            "pot_states_state": safe_get(s, "pot_states_state"),

            # add obs string (your old format)
            "obs": None,

            # positions/orientations (explode later)
            "p0_position": safe_get(s, "p0_position"),
            "p0_orientation": safe_get(s, "p0_orientation"),
            "p1_position": safe_get(s, "p1_position"),
            "p1_orientation": safe_get(s, "p1_orientation"),

            # optional intents as strings
            "p0_intent": intent_id_to_str(s.get("p0_intent_id")),
            "p1_intent": intent_id_to_str(s.get("p1_intent_id")),
        }

        try:
            r["obs"] = flatten_obs_data(s)
        except Exception:
            r["obs"] = None

        rows.append(r)
    return rows

def explode_xy(df: pd.DataFrame, col: str, xname: str, yname: str):
    if col not in df.columns:
        df[xname] = None
        df[yname] = None
        return
    def to_xy(v):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return list(v)
        return [None, None]
    xy = df[col].apply(to_xy)
    df[[xname, yname]] = pd.DataFrame(xy.tolist(), index=df.index)

def main():
    args = parse_args()

    out_dir = dataset_dir(args.out_root, args.fov, args.layout)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{args.fov}.csv")
    pkl_path = os.path.join(out_dir, f"{args.fov}.pkl")

    # Read all JSONs
    json_paths = list_jsons(args.logs_dir)
    if not json_paths:
        print(f"[combine] No JSON files under {args.logs_dir}")
    else:
        print(f"[combine] Found {len(json_paths)} JSON file(s) under {args.logs_dir}")

    rows: List[Dict[str, Any]] = []
    for jp in json_paths:
        try:
            rows.extend(rows_from_json(jp, args.layout))
        except Exception as e:
            print(f"[combine] Skipping {jp}: {e}")

    df_new = pd.DataFrame(rows)
    if df_new.empty:
        if os.path.exists(csv_path) and not os.path.exists(pkl_path):
            df_old = pd.read_csv(csv_path)
            with open(pkl_path, "wb") as f:
                pickle.dump(df_old, f)
        print("[combine] No steps found to write.")
        print(f"[combine] CSV: {csv_path}\n[combine] PKL: {pkl_path}")
        return

    # Expand vectors → columns
    explode_xy(df_new, "p0_position", "p0_position_x", "p0_position_y")
    explode_xy(df_new, "p1_position", "p1_position_x", "p1_position_y")
    explode_xy(df_new, "p0_orientation", "p0_orientation_x", "p0_orientation_y")
    explode_xy(df_new, "p1_orientation", "p1_orientation_x", "p1_orientation_y")

    # Drop original list cols
    for c in ["p0_position", "p1_position", "p0_orientation", "p1_orientation"]:
        if c in df_new.columns:
            df_new.drop(columns=[c], inplace=True)

    # Merge with existing (idempotent)
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True, sort=False)
    else:
        df_all = df_new

    # Deduplicate & sort deterministically (prefer source_json + timestep; fallback to episode + timestep)
    if "source_json" in df_all.columns and "timestep" in df_all.columns:
        df_all.drop_duplicates(subset=["source_json", "timestep"], keep="last", inplace=True)
        df_all.sort_values(by=["source_json", "timestep"], inplace=True, kind="mergesort")
    elif "episode" in df_all.columns and "timestep" in df_all.columns:
        df_all.drop_duplicates(subset=["episode", "timestep"], keep="last", inplace=True)
        df_all.sort_values(by=["episode", "timestep"], inplace=True, kind="mergesort")

    # Reorder to your minimal schema + intents at the end
    minimal_cols = [
        "episode", "timestep",
        "p0_held_object", "p0_action",
        "p1_held_object", "p1_action",
        "chopping_board_state", "grill_states_state", "sink_state", "pot_states_state",
        "obs",
        "p0_position_x", "p0_position_y", "p0_orientation_x", "p0_orientation_y",
        "p1_position_x", "p1_position_y", "p1_orientation_x", "p1_orientation_y",
    ]
    # Keep intent strings too (if available)
    extra_cols = [c for c in ["p0_intent", "p1_intent"] if c in df_all.columns]

    # Ensure all required columns exist
    for c in minimal_cols:
        if c not in df_all.columns:
            df_all[c] = None

    df_export = df_all[minimal_cols + extra_cols]

    # Atomic writes
    tmp_csv = csv_path + ".tmp"
    df_export.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, csv_path)

    with open(pkl_path + ".tmp", "wb") as f:
        pickle.dump(df_export, f)
    os.replace(pkl_path + ".tmp", pkl_path)

    print(f"[combine] Wrote rows: {len(df_export)}")
    print(f"[combine] CSV: {csv_path}\n[combine] PKL: {pkl_path}")

if __name__ == "__main__":
    main()
