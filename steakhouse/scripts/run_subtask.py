#!/usr/bin/env python3
"""
# NO MORE: Infer subtasks (intent) from a replay CSV by doing a rollout and calling
# SteakLimitVisionHumanModel.ml_action(state) at every timestep.

Infer FoV-consistent subtasks (intent) from a replay CSV by doing a rollout
and, at every timestep, computing a subtask for each player *purely from the CSV row's
perception snapshot* (FoV-limited station states, held objects, poses, etc.) 
RETURN NEW CSV AND PKL different from given csv by 2 added columns (p0_intent and p1_intent)

How to run (example)
cd /Users/mishafu/Desktop/steakhouse/Steakhouse-AI
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python run_subtask.py \
  --layout aa \
  --csv "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/third_90.csv" \
         "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/third_120.csv" \
         "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/third_179.csv" \
  --suffix _intent \
  --save-pkl


Notes:
- Subtask selection is *not* omniscient; it uses only the row's columns:
  chopping_board_state, grill_states_state, sink_state, pot_states_state,
  p*_held_object, p*_pose/orientation, etc.
- Copy paste the subtask determining logic from the past scrip
- You can enable state parity checks with --assert-sync or force the sim to match
  the CSV at each step with --hard-sync (useful if the source logs sometimes encode
  non-standard action tokens or start states).
"""
import os, sys, csv, argparse, pickle
from ast import literal_eval as parse

# --- repo path so we can import src/*
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Steakhouse MDP + Action enum
from mdp.steakhouse_mdp import SteakhouseGridworld
from overcooked_ai_py.mdp.actions import Action


# ----------------- CSV helpers -----------------

def read_all_rows(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    return rows, fieldnames

def group_rows_by_episode(rows):
    eps = {}
    for r in rows:
        try:
            e = int(float(r.get("episode", 0)))
        except Exception:
            e = 0
        eps.setdefault(e, []).append(r)
    # sort timesteps within each episode
    for e, lst in eps.items():
        lst.sort(key=lambda rr: int(float(rr.get("timestep", 0))) if rr.get("timestep") not in (None, "") else 0)
    return dict(sorted(eps.items()))  # sort by episode id

def token_to_action(tok):
    """Accept ints (or str of ints), action names ('NORTH'), or Action enum. Defaults to STAY."""
    if isinstance(tok, Action): return tok
    # int-like index
    try:
        idx = int(tok)
        idx2act = {idx: act for act, idx in Action.ACTION_TO_INDEX.items()}
        if idx in idx2act: return idx2act[idx]
    except Exception:
        pass
    # string name
    s = str(tok).strip().upper()
    name2act = {name.upper(): act for name, _ in Action.ACTION_TO_INDEX.items()}
    if s in name2act: return name2act[s]
    try:
        return Action[s]
    except Exception:
        return Action.STAY


# ----------------- FoV-aware subtask inference -----------------

def _held_from_row(row, agent_index):
    key = "p0_held_object" if agent_index == 0 else "p1_held_object"
    val = (row.get(key) or "").strip()
    return val if val else None

def _other_held_from_row(row, agent_index):
    other_key = "p1_held_object" if agent_index == 0 else "p0_held_object"
    val = (row.get(other_key) or "").strip()
    return val if val else None

def _parse_station(row, key, default=None):
    val = row.get(key)
    if val is None or val == "":
        return default if default is not None else {}
    try:
        return parse(val)
    except Exception:
        return default if default is not None else {}

def _bools_from_row(row):
    """Compute FoV-limited booleans like in ml_action but from the row snapshot."""
    chop = _parse_station(row, "chopping_board_state", {"empty": [], "full": [], "ready": []})
    grill = _parse_station(row, "grill_states_state", {"empty": [], "cooking": [], "ready": []})
    sink  = _parse_station(row, "sink_state", {"empty": [], "full": [], "ready": []})
    pot   = _parse_station(row, "pot_states_state", {"empty": [], "cooking": [], "ready": []})

    steak_ready          = len(grill.get("ready", [])) > 0
    cooking_grill        = len(grill.get("cooking", [])) > 0
    steak_nearly_ready   = steak_ready or cooking_grill

    boiled_ready         = len(pot.get("ready", [])) > 0
    cooking_pot          = len(pot.get("cooking", [])) > 0
    chicken_nearly_ready = boiled_ready or cooking_pot

    garnish_ready        = len(chop.get("ready", [])) > 0
    chopping             = len(chop.get("full",  [])) > 0

    clean_plate_ready    = len(sink.get("ready", [])) > 0
    rinsing              = len(sink.get("full",  [])) > 0

    order_idx = 0
    eff_clean_plate_ready = len(sink.get("ready", [])) > order_idx
    eff_rinsing           = len(sink.get("full",  []))  > order_idx

    return {
        "chop": chop, "grill": grill, "sink": sink, "pot": pot,
        "steak_ready": steak_ready,
        "cooking_grill": cooking_grill,
        "steak_nearly_ready": steak_nearly_ready,
        "boiled_ready": boiled_ready,
        "cooking_pot": cooking_pot,
        "chicken_nearly_ready": chicken_nearly_ready,
        "garnish_ready": garnish_ready,
        "chopping": chopping,
        "clean_plate_ready": clean_plate_ready,
        "rinsing": rinsing,
        "eff_clean_plate_ready": eff_clean_plate_ready,
        "eff_rinsing": eff_rinsing
    }

def greedy_subtask_from_row(row, mdp, state, agent_index):
    """
    FoV-aware subtask selection mirroring the greedy if/elif structure,
    but using only CSV row data (no motion planner).
    """
    held = _held_from_row(row, agent_index)
    other_held = _other_held_from_row(row, agent_index)
    other_has = lambda name: (other_held == name)

    B = _bools_from_row(row)
    order_idx = 0
    curr_order = state.order_list[order_idx] if state.num_orders_remaining > 0 else None

    steak_nearly_ready   = B["steak_nearly_ready"]
    steak_ready          = B["steak_ready"]
    chicken_nearly_ready = B["chicken_nearly_ready"]
    boiled_ready         = B["boiled_ready"]
    garnish_ready        = B["garnish_ready"]
    chopping             = B["chopping"]
    eff_clean_plate_ready= B["eff_clean_plate_ready"]
    eff_rinsing          = B["eff_rinsing"]

    if curr_order == "steak_dish" and not other_has("steak"):
        if held is None:
            if (not steak_nearly_ready):
                return "pickup_meat"
            elif (not eff_rinsing) and (not eff_clean_plate_ready) and (not other_has("dirty_plate")) and (not other_has("clean_plate")):
                return "pickup_dirty_plate"
            elif eff_rinsing and not eff_clean_plate_ready:
                return "rinse_plate"
            elif steak_nearly_ready and eff_clean_plate_ready:
                return "pickup_clean_plate"
        else:
            if held == "meat":          return "drop_meat"
            if held == "dirty_plate":   return "drop_dirty_plate"
            if held == "clean_plate":   return "pickup_steak"
            if held == "steak":         return "deliver_dish"

    elif curr_order == "steak_onion_dish":
        if held is None:
            if chopping and not garnish_ready:
                return "chop_onion"
            elif (not chopping) and (not garnish_ready) and (not other_has("onion")):
                return "pickup_onion"
            elif (not steak_nearly_ready):
                return "pickup_meat"
            elif (not eff_rinsing) and (not eff_clean_plate_ready) and (not other_has("dirty_plate")) and (not other_has("clean_plate")):
                return "pickup_dirty_plate"
            elif eff_rinsing and not eff_clean_plate_ready:
                return "rinse_plate"
            elif steak_ready and eff_clean_plate_ready and (not other_has("clean_plate")):
                return "pickup_clean_plate"
        else:
            if held == "onion":         return "drop_onion"
            if held == "meat":          return "drop_meat"
            if held == "dirty_plate":   return "drop_dirty_plate"
            if held == "clean_plate":   return "pickup_steak"
            if held == "steak" and garnish_ready:
                return "add_garnish"
            if held == "steak_onion":   return "deliver_dish"

    elif curr_order == "boiled_chicken_dish" and not other_has("boiled_chicken"):
        if held is None:
            if (not chicken_nearly_ready):
                return "pickup_chicken"
            elif (not eff_rinsing) and (not eff_clean_plate_ready) and (not other_has("dirty_plate")) and (not other_has("clean_plate")):
                return "pickup_dirty_plate"
            elif eff_rinsing and not eff_clean_plate_ready:
                return "rinse_plate"
            elif chicken_nearly_ready and eff_clean_plate_ready:
                return "pickup_clean_plate"
        else:
            if held == "chicken":           return "drop_chicken"
            if held == "dirty_plate":       return "drop_dirty_plate"
            if held == "clean_plate":       return "pickup_boiled_chicken"
            if held == "boiled_chicken":    return "deliver_dish"

    elif curr_order == "boiled_chicken_onion_dish":
        if held is None:
            if chopping and not garnish_ready:
                return "chop_onion"
            elif (not chopping) and (not garnish_ready) and (not other_has("onion")):
                return "pickup_onion"
            elif (not chicken_nearly_ready):
                return "pickup_chicken"
            elif (not eff_rinsing) and (not eff_clean_plate_ready) and (not other_has("dirty_plate")) and (not other_has("clean_plate")):
                return "pickup_dirty_plate"
            elif eff_rinsing and not eff_clean_plate_ready:
                return "rinse_plate"
            elif boiled_ready and eff_clean_plate_ready and (not other_has("clean_plate")):
                return "pickup_clean_plate"
        else:
            if held == "onion":                     return "drop_onion"
            if held == "chicken":                   return "drop_chicken"
            if held == "dirty_plate":               return "drop_dirty_plate"
            if held == "clean_plate":               return "pickup_boiled_chicken"
            if held == "boiled_chicken" and garnish_ready:
                return "add_garnish"
            if held == "boiled_chicken_onion":      return "deliver_dish"

    # Fallbacks
    if held is not None:
        if held == "dirty_plate":   return "drop_dirty_plate"
        if held in ("steak","boiled_chicken","steak_onion","boiled_chicken_onion"):
            return "deliver_dish"
        if held == "onion":         return "drop_onion"
        if held == "meat":          return "drop_meat"
        if held == "chicken":       return "drop_chicken"
        return "drop_dirty_plate"
    else:
        return "pickup_dirty_plate"


# ----------------- Optional parity checks / hard sync -----------------

def _row_pose(row, who):
    px = row.get(f"{who}_position_x"); py = row.get(f"{who}_position_y")
    ox = row.get(f"{who}_orientation_x"); oy = row.get(f"{who}_orientation_y")
    try:
        return (int(float(px)), int(float(py)), int(float(ox)), int(float(oy)))
    except Exception:
        return None

def _held_name(player):
    return player.get_object().name if player.has_object() else ""

def _assert_sync(row, state):
    p0_csv = (row.get("p0_held_object") or "")
    p1_csv = (row.get("p1_held_object") or "")
    p0_sim = _held_name(state.players[0])
    p1_sim = _held_name(state.players[1])
    assert p0_csv == p0_sim, f"p0 held mismatch: csv='{p0_csv}' sim='{p0_sim}'"
    assert p1_csv == p1_sim, f"p1 held mismatch: csv='{p1_csv}' sim='{p1_sim}'"
    p0_pose = _row_pose(row, "p0")
    p1_pose = _row_pose(row, "p1")
    if p0_pose is not None:
        sx, sy = state.players[0].position
        sox, soy = state.players[0].orientation
        assert (sx, sy, sox, soy) == p0_pose, f"p0 pose mismatch: csv={p0_pose} sim={(sx,sy,sox,soy)}"
    if p1_pose is not None:
        sx, sy = state.players[1].position
        sox, soy = state.players[1].orientation
        assert (sx, sy, sox, soy) == p1_pose, f"p1 pose mismatch: csv={p1_pose} sim={(sx,sy,sox,soy)}"

def _hard_sync_from_row(row, state):
    # Only if you need it; will be skipped unless --hard-sync is used
    try:
        from mdp.objects import make_object_by_name
    except Exception:
        make_object_by_name = None

    p0_pose = _row_pose(row, "p0")
    p1_pose = _row_pose(row, "p1")
    if p0_pose is not None:
        x,y,ox,oy = p0_pose
        state.players[0].position = (x,y)
        state.players[0].orientation = (ox,oy)
    if p1_pose is not None:
        x,y,ox,oy = p1_pose
        state.players[1].position = (x,y)
        state.players[1].orientation = (ox,oy)

    def _sync_held(idx, name):
        pl = state.players[idx]
        if pl.has_object():
            pl.remove_object()
        if name and make_object_by_name is not None:
            obj = make_object_by_name(name)
            pl.set_object(obj)

    _sync_held(0, (row.get("p0_held_object") or ""))
    _sync_held(1, (row.get("p1_held_object") or ""))


# ----------------- processing -----------------

def process_episode_rows(layout, rows, assert_sync=False, hard_sync=False):
    mdp = SteakhouseGridworld.from_layout_name(layout)
    state = mdp.get_standard_start_state()

    out_rows = []
    for row in rows:
        if hard_sync:
            _hard_sync_from_row(row, state)

        p0_intent = greedy_subtask_from_row(row, mdp, state, agent_index=0)
        p1_intent = greedy_subtask_from_row(row, mdp, state, agent_index=1)

        a0 = token_to_action(row.get("p0_action", Action.STAY))
        a1 = token_to_action(row.get("p1_action", Action.STAY))
        step_ret = mdp.get_state_transition(state, (a0, a1))
        next_state = step_ret[0] if isinstance(step_ret, tuple) else step_ret
        state = next_state

        if assert_sync:
            _assert_sync(row, state)

        r = dict(row)
        r["p0_intent"] = p0_intent
        r["p1_intent"] = p1_intent
        out_rows.append(r)

    return out_rows

def process_csv_file(layout, csv_path, out_csv_path, save_pkl=False, assert_sync=False, hard_sync=False):
    rows, in_fields = read_all_rows(csv_path)

    # Ensure output fields include intents (last two columns)
    out_fields = list(in_fields)
    for col in ["p0_intent", "p1_intent"]:
        if col not in out_fields:
            out_fields.append(col)

    episodes = group_rows_by_episode(rows)

    all_out_rows = []
    for ep, ep_rows in episodes.items():
        ep_out = process_episode_rows(layout, ep_rows, assert_sync=assert_sync, hard_sync=hard_sync)
        all_out_rows.extend(ep_out)

    # Preserve original overall order (episode then timestep)
    all_out_rows.sort(key=lambda rr: (int(float(rr.get("episode", 0))), int(float(rr.get("timestep", 0)))))

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in all_out_rows:
            w.writerow(r)

    if save_pkl:
        base, _ = os.path.splitext(out_csv_path)
        pkl_path = base + ".pkl"
        try:
            # Prefer pandas DataFrame if available
            import pandas as pd
            df = pd.DataFrame(all_out_rows, columns=out_fields)
            df.to_pickle(pkl_path)
        except Exception:
            # Fallback: plain pickle of list-of-dicts
            with open(pkl_path, "wb") as pf:
                pickle.dump(all_out_rows, pf)

    return out_csv_path


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True, help="Layout name (e.g., aa)")
    ap.add_argument("--csv", nargs="+", required=True, help="One or more CSV files")
    ap.add_argument("--outdir", default="", help="Optional output directory (defaults to input file's dir)")
    ap.add_argument("--suffix", default="_intent", help="Suffix to add before extension (default: _intent)")
    ap.add_argument("--save-pkl", action="store_true", help="Also save a matching .pkl next to the output CSV")
    ap.add_argument("--assert-sync", action="store_true")
    ap.add_argument("--hard-sync", action="store_true")
    args = ap.parse_args()

    for csv_path in args.csv:
        in_dir, in_name = os.path.split(csv_path)
        base, ext = os.path.splitext(in_name)
        ext = ext or ".csv"
        out_dir = args.outdir if args.outdir else in_dir
        out_name = f"{base}{args.suffix}{ext}"
        out_csv_path = os.path.join(out_dir, out_name)

        print(f"[subtask] Processing {csv_path} -> {out_csv_path}")
        process_csv_file(
            layout=args.layout,
            csv_path=csv_path,
            out_csv_path=out_csv_path,
            save_pkl=args.save_pkl,
            assert_sync=args.assert_sync,
            hard_sync=args.hard_sync,
        )
        print(f"[subtask] Done: {out_csv_path}")

if __name__ == "__main__":
    main()