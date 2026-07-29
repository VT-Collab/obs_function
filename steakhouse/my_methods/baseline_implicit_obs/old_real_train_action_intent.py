#!/usr/bin/env python3
"""
Train a shared LSTM to predict BOTH:
  • next subtask/intent (12-way, taken from 'p0_subtask')
  • next low-level action (A-way, inferred from data)

Data split is explicit:
  • TRAIN = 3 PKLs (hard-coded paths below)
  • VAL   = 1 PKL  (hard-coded path below)
No random 90/10 split is used.

Evaluation metrics:
  • train_avg_intent_trueprob, train_avg_action_trueprob
  • val_intent_trueprob, val_action_trueprob
All metrics use TRUE-CLASS PROBABILITIES (soft accuracy), not argmax.


Run from:
  /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_implicit_obs
Run:
  /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python real_train_action_intent.py
"""

import os, sys, pathlib, pickle, random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- Optional: Weights & Biases ----------------
try:
    import wandb
    USE_WANDB = True
except Exception:
    USE_WANDB = False

# ---------------------------------------------------------------------------
# Add project src/ to path so we can import dataset utils
# ---------------------------------------------------------------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add src/ to path
from dataset import obs_to_list, obs_list_to_1D_vec  # add previous vector production 

# ---------------------------------------------------------------------------
# ***** EDIT THESE PATHS IF NEEDED *****
# ---------------------------------------------------------------------------
PKL_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/fov_traj"

TRAIN_FILES = [
    os.path.join(PKL_DIR, "log_90_1_1-4.pkl"),
    os.path.join(PKL_DIR, "log_90_2_1-2.pkl"),
    os.path.join(PKL_DIR, "log_90_2_2-4.pkl"),
]

VAL_FILES = [
    os.path.join(PKL_DIR, "log_90_2_2-5.pkl"),
]

# ---------------------------------------------------------------------------
# Intent (subtask) space (fixed 12 classes, 0-indexed)
# ---------------------------------------------------------------------------
IDX_TO_INTENT = {
    0: 'pickup_meat',
    1: 'pickup_onion',
    2: 'pickup_dirty_plate',
    3: 'drop_meat',
    4: 'drop_onion',
    5: 'drop_dirty_plate',
    6: 'chop_onion',
    7: 'rinse_plate',
    8: 'pickup_clean_plate',
    9: 'pickup_steak',
    10: 'add_garnish',
    11: 'deliver_dish',
}
INTENT_TO_IDX: Dict[str, int] = {name: idx for idx, name in IDX_TO_INTENT.items()}
NUM_INTENTS = len(IDX_TO_INTENT)


def to_intent_idx(x: Any) -> Optional[int]:
    """Maps p0_subtask value to intent index; returns None if not found."""
    s = str(x).strip()
    return INTENT_TO_IDX.get(s, None)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def gather_true_probs(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return per-sample probabilities assigned to the true class."""
    probs = torch.softmax(logits, dim=1)
    return probs.gather(1, y.view(-1, 1)).squeeze(1)  # (B,)

def load_pkl(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        df = pickle.load(f)
    # Attach a stable file_id (use basename to keep things readable)
    base = pathlib.Path(path).name
    df["file_id"] = base
    # make sure episode/timestep are ints (grouping/sorting consistency)
    if "episode" in df.columns:
        df["episode"] = df["episode"].astype(int, errors="ignore")
    if "timestep" in df.columns:
        df["timestep"] = df["timestep"].astype(int, errors="ignore")
    # sort by (file, episode, timestep)
    df = df.sort_values(["file_id", "episode", "timestep"]).reset_index(drop=True)
    return df

def load_split_with_train_val(train_paths: List[str], val_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dfs, val_dfs = [], []
    for p in train_paths:
        if os.path.exists(p):
            train_dfs.append(load_pkl(p))
        else:
            print(f"[WARN] Missing train file: {p}")
    for p in val_paths:
        if os.path.exists(p):
            val_dfs.append(load_pkl(p))
        else:
            print(f"[WARN] Missing val file: {p}")
    if not train_dfs:
        raise RuntimeError("No training PKLs found. Check TRAIN_FILES paths.")
    if not val_dfs:
        raise RuntimeError("No validation PKLs found. Check VAL_FILES paths.")
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df   = pd.concat(val_dfs,   ignore_index=True)
    return train_df, val_df

def build_action_mapping(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    """Build a stable mapping from union(train, val) actions -> indices."""
    uniq = pd.unique(pd.concat([train_df["p0_action"], val_df["p0_action"]], ignore_index=True))
    uniq_sorted = sorted(list(uniq), key=lambda v: str(v))
    action_to_idx = {v: i for i, v in enumerate(uniq_sorted)}
    idx_to_action = {i: v for v, i in action_to_idx.items()}
    return action_to_idx, idx_to_action

def build_episode_tensors_both(df_ep: pd.DataFrame,
                               action_to_idx: Dict[Any, int]) -> Tuple[Optional[np.ndarray],
                                                                        Optional[np.ndarray],
                                                                        Optional[np.ndarray],
                                                                        Optional[list]]:
    """Return (obs_arr, intents, actions, meta) for a single episode.
       - obs_arr:  (T, D)
       - intents:  (T,)
       - actions:  (T,)
       - meta:     list of (file_id, episode, timestep)
    """
    obs_vecs, intents, actions, meta = [], [], [], []
    for _, r in df_ep.iterrows():
        # intent from p0_subtask
        intent_idx = to_intent_idx(r.get("p0_subtask", None))
        if intent_idx is None:
            continue

        # robust action mapping (string/number)
        raw_a = r.get("p0_action", None)
        a_idx = action_to_idx.get(raw_a, None)
        if a_idx is None:
            a_idx = action_to_idx.get(str(raw_a), None)
        if a_idx is None:
            continue

        try:
            obs_list = obs_to_list(r["obs"])       # raw → python lists
            vec      = obs_list_to_1D_vec(obs_list)  # → 1D np.array
        except Exception:
            continue

        obs_vecs.append(vec)
        intents.append(intent_idx)
        actions.append(a_idx)
        meta.append((r["file_id"], int(r["episode"]), int(r["timestep"])))

    if not obs_vecs:
        return None, None, None, None
    return (np.asarray(obs_vecs, dtype=np.float32),
            np.asarray(intents, dtype=np.int64),
            np.asarray(actions, dtype=np.int64),
            meta)

def compute_span_map(intents: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """For each time index, return (rank_within_window, window_span_length)
       where a window is a contiguous run of the same intent label.
    """
    span_map: Dict[int, Tuple[int, int]] = {}
    n = len(intents)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and intents[j+1] == intents[i]:
            j += 1
        span_len = j - i + 1
        for k in range(i, j + 1):
            rank = (k - i) + 1
            span_map[k] = (rank, span_len)
        i = j + 1
    return span_map

def create_sequences_both_for_splits(train_df: pd.DataFrame,
                                     val_df: pd.DataFrame,
                                     L: int,
                                     action_to_idx: Dict[Any, int]):
    """Build sliding windows of length L for BOTH tasks, for TRAIN and VAL."""
    def make_groups(df: pd.DataFrame):
        groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for (fid, epi), g in df.groupby(["file_id", "episode"], sort=False):
            obs_arr, intents, actions, meta = build_episode_tensors_both(g, action_to_idx)
            if obs_arr is None:
                continue
            groups[(fid, int(epi))] = {"obs": obs_arr, "intents": intents, "actions": actions, "meta": meta}
        return groups

    tr_groups = make_groups(train_df)
    va_groups = make_groups(val_df)

    def build(groups):
        X, Y_int, Y_act, M = [], [], [], []
        for key, g in groups.items():
            obs_arr   = g["obs"]
            intents   = g["intents"]
            actions   = g["actions"]
            meta_list = g["meta"]
            T = len(intents)
            if T <= L:
                continue
            for i in range(L, T):
                X.append(torch.tensor(obs_arr[i-L:i], dtype=torch.float32))  # (L,D)
                Y_int.append(torch.tensor(int(intents[i]), dtype=torch.long))
                Y_act.append(torch.tensor(int(actions[i]), dtype=torch.long))
                M.append(meta_list[i])  # (file_id, episode, timestep)
        return X, Y_int, Y_act, M

    trX, trYi, trYa, _   = build(tr_groups)
    vaX, vaYi, vaYa, vaM = build(va_groups)

    # span map for WEIGHTED intent metric on VAL
    val_span_info: Dict[Tuple[str,int,int], Tuple[int,int]] = {}
    for key, g in va_groups.items():
        span_map = compute_span_map(g["intents"])
        for t_idx, meta in enumerate(g["meta"]):
            if t_idx in span_map:
                val_span_info[meta] = span_map[t_idx]

    return trX, trYi, trYa, vaX, vaYi, vaYa, vaM, val_span_info

# ---------------------------------------------------------------------------
# Model: shared LSTM encoder + two classification heads
# ---------------------------------------------------------------------------
class MultiHeadLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layer_dim: int,
                 num_actions: int, num_intents: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layer_dim,
                            batch_first=True)
        self.head_action = nn.Linear(hidden_dim, num_actions)
        self.head_intent = nn.Linear(hidden_dim, num_intents)

    def forward(self, x):
        # x: (B, L, D)
        out, (h, c) = self.lstm(x)      # out: (B, L, H)
        last = out[:, -1, :]            # (B, H)
        logits_action = self.head_action(last)   # (B, A)
        logits_intent = self.head_intent(last)   # (B, 12)
        return logits_action, logits_intent, h, c

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    # ---------------- Config ----------------
    CFG = {
        "seq_length": 3,
        "batch_size": 128,
        "hidden_dim": 128,
        "layer_dim": 1,
        "num_epochs": 200,
        "lr": 1e-3,
        "seed": 7,
        "lambda_intent": 1.0,
        "lambda_action": 1.0,
        "gamma": 0.9,  # for weighted window intent metric
        "project": "action_intent_multitask",
        "entity": "steakteam",
    }

    if USE_WANDB:
        wandb.init(project=CFG["project"], entity=CFG["entity"], config=CFG)
        wandb.define_metric("avg_intent_trueprob", step_metric="epoch")
        wandb.define_metric("avg_action_trueprob", step_metric="epoch")
        wandb.define_metric("WEIGHTED_window_avg_intent_trueprob", step_metric="epoch")

    L        = int(CFG["seq_length"])
    BATCH    = int(CFG["batch_size"])
    HIDDEN   = int(CFG["hidden_dim"])
    LAYERS   = int(CFG["layer_dim"])
    EPOCHS   = int(CFG["num_epochs"])
    LR       = float(CFG["lr"])
    SEED     = int(CFG["seed"])
    LW_INT   = float(CFG["lambda_intent"])
    LW_ACT   = float(CFG["lambda_action"])
    GAMMA    = float(CFG["gamma"])

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # ---------------- Data: explicit split ----------------
    train_df, val_df = load_split_with_train_val(TRAIN_FILES, VAL_FILES)
    print(f"[data] TRAIN rows: {len(train_df)}  VAL rows: {len(val_df)}")
    print(f"[data] TRAIN files: {sorted(train_df['file_id'].unique().tolist())}")
    print(f"[data] VAL   files: {sorted(val_df['file_id'].unique().tolist())}")

    # build ACTION mapping from UNION(train,val) so the head size covers all labels
    ACTION_TO_IDX, IDX_TO_ACTION = build_action_mapping(train_df, val_df)
    NUM_ACTIONS = len(ACTION_TO_IDX)
    print(f"[data] Discovered {NUM_ACTIONS} actions → {ACTION_TO_IDX}")

    # sequences for BOTH tasks
    train_X, train_Yi, train_Ya, val_X, val_Yi, val_Ya, val_meta, val_span = \
        create_sequences_both_for_splits(train_df, val_df, L=L, action_to_idx=ACTION_TO_IDX)

    print(f"[seq] Training sequences:   {len(train_X)}")
    print(f"[seq] Validation sequences: {len(val_X)}")
    if not train_X or not val_X:
        raise RuntimeError("No sequences created. Check data contents and label mappings.")

    # infer input dimension dynamically from first window
    input_dim = int(train_X[0].shape[-1])

    # ---------------- DataLoaders ----------------
    x_tr = torch.stack(train_X)
    yi_tr = torch.stack(train_Yi)
    ya_tr = torch.stack(train_Ya)
    train_loader = DataLoader(TensorDataset(x_tr, yi_tr, ya_tr), batch_size=BATCH, shuffle=True)

    x_va = torch.stack(val_X)
    yi_va = torch.stack(val_Yi)
    ya_va = torch.stack(val_Ya)
    val_loader = DataLoader(TensorDataset(x_va, yi_va, ya_va), batch_size=BATCH, shuffle=False)

    # ---------------- Model / loss / opt ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MultiHeadLSTM(input_dim=input_dim, hidden_dim=HIDDEN, layer_dim=LAYERS,
                           num_actions=NUM_ACTIONS, num_intents=NUM_INTENTS).to(device)
    ce_int = nn.CrossEntropyLoss()
    ce_act = nn.CrossEntropyLoss()
    optim  = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------------- Train ----------------
    for epoch in range(EPOCHS):
        # ----- Train -----
        model.train()
        total_L, total_Li, total_La = 0.0, 0.0, 0.0
        tr_tp_int_sum = 0.0  # sum of true-class probs (intent)
        tr_tp_act_sum = 0.0  # sum of true-class probs (action)
        tr_count      = 0

        for xb, yib, yab in train_loader:
            xb  = xb.to(device)
            yib = yib.to(device)
            yab = yab.to(device)

            optim.zero_grad()
            logits_a, logits_i, _, _ = model(xb)
            loss_i = ce_int(logits_i, yib)
            loss_a = ce_act(logits_a, yab)
            loss   = LW_INT * loss_i + LW_ACT * loss_a
            loss.backward()
            optim.step()

            B = yib.size(0)
            total_L  += loss.item()
            total_Li += loss_i.item()
            total_La += loss_a.item()
            tr_count += B

            tp_i = gather_true_probs(logits_i, yib)  # (B,)
            tp_a = gather_true_probs(logits_a, yab)  # (B,)
            tr_tp_int_sum += tp_i.sum().item()
            tr_tp_act_sum += tp_a.sum().item()

        n_batches = max(1, len(train_loader))
        logs = {
            "epoch": epoch,
            "train_loss_total": total_L / n_batches,
            "train_loss_intent": total_Li / n_batches,
            "train_loss_action": total_La / n_batches,
            "train_avg_intent_trueprob": (tr_tp_int_sum / max(1, tr_count)),
            "train_avg_action_trueprob": (tr_tp_act_sum / max(1, tr_count)),
        }
        if USE_WANDB: wandb.log(logs)

        # ----- Eval -----
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            va_tp_int_sum = 0.0
            va_tp_act_sum = 0.0
            va_count      = 0

            with torch.no_grad():
                for xb, yib, yab in val_loader:
                    xb  = xb.to(device)
                    yib = yib.to(device)
                    yab = yab.to(device)
                    logits_a, logits_i, _, _ = model(xb)

                    tp_i = gather_true_probs(logits_i, yib)
                    tp_a = gather_true_probs(logits_a, yab)

                    va_tp_int_sum += tp_i.sum().item()
                    va_tp_act_sum += tp_a.sum().item()
                    va_count      += yib.size(0)

            avg_int_trueprob = va_tp_int_sum / max(1, va_count)
            avg_act_trueprob = va_tp_act_sum / max(1, va_count)

            # --- Window-weighted intent metric USING PROBS ---
            windows: Dict[Tuple[str,int,int,int,int], Dict[str, float]] = {}
            with torch.no_grad():
                for seq, label_i, meta in zip(val_X, val_Yi, val_meta):
                    logits_a, logits_i, _, _ = model(seq.unsqueeze(0).to(device))
                    p_true = float(gather_true_probs(logits_i, label_i.view(1).to(device)).item())
                    fid, epi, t = meta
                    rank, span_len = val_span.get(meta, (1, 1))
                    w = float(GAMMA ** (rank - 1))
                    start_t = int(t - (rank - 1))
                    end_t   = int(start_t + span_len - 1)
                    key = (fid, int(epi), start_t, end_t, int(label_i.item()))
                    bucket = windows.setdefault(key, {"wp": 0.0, "ws": 0.0})
                    bucket["wp"] += w * p_true
                    bucket["ws"] += w

            if windows:
                per_window_scores = [(b["wp"] / b["ws"]) for b in windows.values() if b["ws"] > 0.0]
                window_weighted_trueprob = float(sum(per_window_scores) / len(per_window_scores)) if per_window_scores else 0.0
            else:
                window_weighted_trueprob = 0.0

            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"val_intent_trueprob={avg_int_trueprob:.4f}  "
                f"val_action_trueprob={avg_act_trueprob:.4f}  "
                f"WEIGHTED_window_avg_intent_trueprob={window_weighted_trueprob:.4f}"
            )
            if USE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "val_intent_trueprob": avg_int_trueprob,
                    "val_action_trueprob": avg_act_trueprob,
                    "WEIGHTED_window_avg_intent_trueprob": window_weighted_trueprob,
                })

    print("Training complete!")

if __name__ == "__main__":
    main()
