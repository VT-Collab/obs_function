#!/usr/bin/env python3
"""
Multi‑task LSTM trainer for Steakhouse‑AI that predicts BOTH:
  • next intent/subtask (12‑way) from p0_intent
  • next low‑level action       (A‑way) from p0_action (A inferred from data)

Data: mixed layouts + FOVs. By default, this script loads the 9 files:
    ../data/fov{90,120,179}_Overcooked2_{1-2,2-4,2-5}/{90,120,179}.pkl
when run from:
    /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_implicit_obs

Key features:
  • Shares one LSTM encoder with two classification heads (action & intent)
  • Episode‑wise split: trains on whole episodes; validates on different episodes
  • Weighted window metric for intents: macro‑averaged across contiguous spans
  • W&B logging for losses, accuracies, and the weighted window intent metric

Run:
  /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python train_action_intent_multitask.py

If your folder names differ, edit make_mixed_file_list().
"""

import os, sys, pathlib, pickle, random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

# ---------------------------------------------------------------------------
# Imports from project (dataset utils)
# ---------------------------------------------------------------------------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add src/
from dataset import obs_to_list, obs_list_to_1D_vec  # noqa

# ---------------------------------------------------------------------------
# Intent space (fixed 12 classes, 0‑indexed)
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
    s = str(x).strip()
    return INTENT_TO_IDX.get(s, None)

# ---------------------------------------------------------------------------
# File list builder (9 files = 3 FOVs × 3 layouts)
# ---------------------------------------------------------------------------

def make_mixed_file_list(root_rel: Optional[str] = None) -> List[str]:
    """Build the 9 mixed layout+FOV paths.
    If root_rel is None, infer the project data dir as: <repo_root>/data
    where <repo_root> = Path(__file__).resolve().parents[2].
    """
    if root_rel is None:
        inferred = pathlib.Path(__file__).resolve().parents[2] / "data"
        root_rel = str(inferred)
    fovs = ["90", "120", "179"]
    layouts = ["Overcooked2_1-2", "Overcooked2_2-4", "Overcooked2_2-5"]
    paths = []
    for f in fovs:
        for lay in layouts:
            paths.append(os.path.join(root_rel, f"fov{f}_{lay}", f"{f}.pkl"))
    return paths

# ---------------------------------------------------------------------------
# Data loading & episode tensorization (yields both actions & intents)
# ---------------------------------------------------------------------------

def load_pkls(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        with open(p, 'rb') as f:
            df = pickle.load(f)
        # Use folder + filename so IDs are unique across the 9 files
        path_obj = pathlib.Path(p)
        df['file_id'] = "/".join(path_obj.parts[-2:])  # e.g., fov120_Overcooked2_1-2/120.pkl
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data['episode'] = data['episode'].astype(int)
    data['timestep'] = data['timestep'].astype(int)
    data = data.sort_values(["file_id", "episode", "timestep"]).reset_index(drop=True)
    return data


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
        intent_idx = to_intent_idx(r.get("p0_intent", None))
        if intent_idx is None:
            continue
        # map action (string or numeric) to 0..A-1
        raw_a = r.get("p0_action", None)
        a_idx = action_to_idx.get(raw_a, None)
        if a_idx is None:
            # try string fallback
            a_idx = action_to_idx.get(str(raw_a), None)
        if a_idx is None:
            continue
        try:
            obs_list = obs_to_list(r["obs"])  # raw → python lists
            vec      = obs_list_to_1D_vec(obs_list)  # → 1D np.array
        except Exception:
            continue
        obs_vecs.append(vec)
        intents.append(intent_idx)
        actions.append(a_idx)
        meta.append((r["file_id"], int(r["episode"]), int(r["timestep"])) )

    if not obs_vecs:
        return None, None, None, None
    return (np.asarray(obs_vecs, dtype=np.float32),
            np.asarray(intents, dtype=np.int64),
            np.asarray(actions, dtype=np.int64),
            meta)


def split_by_episode_keys(data: pd.DataFrame, train_ratio: float = 0.9, seed: int = 0):
    keys = data[["file_id", "episode"]].drop_duplicates().apply(tuple, axis=1).tolist()
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_train = max(1, int(len(keys) * train_ratio))
    return keys[:n_train], keys[n_train:]


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


def create_sequences_both(data: pd.DataFrame,
                          L: int,
                          train_ratio: float,
                          seed: int,
                          action_to_idx: Dict[Any, int]):
    """Build sliding windows of length L for BOTH tasks.
    Returns:
      train_X, train_Y_int, train_Y_act,
      val_X,   val_Y_int,   val_Y_act,
      val_meta, val_span_info
    """
    groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for (fid, epi), g in data.groupby(["file_id", "episode"], sort=False):
        obs_arr, intents, actions, meta = build_episode_tensors_both(g, action_to_idx)
        if obs_arr is None:
            continue
        groups[(fid, int(epi))] = {"obs": obs_arr, "intents": intents, "actions": actions, "meta": meta}

    train_keys, val_keys = split_by_episode_keys(data, train_ratio=train_ratio, seed=seed)

    def build(keys):
        X, Y_int, Y_act, M = [], [], [], []
        for key in keys:
            if key not in groups:
                continue
            g = groups[key]
            obs_arr   = g["obs"]      # (T,D)
            intents   = g["intents"]  # (T,)
            actions   = g["actions"]  # (T,)
            meta_list = g["meta"]     # list of (file_id, epi, t)
            T = len(intents)
            if T <= L:
                continue
            for i in range(L, T):
                X.append(torch.tensor(obs_arr[i-L:i], dtype=torch.float32))  # (L,D)
                Y_int.append(torch.tensor(int(intents[i]), dtype=torch.long))
                Y_act.append(torch.tensor(int(actions[i]), dtype=torch.long))
                M.append(meta_list[i])
        return X, Y_int, Y_act, M

    trX, trYi, trYa, _     = build(train_keys)
    vaX, vaYi, vaYa, vaM   = build(val_keys)

    # span map for WEIGHTED intent accuracy
    val_span_info: Dict[Tuple[str,int,int], Tuple[int,int]] = {}
    for key in val_keys:
        if key not in groups:
            continue
        g = groups[key]
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
    # ---------------- W&B ----------------
    wandb.init(
        project="action_intent_multitask",
        entity="steakteam",
        config={
            "seq_length": 3,
            "train_to_validate_ratio": 0.9,
            "batch_size": 128,
            "hidden_dim": 128,
            "layer_dim": 1,
            "num_epochs": 200,
            "lr": 1e-3,
            "seed": 7,
            "lambda_intent": 1.0,
            "lambda_action": 1.0,
            "gamma": 0.9,  # for weighted window intent metric
        }
    )
    wandb.define_metric("avg_intent_accuracy", step_metric="epoch")
    wandb.define_metric("avg_action_accuracy", step_metric="epoch")
    wandb.define_metric("WEIGHTED_window_avg_intent_accuracy", step_metric="epoch")

    CFG = wandb.config
    L        = int(CFG["seq_length"])
    train_r  = float(CFG["train_to_validate_ratio"])
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

    # ---------------- Data ----------------
    file_paths = make_mixed_file_list()
    data = load_pkls(file_paths)
    print(f"Loaded {len(data)} rows across {data['episode'].nunique()} episodes; files: {sorted(data['file_id'].unique().tolist())}")

    # build ACTION mapping (robust to str or numeric)
    unique_actions = list(pd.unique(data["p0_action"]))
    # preserve order but sort string reps for stability if mixed types
    unique_actions_sorted = sorted(unique_actions, key=lambda v: str(v))
    ACTION_TO_IDX: Dict[Any, int] = {v: i for i, v in enumerate(unique_actions_sorted)}
    IDX_TO_ACTION: Dict[int, Any] = {i: v for v, i in ACTION_TO_IDX.items()}
    NUM_ACTIONS = len(ACTION_TO_IDX)
    print(f"Discovered {NUM_ACTIONS} actions → {ACTION_TO_IDX}")

    # sequences for BOTH tasks
    train_X, train_Yi, train_Ya, val_X, val_Yi, val_Ya, val_meta, val_span = create_sequences_both(
        data, L=L, train_ratio=train_r, seed=SEED, action_to_idx=ACTION_TO_IDX
    )

    print(f"Training sequences:   {len(train_X)}")
    print(f"Validation sequences: {len(val_X)}")
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
        model.train()
        total_L, total_Li, total_La = 0.0, 0.0, 0.0
        for xb, yib, yab in train_loader:
            xb = xb.to(device)
            yib = yib.to(device)
            yab = yab.to(device)

            optim.zero_grad()
            logits_a, logits_i, _, _ = model(xb)
            loss_i = ce_int(logits_i, yib)
            loss_a = ce_act(logits_a, yab)
            loss   = LW_INT * loss_i + LW_ACT * loss_a
            loss.backward()
            optim.step()

            total_L  += loss.item()
            total_Li += loss_i.item()
            total_La += loss_a.item()

        n_batches = max(1, len(train_loader))
        wandb.log({
            "epoch": epoch,
            "train_loss_total": total_L / n_batches,
            "train_loss_intent": total_Li / n_batches,
            "train_loss_action": total_La / n_batches,
        })

        # ---------------- Eval ----------------
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            # plain accuracies on the batched val loader
            c_int = t_int = 0
            c_act = t_act = 0
            with torch.no_grad():
                for xb, yib, yab in val_loader:
                    xb = xb.to(device); yib = yib.to(device); yab = yab.to(device)
                    logits_a, logits_i, _, _ = model(xb)
                    preds_i = logits_i.argmax(dim=1)
                    preds_a = logits_a.argmax(dim=1)
                    c_int  += (preds_i == yib).sum().item(); t_int += yib.size(0)
                    c_act  += (preds_a == yab).sum().item(); t_act += yab.size(0)
            avg_int = c_int / t_int if t_int else 0.0
            avg_act = c_act / t_act if t_act else 0.0

            # window‑weighted macro‑average accuracy for intents (discount earlier‑is‑heavier)
            windows: Dict[Tuple[str,int,int,int,int], Dict[str, float]] = {}
            with torch.no_grad():
                for seq, meta, label_i in zip(val_X, val_meta, val_Yi):
                    logits_a, logits_i, _, _ = model(seq.unsqueeze(0).to(device))
                    pred_i = int(torch.argmax(logits_i, dim=1).item())
                    gt_i   = int(label_i.item())
                    fid, epi, t = meta
                    rank, span_len = val_span.get(meta, (1, 1))
                    w = float(GAMMA ** (rank - 1))
                    start_t = int(t - (rank - 1))
                    end_t   = int(start_t + span_len - 1)
                    key = (fid, int(epi), start_t, end_t, gt_i)
                    bucket = windows.setdefault(key, {"wc": 0.0, "ws": 0.0})
                    bucket["ws"] += w
                    if pred_i == gt_i:
                        bucket["wc"] += w
            if windows:
                per_window_scores = [(b["wc"] / b["ws"]) for b in windows.values() if b["ws"] > 0.0]
                window_weighted_avg = float(sum(per_window_scores) / len(per_window_scores)) if per_window_scores else 0.0
            else:
                window_weighted_avg = 0.0

            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"avg_intent_accuracy={avg_int:.4f}  avg_action_accuracy={avg_act:.4f}  "
                f"WEIGHTED_window_avg_intent_accuracy={window_weighted_avg:.4f}"
            )
            wandb.log({
                "epoch": epoch,
                "avg_intent_accuracy": avg_int,
                "avg_action_accuracy": avg_act,
                "WEIGHTED_window_avg_intent_accuracy": window_weighted_avg,
            })

    print("Training complete!")


if __name__ == "__main__":
    main()
