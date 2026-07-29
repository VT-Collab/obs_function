#!/usr/bin/env python3
"""
Train MLP to predict the NEXT subtask from flattened MiniGrid CSV.
- Uses a single CSV file with per-timestep rows and 'agent_subtask' labels.
- Splits by episode_seed: bottom 20% episode ids -> validation (unseen episodes).
- Inputs are sliding windows of length L over numeric features.
- Target y(t) = subtask at the NEXT step after the input window.

Notes:
- Expects columns produced by your collect_data/write_episode_csv, including:
  ['episode_seed','timestep', ... numeric features ..., 'agent_subtask']
- We exclude 'agent_subtask' from X and use it (at t) for y at (t), where inputs are
  frames [t-L, ..., t-1]. So the model predicts the subtask of the "next" frame
  relative to the last input in the window.
"""

import os, sys, pathlib, random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- Optional: Weights & Biases ----------
try:
    import wandb
    USE_WANDB = True
except Exception:
    USE_WANDB = False

# ---------- Project imports ----------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))      # current dir
sys.path.append(str(pathlib.Path(__file__).resolve().parents[3]))      # repo root
from robot.estimation.mlp.dataset import flatten_record, flatten_episode, expected_dim
from robot.estimation.mlp.model import MLP

# ---------- Config ----------
CSV_PATH = "scripts/runs/lockedroom_success_all.csv"   # <--- point to your big CSV
SEQ_LEN  = 3                                   # sliding window length (L)
BATCH    = 128
HIDDEN   = 128
EPOCHS   = 100
LR       = 1e-3
SEED     = 7
PROJECT  = "minigrid_next_subtask"
RUN_NAME = "csv1_next_subtask_split_bottom20"

# Subtask vocabulary (must match what your generator outputs)
SUBTASK_VOCAB: List[str] = [
    "find_key_room_door",
    "goto_key_room",
    "find_key",
    "pickup_key",
    "find_locked_room",
    "goto_locked_room",
    "find_goal",
    "goto_goal",
]
SUB2IDX: Dict[str, int] = {s: i for i, s in enumerate(SUBTASK_VOCAB)}
NUM_CLASSES = len(SUBTASK_VOCAB)

# ---------- Utils ----------
def to_sub_idx(s: Any) -> Optional[int]:
    if s is None:
        return None
    return SUB2IDX.get(str(s).strip(), None)

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    # enforce ints where applicable
    if "episode_seed" in df.columns:
        df["episode_seed"] = df["episode_seed"].astype(int, errors="ignore")
    if "timestep" in df.columns:
        df["timestep"] = df["timestep"].astype(int, errors="ignore")
    # ensure sorted within episode
    df = df.sort_values(["episode_seed", "timestep"], ascending=[True, True]).reset_index(drop=True)
    return df

def split_by_episode_bottom20(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bottom 20% of episodes by episode_seed -> VAL (unseen), rest -> TRAIN.
    """
    if "episode_seed" not in df.columns:
        raise ValueError("CSV must contain 'episode_seed' column.")
    episodes = sorted(df["episode_seed"].unique().tolist())
    n = len(episodes)
    k = max(1, int(np.floor(0.20 * n)))
    val_episodes = set(episodes[:k])   # bottom (smallest ids) as validation
    tr = df[~df["episode_seed"].isin(val_episodes)].copy()
    va = df[df["episode_seed"].isin(val_episodes)].copy()
    return tr, va

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    All numeric feature columns EXCEPT 'agent_subtask'.
    We keep 'episode_seed' and 'timestep' in the tensor too (they are integers), because
    your flatten vectors included them; keeping them preserves fixed D = expected_dim().
    """
    if "agent_subtask" not in df.columns:
        raise ValueError("CSV must contain 'agent_subtask' as the string label column.")
    # All numeric columns become features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep everything numeric (including episode_seed, timestep) to match expected_dim()
    # Sanity check vs expected_dim (not strictly required, but helpful)
    D_exp = expected_dim()
    # Derive D from numeric columns count:
    D_obs = len(numeric_cols)
    if D_obs != D_exp:
        # Not fatal; just warn so you can align feature construction if needed
        print(f"[warn] observed feature dim {D_obs} != expected_dim() {D_exp}. Proceeding anyway.")
    return numeric_cols

def create_sequences_next_subtask(df: pd.DataFrame, L: int, feat_cols: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int,int]]]:
    """
    Build sliding windows within each episode:
      Inputs:  X[i] = frames [t-L, ..., t-1]  (each frame is the numeric feature vector)
      Target:  y[i] = subtask at time t       (the NEXT subtask)
    Returns:
      - X: list of tensors of shape (L, D)
      - Y: list of scalar Long tensors (class ids)
      - M: list of (episode_seed, t) tuples for optional per-episode metrics
    """
    X, Y, M = [], [], []
    for epi, g in df.groupby("episode_seed", sort=True):
        g = g.sort_values("timestep", ascending=True).reset_index(drop=True)
        # Map labels
        labels = [to_sub_idx(s) for s in g["agent_subtask"].tolist()]
        feats  = g[feat_cols].to_numpy(dtype=np.int64)  # ints -> model will receive float32 later
        T = len(g)
        if T <= L:
            continue
        # Build sequences
        for t in range(L, T):
            y = labels[t]
            if y is None:
                continue
            window = feats[t-L:t]                 # shape (L, D)
            X.append(torch.tensor(window, dtype=torch.float32))
            Y.append(torch.tensor(int(y), dtype=torch.long))
            M.append((int(epi), int(g.loc[t, "timestep"])))
    return X, Y, M

def gather_true_probs(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    return p.gather(1, y.view(-1,1)).squeeze(1)

# ---------- Main ----------
def main():
    CFG = dict(
        seq_length=SEQ_LEN,
        batch_size=BATCH,
        hidden_dim=HIDDEN,
        num_epochs=EPOCHS,
        lr=LR,
        seed=SEED,
        project=PROJECT,
        run_name=RUN_NAME,
    )

    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

    df = load_csv(CSV_PATH)
    train_df, val_df = split_by_episode_bottom20(df)
    print(f"[data] rows: TRAIN={len(train_df)}  VAL={len(val_df)}")
    print(f"[data] episodes: TRAIN={len(train_df['episode_seed'].unique())}  VAL={len(val_df['episode_seed'].unique())}")

    feat_cols = get_feature_columns(df)
    L = CFG["seq_length"]

    trX, trY, _       = create_sequences_next_subtask(train_df, L, feat_cols)
    vaX, vaY, va_meta = create_sequences_next_subtask(val_df,   L, feat_cols)
    if not trX or not vaX:
        raise RuntimeError("No sequences created — check CSV, labels, and SEQ_LEN.")

    input_dim = int(trX[0].shape[-1])   # D
    x_tr = torch.stack(trX)             # (Ntr, L, D)
    y_tr = torch.stack(trY)             # (Ntr,)
    x_va = torch.stack(vaX)             # (Nva, L, D)
    y_va = torch.stack(vaY)             # (Nva,)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=CFG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MLP(seq_len=L, input_dim=input_dim, hidden_dim=CFG["hidden_dim"], output_dim=NUM_CLASSES).to(device)
    crit   = nn.CrossEntropyLoss()
    opt    = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    if USE_WANDB:
        wandb.init(project=CFG["project"], name=CFG["run_name"], config=CFG)
        wandb.define_metric("val_accuracy", step_metric="epoch")

    # ---------- Train ----------
    for epoch in range(CFG["num_epochs"]):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))
        if USE_WANDB: wandb.log({"epoch": epoch, "train_loss": train_loss})

        # periodic val
        if epoch == 1 or epoch % 10 == 0 or epoch == CFG["num_epochs"]-1:
            model.eval()
            correct, tot = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    tot     += yb.size(0)
            val_acc = correct / tot if tot else 0.0
            print(f"Epoch {epoch:3d}/{CFG['num_epochs']}  TrainL={train_loss:.4f}  ValAcc={val_acc:.4f}")
            if USE_WANDB: wandb.log({"epoch": epoch, "val_accuracy": val_acc})

    print("Training complete!")

    # ---------- FINAL: per-episode curves on VAL ----------
    # (i) true-class prob by timestep
    model.eval()
    prob_series = []
    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            logits = model(seq.unsqueeze(0).to(device))  # (1, C)
            p_true = float(gather_true_probs(logits, label.view(1).to(device)).item())
            epi, t = meta
            prob_series.append({"series": f"ep{epi}", "episode_seed": epi, "timestep": t, "trueprob": p_true})

    # (ii) cumulative accuracy by timestep per episode
    preds_map = {}
    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            logits = model(seq.unsqueeze(0).to(device))
            pred   = int(torch.argmax(logits, dim=1).item())
            preds_map[meta] = int(pred == int(label.item()))

    ep_buckets = defaultdict(list)
    for (epi, t), corr in preds_map.items():
        ep_buckets[f"ep{epi}"].append((t, corr))

    cum_lines = []
    for series, items in ep_buckets.items():
        items.sort(key=lambda x: x[0])
        csum = 0
        for i, (t, corr) in enumerate(items, start=1):
            csum += corr
            cum_acc = csum / i
            cum_lines.append({"series": series, "timestep": int(t), "cum_acc": float(cum_acc)})

    # ---------- Log/Save ----------
    out_dir = str(pathlib.Path(CSV_PATH).parent)
    prob_series.sort(key=lambda d: (d["series"], d["timestep"]))
    cum_lines.sort(key=lambda d: (d["series"], d["timestep"]))

    if USE_WANDB:
        sx = defaultdict(lambda: {"x": [], "y": []})
        for r in prob_series:
            sx[r["series"]]["x"].append(r["timestep"])
            sx[r["series"]]["y"].append(r["trueprob"])
        keys = list(sx.keys())
        xs   = [sx[k]["x"] for k in keys]
        ys   = [sx[k]["y"] for k in keys]
        wandb.log({
            "final/val_trueprob_timeseries":
                wandb.plot.line_series(xs=xs, ys=ys, keys=keys,
                                       title="Next-Subtask: true-class prob by timestep (per episode)",
                                       xname="timestep")
        })

        sx2 = defaultdict(lambda: {"x": [], "y": []})
        for r in cum_lines:
            sx2[r["series"]]["x"].append(r["timestep"])
            sx2[r["series"]]["y"].append(r["cum_acc"])
        keys2 = list(sx2.keys())
        xs2   = [sx2[k]["x"] for k in keys2]
        ys2   = [sx2[k]["y"] for k in keys2]
        wandb.log({
            "final/val_cumacc_by_timestep":
                wandb.plot.line_series(xs=xs2, ys=ys2, keys=keys2,
                                       title="Next-Subtask: cumulative accuracy by timestep (per episode)",
                                       xname="timestep")
        })
    else:
        pd.DataFrame(prob_series).to_csv(os.path.join(out_dir, "next_subtask_val_trueprob_timeseries.csv"), index=False)
        pd.DataFrame(cum_lines).to_csv(os.path.join(out_dir, "next_subtask_val_cumacc_timeseries.csv"), index=False)
        print(f"[saved] CSVs in {out_dir}")

if __name__ == "__main__":
    main()
