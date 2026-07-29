#first do normal accuracy 
#giving extra credit for guessing right early
#We also want to experiment with weighted accuracy, where correct predictions earlier in the subtask are given higher value than correct predictions later on.
#takes in obs, return 1 int but its like space of 12 kind of thing, predicting p0_intent

#run from
#/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_action_pred
#script to run
#/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python train_intent.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train an MLP to predict p0_intent (12 classes) from your 65-D obs vector.
Inputs: sequences of length L of obs65 (no intent features).
Label:  p0_intent at time t (0..11).
Logs two metrics you care about:
  - average_accuracy
  - weighted_accuracy   (earlier steps in a subtask count more)
"""

import os, sys, pickle, pathlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

# add src/ so we can import your MLP + dataset helpers
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from model import MLP
from dataset import obs_to_list, obs_list_to_1D_vec

# ---------------------- fixed config (no CLI args) ----------------------

WANDB_PROJECT = "intent_mlp_combo"
WANDB_ENTITY  = "steakteam"

CFG = dict(
    seq_length=3,                 # same as your action baseline (tweak later if you want)
    train_to_validate_ratio=0.9,  # split by episodes
    batch_size=128,
    hidden_dim=40,
    num_epochs=350,
    lr=1e-4,
    seed=0,
    input_dim=65,   # <-- strictly 65-D obs vector
    output_dim=12,  # 12 intents
)

# intent mapping (1..12 → names); we train with labels 0..11
ML_INDEX_ACTION_DICT = {
    1: 'pickup_meat',
    2: 'pickup_onion',
    3: 'pickup_dirty_plate',
    4: 'drop_meat',
    5: 'drop_onion',
    6: 'drop_dirty_plate',
    7: 'chop_onion',
    8: 'rinse_plate',
    9: 'pickup_clean_plate',
    10: 'pickup_steak',
    11: 'add_garnish',
    12: 'deliver_dish'
}
INTENT2IDX = {v: (k-1) for k, v in ML_INDEX_ACTION_DICT.items()}
IDX2INTENT = {idx: name for name, idx in INTENT2IDX.items()}

# ------------------------------ utils --------------------------------

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_df(path):
    """Load PKL (DataFrame) or CSV. Ensure file_id, episode, timestep, p0_intent_idx."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            df = pickle.load(f)
    else:
        df = pd.read_csv(path)

    if "file_id" not in df.columns:
        df["file_id"] = os.path.basename(path)

    df["episode"]  = df["episode"].astype(int)
    df["timestep"] = df["timestep"].astype(int)

    if "obs" not in df.columns:
        raise ValueError(f"{path} missing 'obs' column")

    if "p0_intent" not in df.columns:
        raise ValueError(f"{path} missing 'p0_intent' column")

    df["p0_intent_idx"] = df["p0_intent"].map(INTENT2IDX)
    if df["p0_intent_idx"].isna().any():
        bad = sorted(df.loc[df["p0_intent_idx"].isna(), "p0_intent"].unique())
        raise ValueError(f"Unrecognized intent labels {bad}. Update INTENT2IDX.")
    df["p0_intent_idx"] = df["p0_intent_idx"].astype(int)
    return df

def build_obs65_column(df):
    """Parse the 'obs' string with your helpers and store a 65-D numpy vector per row."""
    def _to65(row):
        obs_list = obs_to_list(str(row["obs"]))         # -> python list with positions, helds, dicts, etc.
        vec65    = obs_list_to_1D_vec(obs_list)         # -> list length 65
        return np.asarray(vec65, dtype=np.float32)
    df["_obs65"] = df.apply(_to65, axis=1)
    return df

def create_sequences_intent(df, L, train_ratio, seed=0):
    """
    Build (X, y, meta) using only obs65.
    X: (N, L, 65) = rows [t-L .. t-1]
    y: (N,)        = p0_intent_idx at time t
    meta: list of (file_id, episode, timestep t)
    """
    # split by (file_id, episode)
    epi_keys = sorted(df[["file_id","episode"]].drop_duplicates().itertuples(index=False, name=None))
    rng = np.random.default_rng(seed); rng.shuffle(epi_keys)
    n_train = int(len(epi_keys) * train_ratio)
    train_keys = set(epi_keys[:n_train]); val_keys = set(epi_keys[n_train:])

    def build(keys):
        X, Y, M = [], [], []
        for (fid, epi) in keys:
            g = df[(df["file_id"]==fid) & (df["episode"]==epi)].sort_values("timestep")
            if g.empty:
                continue
            obs_arr = np.stack(g["_obs65"].to_list(), axis=0).astype(np.float32)  # (T,65)
            intents = g["p0_intent_idx"].to_numpy(dtype=np.int64)                 # (T,)
            ts_list = g["timestep"].to_numpy(dtype=int)                           # (T,)
            T = len(obs_arr)
            if T <= L:
                continue
            for i in range(L, T):
                X.append(torch.tensor(obs_arr[i-L:i], dtype=torch.float32))
                Y.append(torch.tensor(int(intents[i]), dtype=torch.long))
                M.append((fid, int(epi), int(ts_list[i])))
        return X, Y, M

    train_X, train_Y, train_M = build(train_keys)
    val_X,   val_Y,   val_M   = build(val_keys)
    return train_X, train_Y, val_X, val_Y, (train_M, val_M)

def segments_for_episode(df_epi_sorted):
    """Return contiguous segments where p0_intent_idx is constant: [(intent_idx, [t0..tk])]."""
    segs = []
    cur_int, cur_ts = None, []
    for _, r in df_epi_sorted.iterrows():
        ii = int(r["p0_intent_idx"]); t = int(r["timestep"])
        if cur_int is None: cur_int, cur_ts = ii, [t]
        elif ii == cur_int and (not cur_ts or t == cur_ts[-1] + 1):
            cur_ts.append(t)
        else:
            segs.append((cur_int, cur_ts)); cur_int, cur_ts = ii, [t]
    if cur_ts: segs.append((cur_int, cur_ts))
    return segs

def compute_segment_metrics(full_df, val_meta, preds_map):
    """
    average_accuracy:
      For each ground-truth subtask segment, (# correct predictions during that segment) / (# predictions in that segment),
      then average across segments.

    weighted_accuracy:
      Same but within each segment, earlier steps get more weight (1, 1/2, 1/3, ...).
    """
    val_keys = set((fid, epi) for (fid, epi, _) in val_meta)
    per_seg_acc, per_seg_wacc = [], []

    for (fid, epi), g in full_df.groupby(["file_id","episode"]):
        if (fid, epi) not in val_keys:
            continue
        g = g.sort_values("timestep").reset_index(drop=True)
        segs = segments_for_episode(g)
        for intent_idx, ts in segs:
            ts_pred = [t for t in ts if (fid, epi, t) in preds_map]
            if not ts_pred:
                continue
            # unweighted
            corr  = sum(int(preds_map[(fid,epi,t)] == intent_idx) for t in ts_pred)
            total = len(ts_pred)
            per_seg_acc.append(corr / total)

            # weighted: 1, 1/2, 1/3, ...
            ts_pred_sorted = sorted(ts_pred)
            wcorr = 0.0; wtot = 0.0
            for pos, t in enumerate(ts_pred_sorted):
                w = 1.0 / (1.0 + pos)
                wcorr += w * (1.0 if preds_map[(fid,epi,t)] == intent_idx else 0.0)
                wtot  += w
            per_seg_wacc.append(wcorr / wtot if wtot > 0 else 0.0)

    avg_acc  = float(np.mean(per_seg_acc))  if per_seg_acc  else 0.0
    avg_wacc = float(np.mean(per_seg_wacc)) if per_seg_wacc else 0.0
    return avg_acc, avg_wacc

# ---------------------------------- main ----------------------------------

def main():
    set_seed(CFG["seed"])

    # use only the *_intent.{pkl|csv} files
    base_dir = Path(__file__).resolve().parents[1] / "data"
    candidates = [
        ("third_90_intent.pkl",  "third_90_intent.csv"),
        ("third_120_intent.pkl", "third_120_intent.csv"),
        ("third_179_intent.pkl", "third_179_intent.csv"),
    ]
    file_paths = []
    for pkl, csv in candidates:
        pklp = base_dir / pkl
        csvp = base_dir / csv
        if pklp.exists(): file_paths.append(str(pklp))
        elif csvp.exists(): file_paths.append(str(csvp))
    if not file_paths:
        raise FileNotFoundError("Could not find third_*_intent.(pkl|csv) in ../data/")

    dfs = [load_df(p) for p in file_paths]
    data = pd.concat(dfs, ignore_index=True)

    # build the 65-D feature column from your obs strings
    data = build_obs65_column(data)

    print(f"Loaded {len(data)} rows across {data['episode'].nunique()} episodes; "
          f"files: {sorted(data['file_id'].unique())}")
    print("Using 65-D input = obs_list_to_1D_vec(...) only")

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=dict(
            seq_length=CFG["seq_length"],
            train_to_validate_ratio=CFG["train_to_validate_ratio"],
            batch_size=CFG["batch_size"],
            hidden_dim=CFG["hidden_dim"],
            num_epochs=CFG["num_epochs"],
            lr=CFG["lr"],
            model="MLP(65-d)",
        ),
    )
    wandb.define_metric("average_accuracy",  step_metric="epoch")
    wandb.define_metric("weighted_accuracy", step_metric="epoch")

    L = CFG["seq_length"]

    train_X, train_Y, val_X, val_Y, (train_meta, val_meta) = \
        create_sequences_intent(data, L=L, train_ratio=CFG["train_to_validate_ratio"], seed=CFG["seed"])

    print(f"Training sequences:   {len(train_X)}")
    print(f"Validation sequences: {len(val_X)}")
    if len(train_X) != len(train_Y) or len(val_X) != len(val_Y):
        raise ValueError("Input/label length mismatch")

    # DataLoaders
    x_train = torch.stack(train_X)  # (N, L, 65)
    y_train = torch.stack(train_Y).long()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=CFG["batch_size"], shuffle=True)

    x_val = torch.stack(val_X)      # (N_val, L, 65)
    y_val = torch.stack(val_Y).long()
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=CFG["batch_size"], shuffle=False)

    # Model / loss / opt
    model = MLP(seq_len=L, input_dim=CFG["input_dim"], hidden_dim=CFG["hidden_dim"], output_dim=CFG["output_dim"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    for epoch in range(CFG["num_epochs"]):
        # -------- train --------
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))
        wandb.log({"epoch": epoch, "train_loss": train_loss})

        # -------- validate (periodically) --------
        if epoch == 1 or epoch % 10 == 0 or epoch == CFG["num_epochs"] - 1:
            model.eval()
            preds_map = {}  # (fid, epi, t) -> pred idx
            with torch.no_grad():
                idx_meta = 0
                for xb, _ in val_loader:
                    logits = model(xb)
                    preds  = logits.argmax(dim=1)
                    for j in range(preds.size(0)):
                        fid, epi, t = val_meta[idx_meta + j]
                        preds_map[(fid, epi, t)] = int(preds[j].item())
                    idx_meta += preds.size(0)

            avg_acc, avg_wacc = compute_segment_metrics(data, val_meta, preds_map)
            print(f"Epoch {epoch:3d}/{CFG['num_epochs']}  Train L={train_loss:.4f}  "
                  f"average_accuracy={avg_acc:.4f}  weighted_accuracy={avg_wacc:.4f}")
            wandb.log({
                "epoch": epoch,
                "average_accuracy": avg_acc,
                "weighted_accuracy": avg_wacc,  # << WEIGHTED in the name
            })

    print("Training complete!")

if __name__ == "__main__":
    main()
