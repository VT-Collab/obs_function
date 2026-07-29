"""
BNN for fov prediction itself rather than the ultimate subtask. 
Literal bayes gives like 33% right now


Run from:
  /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/bayesian
Run:
  /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python lstm_fov.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn import datasets
import os, sys, pathlib, pickle, random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import wandb
USE_WANDB = True

# ---------- Import your helpers / model ----------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add src/
from dataset import obs_to_list, obs_list_to_1D_vec
from model import LSTMModel

# ---------- Data paths ----------
PKL_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/fov_traj/baseline"

TRAIN_FILES = [
    os.path.join(PKL_DIR, "log_90_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_90_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_90_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-4_dedup.pkl"),
    
]


VAL_FILES = [
    os.path.join(PKL_DIR, "log_90_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-5_dedup.pkl"),
]

#put index on fov
IDX_TO_FOV = {
    0: 90,
    1: 120,
    2: 179,
}

FOV_TO_IDX: Dict[int, int] = {v:k for k,v in IDX_TO_FOV.items()}

#get the number of fov 
NUM_FOV = len(IDX_TO_FOV)

def to_fov_idx(x: Any) -> Optional[int]:
    """
    Converts the raw FOV value from the DataFrame into its class index (0, 1, or 2).
    It safely attempts to cast the value to an integer before lookup.
    """
    if x is None:
        return None
    try:
        # 1. Safely convert the raw value (x) into an integer.
        # This will convert '90' -> 90 or 90 -> 90.
        fov_int = int(str(x).strip())
        
        # 2. Look up the index (0, 1, 2) using the integer FOV value as the key.
        return FOV_TO_IDX.get(fov_int, None)
    except (ValueError, TypeError):
        # Return None if conversion to int fails
        return None

# ---------- Utils ----------
#load the file and sort by file_id, episode, and timestep
#also add an fov field
def load_pkl(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        df = pickle.load(f)
    base = pathlib.Path(path).name
    #adds new file id to the data
    df["file_id"] = base
    
    #turn episode/timestep into int
    if "episode" in df.columns:
        df["episode"] = df["episode"].astype(int, errors="ignore")
    if "timestep" in df.columns:
        df["timestep"] = df["timestep"].astype(int, errors="ignore")
    return df.sort_values(["file_id", "episode", "timestep"]).reset_index(drop=True)

#Condense training and validating sequences
def load_split(train_paths: List[str], val_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = [load_pkl(p) for p in train_paths if os.path.exists(p)]
    va = [load_pkl(p) for p in val_paths   if os.path.exists(p)]
    if not tr: raise RuntimeError("No training PKLs found.")
    if not va: raise RuntimeError("No validation PKLs found.")
    return pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)

#Takes in raw scores of logits of shape BxC: B examples with C classes 
#also takes in y of the integer labels which selects the probability 
#output a 1D tensor of length B where element i is the predicted probability for the correct/true class of sample i
def gather_true_probs(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    return p.gather(1, y.view(-1,1)).squeeze(1)

def create_sequences_fov(df: pd.DataFrame, L: int):
    """
    Build sliding windows of size L within each (file_id, episode):
    X, Y, M all have length N, aka number of windows total
      X: (N, L, D), y: (N,), meta: [(file_id, episode, timestep_t)]
    
    X: sequences of observations of shape (L, D) (D = flattened obs size)
    y: the **FOV index** at the window’s last timestep (index t)
    meta: (file_id, episode, timestep_t) of the label timestep
    
    Target at time t is **FOV** mapped to 0,1,2
    """
    #each group is an entire file
    groups = []
    for (fid, epi), g in df.groupby(["file_id","episode"], sort=False):
        obs_vecs, fov, meta = [], [], []
        for _, r in g.iterrows():
            #get the index of the intent (0-11)
            ii = to_fov_idx(r.get("fov", None))
            if ii is None: continue
            #try turning into 65 dim vector 
            try:
                vec = obs_list_to_1D_vec(obs_to_list(r["obs"]))
            except Exception:
                continue
            
            #Store aligned sequences: same index across these three lists refers to the same timestep
            obs_vecs.append(vec)
            fov.append(ii)
            meta.append((fid, int(epi), int(r["timestep"])))
        if obs_vecs:
            groups.append((np.asarray(obs_vecs, np.float32), np.asarray(fov, np.int64), meta))

    X, Y, M = [], [], []
    for obs_arr, fov, meta in groups:
        #number of rows/total timesteps = T
        T = len(fov)
        #if there aren't at least L+1 timesteps, can't form a full window with a NEXT label at t 
        if T <= L: continue
        #from the sequence length L to total number of timesteps T
        for i in range(L, T):
            """
            obs_arr is a NumPy array of shape (T, D) for one episode:
            T = number of timesteps in that episode
            D = feature dimension after flattening an obs
            obs_arr[i-L:i] uses Python’s slice semantics (start inclusive, end exclusive):
            It returns the last L rows before time i → shape (L, D)
            """
            X.append(torch.tensor(obs_arr[i-L:i], dtype=torch.float32))
            Y.append(torch.tensor(int(fov[i]), dtype=torch.long))
            M.append(meta[i])
    return X, Y, M



# ---------- Main ----------
def main():
    
    #hyperparameters
    CFG = dict(
        seq_length=32,
        batch_size=128,
        hidden_dim=128,
        layer_dim=1,
        num_epochs=300,
        lr=1e-3, 
        kl=1e-5, #was 1e-3
        seed=3,
        project="lstm_fov_full",
        run_name="seq_len_32",
    )
    

    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

    #get rows of training and validating
    train_df, val_df = load_split(TRAIN_FILES, VAL_FILES)
    print(f"[data] TRAIN rows={len(train_df)}  VAL rows={len(val_df)}")
    print(f"[data] TRAIN files={sorted(train_df['file_id'].unique().tolist())}")
    print(f"[data] VAL   files={sorted(val_df['file_id'].unique().tolist())}")

    L = CFG["seq_length"]

    trX, trY, _       = create_sequences_fov(train_df, L)
    vaX, vaY, va_meta = create_sequences_fov(val_df,   L)
    if not trX or not vaX:
        raise RuntimeError("No sequences created — check data and intent mapping.")

    input_dim = int(trX[0].shape[-1])
    x_tr = torch.stack(trX); y_tr = torch.stack(trY)
    x_va = torch.stack(vaX); y_va = torch.stack(vaY)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=CFG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMModel(input_dim=input_dim, hidden_dim=CFG["hidden_dim"],
                       layer_dim=CFG["layer_dim"], output_dim=NUM_FOV).to(device)    
   
    crit   = nn.CrossEntropyLoss()
    opt    = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    
    if USE_WANDB:
        wandb.init(project=CFG["project"], name=CFG["run_name"], config=CFG)
        wandb.define_metric("val_accuracy", step_metric="epoch") #percentage of correct predictions on the entire validation set (y)
    
    # ---------- Train ----------
    for epoch in range(CFG["num_epochs"]):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _, _ = model(xb)
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
                    preds = model(xb)[0].argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    tot     += yb.size(0)
            val_acc = correct / tot if tot else 0.0
            print(f"Epoch {epoch:3d}/{CFG['num_epochs']}  TrainL={train_loss:.4f}  ValAcc={val_acc:.4f}")
            if USE_WANDB: wandb.log({"epoch": epoch, "val_accuracy": val_acc})

    print("Training complete!")

    # ---------- FINAL: per-episode lines ----------
    # 1) TRUE-CLASS prob by timestep
    model.eval()
    prob_series = []
    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            logits, _, _ = model(seq.unsqueeze(0).to(device))
            p_true = float(gather_true_probs(logits, label.view(1).to(device)).item())
            fid, epi, t = meta
            prob_series.append({
                "series": f"{fid}|ep{int(epi)}",
                "file_id": fid,
                "episode": int(epi),
                "timestep": int(t),
                "trueprob": p_true,
            })

    # 2) Cumulative accuracy by timestep (one line per episode)
    preds_map = {}
    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            logits, _, _ = model(seq.unsqueeze(0).to(device))
            pred   = int(torch.argmax(logits, dim=1).item())
            preds_map[meta] = int(pred == int(label.item()))

    ep_buckets = defaultdict(list)
    for (fid, epi, t), corr in preds_map.items():
        ep_buckets[f"{fid}|ep{int(epi)}"].append((int(t), int(corr)))

    cum_lines = []
    for series, items in ep_buckets.items():
        items.sort(key=lambda x: x[0])
        csum = 0
        for i, (t, corr) in enumerate(items, start=1):
            csum += corr
            cum_acc = csum / i
            cum_lines.append({"series": series, "timestep": t, "cum_acc": float(cum_acc)})

    # ---------- Log to W&B or CSV ----------
    prob_series.sort(key=lambda d: (d["series"], d["timestep"]))
    cum_lines.sort(key=lambda d: (d["series"], d["timestep"]))

    if USE_WANDB:
        # True-prob
        series_xy = defaultdict(lambda: {"x": [], "y": []})
        for r in prob_series:
            series_xy[r["series"]]["x"].append(r["timestep"])
            series_xy[r["series"]]["y"].append(r["trueprob"])
        keys = list(series_xy.keys())
        xs   = [series_xy[k]["x"] for k in keys]
        ys   = [series_xy[k]["y"] for k in keys]
        wandb.log({
            "final/val_trueprob_timeseries":
                wandb.plot.line_series(xs=xs, ys=ys, keys=keys,
                                       title="Intent (LSTM): true-class prob by timestep",
                                       xname="timestep")
        })

        # Cumulative accuracy
        series_xy2 = defaultdict(lambda: {"x": [], "y": []})
        for r in cum_lines:
            series_xy2[r["series"]]["x"].append(r["timestep"])
            series_xy2[r["series"]]["y"].append(r["cum_acc"])
        keys2 = list(series_xy2.keys())
        xs2   = [series_xy2[k]["x"] for k in keys2]
        ys2   = [series_xy2[k]["y"] for k in keys2]
        wandb.log({
            "final/val_cumacc_by_timestep":
                wandb.plot.line_series(xs=xs2, ys=ys2, keys=keys2,
                                       title="Intent (LSTM): cumulative accuracy by timestep",
                                       xname="timestep")
        })
    else:
        out_dir = PKL_DIR
        pd.DataFrame(prob_series).to_csv(os.path.join(out_dir, "lstm_intent_val_trueprob_timeseries.csv"), index=False)
        pd.DataFrame(cum_lines).to_csv(os.path.join(out_dir, "lstm_intent_val_cumacc_timeseries.csv"), index=False)
        print("[saved] CSVs in", out_dir)
        
if __name__ == "__main__":
    main()      
        