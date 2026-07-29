"""
BNN for fov prediction itself rather than the ultimate subtask. 
Literal bayes gives like 33% right now


Run from:
  /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/bayesian
Run:
  /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python bnn_fov.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torchbnn as bnn
from sklearn import datasets
import os, sys, pathlib, pickle, random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import wandb
USE_WANDB = True

# ---------- Import your helpers / model ----------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add src/
from dataset import obs_to_list, obs_list_to_1D_vec
from model import BNN

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

#TODO: edit to reflect fov instead of intent/subtask; aka all u need to do is change y
#TODO: bascially change target at time t is y at time t??????
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
        seq_length=3,
        batch_size=128,
        hidden_dim=128,
        num_epochs=30,
        MC = 1,  # try 5–10 for validation
        lr=1e-3, 
        kl=1e-5, #was 1e-3
        seed=3,
        project="bnn_fov_full",
        run_name="seq_len_3",
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
    model  = BNN(seq_len=L, input_dim=input_dim, hidden_dim=CFG["hidden_dim"], output_dim=NUM_FOV).to(device)
    #So ce_loss is the data term
    # kl_loss is the regularization term, 
    # kl_weight controls how much the regularization term matters.
    #regularization
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False) #includes all Bayes layers
    ce_loss   = nn.CrossEntropyLoss()
    opt    = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    kl_weight = CFG["kl"]
    
    if USE_WANDB:
        wandb.init(project=CFG["project"], name=CFG["run_name"], config=CFG)
        wandb.define_metric("val_accuracy", step_metric="epoch") #percentage of correct predictions on the entire validation set (y)
    
    # ---------- Train ----------
    for epoch in range(CFG["num_epochs"]):
        
        model.train()
        total = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            #Run the model on the input batch xb and get its raw outputs (logits)
            logits = model(xb)
            
            #fits the data simple/near prior
            data_loss = ce_loss(logits, yb)
            #keeps data 
            #kll = kl_loss(model) scale KL per batch:
            
            kll = kl_loss(model) / len(train_loader)

            #kl_scale = min(1.0, epoch/50) could multiply by this on kl_weight, kll, etc. 

            #trade-off between fit and simplicity
            loss   = data_loss + kl_weight*kll
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))
        if USE_WANDB: wandb.log({"epoch": epoch, "train_loss": train_loss})

        # periodic val every epoch
        if epoch == 1 or epoch % 1 == 0 or epoch == CFG["num_epochs"]-1:
            model.eval()
            correct, tot = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    if CFG["MC"] > 1:
                        logits = torch.stack([model(xb) for _ in range(CFG["MC"])], dim=0).mean(0)
                    else:
                        logits = model(xb)
                    
                    preds = logits.argmax(dim=1) #directly return the indices of the largest score per row
                    
                    #preds = torch.max(logits.data, 1) #returns tuple (values, indices)
                    correct += (preds == yb).sum().item()
                    tot     += yb.size(0)
            val_acc = correct / tot if tot else 0.0
            print(f"Epoch {epoch:3d}/{CFG['num_epochs']}  TrainL={train_loss:.4f}  ValAcc={val_acc:.4f}")
            if USE_WANDB: wandb.log({"epoch": epoch, "val_accuracy": val_acc})

    print("Training complete!")

    #RECHECK THIS!!!!
    
    #now WandB graphs
    #add the true class cumulative graph
    
    #add the tail accuracy per subtask window graph 
    # ---------- FINAL PLOTS (append-only) ----------
    import torch.nn.functional as F
    from collections import defaultdict as _dd

    model.eval()

    # ===== 1) TRUE-CLASS probability (cumulative) per episode =====
    prob_series = []  # rows: series,fid,epi,timestep,trueprob
    MC = max(1, int(CFG.get("MC", 1)))
    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            xb = seq.unsqueeze(0).to(device)
            # MC-average probabilities (BNN)
            probs = torch.stack([F.softmax(model(xb), dim=1) for _ in range(MC)], dim=0).mean(0)  # [1,C]
            p_true = float(probs[0, int(label.item())].item())
            fid, epi, t = meta
            prob_series.append({
                "series": f"{fid}|ep{int(epi)}",
                "file_id": fid,
                "episode": int(epi),
                "timestep": int(t),
                "trueprob": p_true,
            })

    # build cumulative mean per episode
    series_pts = _dd(list)
    for r in prob_series:
        series_pts[r["series"]].append((r["timestep"], r["trueprob"]))
    series_xy_cum = {}
    for k, pts in series_pts.items():
        pts.sort(key=lambda z: z[0])
        x, y, csum = [], [], 0.0
        for i, (t, p) in enumerate(pts, start=1):
            csum += p
            x.append(t); y.append(csum / i)
        series_xy_cum[k] = {"x": x, "y": y}

    # log/sync
    if USE_WANDB:
        keys = list(series_xy_cum.keys())
        xs   = [series_xy_cum[k]["x"] for k in keys]
        ys   = [series_xy_cum[k]["y"] for k in keys]
        wandb.log({
            "final/intent_trueprob_cumulative_timeseries":
                wandb.plot.line_series(xs=xs, ys=ys, keys=keys,
                                       title="Cumulative mean true-class prob (per episode)",
                                       xname="timestep")
        })
    else:
        out_dir = PKL_DIR
        rows = []
        for k in series_xy_cum:
            for a, b in zip(series_xy_cum[k]["x"], series_xy_cum[k]["y"]):
                rows.append({"series": k, "timestep": a, "cum_trueprob": b})
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "intent_val_trueprob_cumulative_timeseries.csv"), index=False)

    # # ===== 2) Subtask "start-to-always-correct" tail percentage histogram (10% bins) =====
    
    # # recompute predictions (MC-avg probs -> argmax) to get correctness per (fid,epi,t)
    # preds_map = {}
    # with torch.no_grad():
    #     for seq, label, meta in zip(vaX, vaY, va_meta):
    #         xb = seq.unsqueeze(0).to(device)
    #         probs = torch.stack([F.softmax(model(xb), dim=1) for _ in range(MC)], dim=0).mean(0)
    #         pred  = int(probs.argmax(dim=1).item())
    #         preds_map[meta] = int(pred == int(label.item()))

    # # choose subtask label column
    # SUBTASK_COL = None
    # for cand in ["subtask", "p0_intent"]:
    #     if cand in val_df.columns:
    #         SUBTASK_COL = cand
    #         break

    # _meta_to_label = {}
    # if SUBTASK_COL is not None:
    #     for _, r in val_df.iterrows():
    #         _meta_to_label[(r["file_id"], int(r["episode"]), int(r["timestep"]))] = str(r[SUBTASK_COL])
    # else:
    #     # fallback: use intent index string
    #     for _, r in val_df.iterrows():
    #         _meta_to_label[(r["file_id"], int(r["episode"]), int(r["timestep"]))] = f"INTENT:{r.get('p0_intent')}"

    # # build per-episode, ordered by timestep
    # ep_rows = _dd(list)  # "fid|epX" -> [{t, correct, subtask}, ...]
    # for (fid, epi, t), corr in preds_map.items():
    #     ep_rows[f"{fid}|ep{int(epi)}"].append({
    #         "fid": fid, "epi": int(epi), "t": int(t),
    #         "correct": int(corr),
    #         "subtask": _meta_to_label.get((fid, int(epi), int(t)), "unknown"),
    #     })
    # for k in ep_rows:
    #     ep_rows[k].sort(key=lambda d: d["t"])

    # # tail fraction helper
    # def tail_always_correct_fraction(correct_list):
    #     if not correct_list or correct_list[-1] == 0:
    #         return 0.0
    #     tail_len = 0
    #     for c in reversed(correct_list):
    #         if c == 1: tail_len += 1
    #         else: break
    #     return tail_len / len(correct_list)

    # # compute per-subtask-window tail fractions
    # tail_fracs = []
    # for _, rows in ep_rows.items():
    #     i, n = 0, len(rows)
    #     while i < n:
    #         j = i
    #         cur = rows[i]["subtask"]
    #         window = []
    #         while j < n and rows[j]["subtask"] == cur:
    #             window.append(rows[j]["correct"])
    #             j += 1
    #         tail_fracs.append(tail_always_correct_fraction(window))
    #         i = j

    # # bin into 10% buckets
    # total = max(1, len(tail_fracs))
    # zeros = sum(1 for v in tail_fracs if v == 0.0)
    # ones  = sum(1 for v in tail_fracs if v == 1.0)
    # mids  = [v for v in tail_fracs if 0.0 < v < 1.0]

    # edges = np.linspace(0.0, 1.0, 11)  # 0.0,0.1,...,1.0
    # labels_mid = [f"{int(edges[i]*100)}–{int(edges[i+1]*100)}%" for i in range(10)]
    # if len(mids) > 0:
    #     mids_s = pd.Series(mids, dtype="float64")
    #     binned = pd.cut(mids_s, bins=edges, right=True, include_lowest=False)
    #     vc = binned.value_counts(sort=False)
    #     mid_counts = [int(vc.iloc[i]) for i in range(len(vc))]
    # else:
    #     mid_counts = [0]*10

    # bins_lbls = ["0%"] + labels_mid + ["100%"]
    # counts    = [zeros] + mid_counts + [ones]
    # percents  = [(c / total) * 100.0 for c in counts]
    # hist_df   = pd.DataFrame({"bin": bins_lbls, "percent": percents})

    # if USE_WANDB:
    #     wandb.log({
    #         "final/subtask_always_correct_tail_hist":
    #             wandb.plot.bar(
    #                 wandb.Table(dataframe=hist_df[["bin","percent"]]),
    #                 "bin", "percent",
    #                 title="Where within a subtask it becomes consistently correct (0% & 100% isolated)"
    #             )
    #     })
    # else:
    #     out_dir = PKL_DIR
    #     hist_df.to_csv(os.path.join(out_dir, "subtask_tail_always_correct_hist.csv"), index=False)



if __name__ == "__main__":
    main()
