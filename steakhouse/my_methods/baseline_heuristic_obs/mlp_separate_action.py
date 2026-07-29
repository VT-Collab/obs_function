#!/usr/bin/env python3

'''
Run from:
  /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_heuristic_obs
Run:
  /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python mlp_separate_action.py

'''



import os, sys, pathlib, pickle, random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import wandb

# add src/ to path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataset import obs_to_list, obs_list_to_1D_vec
from model import MLP

# ----------------------------- paths (edit) -----------------------------
PKL_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/fov_traj"

TRAIN_FILES = [
    os.path.join(PKL_DIR, "log_90_1_1-4.pkl"),
    os.path.join(PKL_DIR, "log_90_2_1-2.pkl"),
    os.path.join(PKL_DIR, "log_90_2_2-4.pkl"),

    os.path.join(PKL_DIR, "log_120_1_1-4.pkl"),
    os.path.join(PKL_DIR, "log_120_2_1-2.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-4.pkl"),

    os.path.join(PKL_DIR, "log_179_1_1-4.pkl"),
    os.path.join(PKL_DIR, "log_179_2_1-2.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-4.pkl"),
]

VAL_FILES = [
    os.path.join(PKL_DIR, "log_90_2_2-5.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-5.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-5.pkl"),
]

WANDB_PROJECT = "steakhouse_trueprob"
WANDB_ENTITY  = "steakteam"
RUN_NAME      = "mlp_separate_action"

# ----------------------------- utils -----------------------------
def load_pkl(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        df = pickle.load(f)
    df["file_id"]  = pathlib.Path(path).name
    df["episode"]  = df["episode"].astype(int)
    df["timestep"] = df["timestep"].astype(int)
    return df.sort_values(["file_id","episode","timestep"]).reset_index(drop=True)

def build_action_mapping(dfs: List[pd.DataFrame]) -> Tuple[Dict[Any,int], Dict[int,Any]]:
    uniq = pd.unique(pd.concat([d["p0_action"] for d in dfs], ignore_index=True))
    uniq_sorted = sorted(list(uniq), key=lambda v: str(v))
    a2i = {v:i for i,v in enumerate(uniq_sorted)}
    i2a = {i:v for v,i in a2i.items()}
    return a2i, i2a

def create_sequences_action(df: pd.DataFrame, L: int, action_to_idx: Dict[Any,int]):
    X, Y, M = [], [], []
    for (fid, epi), g in df.groupby(["file_id","episode"], sort=False):
        vecs, acts, ts = [], [], []
        for _, r in g.iterrows():
            try:
                v = obs_list_to_1D_vec(obs_to_list(r["obs"]))
            except Exception:
                continue
            a_idx = action_to_idx.get(r["p0_action"], action_to_idx.get(str(r["p0_action"]), None))
            if a_idx is None: continue
            vecs.append(v); acts.append(a_idx); ts.append(int(r["timestep"]))
        T = len(acts)
        if T <= L: continue
        for i in range(L, T):
            X.append(torch.tensor(np.asarray(vecs[i-L:i], dtype=np.float32)))
            Y.append(torch.tensor(int(acts[i]), dtype=torch.long))
            M.append((fid, int(epi), ts[i]))
    return X, Y, M

def gather_true_probs(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return probs.gather(1, y.view(-1,1)).squeeze(1)

def log_episode_lines_softprob(tag_prefix: str, series_rows: list):
    # per-episode multi-line plot + avg over episodes at each timestep
    series_rows.sort(key=lambda d: (d["series"], d["timestep"]))
    # 1) multi-line
    from collections import defaultdict
    series_xy = defaultdict(lambda: {"x": [], "y": []})
    for r in series_rows:
        series_xy[r["series"]]["x"].append(r["timestep"])
        series_xy[r["series"]]["y"].append(r["trueprob"])
    keys = list(series_xy.keys())
    xs = [series_xy[k]["x"] for k in keys]
    ys = [series_xy[k]["y"] for k in keys]
    wandb.log({f"{tag_prefix}/per_episode_line":
        wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title=f"{tag_prefix} — true prob by timestep", xname="timestep")
    })
    # 2) avg across episodes at each timestep
    by_t = defaultdict(list)
    for r in series_rows:
        by_t[r["timestep"]].append(r["trueprob"])
    tbl = wandb.Table(columns=["timestep","avg_trueprob"])
    for t in sorted(by_t):
        tbl.add_data(t, float(np.mean(by_t[t])))
    wandb.log({f"{tag_prefix}/avg_by_timestep":
        wandb.plot.line(tbl, "timestep", "avg_trueprob", title=f"{tag_prefix} — avg true prob by timestep")
    })

# ----------------------------- main -----------------------------
def main():
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=RUN_NAME, config=dict(
        seq_length=3, batch_size=128, hidden_dim=128, num_epochs=200, lr=1e-3
    ))
    CFG = wandb.config
    L, B, H, E, LR = int(CFG.seq_length), int(CFG.batch_size), int(CFG.hidden_dim), int(CFG.num_epochs), float(CFG.lr)

    # pre-load everything once to build a global action mapping (train+val)
    all_dfs = [load_pkl(p) for p in TRAIN_FILES + VAL_FILES if os.path.exists(p)]
    a2i, _ = build_action_mapping(all_dfs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train a separate model per TRAIN file and evaluate on every VAL file
    for train_path in TRAIN_FILES:
        if not os.path.exists(train_path):
            print(f"[skip missing train] {train_path}"); continue
        tr_df = load_pkl(train_path)
        trX, trY, _ = create_sequences_action(tr_df, L, a2i)
        if not trX:
            print(f"[no sequences in] {train_path}"); continue

        x_tr = torch.stack(trX); y_tr = torch.stack(trY)
        tr_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=B, shuffle=True)

        # infer input dim from first window
        D = int(x_tr.shape[-1])

        model = MLP(seq_len=L, input_dim=D, hidden_dim=H, output_dim=len(a2i)).to(device)
        ce = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        # quick train loop
        for epoch in range(E):
            model.train(); total = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = ce(logits, yb)
                loss.backward(); opt.step()
                total += loss.item()
            if epoch % 10 == 0 or epoch == E-1:
                wandb.log({"epoch": epoch, f"{pathlib.Path(train_path).name}/train_loss": total/max(1,len(tr_loader))})

        # post-training: evaluate soft accuracy lines on every VAL file
        model.eval()
        for val_path in VAL_FILES:
            if not os.path.exists(val_path):
                print(f"[skip missing val] {val_path}"); continue
            va_df = load_pkl(val_path)
            vaX, vaY, vaM = create_sequences_action(va_df, L, a2i)
            if not vaX:
                print(f"[no val sequences in] {val_path}"); continue

            series = []
            with torch.no_grad():
                for seq, lab, meta in zip(vaX, vaY, vaM):
                    logits = model(seq.unsqueeze(0).to(device))
                    p_true = float(gather_true_probs(logits, lab.view(1).to(device)).item())
                    fid, epi, t = meta
                    series.append({
                        "series": f"{pathlib.Path(train_path).name}→{fid}|ep{epi}",
                        "file_id": fid, "episode": int(epi), "timestep": int(t),
                        "trueprob": p_true
                    })
            tag = f"separate_action/{pathlib.Path(train_path).name}/on/{pathlib.Path(val_path).name}"
            log_episode_lines_softprob(tag, series)

if __name__ == "__main__":
    main()
