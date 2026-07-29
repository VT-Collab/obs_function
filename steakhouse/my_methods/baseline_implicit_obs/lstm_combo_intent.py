#!/usr/bin/env python3
"""
LSTM (combo) — Predict INTENT (12-way) from obs sequences.
- Train on multiple files, validate on held-out files.
- Log per-episode cumulative accuracy by timestep (lines).
- Log per-episode TRUE-CLASS probability by timestep (lines).



Run from:
  /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_implicit_obs
Run:
  /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python lstm_combo_intent.py

"""

import os, sys, pathlib, pickle, random
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

# ---------- Import helpers / model ----------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add src/
from dataset import obs_to_list, obs_list_to_1D_vec
from model import LSTMModel  # <-- your LSTM (do not modify here)

# ---------- Data paths ----------
PKL_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/fov_traj"

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
    
    # os.path.join(PKL_DIR, "25_log_90_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "25_log_90_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "25_log_90_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "25_log_120_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "25_log_120_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "25_log_120_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "25_log_179_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "25_log_179_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "25_log_179_2_2-4_dedup.pkl"),
    
    # os.path.join(PKL_DIR, "50_log_90_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "50_log_90_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "50_log_90_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "50_log_120_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "50_log_120_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "50_log_120_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "50_log_179_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "50_log_179_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "50_log_179_2_2-4_dedup.pkl"),
    
    # os.path.join(PKL_DIR, "75_log_90_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "75_log_90_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "75_log_90_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "75_log_120_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "75_log_120_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "75_log_120_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "75_log_179_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "75_log_179_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "75_log_179_2_2-4_dedup.pkl"),
    
    # os.path.join(PKL_DIR, "mixed_log_90_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "mixed_log_90_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "mixed_log_90_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "mixed_log_120_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "mixed_log_120_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "mixed_log_120_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "mixed_log_179_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "mixed_log_179_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "mixed_log_179_2_2-4_dedup.pkl"),
    
    
]

VAL_FILES = [
    os.path.join(PKL_DIR, "log_90_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-5_dedup.pkl"),
]

# ---------- Intent space ----------
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
INTENT_TO_IDX: Dict[str,int] = {v:k for k,v in IDX_TO_INTENT.items()}
NUM_INTENTS = len(IDX_TO_INTENT)

def to_intent_idx(x: Any) -> Optional[int]:
    return INTENT_TO_IDX.get(str(x).strip(), None)

# ---------- Utils ----------
def load_pkl(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        df = pickle.load(f)
    base = pathlib.Path(path).name
    df["file_id"] = base
    if "episode" in df.columns:
        df["episode"] = df["episode"].astype(int, errors="ignore")
    if "timestep" in df.columns:
        df["timestep"] = df["timestep"].astype(int, errors="ignore")
    return df.sort_values(["file_id", "episode", "timestep"]).reset_index(drop=True)

def load_split(train_paths: List[str], val_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = [load_pkl(p) for p in train_paths if os.path.exists(p)]
    va = [load_pkl(p) for p in val_paths   if os.path.exists(p)]
    if not tr: raise RuntimeError("No training PKLs found.")
    if not va: raise RuntimeError("No validation PKLs found.")
    return pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True)

def gather_true_probs(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    return p.gather(1, y.view(-1,1)).squeeze(1)

def create_sequences_intent(df: pd.DataFrame, L: int):
    """
    Build sliding windows of size L within each (file_id, episode):
      X: (N, L, D), y: (N,), meta: [(file_id, episode, timestep_t)]
    Target at time t is p0_subtask mapped to 0..11.
    """
    groups = []
    for (fid, epi), g in df.groupby(["file_id","episode"], sort=False):
        obs_vecs, intents, meta = [], [], []
        for _, r in g.iterrows():
            ii = to_intent_idx(r.get("p0_subtask", None))
            if ii is None: continue
            try:
                vec = obs_list_to_1D_vec(obs_to_list(r["obs"]))
            except Exception:
                continue
            obs_vecs.append(vec)
            intents.append(ii)
            meta.append((fid, int(epi), int(r["timestep"])))
        if obs_vecs:
            groups.append((np.asarray(obs_vecs, np.float32), np.asarray(intents, np.int64), meta))

    X, Y, M = [], [], []
    for obs_arr, intents, meta in groups:
        T = len(intents)
        if T <= L: continue
        for i in range(L, T):
            X.append(torch.tensor(obs_arr[i-L:i], dtype=torch.float32))
            Y.append(torch.tensor(int(intents[i]), dtype=torch.long))
            M.append(meta[i])
    return X, Y, M

# ---------- Main ----------
def main():
    CFG = dict(
        seq_length=3,
        batch_size=128,
        hidden_dim=128,
        layer_dim=1,
        num_epochs=300,
        lr=1e-3,
        seed=7,
        project="9_25_steakhouse_combo",
        run_name="mixed_lstm_combo_intent",
    )

    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

    train_df, val_df = load_split(TRAIN_FILES, VAL_FILES)
    print(f"[data] TRAIN rows={len(train_df)}  VAL rows={len(val_df)}")
    print(f"[data] TRAIN files={sorted(train_df['file_id'].unique().tolist())}")
    print(f"[data] VAL   files={sorted(val_df['file_id'].unique().tolist())}")

    L = CFG["seq_length"]

    trX, trY, _       = create_sequences_intent(train_df, L)
    vaX, vaY, va_meta = create_sequences_intent(val_df,   L)
    if not trX or not vaX:
        raise RuntimeError("No sequences created — check data and intent mapping.")

    input_dim = int(trX[0].shape[-1])
    x_tr = torch.stack(trX); y_tr = torch.stack(trY)
    x_va = torch.stack(vaX); y_va = torch.stack(vaY)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=CFG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMModel(input_dim=input_dim, hidden_dim=CFG["hidden_dim"],
                       layer_dim=CFG["layer_dim"], output_dim=NUM_INTENTS).to(device)
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
    
    # ==============================
    # DROP-IN MIDDLE-OF-EPISODE EVAL — LSTM INTENT
    # ==============================
    # if USE_WANDB:
    #     wandb.finish()
    #     wandb.init(project=CFG["project"], name=CFG["run_name"] + "_dropin", config=CFG)

    # FRACTIONS = [0.10, 0.25, 0.50, 0.75]

    # #get the max timestep
    # epi_max_t = {}
    # for (fid, epi), g in val_df.groupby(["file_id","episode"], sort=False):
    #     epi_max_t[(fid, int(epi))] = int(g["timestep"].max())

    # #drop everything before; aka only keep at that fraction and after
    # def filter_dropin(vaX, vaY, va_meta, frac):
    #     keepX, keepY, keepM = [], [], []
    #     for seq, y, meta in zip(vaX, vaY, va_meta):
    #         fid, epi, t = meta
    #         t0 = max(L, int(epi_max_t[(fid, int(epi))] * frac))
    #         if t >= t0:
    #             keepX.append(seq); keepY.append(y); keepM.append(meta)
    #     return keepX, keepY, keepM

    # from collections import defaultdict as _dd

    # model.eval()
    # with torch.no_grad():
    #     for f in FRACTIONS:
    #         Xf, Yf, Mf = filter_dropin(vaX, vaY, va_meta, f)
    #         if not Xf:
    #             continue

    #         series_tp = _dd(lambda: {"x": [], "y": []})
    #         per_ep = _dd(list)

    #         for seq, y, meta in zip(Xf, Yf, Mf):
    #             logits, _, _ = model(seq.unsqueeze(0).to(device))  # LSTMModel: (logits, h, c)
    #             p_true = float(gather_true_probs(logits, y.view(1).to(device)).item())
    #             fid, epi, t = meta
    #             key = f"{fid}|ep{int(epi)}"
    #             series_tp[key]["x"].append(int(t))
    #             series_tp[key]["y"].append(p_true)

    #             pred = int(torch.argmax(logits, dim=1).item())
    #             per_ep[key].append((int(t), int(pred == int(y.item()))))

    #         series_ca = _dd(lambda: {"x": [], "y": []})
    #         for key, items in per_ep.items():
    #             items.sort(key=lambda x: x[0])
    #             csum = 0
    #             for i, (t, corr) in enumerate(items, start=1):
    #                 csum += corr
    #                 series_ca[key]["x"].append(t)
    #                 series_ca[key]["y"].append(csum / i)
            
    #         tag = str(int(f * 100))
    #         if USE_WANDB:
    #             xs = [series_tp[k]["x"] for k in series_tp]
    #             ys = [series_tp[k]["y"] for k in series_tp]
    #             wandb.log({f"dropin@{tag}/trueprob_timeseries":
    #                     wandb.plot.line_series(xs=xs, ys=ys, keys=list(series_tp.keys()),
    #                                         title=f"Intent LSTM drop-in @{tag}%", xname="timestep")})

    #             xs2 = [series_ca[k]["x"] for k in series_ca]
    #             ys2 = [series_ca[k]["y"] for k in series_ca]
    #             wandb.log({f"dropin@{tag}/cumacc_timeseries":
    #                     wandb.plot.line_series(xs=xs2, ys=ys2, keys=list(series_ca.keys()),
    #                                         title=f"Intent LSTM cum-acc drop-in @{tag}%", xname="timestep")})


if __name__ == "__main__":
    main()
