#!/usr/bin/env python3
"""
MLP (combo) — Predict ACTION from obs sequences.
- Train on multiple files, validate on held-out files.
- Log per-episode cumulative accuracy by timestep (lines).
- Log per-episode TRUE-CLASS probability by timestep (lines).

Run from:
  /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_action_pred
Run:
  /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python mlp_combo_action.py
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

# ---------- Import your helpers / model ----------
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # add src/
from dataset import obs_to_list, obs_list_to_1D_vec
from model import MLP  # <-- your MLP (do not modify)

# ---------- Data paths ----------
PKL_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/data/fov_traj"

TRAIN_FILES = [
    # os.path.join(PKL_DIR, "log_90_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "log_90_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "log_90_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "log_120_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "log_120_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "log_120_2_2-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "log_179_1_1-4_dedup.pkl"),
    # os.path.join(PKL_DIR, "log_179_2_1-2_dedup.pkl"),
    # os.path.join(PKL_DIR, "log_179_2_2-4_dedup.pkl"),
    
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
    
    os.path.join(PKL_DIR, "mixed_log_90_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "mixed_log_90_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "mixed_log_90_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "mixed_log_120_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "mixed_log_120_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "mixed_log_120_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "mixed_log_179_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "mixed_log_179_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "mixed_log_179_2_2-4_dedup.pkl"),
    
    
]

VAL_FILES = [
    os.path.join(PKL_DIR, "log_90_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-5_dedup.pkl"),
]

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

def build_action_mapping(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Dict[Any,int], Dict[int,Any]]:
    uniq = pd.unique(pd.concat([train_df["p0_action"], val_df["p0_action"]], ignore_index=True))
    uniq_sorted = sorted(list(uniq), key=lambda v: str(v))
    a2i = {v:i for i,v in enumerate(uniq_sorted)}
    i2a = {i:v for v,i in a2i.items()}
    return a2i, i2a

def gather_true_probs(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    return p.gather(1, y.view(-1,1)).squeeze(1)

def create_sequences_action(df: pd.DataFrame, L: int, action_to_idx: Dict[Any,int]):
    """
    Build sliding windows of size L within each (file_id, episode):
      X: (N, L, D), y: (N,), meta: [(file_id, episode, timestep_t)]
    Target at time t (the row just after the window).
    """
    groups = []
    for (fid, epi), g in df.groupby(["file_id","episode"], sort=False):
        obs_vecs, act_idxs, meta = [], [], []
        for _, r in g.iterrows():
            a_idx = action_to_idx.get(r.get("p0_action"), action_to_idx.get(str(r.get("p0_action")), None))
            if a_idx is None: continue
            try:
                vec = obs_list_to_1D_vec(obs_to_list(r["obs"]))
            except Exception:
                continue
            obs_vecs.append(vec)
            act_idxs.append(a_idx)
            meta.append((fid, int(epi), int(r["timestep"])))
        if obs_vecs:
            groups.append((np.asarray(obs_vecs, np.float32), np.asarray(act_idxs, np.int64), meta))

    X, Y, M = [], [], []
    for obs_arr, acts, meta in groups:
        T = len(acts)
        if T <= L: continue
        for i in range(L, T):
            X.append(torch.tensor(obs_arr[i-L:i], dtype=torch.float32))
            Y.append(torch.tensor(int(acts[i]), dtype=torch.long))
            M.append(meta[i])  # (fid, epi, timestep t)
    return X, Y, M

# ---------- Main ----------
def main():
    CFG = dict(
        seq_length=3,
        batch_size=128,
        hidden_dim=128,
        num_epochs=300,
        lr=1e-3,
        seed=7,
        project="10_10_testing_steakhouse_combo",
        run_name="mixed_mlp_combo_action_4",
    )

    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

    train_df, val_df = load_split(TRAIN_FILES, VAL_FILES)
    print(f"[data] TRAIN rows={len(train_df)}  VAL rows={len(val_df)}")
    print(f"[data] TRAIN files={sorted(train_df['file_id'].unique().tolist())}")
    print(f"[data] VAL   files={sorted(val_df['file_id'].unique().tolist())}")

    ACTION_TO_IDX, IDX_TO_ACTION = build_action_mapping(train_df, val_df)
    NUM_ACTIONS = len(ACTION_TO_IDX)
    print(f"[data] actions={NUM_ACTIONS} → {ACTION_TO_IDX}")

    L = CFG["seq_length"]

    trX, trY, _       = create_sequences_action(train_df, L, ACTION_TO_IDX)
    vaX, vaY, va_meta = create_sequences_action(val_df,   L, ACTION_TO_IDX)
    if not trX or not vaX:
        raise RuntimeError("No sequences created — check data.")

    input_dim = int(trX[0].shape[-1])
    x_tr = torch.stack(trX); y_tr = torch.stack(trY)
    x_va = torch.stack(vaX); y_va = torch.stack(vaY)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=CFG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MLP(seq_len=L, input_dim=input_dim, hidden_dim=CFG["hidden_dim"], output_dim=NUM_ACTIONS).to(device)
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

    # ---------- FINAL: per-episode lines ----------
    # 1) TRUE-CLASS prob by timestep
    model.eval()
    prob_series = []
    acc_cum_series_by_episode = defaultdict(list)  # for cumulative accuracy

    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            logits = model(seq.unsqueeze(0).to(device))
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
    #    (we must re-run predictions in meta-order)
    preds_map = {}
    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            logits = model(seq.unsqueeze(0).to(device))
            pred   = int(torch.argmax(logits, dim=1).item())
            preds_map[meta] = int(pred == int(label.item()))

    ep_buckets = defaultdict(list)  # key: "fid|epX" -> [(t, correct), ...]
    for (fid, epi, t), corr in preds_map.items():
        ep_buckets[f"{fid}|ep{int(epi)}"].append((int(t), int(corr)))

    cum_lines = []  # rows with (series, timestep, cum_acc)
    for series, items in ep_buckets.items():
        items.sort(key=lambda x: x[0])
        csum = 0
        for i, (t, corr) in enumerate(items, start=1):
            csum += corr
            cum_acc = csum / i
            cum_lines.append({"series": series, "timestep": t, "cum_acc": float(cum_acc)})

    # ---------- Log to W&B (tables + real line plots) ----------
    prob_series.sort(key=lambda d: (d["series"], d["timestep"]))
    cum_lines.sort(key=lambda d: (d["series"], d["timestep"]))

    if USE_WANDB:
        # A) True-prob line plot
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
                                       title="Action: true-class prob by timestep (per episode)",
                                       xname="timestep")
        })

        # B) Cumulative accuracy line plot
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
                                       title="Action: cumulative accuracy by timestep (per episode)",
                                       xname="timestep")
        })
    else:
        out_dir = PKL_DIR
        pd.DataFrame(prob_series).to_csv(os.path.join(out_dir, "action_val_trueprob_timeseries.csv"), index=False)
        pd.DataFrame(cum_lines).to_csv(os.path.join(out_dir, "action_val_cumacc_timeseries.csv"), index=False)
        print("[saved] CSVs in", out_dir)
        
        
    #cumulative true class probability graph overall for each episode
    #separate with each subtask transition (later?)
    # ---- Cumulative true-class probability (per-episode) ----
    from collections import defaultdict as _dd

    series_pts = _dd(list)  # series -> [(t, p_true)]
    for r in prob_series:
        series_pts[r["series"]].append((int(r["timestep"]), float(r["trueprob"])))

    series_xy_cum = {}
    for k, pts in series_pts.items():
        pts.sort(key=lambda z: z[0])
        x, y, csum = [], [], 0.0
        for i, (t, p) in enumerate(pts, start=1):
            csum += p
            x.append(t); y.append(csum / i)
        series_xy_cum[k] = {"x": x, "y": y}

    if USE_WANDB:
        keys = list(series_xy_cum.keys())
        xs   = [series_xy_cum[k]["x"] for k in keys]
        ys   = [series_xy_cum[k]["y"] for k in keys]
        wandb.log({
            "final/val_trueprob_cumulative_timeseries":
                wandb.plot.line_series(xs=xs, ys=ys, keys=keys,
                                    title="Cumulative mean true-class prob (per episode)",
                                    xname="timestep")
        })
    else:
        import matplotlib.pyplot as plt
        out_dir = PKL_DIR
        rows = []
        for k in series_xy_cum:
            for a, b in zip(series_xy_cum[k]["x"], series_xy_cum[k]["y"]):
                rows.append({"series": k, "timestep": a, "cum_trueprob": b})
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "action_val_trueprob_cumulative_timeseries.csv"), index=False)

        plt.figure()
        for k in sorted(series_xy_cum.keys()):
            plt.plot(series_xy_cum[k]["x"], series_xy_cum[k]["y"], label=k)
        plt.title("Cumulative mean true-class prob (per episode)")
        plt.xlabel("timestep"); plt.ylabel("cumulative mean true-prob")
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "action_val_trueprob_cumulative_timeseries.png"))
        plt.close()

    # ---- End Cumulative true-class probability (per-episode) ----

    #get a number like average percentage where it starts to be all correct.
    #bar graph of when it starts to be all correct
    # ---- Subtask "start-to-always-correct" tail percentage histogram (10% bins) ----
    from collections import defaultdict as _dd

    # 1) Decide subtask label per timestep (prefer 'subtask' or 'p0_intent'; else fallback to action)
    SUBTASK_COL = None
    for cand in ["subtask", "p0_intent"]:
        if cand in val_df.columns:
            SUBTASK_COL = cand
            break

    _meta_to_label = {}
    if SUBTASK_COL is not None:
        for _, r in val_df.iterrows():
            _meta_to_label[(r["file_id"], int(r["episode"]), int(r["timestep"]))] = str(r[SUBTASK_COL])
    else:
        # Fallback: treat each action as its own "subtask" label
        for _, r in val_df.iterrows():
            _meta_to_label[(r["file_id"], int(r["episode"]), int(r["timestep"]))] = f"ACTION:{r.get('p0_action')}"

    # 2) Build per-episode rows with correctness + subtask label
    ep_rows = _dd(list)  # "fid|epX" -> list of dict sorted by timestep
    for (fid, epi, t), corr in preds_map.items():
        ep_rows[f"{fid}|ep{int(epi)}"].append({
            "fid": fid, "epi": int(epi), "t": int(t),
            "correct": int(corr),
            "subtask": _meta_to_label.get((fid, int(epi), int(t)), "unknown"),
        })
    for k in ep_rows:
        ep_rows[k].sort(key=lambda d: d["t"])

    # 3) For each contiguous subtask window, compute tail fraction where all remaining are correct
    def tail_always_correct_fraction(correct_list):
        # If last step is wrong -> 0% by spec
        if not correct_list or correct_list[-1] == 0:
            return 0.0
        # Walk from end; when the tail ceases to be all-correct, stop
        tail_len = 0
        for c in reversed(correct_list):
            if c == 1:
                tail_len += 1
            else:
                break
        # Now find earliest index from which the tail (all-correct) starts
        # tail_len >= 1 guaranteed here; fraction = tail_len / total_len
        return tail_len / len(correct_list)

    tail_fracs = []  # collect one value per subtask window
    for _, rows in ep_rows.items():
        i, n = 0, len(rows)
        while i < n:
            j = i
            cur = rows[i]["subtask"]
            window = []
            while j < n and rows[j]["subtask"] == cur:
                window.append(rows[j]["correct"])
                j += 1
            # compute tail fraction for this window
            tf = tail_always_correct_fraction(window)
            tail_fracs.append(tf)
            i = j

    # 4) Bin into 10% buckets and plot
    total = max(1, len(tail_fracs))
    zeros = sum(1 for v in tail_fracs if v == 0.0)
    ones  = sum(1 for v in tail_fracs if v == 1.0)
    mids  = [v for v in tail_fracs if 0.0 < v < 1.0]

    # interior 10% bins: (0,10%], (10%,20%], ..., (90%,100%]
    edges = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    labels_mid = [f"{int(edges[i]*100)}–{int(edges[i+1]*100)}%" for i in range(10)]

    if len(mids) > 0:
        mids_s = pd.Series(mids, dtype="float64")
        # right-closed bins; exclude 0 and 1 which we handled separately
        binned = pd.cut(mids_s, bins=edges, right=True, include_lowest=False)
        vc = binned.value_counts(sort=False)
        mid_counts = [int(vc.iloc[i]) for i in range(len(vc))]
    else:
        mid_counts = [0]*10

    bins_lbls = ["0%"] + labels_mid + ["100%"]
    counts    = [zeros] + mid_counts + [ones]
    percents  = [(c / total) * 100.0 for c in counts]

    hist_df = pd.DataFrame({"bin": bins_lbls, "count": counts, "percent": percents})

    if USE_WANDB:
        wandb.log({
            "final/subtask_always_correct_tail_hist":
                wandb.plot.bar(
                    wandb.Table(dataframe=hist_df[["bin","percent"]]),
                    "bin", "percent",
                    title="Where within a subtask it becomes consistently correct (0% & 100% isolated)"
                )
        })
    else:
        import matplotlib.pyplot as plt, os
        out_dir = PKL_DIR
        hist_df.to_csv(os.path.join(out_dir, "subtask_tail_always_correct_hist.csv"), index=False)

        plt.figure()
        x = range(len(hist_df))
        plt.bar(list(x), hist_df["percent"].values)
        plt.xticks(list(x), hist_df["bin"].values, rotation=45, ha="right")
        plt.ylabel("segments (%)")
        plt.title("Where within a subtask it becomes consistently correct (0% & 100% isolated)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "subtask_tail_always_correct_hist.png"))
        plt.close()


# ---- END Subtask "start-to-always-correct" tail percentage histogram (10% bins) ----

        
    

if __name__ == "__main__":
    main()
