
"""
Training a BNN model to predict fov number based on past sequence

Each file contains multiple layouts b/c
Since each episode is one unique seed, meaning each episode is one unique layout, we can
split within file instead of multiple files

"""

#!/usr/bin/env python3
"""
BNN to predict FOV (90 / 120 / 179) from flattened MiniGrid CSV.

- Uses 3 CSV files, each corresponding to a single FOV:
    scripts/runs/lockedroom_success_all_90.csv
    scripts/runs/lockedroom_success_all_120.csv
    scripts/runs/lockedroom_success_all_179.csv

- Each file contains per-timestep rows; all rows in a file share the same FOV.
- We infer the FOV from the filename and treat it as the label.

- Split by episode_seed: bottom 20% episode ids -> validation (unseen episodes).
- Inputs are sliding windows of length L over numeric features.
- Target y(t) = FOV class at time t (the "next" frame relative to last input).

Run from repo root or scripts directory that can see:
  - dataset.expected_dim()
  - model.BNN
"""

import os, sys, pathlib, random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchbnn as bnn
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
from robot.estimation.bnn.dataset import flatten_record, flatten_episode, expected_dim
from robot.estimation.bnn.model import BNN   # <--- BNN instead of LSTMModel

# ---------- Config ----------
ROOT = pathlib.Path(__file__).resolve().parents[2]  # /Users/.../minigrid-ai

CSV_PATHS = [
    str(ROOT / "scripts" / "runs" / "lockedroom_success_all_90.csv"),
    str(ROOT / "scripts" / "runs" / "lockedroom_success_all_120.csv"),
    str(ROOT / "scripts" / "runs" / "lockedroom_success_all_179.csv"),
]

SEQ_LEN  = 3
BATCH    = 128
HIDDEN   = 128
EPOCHS   = 10

# BNN-specific hyperparams (matching your other BNN use)
CFG = dict(
    seq_length=SEQ_LEN,
    batch_size=BATCH,
    hidden_dim=HIDDEN,
    num_epochs=EPOCHS,
    MC=1,          # MC forward passes at eval; set >1 if you want stochastic averaging
    lr=1e-3,
    kl=1e-5,       # KL weight
    seed=7,
    project="minigrid_bnn_fov",
    run_name="2_bnn_fov_seq3_split_bottom20",
)

SEED = CFG["seed"]

# FOV vocabulary
FOV_VOCAB: List[str] = ["90", "120", "179"]
FOV2IDX: Dict[str, int] = {s: i for i, s in enumerate(FOV_VOCAB)}
NUM_CLASSES = len(FOV_VOCAB)

# ---------- Utils ----------
def to_fov_idx(s: Any) -> Optional[int]:
    if s is None:
        return None
    return FOV2IDX.get(str(s).strip(), None)

def infer_fov_from_path(path: str) -> str:
    """
    Infer FOV string ("90", "120", "179") from the CSV filename
    by checking which token appears in the basename.
    """
    base = os.path.basename(path)
    for k in FOV_VOCAB:
        if k in base:
            return k
    raise ValueError(f"Could not infer FOV from filename: {base}")

def load_csv(paths: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files.
    Each file's rows get a string 'fov' column inferred from its filename.
    We also mirror 'fov' into 'agent_subtask' to reuse the old label pipeline.
    """
    dfs = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)

        # enforce ints where applicable
        if "episode_seed" in df.columns:
            df["episode_seed"] = df["episode_seed"].astype(int, errors="ignore")
        if "timestep" in df.columns:
            df["timestep"] = df["timestep"].astype(int, errors="ignore")

        # infer FOV from filename
        fov_str = infer_fov_from_path(path)
        df["fov"] = fov_str            # string label (not numeric)
        df["agent_subtask"] = df["fov"]  # reuse the same label column name

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSVs loaded; check CSV_PATHS.")
    df_all = pd.concat(dfs, ignore_index=True)

    # ensure sorted within episode
    df_all = df_all.sort_values(
        ["episode_seed", "timestep"], ascending=[True, True]
    ).reset_index(drop=True)
    return df_all

def split_by_episode_bottom20(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bottom 20% of episodes by episode_seed -> VAL (unseen), rest -> TRAIN.
    """
    if "episode_seed" not in df.columns:
        raise ValueError("CSV must contain 'episode_seed' column.")
    episodes = sorted(df["episode_seed"].unique().tolist())
    n = len(episodes)
    k = max(1, int(np.floor(0.20 * n)))
    val_episodes = set(episodes[:k])   # bottom ids as validation
    tr = df[~df["episode_seed"].isin(val_episodes)].copy()
    va = df[df["episode_seed"].isin(val_episodes)].copy()
    return tr, va

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    All numeric feature columns.
    'agent_subtask' and 'fov' are strings, so they are NOT included -> no label leakage.
    """
    if "agent_subtask" not in df.columns:
        raise ValueError("CSV must contain 'agent_subtask' column (here used for FOV labels).")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    D_exp = expected_dim()
    D_obs = len(numeric_cols)
    if D_obs != D_exp:
        print(f"[warn] observed feature dim {D_obs} != expected_dim() {D_exp}. Proceeding anyway.")
    return numeric_cols

def create_sequences_fov(
    df: pd.DataFrame, L: int, feat_cols: List[str]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int,int]]]:
    """
    Build sliding windows within each episode:
      Inputs:  X[i] = frames [t-L, ..., t-1]  (each frame = numeric feature vector)
      Target:  y[i] = FOV at time t (class index 0/1/2)
    Returns:
      - X: list of tensors of shape (L, D)
      - Y: list of scalar Long tensors
      - M: list of (episode_seed, timestep) tuples (meta)
    """
    X, Y, M = [], [], []
    for epi, g in df.groupby("episode_seed", sort=True):
        g = g.sort_values("timestep", ascending=True).reset_index(drop=True)
        # 'agent_subtask' actually stores FOV string here
        labels = [to_fov_idx(s) for s in g["agent_subtask"].tolist()]
        feats  = g[feat_cols].to_numpy(dtype=np.int64)  # ints -> cast to float32 later
        T = len(g)
        if T <= L:
            continue
        for t in range(L, T):
            y = labels[t]
            if y is None:
                continue
            window = feats[t-L:t]  # shape (L, D)
            X.append(torch.tensor(window, dtype=torch.float32))
            Y.append(torch.tensor(int(y), dtype=torch.long))
            M.append((int(epi), int(g.loc[t, "timestep"])))
    return X, Y, M

def gather_true_probs(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    return p.gather(1, y.view(-1,1)).squeeze(1)

# ---------- Main ----------
def main():
    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])

    df = load_csv(CSV_PATHS)
    train_df, val_df = split_by_episode_bottom20(df)
    print(f"[data] rows: TRAIN={len(train_df)}  VAL={len(val_df)}")
    print(f"[data] episodes: TRAIN={len(train_df['episode_seed'].unique())}  VAL={len(val_df['episode_seed'].unique())}")

    feat_cols = get_feature_columns(df)
    L = CFG["seq_length"]

    trX, trY, _       = create_sequences_fov(train_df, L, feat_cols)
    vaX, vaY, va_meta = create_sequences_fov(val_df,   L, feat_cols)
    if not trX or not vaX:
        raise RuntimeError("No sequences created — check CSVs, labels, and SEQ_LEN.")

    input_dim = int(trX[0].shape[-1])   # D
    x_tr = torch.stack(trX)             # (Ntr, L, D)
    y_tr = torch.stack(trY)             # (Ntr,)
    x_va = torch.stack(vaX)             # (Nva, L, D)
    y_va = torch.stack(vaY)             # (Nva,)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(x_va, y_va), batch_size=CFG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- BNN setup (copied style from your other BNN script) ----------
    model  = BNN(seq_len=L, input_dim=input_dim, hidden_dim=CFG["hidden_dim"], output_dim=NUM_CLASSES).to(device)

    # CE = data term, KL = regularization term over Bayesian layers
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    opt     = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    kl_weight = CFG["kl"]
    MC = int(CFG.get("MC", 1))

    if USE_WANDB:
        wandb.init(project=CFG["project"], name=CFG["run_name"], config=CFG)
        wandb.define_metric("val_accuracy", step_metric="epoch")

    # ---------- Train ----------
    for epoch in range(CFG["num_epochs"]):
        model.train()
        total = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)                              # (B, C)
            data_loss = ce_loss(logits, yb)                 # fit term
            kll = kl_loss(model) / len(train_loader)        # KL averaged per batch
            loss = data_loss + kl_weight * kll              # trade-off

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        train_loss = total / max(1, len(train_loader))
        if USE_WANDB:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_data_loss": data_loss.item(),
                "train_kl_loss": kll.item(),
            })

        # ---------- Validation ----------
        if epoch == 1 or epoch % 1 == 0 or epoch == CFG["num_epochs"] - 1:
            model.eval()
            correct, tot = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    if MC > 1:
                        # MC-average logits (or probs) across stochastic passes
                        logits_mc = torch.stack([model(xb) for _ in range(MC)], dim=0).mean(0)
                        logits_eval = logits_mc
                    else:
                        logits_eval = model(xb)
                    preds = logits_eval.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    tot     += yb.size(0)
            val_acc = correct / tot if tot else 0.0
            print(f"Epoch {epoch:3d}/{CFG['num_epochs']}  TrainL={train_loss:.4f}  ValAcc={val_acc:.4f}")
            if USE_WANDB:
                wandb.log({"epoch": epoch, "val_accuracy": val_acc})

    print("Training complete!")

    # ---------- FINAL: per-episode curves on VAL ----------
    # (same style as before, but can optionally use MC averaging)
    model.eval()
    prob_series = []
    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            xb = seq.unsqueeze(0).to(device)  # (1, L, D)
            if MC > 1:
                logits_mc = torch.stack([model(xb) for _ in range(MC)], dim=0).mean(0)
                logits_eval = logits_mc
            else:
                logits_eval = model(xb)
            p_true = float(gather_true_probs(logits_eval, label.view(1).to(device)).item())
            epi, t = meta
            prob_series.append({
                "series": f"ep{epi}",
                "episode_seed": epi,
                "timestep": t,
                "trueprob": p_true
            })

    # cumulative accuracy by timestep per episode
    preds_map = {}
    with torch.no_grad():
        for seq, label, meta in zip(vaX, vaY, va_meta):
            xb = seq.unsqueeze(0).to(device)
            if MC > 1:
                logits_mc = torch.stack([model(xb) for _ in range(MC)], dim=0).mean(0)
                logits_eval = logits_mc
            else:
                logits_eval = model(xb)
            pred = int(torch.argmax(logits_eval, dim=1).item())
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
    out_dir = str(pathlib.Path(CSV_PATHS[0]).parent)
    prob_series.sort(key=lambda d: (d["series"], d["timestep"]))
    cum_lines.sort(key=lambda d: (d["series"], d["timestep"]))

    if USE_WANDB:
        # per-episode true-class prob
        sx = defaultdict(lambda: {"x": [], "y": []})
        for r in prob_series:
            sx[r["series"]]["x"].append(r["timestep"])
            sx[r["series"]]["y"].append(r["trueprob"])
        keys = list(sx.keys())
        xs   = [sx[k]["x"] for k in keys]
        ys   = [sx[k]["y"] for k in keys]
        wandb.log({
            "final/fov_trueprob_timeseries":
                wandb.plot.line_series(xs=xs, ys=ys, keys=keys,
                                       title="FOV: true-class prob by timestep (per episode)",
                                       xname="timestep")
        })

        # per-episode cumulative accuracy
        sx2 = defaultdict(lambda: {"x": [], "y": []})
        for r in cum_lines:
            sx2[r["series"]]["x"].append(r["timestep"])
            sx2[r["series"]]["y"].append(r["cum_acc"])
        keys2 = list(sx2.keys())
        xs2   = [sx2[k]["x"] for k in keys2]
        ys2   = [sx2[k]["y"] for k in keys2]
        wandb.log({
            "final/fov_cumacc_by_timestep":
                wandb.plot.line_series(xs=xs2, ys=ys2, keys=keys2,
                                       title="FOV: cumulative accuracy by timestep (per episode)",
                                       xname="timestep")
        })
    else:
        pd.DataFrame(prob_series).to_csv(os.path.join(out_dir, "fov_val_trueprob_timeseries.csv"), index=False)
        pd.DataFrame(cum_lines).to_csv(os.path.join(out_dir, "fov_val_cumacc_timeseries.csv"), index=False)
        print(f"[saved] CSVs in {out_dir}")

if __name__ == "__main__":
    main()
