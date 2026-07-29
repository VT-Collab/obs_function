#Predict next true intention/subtask
# #run from
#/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_implicit_obs
#script to run
#/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python train_intent.py

#!/usr/bin/env python3
# LSTM intent prediction (p0_intent) using  65 or smt else-D obs vectors built by dataset.py
# - Inputs:   sequences of length L of obs65 or something else dimesion (t-L .. t-1)
# - Target:   p0_intent at time t (12-way classification)
# - Files:    ../data/third_90_intent.pkl, ../data/third_120_intent.pkl, ../data/third_179_intent.pkl
# - Logs:     avg_intent_accuracy, WEIGHTED_avg_intent_accuracy (higher weight earlier in a subtask)

#with a starting accuracy of 80 then up to 99%, we see that model can very quickly learn (almost) the same deterministic rules you used to create p0_intent.

import os, sys, pathlib, pickle, random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

#add grandparent folder to python directory
#.Path turns it into path object, .resolve turns relative path to absolute file path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from dataset import obs_to_list, obs_list_to_1D_vec
from model import LSTMModel 

#subtask dictionary staring from 0
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
    9:'pickup_steak',
    10:'add_garnish',
    11:'deliver_dish'
}

#backwards mapping starting from 0
#invert dict general syntax: inverted = {v: k for k, v in my_dict.items()}
INTENT_TO_IDX = {name: idx for idx, name in IDX_TO_INTENT.items()}

#number of classes
NUM_INTENTS = len(IDX_TO_INTENT)

#load from pickles given paths
#add file_id column to every row, fill with the file name as is
#sort by episode and timestep
def load_pkls(paths):
    dfs = []
    #goes through each file
    for p in paths:
        with open(p, 'rb') as f:
            df = pickle.load(f)
        #file read closes now
        
        
        df['file_id'] = os.path.basename(p)
        dfs.append(df)
    #put everything into one big dataframe
    data = pd.concat(dfs, ignore_index=True)

    #convert to int; .astype(int) casts the series to dtype int
    data['episode'] = data['episode'].astype(int)
    data['timestep'] = data['timestep'].astype(int)
    
    #first sort values then drop orginal index
    data = data.sort_values(["file_id", "episode", "timestep"]).reset_index(drop=True)
    return data

#takes string subtask like pickup_meat and turn it into int value
def to_intent_idx(s):
    s = str(s).strip()
    #.get(key_name, default) returns the value if the key exists; otherwise returns default without raising KeyError ; safer than directly doing like dic['key_name']
    return INTENT_TO_IDX.get(s, None)

#given one episode with all its rows, return all the rows of 65_dim vector, intent, and meta (file_id, episode, and timestep) as one list
def build_episode_tensors(df_ep):
    obs_vecs, intents, meta = [], [], []
    for _, r in df_ep.iterrows():
        idx = to_intent_idx(r["p0_intent"])
        if idx is None:
            continue
        try:
            obs_list = obs_to_list(r["obs"])
            the_vec    = obs_list_to_1D_vec(obs_list)
        except Exception:
            continue

        obs_vecs.append(the_vec)
        intents.append(idx)
        meta.append((r["file_id"], int(r["episode"]), int(r["timestep"])))
    if not obs_vecs:
        return None, None, None
    return np.asarray(obs_vecs, dtype=np.float32), np.asarray(intents, dtype=np.int64), meta

#split episodes into training/validation; first shuffle them then split top bottom; each a list only consisting of (file_id, episode) pairs.
def split_by_episode_keys(data, train_ratio=0.9, seed=0):
    keys = data[["file_id","episode"]].drop_duplicates().apply(tuple, axis=1).tolist()
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_train = max(1, int(len(keys) * train_ratio))
    return keys[:n_train], keys[n_train:]

#go through intent and basically create map that records each segments of same intent in order to compute for later 
#intents = ['pickup','pickup','drop','drop','drop','rinse','rinse','pickup']
# indexes:  0         1        2      3      4       5       6       7
#Output (index → (rank, span_len)) or index overall in the file -> rank within the same subtask/span_len (total length of the same subtask)
# 0→(1,2), 1→(2,2),
# 2→(1,3), 3→(2,3), 4→(3,3),
# 5→(1,2), 6→(2,2),
# 7→(1,1)
def compute_span_map(intents):
    span_map = {}
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

#group data by file and episode
#returns processed training/validation sequence results
# L = seq_length
# T = total time steps in an episode

# train_X: Each item is a window of seq_length consecutive timesteps.
# train_Y: Each item is one integer = the “next intent” class index (0..11) that occurs right after the train_X window.

# val_X: same as train_X but for validation
# val_Y: same as train_Y but for validation

# val_M: list of metadata tuples (file_id, episode, timestep) aligned with val_X/val_Y

# val_span_info: dict mapping metadata → (rank_within_span, span_length) for validation episodes
# (This is for the weighted-accuracy metric within contiguous runs of the same intent.)
def create_sequences_intent(data, L, train_ratio, seed):
    groups = {}
    for (fid, epi), g in data.groupby(["file_id","episode"], sort=False):
        obs_arr, intents, meta = build_episode_tensors(g)
        if obs_arr is None:
            continue
        groups[(fid, int(epi))] = {"obs": obs_arr, "intents": intents, "meta": meta}

    train_keys, val_keys = split_by_episode_keys(data, train_ratio=train_ratio, seed=seed)

    def build(keys):
        X, Y, M = [], [], []
        for key in keys:
            if key not in groups:
                continue
            g = groups[key]
            obs_arr   = g["obs"]      # (T, one_row_obs_dim_vector length)
            intents   = g["intents"]  # (T,)
            meta_list = g["meta"]     # list of (file_id, epi, t)
            T = len(intents)
            if T <= L:
                continue
            for i in range(L, T):
                X.append(torch.tensor(obs_arr[i-L:i], dtype=torch.float32))  # (L, one_row_obs_dim_vector length)
                Y.append(torch.tensor(int(intents[i]), dtype=torch.long))    # ()
                M.append(meta_list[i])
        return X, Y, M

    train_X, train_Y, _      = build(train_keys)
    val_X,   val_Y,   val_M  = build(val_keys)

    # Build span info for weighted accuracy
    val_span_info = {}
    for key in val_keys:
        if key not in groups:
            continue
        g = groups[key]
        span_map = compute_span_map(g["intents"])
        for t_idx, meta in enumerate(g["meta"]):
            if t_idx in span_map:
                val_span_info[meta] = span_map[t_idx]

    return train_X, train_Y, val_X, val_Y, val_M, val_span_info



#main training part now
def main():
    # Data files
    file_paths = [
        '../data/third_90_intent.pkl',
        '../data/third_120_intent.pkl',
        '../data/third_179_intent.pkl',
    ]
    data = load_pkls(file_paths)
    print(f"Loaded {len(data)} rows across {data['episode'].nunique()} episodes; files: {sorted(data['file_id'].unique().tolist())}")

    # W&B config
    wandb_project_name = "intent_lstm"
    wandb_entity       = "steakteam"
    wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        config={
            "seq_length": 3,
            "train_to_validate_ratio": 0.9,
            "batch_size": 128,
            "hidden_dim": 128,
            "layer_dim": 1,
            "num_epochs": 200,
            "lr": 1e-3,
            "seed": 7,
            "model": "LSTM",
        }
    )
    wandb.define_metric("avg_intent_accuracy", step_metric="epoch")
    wandb.define_metric("WEIGHTED_window_avg_intent_accuracy", step_metric="epoch")

    CFG        = wandb.config
    L          = int(CFG["seq_length"])
    train_r    = float(CFG["train_to_validate_ratio"])
    BATCH      = int(CFG["batch_size"])
    HIDDEN     = int(CFG["hidden_dim"])
    LAYERS     = int(CFG["layer_dim"])
    EPOCHS     = int(CFG["num_epochs"])
    LR         = float(CFG["lr"])
    SEED       = int(CFG["seed"])

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # Sequences
    train_X, train_Y, val_X, val_Y, val_meta, val_span = create_sequences_intent(
        data, L=L, train_ratio=train_r, seed=SEED
    )
    
    print(f"Training sequences:   {len(train_X)}")
    print(f"Validation sequences: {len(val_X)}")
    if not train_X or not val_X:
        raise RuntimeError("No sequences created. Check data contents and p0_intent mapping.")

    # DataLoaders
    x_tr = torch.stack(train_X)
    y_tr = torch.stack(train_Y)
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True)

    x_va = torch.stack(val_X)
    y_va = torch.stack(val_Y)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=BATCH, shuffle=False)

    # Model / opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMModel(input_dim=65, hidden_dim=HIDDEN, layer_dim=LAYERS, output_dim=NUM_INTENTS).to(device)
    crit   = nn.CrossEntropyLoss()
    optim  = torch.optim.Adam(model.parameters(), lr=LR)

    #we validate/test in every 10 step of the training loop
    # Train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            # UNPACK: LSTMModel returns (logits, h, c); we only need logits
            logits, _, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        train_loss = total_loss / max(1, len(train_loader))
        wandb.log({"epoch": epoch, "train_loss": train_loss})

        # Eval
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()
            # 1) unweighted accuracy
            correct = 0
            total   = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    logits, _, _ = model(xb)            # UNPACK
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total   += yb.size(0)
            avg_acc = correct / total if total else 0.0
            

            # 2) weighted accuracy (earlier steps in a GT subtask get higher weight)
            
            # ---- NEW: window-weighted (discounted) accuracy, averaged across windows ----
            #w = gamma^(rank-1) 
            
            gamma = float(CFG.get("gamma", 0.9))

            # We’ll accumulate per-window discounted accuracy.
            # A window = one contiguous run of the same GT intent.
            # Key it by (file_id, episode, window_start_t, window_end_t, gt_intent_idx).
            windows = {}  # key -> {'wc': weighted_correct_sum, 'ws': weighted_sum}

            with torch.no_grad():
                for seq, meta, label in zip(val_X, val_meta, val_Y):
                    # predict for this single target timestep
                    logits, _, _ = model(seq.unsqueeze(0).to(device))  # (1, NUM_INTENTS)
                    pred         = int(torch.argmax(logits, dim=1).item())
                    gt     = int(label.item())
                    fid, epi, t = meta

                    # get rank within its GT window, and the span length, from the prebuilt map
                    rank, span_len = val_span.get(meta, (1, 1))

                    # discount weight: earlier steps (smaller rank) get more weight
                    w = float(gamma ** (rank - 1))

                    # reconstruct this window’s start/end to key windows consistently
                    start_t = int(t - (rank - 1))
                    end_t   = int(start_t + span_len - 1)
                    key     = (fid, int(epi), start_t, end_t, gt)

                    bucket = windows.setdefault(key, {"wc": 0.0, "ws": 0.0})
                    bucket["ws"] += w
                    if pred == gt:
                        bucket["wc"] += w

            # macro average across windows: mean over windows of (weighted_correct / weighted_sum)
            if windows:
                per_window_scores = [
                    (b["wc"] / b["ws"]) for b in windows.values() if b["ws"] > 0.0
                ]
                window_weighted_avg = float(sum(per_window_scores) / len(per_window_scores)) if per_window_scores else 0.0
            else:
                window_weighted_avg = 0.0

            print(
                f"Epoch {epoch:3d}/{EPOCHS}  Train L={train_loss:.4f}  "
                f"avg_intent_accuracy={avg_acc:.4f}  WEIGHTED_window_avg_intent_accuracy={window_weighted_avg:.4f}"
            )
            wandb.log({
                "epoch": epoch,
                "avg_intent_accuracy": avg_acc,
                "WEIGHTED_window_avg_intent_accuracy": window_weighted_avg
            })

    print("Training complete!")

if __name__ == "__main__":
    main()
