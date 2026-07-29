#How to run
#/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_heuristic_obs
#(steakhouse-ai) (miniconda3)mishafu@usc-guestwireless-upc-newsc21240 baseline_heuristic_obs % /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python train.py 

#Workflow
#so this uses MLP but trains on each file SEPARATELY
#train 3 networks, each for one file, and test/validate on itself and the others 
#Two simple loops, one over "train" FOVs, and then an inner loop over all three "test" FOVs (while parameterizing your WandB run names)

#Next steps
#challenge: add a graph where the test on self and others are all together just represented by different lines
#add within episode accruacy li


#same imports as before

import pickle
import os
import random
import sys
import pathlib
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from baseline_implicit_obs.train import create_sequences
from model import MLP

FILE_PATHS = {
     #old set of files
    # "fov90":  "../data/fov-90.pkl",
    # "fov120": "../data/fov-120.pkl",
    # "fov179": "../data/fov-179.pkl",
    
    #new set of files with finer time steps
    "fov90":  "../data/90.pkl",
    "fov120": "../data/120.pkl",
    "fov179": "../data/179.pkl",
    
}

def get_loaders(path, seq_len, train_ratio, batch_size):
    with open(path, 'rb') as file:
        df = pickle.load(file)

    df['file_id'] = os.path.basename(path)
    train_seq, train_t, val_seq, val_t, val_meta = create_sequences(df, seq_len, train_ratio)

    if train_ratio > 0:
        x_train = torch.stack(train_seq)
        y_train = torch.stack(train_t).long()
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    else:
        train_loader = None

    x_val = torch.stack(val_seq)
    y_val = torch.stack(val_t).long()
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    return train_loader, val_loader, val_seq, val_t, val_meta

def train_and_log(train_name, train_path, seq_len, batch_size, epochs, lr, TRAIN_RATIO):
    model = MLP(seq_len=seq_len, input_dim=65, hidden_dim=40, output_dim=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader, val_seqs, val_targets, val_meta = get_loaders(
        train_path, seq_len, TRAIN_RATIO, batch_size
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})

        if epoch % 10 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            self_acc = (correct / total) if total else 0.0
            wandb.log({f"{train_name}_val_acc": self_acc, "epoch": epoch})

            results = {f"{train_name}_on_{train_name}": self_acc}
            for test_name, test_path in FILE_PATHS.items():
                if test_name == train_name:
                    continue
                _, validation_loader, _, _, _ = get_loaders(test_path, seq_len, train_ratio=0.0, batch_size=batch_size)
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in validation_loader:
                        preds = model(xb).argmax(dim=1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                acc = (correct / total) if total else 0.0
                wandb.log({f"{train_name}_on_{test_name}_acc": acc, "epoch": epoch})
                results[f"{train_name}_on_{test_name}"] = acc

            table = wandb.Table(columns=["epoch", "accuracy", "dataset"])
            for ds, ds_acc in results.items():
                table.add_data(epoch, ds_acc, ds)
            wandb.log({
                f"{train_name}_combined_acc": wandb.plot.line(
                    table,
                    "epoch",
                    "accuracy",
                    "dataset",
                    title=f"Combined eval for {train_name}"
                )
            })

    # After training: exact-timestep accuracy averaged across episodes
    timestep_results = defaultdict(list)  # {timestep: [0, 1, 1, 0, ...]} (corrects only)

    model.eval()
    with torch.no_grad():
        for seq, meta, label in zip(val_seqs, val_meta, val_targets):
            pred = model(seq.unsqueeze(0)).argmax(dim=1).item()
            correct = int(pred == label.item())
            _, _, timestep = meta
            timestep_results[timestep].append(correct)

    # Average at each timestep across all episodes
    avg_table = wandb.Table(columns=["timestep", "accuracy"])
    for t in sorted(timestep_results):
        acc = sum(timestep_results[t]) / len(timestep_results[t])
        avg_table.add_data(t, acc)

    wandb.log({
        f"{train_name}_exact_acc_by_timestep":
            wandb.plot.line(
                avg_table,
                "timestep",
                "accuracy",
                title=f"Exact timestep accuracy average — {train_name}"
            )
    })
    
    # Cumulative timestep accuracy averaged across episodes
    cumulative_accs = defaultdict(list)  # {timestep: [acc₁, acc₂, ...]} from each episode

    episode_data = defaultdict(list)  # {episode_id: [(timestep, correct)]}
    for seq, meta, label in zip(val_seqs, val_meta, val_targets):
        pred = model(seq.unsqueeze(0)).argmax(dim=1).item()
        correct = int(pred == label.item())
        _, episode_id, timestep = meta
        episode_data[episode_id].append((timestep, correct))

    for ep_id, steps in episode_data.items():
        steps.sort(key=lambda x: x[0])
        cumsum = 0
        for i, (t, corr) in enumerate(steps, start=1):
            cumsum += corr
            acc = cumsum / i
            cumulative_accs[t].append(acc)

    cum_table = wandb.Table(columns=["timestep", "avg_cumulative_accuracy"])
    for t in sorted(cumulative_accs):
        avg = sum(cumulative_accs[t]) / len(cumulative_accs[t])
        cum_table.add_data(t, avg)

    wandb.log({
        f"{train_name}_cumulative_acc_by_timestep":
            wandb.plot.line(
                cum_table,
                "timestep",
                "avg_cumulative_accuracy",
                title=f"Cumulative timestep accuracy average — {train_name}"
            )
    })


    return model

if __name__ == "__main__":
    SEQ_LEN     = 3 #32
    TRAIN_RATIO = 0.9
    BATCH_SIZE  = 128
    EPOCHS      = 350
    LR          = 1e-4

    for train_name, train_path in FILE_PATHS.items():
        wandb.init(
            project="new_small_seqs_baseline_heuristic_obs",
            name=f"train_{train_name}",
            reinit=True,
            config={
                "train_fov": train_name,
                "seq_len": SEQ_LEN,
                "train_ratio": TRAIN_RATIO,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
            }
        )

        train_and_log(train_name, train_path, SEQ_LEN, BATCH_SIZE, EPOCHS, LR, TRAIN_RATIO)
        wandb.finish()
