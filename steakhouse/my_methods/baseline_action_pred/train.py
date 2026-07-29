#this is the action prediction baseline testing file
#almost the same as baseline_implicit_obs except the model
    # ACTION_TO_CHAR = {
    # 0    Direction.NORTH: "↑",
    #     Direction.SOUTH: "↓",
    #     Direction.EAST: "→",
    #     Direction.WEST: "←",
    #     STAY: "stay",
    #     INTERACT: INTERACT,
    # }

import argparse

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset

from model import MLP #THIS IS DIFFERENT 
from dataset import * #import all functions from dataset.py
import matplotlib.pyplot as plt
import wandb
from collections import Counter
import os
import random
from collections import defaultdict

import sys, pathlib
from pathlib import Path
# add src/ (the parent of both packages) to the module search path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from baseline_implicit_obs.train import create_sequences

#How to run
#/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_action_pred
#(steakhouse-ai) (miniconda3)mishafu@usc-guestwireless-upc-newsc21240 baseline_action_pred % /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python train.py 

#Workflow
# Step 1: Split raw data rows into input output of seq_length to 1 (like 10 or 32  sequential 65-dim obs vector to one immediate after row's po_action). Each size-10 sequence has to be in the same episode
# Step 2: group sequences by file and episode, randomly shuffle the episodes, then split those episodes into 90% portion for training and 10% portion for testing accuracy, so we test on completely unseen episodes without mixing across files
# Step 3: Randomly group, say, 128 of the sequences into one batch (overall, across episodes), and feed into mlp to train. Do that until the 90% portion is used up
# Step 4: after mlp is done training every 10 epochs, take the rest 10% from step 2 and test mlp output on them (no training here just pure testing)


def main():
    
    #load data
    #file_path = '../data/fov-90.pkl' #from baseline_action_pred
    file_paths = [
        #old set of files
        # '../data/fov-90.pkl',
        # '../data/fov-120.pkl',
        # '../data/fov-179.pkl'
        
        #new set of files with finer time steps
        '../data/90.pkl',
        '../data/120.pkl',
        '../data/179.pkl'
    ]
    dataframes = []
    for path in file_paths:
        with open(path, 'rb') as file:
            df = pickle.load(file)
            df["file_id"] = os.path.basename(path)
            dataframes.append(df)
            
    data = pd.concat(dataframes, ignore_index = True)
    # with open(file_path, 'rb') as file:
    #     data = pickle.load(file)
    
    print(f"Loaded data with {len(data)} rows")
    print(f"Episodes: {data['episode'].nunique()}")
    
    #start wandb
    wandb_project_name: str = "new_small_seqs_baseline_action_pred_steakhouse-trajectory-training"
    wandb_entity: str = "steakteam"

    # Initialize wandb
    wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        config={
            "seq_length": 3, #change this to 32 next, last one 
            "train_to_validate_ratio": 0.9,
            "batch_size": 128,
            "hidden_dim": 40,
            #"layer_dim": 1,
            "num_epochs": 350,
            "lr": 0.0001,
            "model": "MLP",
        }
    )
    
    #start added
    # tell WandB to use our 'epoch' field as the step for these metrics
    for m in [
        "val_accuracy",
        "fov-90_val_accuracy",
        "fov-120_val_accuracy",
        "fov-179_val_accuracy"
    ]:
        wandb.define_metric(m, step_metric="epoch")
    #end added
    
    
    config = wandb.config
    seq_length = config.seq_length
    train_to_validate_ratio = config.train_to_validate_ratio
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    #layer_dim = config.layer_dim
    num_epochs = config.num_epochs
    lr         = config.lr

    #get the data
    train_seqs, train_targs, val_seqs, val_targs, val_meta = create_sequences(data, seq_length, train_to_validate_ratio)
    
    print(f"Training sequences:   {len(train_seqs)}")
    print(f"Validation sequences: {len(val_seqs)}")

    #these lengths should match input output should match
    if len(train_seqs) != len(train_targs):
        raise ValueError("ERROR ERRROOOORRRRRR INPUT OUTPUT DON'T MATCH")
    if len(val_seqs) != len(val_targs):
        raise ValueError("ERROR ERRROOOORRRRRR INPUT OUTPUT DON'T MATCH")
    
    #stack the input output
    x = torch.stack(train_seqs) #this wil create a tensor of (N, batch_input dim which is (10, 65)) so like instead of N separate sequences group them into one (each sequence at a new dimention)
    y = torch.stack(train_targs).long()
    #dataset that pairs x and y
    train_ds = TensorDataset(x, y)
    
    #takes dataset and makes it in batches
    train_loader = DataLoader(train_ds, batch_size=batch_size) #, shuffle=True makes what goes in each batch's ordering random, initial random weights also introduce randomness

    x_val   = torch.stack(val_seqs)             # (N_val, seq_length, 65)
    y_val   = torch.stack(val_targs).long()    # (N_val,)
    #dataset that pairs x and y
    val_ds  = TensorDataset(x_val, y_val)
    #takes dataset and makes it in batches
    val_loader = DataLoader(val_ds, batch_size=batch_size)    
    
    #-----------------------------Initializing Model, LOSS FUNCTION and OPTIMIZER--------------------------------------------------------------------------
    # we can tweak hidden_dim and layer_dim we can tweak to test performance 
    model = MLP(seq_len = seq_length, input_dim=65, hidden_dim = hidden_dim, output_dim=6) 
    criterion = nn.CrossEntropyLoss() #mean squared error loss function, (predicted-actual)^2 and average accross batch
    
    #use adam (adaptive learning rate optimizer) and pass in model parameters and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    
    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        
        #goes through the batch !!!!!???????
        for xb, yb in train_loader:
            #clears greadient from previous batch
            optimizer.zero_grad()

            #forward pass, gets prediction output from model from input
            output = model(xb)
            #preds = output.squeeze(-1)
            
            #calculate difference between prediction and actual
            loss = criterion(output, yb)
            
            #backpropagate and add to loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        #Divides by number of batches to get average loss per batch
        train_loss /= len(train_loader)

        wandb.log({"epoch": epoch, "train_loss": train_loss}) #wandb

        #Prints progress every 10 epochs (and epoch 1)
        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs-1:
            print(f"Epoch {epoch:3d}/{num_epochs}  "
                f"Train L={train_loss:.4f}")
        # ——— Finally get accuracy on validation set ———
            model.eval()
            correct = 0
            total   = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    #get model predictions
                    outputs = model(xb)
                    
                    #classification
                    preds = outputs.argmax(dim=1)  # gets index of max logit
                    
                    #convert to long again
                    labels = yb.long()
                    
                    #add to correct if they are same
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)

            accuracy = correct / total
            print(f"Validation accuracy: {accuracy:.4f}")
            
            wandb.log({"epoch": epoch, "val_accuracy by epoch": accuracy})

            # ——— 1b: combined per-file accuracy plot ———
            model.eval()
            file_counts = defaultdict(lambda: [0, 0])
            with torch.no_grad():
                for seq, meta, label in zip(val_seqs, val_meta, val_targs):
                    outputs = model(seq.unsqueeze(0))             # <- no unpack here
                    pred    = outputs.argmax(dim=1).item()
                    fid, _, _ = meta
                    file_counts[fid][1] += 1
                    file_counts[fid][0] += int(pred == label.item())

            # pack everything into one dict
            log_dict = {"val_accuracy": accuracy}
            for fid, (correct_count, total_count) in file_counts.items():
                log_dict[f"{fid}_val_accuracy"] = (
                    correct_count / total_count if total_count else 0.0
                )

            # include the epoch so WandB plots it on the x‐axis
            log_dict["epoch"] = epoch

            # log the combined 3‐line plot
            wandb.log(log_dict)
            # ——— end combined per-file accuracy plot



    print("Training complete!")


    #more analysis, including prediciton outcomes distribution graph 
    preds_list = []
    model.eval()
    with torch.no_grad():
        for xb, _ in val_loader:
            outputs = model(xb)
            preds = outputs.argmax(dim=1).cpu().tolist()
            preds_list.extend(preds)

    #start wandb graphing
    # 2. Count each predicted action
    counts = Counter(preds_list)  # e.g., {0: 300, 1: 200, 2: 50, ...}

    # Sort the keys to get consistent order
    sorted_actions = sorted(counts.keys())
    frequencies = [counts[a] for a in sorted_actions]

    # 3a. Log predicted in a historgram in WandB
    wandb.log({"predicted_action_bar": wandb.plot.bar(
        wandb.Table(data=[[a, c] for a, c in zip(sorted_actions, frequencies)], columns=["Action", "Count"]),
        "Action",
        "Count",
        title="Predicted Action Distribution"
    )})
    #3b.  log actual in a historgram in WandB
    true_actions = y_val.cpu().tolist()
    true_counts = Counter(true_actions)
    sorted_true_actions = sorted(true_counts.keys())
    true_frequencies = [true_counts[a] for a in sorted_true_actions]

    wandb.log({"true_action_bar": wandb.plot.bar(
        wandb.Table(data=[[a, c] for a, c in zip(sorted_true_actions, true_frequencies)], columns=["Action", "Count"]),
        "Action",
        "Count",
        title="True Action Distribution"
    )})
    
    #3c. line plot of predicted and actual together in one graph
    pred_counts = Counter(preds_list)
    true_counts = Counter(y_val.cpu().tolist())

    table = wandb.Table(columns=["Action", "Count", "series"])
    for action, cnt in true_counts.items():
        table.add_data(action, cnt, "true")
    for action, cnt in pred_counts.items():
        table.add_data(action, cnt, "predicted")

    wandb.log({
        "action_counts_line": wandb.plot.line(
            table,
            "Action",      # x-axis
            "Count",       # y-axis
            "series",      # ← grouping column (positional)
            title="True vs Predicted Action Counts"
        )
    })
    
    # ——————————————————————————————————————————————————————————
    # 4) Per‑episode cumulative‑accuracy line plots, one graph per file
    # collect per-episode, per-timestep correctness
    episode_results = defaultdict(lambda: defaultdict(list))
    model.eval()
    with torch.no_grad():
        for seq, meta, label in zip(val_seqs, val_meta, val_targs):
            logits = model(seq.unsqueeze(0))
            pred         = logits.argmax(dim=1).item()
            correct      = int(pred == label.item())
            file_id, epi, t = meta
            episode_results[file_id][epi].append((t, correct))

    # for each file, build a W&B Table & log a line plot (one line per episode)
    for file_id, epis in episode_results.items():
        table = wandb.Table(columns=["timestep","accuracy","episode"])
        for epi, data in epis.items():
            data.sort(key=lambda x: x[0])  # by timestep
            cumsum = 0
            for i, (t, corr) in enumerate(data, start=1):
                cumsum += corr
                acc     = cumsum / i
                table.add_data(t, acc, epi)
        wandb.log({
            f"{file_id}_acc_by_timestep":
                #you have to give all the steps here, if you want to do one at a time, do log instead
                wandb.plot.line(
                    table,
                    "timestep",                     # x‑axis
                    "accuracy",                     # y‑axis
                    "episode",                      # ← GROUP column, positional
                    title=f"Cumulative acc by timestep — {file_id}"
                )
        })
        
    # ——————————————————————————————————————————————————————————

    #5 #AVERAGE ACCURACY AT EACH TIMESTEP so for example at timestep 11 average across all episodes, at timestep 12 average acorss all episode (should be 8 episodes)
    #CUMULATIVE ACCURACY AT EACH TIMESTEP
    for file_id, epis in episode_results.items():
        # collect cumulative‑accuracy per episode as before,
        # then bucket those cumulative accuracies by timestep
        timestep_accs = defaultdict(list)
        for epi, data in epis.items():
            data.sort(key=lambda x: x[0])   # by timestep
            cumsum = 0
            for i, (t, corr) in enumerate(data, start=1):
                cumsum += corr
                acc = cumsum / i
                timestep_accs[t].append(acc)

        # build a W&B table of average accuracy at each timestep
        avg_table = wandb.Table(columns=["timestep", "avg_accuracy"])
        for t in sorted(timestep_accs):
            avg = sum(timestep_accs[t]) / len(timestep_accs[t])
            avg_table.add_data(t, avg)

        # log it as a single‑line plot
        wandb.log({
            f"{file_id}_avg_acc_by_timestep":
              wandb.plot.line(
                avg_table,
                "timestep",         # x‑axis
                "avg_accuracy",     # y‑axis
                title=f"Avg accuracy by individual timestep — {file_id}"
              )
        })
    
if __name__ == "__main__":
    main()

