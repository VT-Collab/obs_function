#predict next po_action
import argparse
from pathlib import Path

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import TensorDataset

#from model import LSTMModel
from dataset import * #import all functions from dataset.py
import matplotlib.pyplot as plt
import wandb
from collections import Counter
import os
import random
from collections import defaultdict

#Report
#swtiched to full unseen episodes
#implemented mlp with all files mixed train (action-prediction) and only one file at a time train (heuristic obs)
    #fov-90, 120, 179 individually on like a mix of all 3 or like individually 


#accruracy is like 90% when validating on completely unseen episodes rather than validating on last 10% WITHIN an episode, in the latter case accuracy is only like 80%
#probabily b/c the pattern at the end of each episdoe is very important and the model was denied that before
#again we want full unseen episodes/we want to predict a full new episode/FULL UNSEEN EPISODE b/c we are not dealing with continuous data but rather individual episodes that are different 

#How to run
#/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/baseline_implicit_obs
#(steakhouse-ai) (miniconda3)mishafu@usc-guestwireless-upc-newsc21240 baseline_implicit_obs % /Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python train.py 

#Workflow
# Step 1: Split raw data rows into input output of seq_length to 1 (like 10 or 32  sequential 65-dim obs vector to one immediate after row's po_action). Each size-10 sequence has to be in the same episode
# Step 2: group sequences by file and episode, randomly shuffle the episodes, then split those episodes into 90% portion for training and 10% portion for testing accuracy, so we test on completely unseen episodes without mixing across files
# Step 3: Randomly group, say, 128 of the sequences into one batch (overall, across episodes), and feed into lstm to train. Do that until the 90% portion is used up
# Step 4: after lstm is done training every 10 epochs, take the rest 10% from step 2 and test lstm output on them (no training here just pure testing)

#Next Steps
#keep all files clean
#do 1 (action predction)-> just feed into mlp  (basically treat all 3 files the same, reshuffle) (treat everything the same, just no hidden)
#do 2 (Heuristic Observation Baseline) -> predefinee assumption on observation model  (explocitely say i assume it is this; train 3 networks, each for one file, and then test each of the 3 on the 3 files so 3*3 = 9 results)
    #misalighnment between observation and prediction
    #train mlp with different files, mark them as fov90, fov120, etc. \, do prediciton pairs with different ones and do kind of teh same as number 3
#do 3 -> what i am doing now, optimizec (test on unseen episodes)
    #x - axis = timestep in the episode, y-axis is the accruacy 
    #within an episode is what really matters not throughout 
    #why is it going up, when within the episode
    # 0 -200 timesteps anyawys 
#3-> 1 -> 2
#4 = belief head NEXT

#Note
#set fixed seed (seeds matter in RL LOL, run accross different seed)

#visualization manipulation only, no difference in training or validating:
#1
#overall validation and loss curves for different files? (note just for visualization NOT for training as we are blending all files into training so no distinction in training)
#again one line per file, all on one graph with legends fov-90, fov-120, and fov-179
#2
#add a true action distribution file
#add grouping so true vs predicted action distribution overlap


#visualization + we need do validation differently (actually no just plot it like a line and connect them? just that it wont be left to right but like scattered; shouldn't )
#3
#to accomplish below, we need to train in sequential sliding window in the validation set episodes (cuz we are not training and letting model "see" the validatin set anyways)
#and validating sequentially allow us to visualize accracy anyways; we do averages? so at the first one it is like 0 or 100% -> last one it is like all accuracy/total times tested in episode so far?
#a line for each episode, so multiple lines, # lines = # episodes, and x-axis = time step (0-250 max), y axis = accuracy 

#track prediciton accuracy of an episode
#Track when it becomes higher ACCROSS DIFFERENT FOV (DIFFERENT FILES) -> WHEN EXACTLY IT BECOMES HIGHER

#x - axis = timestep in the episode, y-axis is the accruacy 
# Think about the right way of plotting 
# different ways of plotting things (why pick this and not others)

#OVERALL PAPER = learning obseration function and see how it could help us better
# grouping in wandb reports
#group truth for action vs prediction
#within an episode is what really matters not throughout 
#why is it going up, when within the episode
# 0 -200 timesteps anyawys 


#return training and validation sets, also validation metadata, val_meta = (file_id, episode, prediction_timestep)
def create_sequences(data, seq_length, train_ratio):
    
    #this seed produce 95% accuracy 
    random.seed(42)
    
    # build per-file episode splits
    file_train_eps = {}
    file_val_eps   = {}
    for fid, df_file in data.groupby("file_id"):
        eps = sorted(df_file["episode"].unique())
        random.shuffle(eps)
        split = int(train_ratio * len(eps))
        file_train_eps[fid] = set(eps[:split])
        file_val_eps[fid]   = set(eps[split:])
    
    train_sequence = []
    train_action = []
    
    val_sequence = []
    val_action = []
    val_meta = []

    #group by episodes and file
    episode_data = data.groupby(['file_id', 'episode'])

    # #Get all (file_id, episode) pairs and shuffle globally over all files ok but without putting different files together
    # all_episode_keys = list(episode_data.groups.keys())
    # random.shuffle(all_episode_keys)
    
    # #split into training and validation episode keys, or take 90% of all episodes to train and 10% of remaining whole episodes to validate on 
    # split_idx = int(train_ratio*len(all_episode_keys))
    # #first 90%
    # train_keys = set(all_episode_keys[:split_idx])
    # #remaining 
    # val_keys = set(all_episode_keys[split_idx:])
    
    #iterate over each index and its contents in episode_data
    for key, epi_data in episode_data:
        
        #key is a tuple : (file_id, episode_number)
        fid, epi_num = key
        is_train = (epi_num in file_train_eps[fid])


        total_sequence_episode = []
        total_action_episode = []
        total_meta_episode     = []
        
        #sort by timestep, kinda useless cuz it's alr sorted, but wouldn't hurt
        epi_data = epi_data.sort_values('timestep')
        
        #below each index is corresponding row's info
        vectors = [] #observation vectors
        actions = [] #corresponding actions to the above observation vectors
        timesteps = [] #corresponding timesteps to the obs/actions
        
        #_, is index, but like we dont really need it so we just do _, for ignore
        for _, row in epi_data.iterrows():
            obs_list = obs_to_list(row['obs'])
            vector = obs_list_to_1D_vec(obs_list) #converted final 65-dim observations
            vectors.append(vector) #array of dimention vectors
            actions.append(row['p0_action'])
            timesteps.append(row['timestep'])
    
        #loop over from total # of obs - of size seg_length because we use sliding window of size seg_length
        for i in range(len(vectors) - seq_length):
            #takes chunk of time steps, eg, frost from 0-10 (0:10 means 0 to 10) then 1-11 then 2-12, we need 11 instead of 10 b/c we need the 11th one's p0_action
            timestep_slice = timesteps[i:i + seq_length+1]
            
            #check if timesteps are consequtive, only then, add to final result b/c I noticed it always skip from 1 to 8 at the beginning
            #all() returns true if bool(x) is True for all values x in the iterable
            is_consecutive = all(
                # for j in range(len(timestep_slice)):
                #     timestep_slice[j+1] - timestep_slice[j] == 1 
                timestep_slice[j+1] - timestep_slice[j] == 1 
                    for j in range(len(timestep_slice) - 1)
            )
            if is_consecutive:
                #this will be input, one sequence of seg_length dim-65 vectors
                sequence = vectors[i:i + seq_length]
                #this will be what we predict, one corresponding output of p0_action
                target = actions[i + seq_length] 

                #add this one pair of input output to the overall things
                # total_sequence.append(torch.tensor(sequence, dtype=torch.float32))
                # total_target.append(torch.tensor(target, dtype=torch.float32))
                
                total_sequence_episode.append(torch.tensor(sequence, dtype=torch.float32))
                total_action_episode.append(torch.tensor(target, dtype=torch.float32))
                total_meta_episode.append((key[0], key[1], timestep_slice[-1])) # record which file/episode and at what timestep we predicted
        
        #add to training set if its key is determined to be in training set, already determined above randomly 
        if is_train:
            train_sequence.extend(total_sequence_episode)
            train_action.extend(total_action_episode)
        else:
            val_sequence.extend(total_sequence_episode)
            val_action.extend(total_action_episode)
            val_meta.extend(total_meta_episode)
    
    return train_sequence, train_action, val_sequence, val_action, val_meta
    

def main():
    
    #load data
    #file_path = '../data/fov-90.pkl' #from baseline_implicit_obs
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
    wandb_project_name: str = "new_small_seqs_baseline_implicit_obs_steakhouse-trajectory-training"
    wandb_entity: str = "steakteam"

    # Initialize wandb
    wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        config={
            "seq_length": 3, #change this to 32; running 32
            "train_to_validate_ratio": 0.9,
            "batch_size": 128,
            "hidden_dim": 40,
            "layer_dim": 1,
            "num_epochs": 350, #change to 300 after
            "lr": 0.0001,
            "model": "LSTMModel",
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
    layer_dim = config.layer_dim
    num_epochs = config.num_epochs
    lr         = config.lr

    #get the data
    train_seqs, train_targs, val_seqs, val_targs, val_meta = create_sequences(data, seq_length, train_to_validate_ratio)
    
    print(f"Training sequences:   {len(train_seqs)}")
    print(f"Validation sequences: {len(val_seqs)}")

    #these lengths should match
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
    model = LSTMModel(input_dim=65, hidden_dim = hidden_dim, layer_dim = layer_dim, output_dim=6) 
    criterion = nn.CrossEntropyLoss() #mean squared error loss function, (predicted-actual)^2 and average accross batch
    
    #use adam (adaptive learning rate optimizer) and pass in model parameters and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    # h0, c0 = None, None
    
    for epoch in range(num_epochs):
        
        model.train()
        train_loss = 0.0
        
        #goes through the batch !!!!!???????
        for xb, yb in train_loader:
            #clears greadient from previous batch
            optimizer.zero_grad()

            #forward pass, gets prediction output from model from input
            output, hn, cn = model(xb)
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
        # ——— 1. Finally get accuracy on validation set ———
            model.eval()
            correct = 0
            total   = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    #get model predictions
                    outputs, hn, cn = model(xb)
                    
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
            # inside your epoch‐check block, right after wandb.log({... "val_accuracy by epoch": accuracy})
            model.eval()
            file_counts = defaultdict(lambda: [0, 0])
            with torch.no_grad():
                for seq, meta, label in zip(val_seqs, val_meta, val_targs):
                    logits, _, _ = model(seq.unsqueeze(0))
                    pred         = logits.argmax(dim=1).item()
                    fid, _, _    = meta
                    file_counts[fid][1] += 1
                    file_counts[fid][0] += int(pred == label.item())

            # pack everything into one dict
            log_dict = {"val_accuracy": accuracy}
            for fid, (correct_count, total_count) in file_counts.items():
                log_dict[f"{fid}_val_accuracy"] = correct_count / total_count if total_count else 0.0

            # log all metrics at this epoch
            log_dict["epoch"] = epoch
            wandb.log(log_dict)
            # ——— end combined per-file accuracy plot

    print("Training complete!")


    #more analysis, including prediciton outcomes distribution graph 
    preds_list = []
    model.eval()
    with torch.no_grad():
        for xb, _ in val_loader:
            outputs, hn, cn = model(xb)
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
            logits, _, _ = model(seq.unsqueeze(0))
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
                wandb.plot.line(
                    table,
                    "timestep",                     # x‑axis
                    "accuracy",                     # y‑axis
                    "episode",                      # ← GROUP column, positional
                    title=f"Accuracy by timestep — {file_id}"
                )
        })
        
    # ——————————————————————————————————————————————————————————

    #5 #AVERAGE ACCURACY AT EACH TIMESTEP so for example at timestep 11 average across all episodes, at timestep 12 average acorss all episode (should be 8 episodes)
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
                title=f"Avg accuracy by timestep — {file_id}"
              )
        })
    
if __name__ == "__main__":
    from model import LSTMModel
    
    main()