"""

Current: MSE Loss but only a few are continuous and a lot are 0/1
Total numbers predicted per timestep: 39 times 15 times 10 = 5,850 numbers

If your total validation MSE is 270, that means the average squared error per grid cell is sqrt(270/5850) = 0.021
or 0.021 off each grid 

0 1
0.02 1.02

2.3 -> 2 rather than 0/1

29
29.02


=============================================================================
   TRAINING SCRIPT: HUMAN BELIEF CVAE
=============================================================================
High-Level Goal: 
We are training a Conditional Variational Autoencoder (CVAE) to act as a 
forward dynamics model. It learns to predict the human player's *current* partial observation, based on:
  1. The robot's current full observation (Context)
  2. The human's previous partial observation (Context)
  3. The action the human just took (Context)

Key Technical Detail: 
The environment grids contain both binary flags (e.g., "Is there an onion? 0/1") 
and continuous timers (e.g., "Soup cook time: 29"). Because of these timers, 
we CANNOT use standard Binary Cross-Entropy (BCE). Instead, we use Mean Squared 
Error (MSE) combined with a KL Divergence penalty.

--- QUICK START / MANUAL TOGGLES ---

1. File Paths (Update these for new datasets):
   - TRAIN_FILE: Path to the .pkl file used for training.
   - VAL_FILE: Path to the .pkl file used for validation.

2. Weights & Biases (WandB) Settings:
   - WANDB_PROJECT: Name of the project dashboard.
   - WANDB_ENTITY: Your WandB team/username.
   - RUN_NAME: Descriptive name for this specific training run.

3. Hyperparameters (Located inside the `wandb.init` config dictionary):
   - epochs: (Default 100) How many passes over the full dataset.
   - batch_size: (Default 64) Number of states to process at once.
   - learning_rate: (Default 1e-3) Optimizer step size.
   - latent_dim: (Default 64) The size of the compressed "concept" bottleneck.
   - action_emb_dim: (Default 16) Size of the dense vector representing the human's action.
   - beta_kld: (Default 0.1) The weight applied to the KL Divergence loss. 
     * Tuning Tip: Turn this UP if the latent space needs to be smoother. 
                   Turn this DOWN if the model's reconstructions are too blurry.
=============================================================================
"""

import os
import torch
import torch.nn as nn
#optimizer that updates the model's weights
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np

# Import your custom classes from the same folder
from vae_dataset import SteakhouseVAEDataset
from vae_class import HumanBeliefCVAE

# ----------------------------- PATHS & CONFIG -----------------------------
TRAIN_FILE = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj/full_90_2_2-4.pkl"
VAL_FILE   = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj/full_179_2_2-5.pkl"

WANDB_PROJECT = "steakhouse_vae" # Updated project name
WANDB_ENTITY  = "steakteam"
RUN_NAME      = "new_cvae_human_belief_mse"


def vae_loss_function(recon_x, x, mu, logvar, beta=0.1):
    """
    Computes the VAE loss function: Weighted BCE + MSE + Beta * KL Divergence
    Model tries to make the calculated loss as small as possible
    
    Parameters:
        recon_x — the model's prediction (what it THINKS the human sees)
        x — the ground truth (what the human ACTUALLY sees)
        mu — the mean from the encoder (from the VAE architecture)
        logvar — the log variance from the encoder
        beta=0.1 — a weight that controls how much one part of the loss matters (default 0.1)
        
    
    Goal: 
    - Use BCE for binary masks (0/1) to keep the kitchen layout sharp.
    - Use MSE for normalized timers (0-1) to maintain numeric precision.
    - Weigh Player 0 and Player 1 more heavily to prevent them from being "ignored" 
      in the sparse 15x10 grid.
    """
    timer_indices = [28, 30, 32]
    # From get_obs: 0 is 'human_loc', 1 is 'other_player_loc'
    player_layer_indices = [0, 1] 
    
    # Identify which layers are strictly binary (all layers except the 3 timers)
    # different layers is different math
    binary_indices = [i for i in range(x.shape[1]) if i not in timer_indices]

    # --- 1. BCE for Logic/Players/Objects (Binary Layers) ---
    # These layers are only 0 OR 1.
    #slice out only binary layers
    # [:, binary_indices, :, :] means: "take all batches, only these layer indices, all rows, all columns.
    recon_bin = recon_x[:, binary_indices, :, :]
    target_bin = x[:, binary_indices, :, :]
    
    # We use reduction='none' because we need the raw error for every single pixel.
    # If we used 'sum' here, we couldn't multiply the player layers specifically.
    #the standard math formula for grading 0/1 predictions. It's designed specifically for probabilities between 0 and 1.
    bce_loss_raw = nn.functional.binary_cross_entropy(recon_bin, target_bin, reduction='none')

    # --- APPLY WEIGHTS TO THE PLAYER LAYERS ---
    # Since 'recon_bin' only contains 36 layers, we need to find which of those 
    # 36 correspond to the original 'human_loc' (0) and 'other_player_loc' (1).
    for i, original_idx in enumerate(binary_indices):
        if original_idx in player_layer_indices:
            # 200x weight: This tells the optimizer that missing a player is 
            # 200 times worse than missing a piece of floor or a counter.
            # This forces the model to reconstruct the player pixels clearly.
            bce_loss_raw[:, i, :, :] *= 300.0 
            
            #OHHH if you predict everything else as 0 in those layers you are still MOSTLY correct
            #even if you get the 1 (where the player is) WRONG
            #CUZ the layout is really really big 
    
    # After weighting the important layers, we sum everything into one scalar.
    #torch.sum() adds up every single number in bce_loss_raw — across all batch items, all layers, all rows, all columns — into one single scalar (a single number).
    BCE = torch.sum(bce_loss_raw)

    # --- 2. MSE for Timers (Already normalized in Dataset) ---
    # These values are continuous BETWEEN 0 and 1.
    #slices so it grabs only the continuous layers 
    recon_time = recon_x[:, timer_indices, :, :]
    target_time = x[:, timer_indices, :, :]
    
    # We sum the squared error for the timers. 
    # We multiply by 5.0 to ensure the timer precision stays relevant 
    # now that we've boosted the BCE side.
    # calculate the mse loss 
    MSE = nn.functional.mse_loss(recon_time, target_time, reduction='sum')

    # --- 3. KL Divergence ---
    # Forces the latent space to follow a normal distribution (forces smoothness).
    # the KL Divergence between our learned distribution (defined by mu and logvar) and a standard normal distribution (bell curve centered at 0).
    # aka "how different is our learned distribution from a perfect bell curve?"
    # prevents distribution from being scaled too far apart
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss: Combined hybrid objectives.
    # Three components get added together:
    #   1. BCE — error on binary layers (with player cells weighted 300×)
    #   2. MSE * 5.0 — error on timer layers, multiplied by 5 to keep it "relevant" now that BCE is boosted by the 300× player weight
    #   3. beta * KLD — the latent space regularization penalty, scaled by beta (default 0.1 or 0.05)
    total_loss = BCE + (MSE * 5.0) + (beta * KLD)
    
    return total_loss, BCE, KLD

# ----------------------------- MAIN LOOP -----------------------------
def main():
    # 1. Initialize Weights & Biases
    wandb.init(
        project=WANDB_PROJECT, 
        entity=WANDB_ENTITY, 
        name=RUN_NAME, 
        config={
            "batch_size": 64,
            "epochs": 6,
            "learning_rate": 1e-3,
            "latent_dim": 128,
            "action_emb_dim": 16,
            "beta_kld": 0.05, # Weight for KL Divergence
            "human_player_idx": 0 # Assuming p0 is the human
        }
    )
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # 2. Load Datasets & DataLoaders
    # The dataset handles all the t and t-1 pairing internally!
    #1. Target (X): Human's partial observation at time t (h_s_t)
    train_dataset = SteakhouseVAEDataset(TRAIN_FILE, human_player_idx=cfg.human_player_idx)
    val_dataset   = SteakhouseVAEDataset(VAL_FILE, human_player_idx=cfg.human_player_idx)

    #shuffle traning but not validation
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 3. Dynamically Initialize Model
    print("\nInitializing model...")
    #gets the first target value from the dataset
    sample_obs = train_dataset[0]['X'] # Just pass the raw 3D tensor!    
    model = HumanBeliefCVAE(
        sample_obs=sample_obs, 
        latent_dim=cfg.latent_dim, #different here than defined in model
        action_emb_dim=cfg.action_emb_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # Watch model gradients and topology in wandb
    wandb.watch(model, log="all", log_freq=10)

    # 4. Training & Validation Loop
    #sets best score so far as infinity
    best_val_loss = float('inf')

    print("\n🔥 Starting Training Loop...")
    #loops through the number of epochs (each epoch = entire training set)
    for epoch in range(cfg.epochs):
        
        # --- TRAIN PASS ---
        model.train()
        
        #add each batch's loss to them, at the end of epoch divide by dataset size to get an average
        tr_loss_total = 0.0
        tr_mse_total = 0.0
        tr_kld_total = 0.0
        
        for batch in train_loader:
            
            X       = batch['X'].to(device)
            r_s_t   = batch['r_s_t'].to(device)
            h_s_t_1 = batch['h_s_t_1'].to(device)
            h_a_t   = batch['h_a_t'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            recon_batch, mu, logvar = model(h_s_t=X, r_s_t=r_s_t, h_s_t_1=h_s_t_1, h_a_t=h_a_t)
            
            # Loss
            loss, mse, kld = vae_loss_function(recon_batch, X, mu, logvar, beta=cfg.beta_kld)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            tr_loss_total += loss.item()
            tr_mse_total  += mse.item()
            tr_kld_total  += kld.item()
            
        # Average Train Metrics
        avg_tr_loss = tr_loss_total / len(train_dataset)
        avg_tr_mse  = tr_mse_total / len(train_dataset)
        avg_tr_kld  = tr_kld_total / len(train_dataset)

        # --- VALIDATION PASS ---
        model.eval()
        val_loss_total = 0.0
        val_mse_total = 0.0
        val_kld_total = 0.0
        
        with torch.no_grad():
            for batch in val_loader: #validation loader
                X       = batch['X'].to(device)
                r_s_t   = batch['r_s_t'].to(device)
                h_s_t_1 = batch['h_s_t_1'].to(device)
                h_a_t   = batch['h_a_t'].to(device)
                
                recon_batch, mu, logvar = model(h_s_t=X, r_s_t=r_s_t, h_s_t_1=h_s_t_1, h_a_t=h_a_t)
                loss, mse, kld = vae_loss_function(recon_batch, X, mu, logvar, beta=cfg.beta_kld)
                
                val_loss_total += loss.item()
                val_mse_total  += mse.item()
                val_kld_total  += kld.item()

        # Average Val Metrics
        avg_val_loss = val_loss_total / len(val_dataset)
        avg_val_mse  = val_mse_total / len(val_dataset)
        avg_val_kld  = val_kld_total / len(val_dataset)

        # --- LOGGING ---
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | Train Loss: {avg_tr_loss:.2f} | Val Loss: {avg_val_loss:.2f}")
        
        wandb.log({
            "epoch": epoch + 1,
            
            "Train/Total_Loss": avg_tr_loss,
            "Train/Reconstruction_MSE": avg_tr_mse,
            "Train/KL_Divergence": avg_tr_kld,
            
            "Val/Total_Loss": avg_val_loss,
            "Val/Reconstruction_MSE": avg_val_mse,
            "Val/KL_Divergence": avg_val_kld
        })

        # --- SAVE BEST MODEL ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_human_belief_cvae.pth")
            wandb.run.summary["best_val_loss"] = best_val_loss
            # print(f"  -> Saved new best model! (Val Loss: {best_val_loss:.2f})")

    print("\n✅ Training Complete!")
    wandb.finish()

if __name__ == "__main__":
    main()