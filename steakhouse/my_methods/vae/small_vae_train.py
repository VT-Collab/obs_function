"""
small_vae_train.py
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np

# Import from your localized files
from small_vae_dataset import SteakhouseVAEDataset
from small_vae_class import HumanBeliefCVAE

# ----------------------------- PATHS & CONFIG -----------------------------
BASE_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj"

TRAIN_FILES = [
    os.path.join(BASE_DIR, "90_small_small1.pkl"),
    os.path.join(BASE_DIR, "90_small_small2.pkl")
]

VAL_FILES = [
    os.path.join(BASE_DIR, "90_small_small5.pkl")
]

WANDB_PROJECT = "steakhouse_vae_small"
WANDB_ENTITY  = "steakteam"
RUN_NAME      = "dynamic_change_loss"

# ----------------------------- LOSS FUNCTION -----------------------------
def vae_loss_function(recon_x, x, h_s_t_1, mu, logvar, beta=0.1):
    """
    Dynamic Change VAE loss. Heavily penalizes getting the *transitions* wrong.
    """
    timer_indices = [14, 16, 18] 
    player_layer_indices = [0, 1] 
    binary_indices = [i for i in range(x.shape[1]) if i not in timer_indices]

    recon_bin = recon_x[:, binary_indices, :, :]
    target_bin = x[:, binary_indices, :, :]
    prev_bin = h_s_t_1[:, binary_indices, :, :] # <-- We now look at the past!
    
    # 1. Raw BCE
    bce_raw = nn.functional.binary_cross_entropy(recon_bin, target_bin, reduction='none')

    # 2. DYNAMIC CHANGE WEIGHTING (The Fix)
    # Identify exactly which pixels changed state from t-1 to t (0->1 or 1->0)
    changed_mask = (target_bin != prev_bin).float()
    
    # Base weight is 1. If it changed, multiply penalty by 50.
    # This stops the model from guessing all 1s, because false positives on static 0s 
    # will no longer be overpowered by a blanket 1s multiplier.
    weight_mask = torch.ones_like(target_bin) + (changed_mask * 49.0) 

    # 3. PLAYER WEIGHTING
    # Multiply the player layers by an extra 2x so it tracks movement closely.
    for i, original_idx in enumerate(binary_indices):
        if original_idx in player_layer_indices:
            weight_mask[:, i, :, :] *= 2.0 

    BCE = torch.sum(bce_raw * weight_mask)

    # 4. MSE FOR TIMERS
    recon_time = recon_x[:, timer_indices, :, :]
    target_time = x[:, timer_indices, :, :]
    MSE = nn.functional.mse_loss(recon_time, target_time, reduction='sum')

    # 5. KL DIVERGENCE
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + (MSE * 5.0) + (beta * KLD), BCE, KLD

# ----------------------------- MAIN LOOP -----------------------------
def main():
    wandb.init(
        project=WANDB_PROJECT, 
        entity=WANDB_ENTITY, 
        name=RUN_NAME, 
        config={
            "batch_size": 64,
            "epochs": 39,
            "learning_rate": 1e-3,
            "latent_dim": 128,
            "action_emb_dim": 16,
            "beta_kld": 0.05, 
            "human_player_idx": 0 
        }
    )
    cfg = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Load Data
    train_dataset = SteakhouseVAEDataset(TRAIN_FILES, human_player_idx=cfg.human_player_idx)
    val_dataset   = SteakhouseVAEDataset(VAL_FILES, human_player_idx=cfg.human_player_idx)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    sample_obs = train_dataset[0]['X']  
    model = HumanBeliefCVAE(
        sample_obs=sample_obs, 
        latent_dim=cfg.latent_dim, 
        action_emb_dim=cfg.action_emb_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    wandb.watch(model, log="all", log_freq=10)

    best_val_loss = float('inf')

    print("\n🔥 Starting Training Loop...")
    for epoch in range(cfg.epochs):
        
        # --- TRAIN ---
        model.train()
        tr_loss_total, tr_bce_total, tr_kld_total = 0.0, 0.0, 0.0
        
        for batch in train_loader:
            X       = batch['X'].to(device)
            r_s_t   = batch['r_s_t'].to(device)
            h_s_t_1 = batch['h_s_t_1'].to(device)
            h_a_t   = batch['h_a_t'].to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(h_s_t=X, r_s_t=r_s_t, h_s_t_1=h_s_t_1, h_a_t=h_a_t)
            
            # 👇 Pass h_s_t_1 into the loss function here!
            loss, bce, kld = vae_loss_function(recon_batch, X, h_s_t_1, mu, logvar, beta=cfg.beta_kld)
            
            loss.backward()
            optimizer.step()
            
            tr_loss_total += loss.item()
            tr_bce_total  += bce.item()
            tr_kld_total  += kld.item()
            
        avg_tr_loss = tr_loss_total / len(train_dataset)
        avg_tr_bce  = tr_bce_total / len(train_dataset)
        avg_tr_kld  = tr_kld_total / len(train_dataset)

        # --- VALIDATE ---
        model.eval()
        val_loss_total, val_bce_total, val_kld_total = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                X       = batch['X'].to(device)
                r_s_t   = batch['r_s_t'].to(device)
                h_s_t_1 = batch['h_s_t_1'].to(device)
                h_a_t   = batch['h_a_t'].to(device)
                
                recon_batch, mu, logvar = model(h_s_t=X, r_s_t=r_s_t, h_s_t_1=h_s_t_1, h_a_t=h_a_t)
                
                # 👇 Pass h_s_t_1 into the loss function here!
                loss, bce, kld = vae_loss_function(recon_batch, X, h_s_t_1, mu, logvar, beta=cfg.beta_kld)
                
                val_loss_total += loss.item()
                val_bce_total  += bce.item()
                val_kld_total  += kld.item()

        avg_val_loss = val_loss_total / len(val_dataset)
        avg_val_bce  = val_bce_total / len(val_dataset)
        avg_val_kld  = val_kld_total / len(val_dataset)

        # --- LOGGING ---
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | Train Loss: {avg_tr_loss:.2f} | Val Loss: {avg_val_loss:.2f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "Train/Total_Loss": avg_tr_loss,
            "Train/Weighted_BCE": avg_tr_bce,
            "Train/KL_Divergence": avg_tr_kld,
            "Val/Total_Loss": avg_val_loss,
            "Val/Weighted_BCE": avg_val_bce,
            "Val/KL_Divergence": avg_val_kld
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_small_human_belief_cvae.pth")
            wandb.run.summary["best_val_loss"] = best_val_loss

    print("\n✅ Training Complete!")
    wandb.finish()

if __name__ == "__main__":
    main()