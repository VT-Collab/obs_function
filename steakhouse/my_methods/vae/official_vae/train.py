"""
Training script that trains the CVAE on data
Loss is computed as the difference between predicted and ground truth partial observatoin
   Predicted_Belief = (mask * r_s_t) + ((1 - mask) * h_s_t_1)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
import torch.nn.functional as F

# Import from your localized files
from dataset import SteakhouseVAEDataset
from vae_class import FixedSpotlightCVAE

# ----------------------------- PATHS & CONFIG -----------------------------
BASE_DIR = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj"

TRAIN_FILES = [
    os.path.join(BASE_DIR, "90_small_small1.pkl"),
    os.path.join(BASE_DIR, "90_small_small2.pkl"),
    os.path.join(BASE_DIR, "full_90_2_2-4.pkl"),
    os.path.join(BASE_DIR, "90_small_small5.pkl"),
    os.path.join(BASE_DIR, "stay_full_90_2_2-4.pkl"),
]

VAL_FILES = [
    #os.path.join(BASE_DIR, "full_179_2_2-5.pkl")
    os.path.join(BASE_DIR, "stay_full_90_2_2-5.pkl")
]

WANDB_PROJECT = "steakhouse_vae_small"
WANDB_ENTITY  = "steakteam"
RUN_NAME      = "fixed_spotlight_2_2-5_run"

# ----------------------------- LOSS FUNCTION -----------------------------
def vae_loss_function(h_s_t_pred, target_x, mu, logvar, mask, pad_mask, beta=0.0, sharpness_weight=1.0): 
    timer_indices_new = [16, 18, 20, 22]
    binary_indices_new = [i for i in range(24) if i not in timer_indices_new]

    pred_bin = h_s_t_pred[:, binary_indices_new, :, :]
    target_bin = target_x[:, binary_indices_new, :, :]
    pred_cont = h_s_t_pred[:, timer_indices_new, :, :]
    target_cont = target_x[:, timer_indices_new, :, :]

    # 1. Weighted BCE
    pos_weight = 50.0  
    bce_weights = torch.ones_like(target_bin)
    bce_weights[target_bin > 0.5] = pos_weight 
    
    # Calculate unreduced BCE so we can mask out the void
    unreduced_bce = F.binary_cross_entropy(pred_bin, target_bin, weight=bce_weights, reduction='none')
    masked_bce = unreduced_bce * pad_mask
    
    # Average ONLY over the valid (non-padded) cells
    BCE = masked_bce.sum() / (pad_mask.sum() * pred_bin.size(1))
    
    # 2. Sharpness Penalty
    sharpness_matrix = (mask * (1.0 - mask)) * pad_mask
    sharpness_penalty = sharpness_matrix.sum() / pad_mask.sum()

    # 3. Continuous Timer MSE
    unreduced_mse = F.mse_loss(pred_cont, target_cont, reduction='none')
    masked_mse = unreduced_mse * pad_mask
    MSE = masked_mse.sum() / (pad_mask.sum() * pred_cont.size(1))

    # 4. KL Divergence (Standard)
    batch_size = target_x.size(0)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    total_loss = BCE + (sharpness_weight * sharpness_penalty) + (MSE * 5.0) + (beta * KLD)
    return total_loss, BCE, KLD

# ----------------------------- MAIN LOOP -----------------------------
def main():
    wandb.init(
        project=WANDB_PROJECT, 
        entity=WANDB_ENTITY, 
        name=RUN_NAME, 
        config={
            "batch_size": 15, 
            "epochs": 10,    
            "learning_rate": 1e-3,
            "latent_dim": 16, #or 8    
            "action_emb_dim": 4, 
            "beta_kld": 0.05, 
            "human_player_idx": 0 
        }
    )
    cfg = wandb.config

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Load Data
    train_dataset = SteakhouseVAEDataset(TRAIN_FILES, human_player_idx=cfg.human_player_idx)
    val_dataset   = SteakhouseVAEDataset(VAL_FILES, human_player_idx=cfg.human_player_idx)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    sample_obs = train_dataset[0]['h_s_t']  
    
    model = FixedSpotlightCVAE(
        vae_layers=sample_obs.shape[0], 
        latent_dim=cfg.latent_dim, 
        action_emb_dim=cfg.action_emb_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    wandb.watch(model, log="all", log_freq=10)

    best_val_loss = float('inf')

    print("\n🔥 Starting Training Loop...")
    
    # --- KL ANNEALING SETUP ---
    warmup_epochs = 50 
    target_beta = cfg.beta_kld

    for epoch in range(cfg.epochs):
        
        # --- CALCULATE DYNAMIC BETA ---
        current_beta = target_beta * min(1.0, epoch / warmup_epochs)
        
        # ==========================================
        #                 TRAINING
        # ==========================================
        model.train()
        tr_loss_total, tr_bce_total, tr_kld_total = 0.0, 0.0, 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            X        = batch['h_s_t'].to(device).float()       
            r_s_t    = batch['r_s_t'].to(device).float()
            h_s_t_1  = batch['h_s_t_1'].to(device).float()
            h_a_t    = batch['h_a_t'].to(device).long()    
            pad_mask = batch['pad_mask'].to(device).float() 
            
            optimizer.zero_grad()
            
            h_s_t_pred, mu, logvar, mask = model(h_s_t=X, r_s_t=r_s_t, h_s_t_1=h_s_t_1, h_a_t=h_a_t)
            
            loss, bce, kld = vae_loss_function(
                h_s_t_pred, X, mu, logvar, 
                mask=mask, pad_mask=pad_mask, beta=current_beta
            )
            
            # # 👇 --- DEBUG: PRINT MASKS THROUGHOUT THE BATCH --- 👇
            # if epoch % 10 == 0 and batch_idx == 0: 
            #     print(f"\n[{epoch+1}/{cfg.epochs}] Train Spotlight Masks across the batch:")
                
            #     current_batch_size = mask.shape[0]
            #     indices_to_print = [0, 15, 30, 45]
                
            #     for idx in indices_to_print:
            #         if idx < current_batch_size: 
            #             single_mask = mask[idx, 0].detach().cpu().numpy()
            #             print(f"--- Train Mask at batch index {idx} ---")
            #             print(np.round(single_mask, 2))
            #     print("-----------------------------------------\n")
            # # 👆 ------------------------------------------------ 👆

            loss.backward()
            optimizer.step()
            
            tr_loss_total += loss.item()
            tr_bce_total  += bce.item()
            tr_kld_total  += kld.item()
            
        avg_tr_loss = tr_loss_total / len(train_dataset)
        avg_tr_bce  = tr_bce_total / len(train_dataset)
        avg_tr_kld  = tr_kld_total / len(train_dataset)

        # ==========================================
        #        VALIDATION (PURE INFERENCE)
        # ==========================================
        model.eval()
        val_loss_total, val_bce_total, val_kld_total = 0.0, 0.0, 0.0
        
        current_belief = None

        # ==========================================
        #        VALIDATION (ROLLING AUTOREGRESSION)
        # ==========================================
        model.eval()
        val_loss_total, val_bce_total, val_kld_total = 0.0, 0.0, 0.0
        
        current_belief = None
        prev_episode = None
        prev_timestep = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader): 
                X        = batch['h_s_t'].to(device).float()
                r_s_t    = batch['r_s_t'].to(device).float()
                h_s_t_1  = batch['h_s_t_1'].to(device).float()
                h_a_t    = batch['h_a_t'].to(device).long()
                pad_mask = batch['pad_mask'].to(device).float() 
                
                # Extract tracking info (Batch size is 1, so we grab index 0)
                current_episode = batch['episode'][0] 
                current_timestep = batch['timestep'][0].item() 
                
                # --- THE RESET LOGIC ---
                # Check 1: Is this a brand new episode?
                is_new_episode = (prev_episode is None) or (current_episode != prev_episode)
                
                # Check 2: Did the timestep jump/skip? (e.g., went from 5 to 10)
                is_jump = (prev_timestep is not None) and (current_timestep != prev_timestep + 1)
                
                if is_new_episode or is_jump:
                    # RESET TO r_s_t as requested!
                    current_belief = r_s_t
                    
                # 🚨 PURE INFERENCE: Sample random noise z 🚨
                z = torch.randn(X.size(0), cfg.latent_dim).to(device)
                
                # Use our rolling 'current_belief'
                h_s_t_pred, mask = model.inference(z=z, r_s_t=r_s_t, h_s_t_1=current_belief, h_a_t=h_a_t)
                
                # Update belief for the NEXT loop and detach from graph
                current_belief = h_s_t_pred.detach()
                
                # Update trackers for the NEXT loop
                prev_episode = current_episode
                prev_timestep = current_timestep
                
                # Calculate Loss (using dummy vars for the missing Encoder)
                dummy_mu = torch.zeros(X.size(0), cfg.latent_dim).to(device)
                dummy_logvar = torch.zeros(X.size(0), cfg.latent_dim).to(device)
                
                loss, bce, kld = vae_loss_function(
                    h_s_t_pred, X, dummy_mu, dummy_logvar, 
                    mask=mask, pad_mask=pad_mask, beta=current_beta
                )
                
                # 👇 --- DEBUG: PRINT TRUE CLEAN MASKS DURING VALIDATION --- 👇
                # Note: Changed indices_to_print since batch_size=1 limits the dimension
                #if epoch % 5 == 0 and batch_idx in [0, 15, 30]:
                if epoch % 9 == 0 and batch_idx in [0, 15, 30]:
                    print(f"\n[Epoch {epoch+1}] VAL Spotlight Mask at Val Step {batch_idx}:")
                    single_mask = mask[0, 0].cpu().numpy() 
                    print(np.round(single_mask, 2))
                    print("--------------------------------------------------\n")
                # 👆 -------------------------------------------------------- 👆

                val_loss_total += loss.item()
                val_bce_total  += bce.item()
                val_kld_total  += kld.item()

        avg_val_loss = val_loss_total / len(val_dataset)
        avg_val_bce  = val_bce_total / len(val_dataset)
        avg_val_kld  = val_kld_total / len(val_dataset)

        # --- LOGGING ---
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | Beta: {current_beta:.4f} | Train Loss: {avg_tr_loss:.2f} | Val Loss: {avg_val_loss:.2f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "Current_Beta": current_beta, 
            "Train/Total_Loss": avg_tr_loss,
            "Train/Weighted_BCE": avg_tr_bce,
            "Train/KL_Divergence": avg_tr_kld,
            "Val/Total_Loss": avg_val_loss,
            "Val/Weighted_BCE": avg_val_bce,
            "Val/KL_Divergence": avg_val_kld
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_fixed_spotlight_cvae.pth") 
            wandb.run.summary["best_val_loss"] = best_val_loss

    print("\n✅ Training Complete!")
    wandb.finish()

if __name__ == "__main__":
    main()