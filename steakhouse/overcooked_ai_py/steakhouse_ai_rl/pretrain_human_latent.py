from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import shutil

from dataset import TrajectoryDataset
def setup_layouts(layout_names):
    """Setup layout files"""
    for layout_name in layout_names:
        layout_file_name = layout_name + ".layout"
        base_folder = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir))
        path_from = os.path.join(base_folder, "src/data", "layout", layout_file_name)
        path_to = os.path.join(base_folder, "overcooked_ai", "src", "overcooked_ai_py", "data", "layouts", layout_file_name)
        shutil.copy(path_from, path_to)


class HumanLatentStateEstimator(nn.Module):
    def __init__(self, input_dim, sequence_length, latent_dim, hidden_dim=256):
        super(HumanLatentStateEstimator, self).__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * sequence_length)
        )

        # Decoder for ml_actions
        self.ml_action_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        batch_size = x.size(0)
        # Flatten the sequence
        x_flat = x.view(batch_size, -1)
        
        # Encode
        latent = self.encoder(x_flat)
        
        # Decode
        decoded_flat = self.decoder(latent)
        decoded = decoded_flat.view(batch_size, self.sequence_length, self.input_dim)

        # Decode ml_actions
        ml_action_flat = self.ml_action_decoder(latent)
        ml_actions = ml_action_flat.view(batch_size, self.sequence_length)
        
        return latent, decoded, ml_actions

def contrastive_loss(latent, ml_actions, margin=1.0, window_size=3):
    # compare different latent states with each other and compute the loss guided by their difference in ml_action
    batch_size = latent.size(0)
    # Compute pairwise differences between all latent vectors
    # Shape: [batch_size, batch_size, latent_dim]
    diff_latent = latent.unsqueeze(1) - latent.unsqueeze(0)
    
    # Compute L2 distance between latent vectors
    # Shape: [batch_size, batch_size]
    dist_latent = torch.norm(diff_latent, dim=-1)
    
    # Expand sequences for batch comparison
    # Shape: [batch_size, 1, seq_length] and [1, batch_size, seq_length]
    similarity_ml_actions = torch.eq(ml_actions.unsqueeze(1), ml_actions.unsqueeze(0)).float().sum(dim=-1)
    
    # Convert similarity to distance (1 - similarity)
    action_diff = (10.0 - similarity_ml_actions)/10.0
    
    # Compute loss
    loss = (1 - action_diff) * dist_latent + \
           action_diff * torch.clamp(margin - dist_latent, min=0.0)
    
    # Return mean loss, excluding diagonal elements
    mask = 1 - torch.eye(batch_size)
    return (loss * mask).sum() / (mask.sum() + 1e-6)

def train_latent_estimator(num_epochs=20, batch_size=32, lr=1e-3, sequence_length=10, 
                          contrastive_weight=1.0, reconstruction_weight=0.1, ml_reconstruction_weight=1.0):
    # Load data and setup layouts
    dataset = TrajectoryDataset(sequence_length=sequence_length)
    setup_layouts(dataset.layout_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = HumanLatentStateEstimator(
        input_dim=dataset.input_dim,
        sequence_length=sequence_length,
        latent_dim=dataset.latent_dim
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reconstruction_criterion = nn.MSELoss()
    ml_reconstruction_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_ml_reconstruction_loss = 0.0

        for batch_idx, (sequences, times, ml_actions) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            latent, decoded, ml_actions_pred = model(sequences)
            
            # Compute losses
            contr_loss = contrastive_loss(latent, ml_actions)
            recon_loss = reconstruction_criterion(decoded, sequences)
            ml_action_loss = ml_reconstruction_criterion(ml_actions_pred, ml_actions)
            
            # Combined loss
            loss = contrastive_weight * contr_loss + reconstruction_weight * recon_loss + ml_reconstruction_weight * ml_action_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            batch_size = sequences.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_contrastive_loss += contr_loss.item() * batch_size
            epoch_reconstruction_loss += recon_loss.item() * batch_size
            epoch_ml_reconstruction_loss += ml_action_loss.item() * batch_size
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}")
                print(f"Total Loss: {loss.item():.4f}, "
                      f"Contrastive: {contr_loss.item():.4f}, "
                      f"Reconstruction: {recon_loss.item():.4f}, "
                      f"ML Reconstruction: {ml_action_loss.item():.4f}")

        
        # Compute average epoch losses
        num_samples = len(dataset)
        epoch_loss /= num_samples
        epoch_contrastive_loss /= num_samples
        epoch_reconstruction_loss /= num_samples
        epoch_ml_reconstruction_loss /= num_samples

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Average Contrastive Loss: {epoch_contrastive_loss:.4f}")
        print(f"Average Reconstruction Loss: {epoch_reconstruction_loss:.4f}")
        print(f"Average ML Reconstruction Loss: {epoch_ml_reconstruction_loss:.4f}\n")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_dim': dataset.input_dim,
                'latent_dim': dataset.latent_dim,
                'sequence_length': sequence_length,
                'layout_names': dataset.layout_names,
                'epoch': epoch
            }
            torch.save(checkpoint, f"human_latent_estimator_epoch_{epoch+1}.pth")
    
    # Save final model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': dataset.input_dim,
        'latent_dim': dataset.latent_dim,
        'sequence_length': sequence_length,
        'layout_names': dataset.layout_names,
        'epoch': num_epochs
    }
    torch.save(checkpoint, "human_latent_estimator_final.pth")
    print("Training completed. Model saved as human_latent_estimator_final.pth")

if __name__ == "__main__":
    train_latent_estimator()