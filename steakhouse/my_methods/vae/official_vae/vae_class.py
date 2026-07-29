"""
A CVAE class that predicts the observation mask over the layout at any given timestep.
Designed strictly for a fixed 15x15 zero-padded layout size and optimized for ultra-small datasets.

Predicted_Belief = (mask * r_s_t) + ((1 - mask) * h_s_t_1)
"""

import torch
import torch.nn as nn


class FixedSpotlightCVAE(nn.Module):
    def __init__(self, vae_layers=24, latent_dim=16, action_emb_dim=4, max_grid_size=15):
        super().__init__()
        
        # Dimensions
        self.vae_layers = vae_layers
        self.latent_dim = latent_dim # Bumped up to 16 to handle the larger 15x15 map
        self.max_grid_size = max_grid_size
        
        
        # 1. Action Embedding (6 actions fit perfectly into 4 continuous dimensions)
        # IN: (Batch) integer -> OUT: (Batch, 4)
        self.action_emb = nn.Embedding(num_embeddings=6, embedding_dim=action_emb_dim)
        
        # ====================
        # MICRO ENCODER (The Funnel: 72 -> 16 -> 8 -> 1800)
        # ====================
        self.enc_conv = nn.Sequential(
            # IN: 3 grids (Target, Robot, Past) * 24 layers = 72 layers
            # Shape: (Batch, 72, 15, 15) -> (Batch, 16, 15, 15)
            nn.Conv2d(self.vae_layers * 3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 25% Dropout: Forces general rule learning instead of memorizing exact pixels
            nn.Dropout2d(0.25), 
            
            # Shape: (Batch, 16, 15, 15) -> (Batch, 8, 15, 15)
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            
            # Flatten the spatial grid into a 1D vector
            # Shape: (Batch, 8, 15, 15) -> (Batch, 1800) 
            nn.Flatten()
        )
        
        # New calculation: 8 channels * 15 * 15 = 1800
        self.flatten_size = 8 * self.max_grid_size * self.max_grid_size
        
        # The flattened grid (1800 numbers) + the action embedding (4 numbers) = 1804
        # Shape: (Batch, 1804) -> (Batch, 16)
        self.enc_fc_mu = nn.Linear(self.flatten_size + action_emb_dim, latent_dim)        
        self.enc_fc_logvar = nn.Linear(self.flatten_size + action_emb_dim, latent_dim)      
        
          
        # ====================
        # MICRO DECODER (Outputs the 1-Layer 15x15 Spotlight Mask)
        # ====================
        self.dec_conv = nn.Sequential(
            # IN: Broadcasted Latent/Action + (16+4=20 numbers where each of the 20 numbers is stretched to 15x15 layout) 
            # + Robot (24) + Past (24) = 68 channels!
            # Shape: (Batch, 68, 15, 15) -> (Batch, 16, 15, 15)
            nn.Conv2d(latent_dim + action_emb_dim + (self.vae_layers * 2), 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            
            # Shape: (Batch, 16, 15, 15) -> (Batch, 8, 15, 15)
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # Final output is exactly 1 layer (The Spotlight mask)
            # Shape: (Batch, 8, 15, 15) -> (Batch, 1, 15, 15)
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() # Forces spotlight values to probabilities between 0.0 and 1.0
        )

    def encode(self, h_s_t, r_s_t, h_s_t_1, h_a_t):
        # 1. Action: (Batch) -> (Batch, 4)
        a_emb = self.action_emb(h_a_t) 
        
        # 2. Stack grids: 3 * (Batch, 24, 15, 15) -> (Batch, 72, 15, 15)
        x = torch.cat([h_s_t, r_s_t, h_s_t_1], dim=1) 
        
        # 3. CNN + Flatten: (Batch, 72, 15, 15) -> (Batch, 1800)
        x = self.enc_conv(x) 
        
        # 4. Combine with action: (Batch, 1800) + (Batch, 4) -> (Batch, 1804)
        combined = torch.cat([x, a_emb], dim=1) 
        
        # 5. Project to latent bottleneck: (Batch, 1804) -> (Batch, 16)
        mu = self.enc_fc_mu(combined)
        logvar = self.enc_fc_logvar(combined)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Standard VAE sampling trick
        # IN: mu (Batch, 16), logvar (Batch, 16) -> OUT: z (Batch, 16)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, r_s_t, h_s_t_1, h_a_t):
        # We know it's fixed 15x15, so we extract H and W dynamically but safely
        B, _, H, W = r_s_t.shape # H=15, W=15
        
        # 1. Action: (Batch) -> (Batch, 4)
        a_emb = self.action_emb(h_a_t) 
        
        # 2. Combine Latent 'z' and Action: (Batch, 16) + (Batch, 4) -> (Batch, 20)
        z_and_action = torch.cat([z, a_emb], dim=1) 
        
        # 3. BROADCASTING:
        # WHAT THIS IS doing: `z_and_action` is just a list of 20 numbers right now. 
        # But we need to stack it on top of our image maps! You can't stack a 1D list on a 2D image.
        # `.view` turns it into a 1x1 pixel image with 20 layers.
        # `.expand` copies that 1x1 pixel across the whole board to make a 15x15 image.
        # So every single tile on the 15x15 grid now knows what the 20 latent/action numbers are!
        z_broadcast = z_and_action.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # 4. Stack everything: Broadcasted concept (20) + Robot View (24) + Past View (24) 
        # Shape: (Batch, 20, 15, 15) + (Batch, 24, 15, 15) + (Batch, 24, 15, 15) -> (Batch, 68, 15, 15)
        decoder_input = torch.cat([z_broadcast, r_s_t, h_s_t_1], dim=1) 
        
        # 5. Output the Spotlight Mask
        # Shape: (Batch, 68, 15, 15) -> (Batch, 1, 15, 15)
        mask = self.dec_conv(decoder_input) 
        
        return mask

    def forward(self, h_s_t, r_s_t, h_s_t_1, h_a_t):
        # 1. Get Latent variables from Encoder
        mu, logvar = self.encode(h_s_t, r_s_t, h_s_t_1, h_a_t)
        
        # 2. Sample from the distribution
        z = self.reparameterize(mu, logvar)
        
        # 3. Predict the 1x15x15 Spotlight Mask from Decoder
        mask = self.decode(z, r_s_t, h_s_t_1, h_a_t)
        
        # 4. THE SPOTLIGHT MATH (Merge Reality and Memory)
        visible_now = mask * r_s_t               # (Batch, 1, 15, 15) * (Batch, 24, 15, 15)
        remembered_past = (1.0 - mask) * h_s_t_1 # (Batch, 1, 15, 15) * (Batch, 24, 15, 15)
        
        # Final mapped belief state: (Batch, 24, 15, 15)
        h_s_t_pred = visible_now + remembered_past
        
        return h_s_t_pred, mu, logvar, mask
    
    # 👇 ADD THIS NEW METHOD RIGHT HERE 👇
    def inference(self, z, r_s_t, h_s_t_1, h_a_t):
        """
        PURE INFERENCE: Bypasses the Encoder completely! 
        Used during validation and actual gameplay.
        Takes a random 'z' dart and decoder conditions to generate the belief state.
        """
        # 1. Predict the 1x15x15 Spotlight Mask straight from the Decoder
        mask = self.decode(z, r_s_t, h_s_t_1, h_a_t)
        
        # 2. THE SPOTLIGHT MATH (Merge Reality and Memory)
        visible_now = mask * r_s_t               
        remembered_past = (1.0 - mask) * h_s_t_1 
        
        # Final mapped belief state
        h_s_t_pred = visible_now + remembered_past
        
        # Notice we only return the prediction and the mask (no mu or logvar here)
        return h_s_t_pred, mask