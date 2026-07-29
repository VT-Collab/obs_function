"""
A CVAE class that predicts the observation mask over the layout at any given timestep 
    it takes in (ultimately different) layout size and outputs a different layout size
"""

import torch
import torch.nn as nn

class SpotlightCVAE(nn.Module):
    """
    vae_layers = fixed; 24 layers no matter the layout
    latent_dim = how many numbers we ultimately compress into the latent
    action_emb_dim = how many numbers we append to the output. Can consider adding not appending later
    total vector size = latent_dim + action_emb_dim
    """
    def __init__(self, vae_layers=24, latent_dim=64, action_emb_dim=16):
        super().__init__()
        
        self.vae_layers = vae_layers
        self.latent_dim = latent_dim
        
        # Action Embedding (6 possible actions -> 16 numbers)
        self.action_emb = nn.Embedding(num_embeddings=6, embedding_dim=action_emb_dim)
        
        # ====================
        # ENCODER (Training Only)
        # Inputs: Target (24) + Robot (24) + Past (24) = 72 layers
        # ====================
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.vae_layers * 3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # THE MAGIC TRICK: This shrinks ANY Height x Width down to 1x1 (so we can have a latent).
            # This is what makes the model immune to different layout sizes!
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        # Latent projections (Input is 64 from CNN + 16 from Action)
        self.enc_fc_mu = nn.Linear(64 + action_emb_dim, latent_dim)
        self.enc_fc_logvar = nn.Linear(64 + action_emb_dim, latent_dim)
        
        # ====================
        # DECODER (Outputs the 1-Layer Spotlight Mask)
        # Inputs: Broadcasted Latent/Action (64+16=80) + Robot (24) + Past (24) = 128 layers
        # ====================
        self.dec_conv = nn.Sequential(
            nn.Conv2d(latent_dim + action_emb_dim + (self.vae_layers * 2), 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Final output is exactly 1 layer (The Spotlight)
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid() # Forces spotlight values between 0 (Blind) and 1 (Visible)
        )

    def encode(self, h_s_t, r_s_t, h_s_t_1, h_a_t):
        # 1. Embed Action
        a_emb = self.action_emb(h_a_t) # Shape: (Batch, 16)
        
        # 2. Stack grids
        x = torch.cat([h_s_t, r_s_t, h_s_t_1], dim=1) # Shape: (Batch, 72, H, W)
        
        # 3. Run through CNN and Global Average Pool
        x = self.enc_conv(x) # Shape: (Batch, 64, 1, 1)
        x = x.view(x.size(0), -1) # Flatten to (Batch, 64)
        
        # 4. Combine with action and get latent distribution
        combined = torch.cat([x, a_emb], dim=1) # Shape: (Batch, 80)
        mu = self.enc_fc_mu(combined)
        logvar = self.enc_fc_logvar(combined)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, r_s_t, h_s_t_1, h_a_t):
        # Get dynamic Height and Width from the Robot's current view
        B, C, H, W = r_s_t.shape
        
        # 1. Embed Action
        a_emb = self.action_emb(h_a_t) # Shape: (Batch, 16)
        
        # 2. Combine Latent 'z' and Action
        z_and_action = torch.cat([z, a_emb], dim=1) # Shape: (Batch, 80)
        
        # 3. BROADCASTING: Stretch the 80-number vector across the whole HxW grid
        # Reshape to (Batch, 80, 1, 1) then expand to (Batch, 80, H, W)
        z_broadcast = z_and_action.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        # 4. Stack everything together: The broadcasted concept + Robot View + Past View
        decoder_input = torch.cat([z_broadcast, r_s_t, h_s_t_1], dim=1) # (Batch, 128, H, W)
        
        # 5. Output the Spotlight Mask!
        mask = self.dec_conv(decoder_input) # Shape: (Batch, 1, H, W)
        
        return mask

    def forward(self, h_s_t, r_s_t, h_s_t_1, h_a_t):
        # 1. Get Latent variables
        mu, logvar = self.encode(h_s_t, r_s_t, h_s_t_1, h_a_t)
        z = self.reparameterize(mu, logvar)
        
        # 2. Predict the Spotlight Mask
        mask = self.decode(z, r_s_t, h_s_t_1, h_a_t)
        
        # 3. THE SPOTLIGHT MATH (Deduce the new state)
        # Current vision is grounded in the Robot's perfect view
        visible_now = mask * r_s_t
        # Past memory is kept where the mask is 0
        remembered_past = (1.0 - mask) * h_s_t_1
        
        # Combine for the final prediction!
        h_s_t_pred = visible_now + remembered_past
        
        # Return the prediction, latent vars (for KL loss), and the mask (for visualization)
        return h_s_t_pred, mu, logvar, mask