"""
small_vae_class.py
"""
import torch
import torch.nn as nn

class HumanBeliefCVAE(nn.Module):
    def __init__(self, sample_obs, latent_dim=64, action_emb_dim=16):
        """
        sample_obs: A single 3D tensor from your dataset.
        Dynamically extracts Channels (21), Height (5), and Width (5).
        """
        super().__init__()
        
        self.channels, self.height, self.width = sample_obs.shape
        print(f"VAE dynamically configured for pruned shape: C={self.channels}, H={self.height}, W={self.width}")
        
        self.latent_dim = latent_dim
        
        # Action Embedding (Locked to 6 discrete actions)
        self.action_emb = nn.Embedding(num_embeddings=6, embedding_dim=action_emb_dim)
        
        # ====================
        # ENCODER (Target + Robot + Past = 3 * C channels)
        # ====================
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.channels * 3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.2)
        )
        
        # ====================
        # DECODER CONDITIONS (Robot + Past = 2 * C channels)
        # ====================
        self.dec_cond_conv = nn.Sequential(
            nn.Conv2d(self.channels * 2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Dynamically calculate flattened size
        dummy_input = torch.zeros(1, self.channels * 3, self.height, self.width)
        conv_out = self.enc_conv(dummy_input)
        self.conv_out_shape = conv_out.shape[1:] 
        self.flat_size = self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]

        # Encoder Linear Layers
        self.enc_fc_mu = nn.Linear(self.flat_size + action_emb_dim, latent_dim)
        self.enc_fc_logvar = nn.Linear(self.flat_size + action_emb_dim, latent_dim)
        
        # Decoder Linear Layer
        self.dec_fc = nn.Linear(latent_dim + action_emb_dim + self.flat_size, self.flat_size)
        
        # ====================
        # DECODER RECONSTRUCTION
        # ====================
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(self.conv_out_shape[0], 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # <-- Vital for BCE Loss!
        )

    def encode(self, h_s_t, r_s_t, h_s_t_1, h_a_t):
        a_emb = self.action_emb(h_a_t) 
        
        x = torch.cat([h_s_t, r_s_t, h_s_t_1], dim=1) 
        x = self.enc_conv(x)
        x = x.view(x.size(0), -1) 
        
        combined = torch.cat([x, a_emb], dim=1)
        mu = self.enc_fc_mu(combined)
        logvar = self.enc_fc_logvar(combined)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, r_s_t, h_s_t_1, h_a_t):
        a_emb = self.action_emb(h_a_t)
        
        cond = torch.cat([r_s_t, h_s_t_1], dim=1)
        cond_features = self.dec_cond_conv(cond)
        cond_features = cond_features.view(cond_features.size(0), -1)
        
        combined = torch.cat([z, a_emb, cond_features], dim=1)
        x = self.dec_fc(combined)
        
        x = x.view(x.size(0), *self.conv_out_shape)
        h_s_t_pred = self.dec_deconv(x)
        
        # Force exact match to original height/width (crops 6x6 down to 5x5)
        if h_s_t_pred.shape[2:] != (self.height, self.width):
            h_s_t_pred = h_s_t_pred[:, :, :self.height, :self.width]
            
        return h_s_t_pred

    def forward(self, h_s_t, r_s_t, h_s_t_1, h_a_t):
        mu, logvar = self.encode(h_s_t, r_s_t, h_s_t_1, h_a_t)
        z = self.reparameterize(mu, logvar)
        h_s_t_pred = self.decode(z, r_s_t, h_s_t_1, h_a_t)
        return h_s_t_pred, mu, logvar