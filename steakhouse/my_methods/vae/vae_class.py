"""


=============================================================================
   HUMAN BELIEF CONDITIONAL VARIATIONAL AUTOENCODER (CVAE)
=============================================================================
Goal: Forward dynamics model. Predicts what the human sees *now*, given 
      the world state, what they saw *before*, and the action they just took.

--- INPUTS ---
[Target]  h_s_t   : Human's partial view NOW (The Ground Truth / Answer Key)
[Cond 1]  r_s_t   : Robot's full view NOW (World State)
[Cond 2]  h_s_t_1 : Human's partial view 1 TIMESTEP AGO (Past State)
[Cond 3]  h_a_t   : Human's action JUST TAKEN (Integer 0-5)

--- ARCHITECTURE FLOW ---

PART 1: THE ENCODER (Training Only)
[Compresses Target + Context into a "concept" distribution]

  -> Action (h_a_t)   => Embeds integer into a dense 16-dim vector.
  -> Grids (All 3)    => Stacks h_s_t, r_s_t, and h_s_t_1 along layers.
  -> CNN Compressor   => Shrinks spatial size, extracts visual features.
  -> Flatten & Merge  => Flattens grid => concatenates with Action vector.
  -> Output Dist.     => Linear layers output Mean (mu) & Variance (logvar).


PART 2: THE BOTTLENECK (Reparameterization Trick)
[Forces model to learn a continuous, generalizable space]

  -> Sample 'z'       => Randomly draws latent vector 'z' from N(mu, logvar).


PART 3: THE DECODER (Training & Inference)
[Rebuilds the Target using only the Concept 'z' + Context]

  -> Context Grids    => Mini-CNN processes r_s_t and h_s_t_1 independently.
  -> The Big Merge    => Glues together: [Concept 'z'] + [Action] + [Context].
  -> Expand & Reshape => Linear layer blows it up => folds back into 3D shape.
  -> Deconvolution    => Upsamples (ConvTranspose2d) back to exact kitchen size.


--- OUTPUT ---
Predictions (h_s_t_pred): Outputs real, continuous numbers. 
                          We use MSELoss during training because the grid 
                          contains a mix of binary flags (0/1) and continuous 
                          timers (e.g., up to 29).
=============================================================================
"""

import torch
import torch.nn as nn

class HumanBeliefCVAE(nn.Module):
    def __init__(self, sample_obs, latent_dim=64, action_emb_dim=16):
        """
        sample_obs: A single 3D tensor from your dataset (e.g., dataset[0]['X']).
                    The class will dynamically extract layers, Height, and Width.
                    
        latent_dim: model compresses everything down to 64 numbers; the bottleneck
        action_emb_dim: actions get represented as (streched to) 16 numbers. 
        
        """
        super().__init__()
        
        # 1. Dynamically extract the exact dimensions from the sample!
        self.layers, self.height, self.width = sample_obs.shape
        print(f"VAE dynamically configured for kitchen shape: C={self.layers}, H={self.height}, W={self.width}")
        
        self.latent_dim = latent_dim
        
        # 2. Action Embedding (Locked to 6 discrete actions)
        #setting up turning the 6 actions which were single numbers to 6 embeddings of size 16;
        #filled with random numbers rn
        self.action_emb = nn.Embedding(num_embeddings=6, embedding_dim=action_emb_dim)
        
        # ====================
        # ENCODER - Training only
        # Inputs: Target (C) + Robot (C) + Past (C) = 3 * C layers
        # Sequential is a container that runs layers one after another in order.
        # nn.Conv2d is a convolutional layer that slides a filter/magnifying class across the grid to detect patterrns
        # ====================
        self.enc_conv = nn.Sequential(
            #takes in #layers x 3 of say 10x10 layout, so if 39 layers its 3 times 39 = 117 (117x10x10), output 16x 10x10, 
            # with filter size of 3x3, stride 1 pixel at a time, and padding border of 0's 
            # so with the filter its like a cookie cutter that cuts down the piece 3x3 down ALL the 117 layers and spits out one single number that is one pixel in the output image
            # there are 16 different cookie cutters
            nn.Conv2d(self.layers * 3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #from 16x 10x10 to 32x 5x5 (b/c stride of 2 means that we skip to every other pixel)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            #It randomly "turns off" 20% of the 32 layers during practice.
            nn.Dropout2d(p=0.2)  # <--- new added misha
        )
        
        # ====================
        # DECODER CONDITIONS - Training and inference
        # Inputs: Robot (C) + Past (C) = 2 * C layers
        # Nearly identical to encoder but it processes only 2 grids
        # ====================
        self.dec_cond_conv = nn.Sequential(
            #takes in only 2 39 layers b/c this also supports inference
            #spits out 16 feature maps so reduce layer
            #2*39 x 10x10 -> 16x10x10 (16 cookie cutter filters)
            nn.Conv2d(self.layers * 2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            #16x10x10 -> 32 x 5x5
            #each filter (16 of them) slides across full image and recreates the layout
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # --- NO GUESSWORK MATH ---
        #After a CNN processes data, it shrinks the spatial dimensions (because of stride=2). 
        # But calculating the exact output size mathematically is tedious and error-prone. 
        
        # Pass a fake tensor of zeroes through the CNN to dynamically figure out 
        # the flattened size needed for the Linear layers.
        
        #create the fake all-0 tensor 
        dummy_input = torch.zeros(1, self.layers * 3, self.height, self.width)
        #pass the fake tensor through the encoder cnn
        conv_out = self.enc_conv(dummy_input)
        
        #get the shape that comes out. 
        #conv_out.shape gives the shape of the output. 
        #[1:] means "skip the first dimension (batch size)" and save just (channels, height, width)
        self.conv_out_shape = conv_out.shape[1:] # (layers, H, W) after convolutions
        
        #Then flat_size = channels × height × width — the total number of numbers when you flatten the 3D output into a 1D list
        self.flat_size = self.conv_out_shape[0] * self.conv_out_shape[1] * self.conv_out_shape[2]
        # -------------------------

        # Encoder Linear Layers
        #nn.Linear is the simplest type of neural network layer — it takes N numbers in, outputs M numbers out, using a learned weight matrix. "fc" stands for "fully connected" — every input connects to every output.
        #squash vae output and action embedding down to just 64 numbers (or whatever latent_dim is)
        self.enc_fc_mu = nn.Linear(self.flat_size + action_emb_dim, latent_dim)
        self.enc_fc_logvar = nn.Linear(self.flat_size + action_emb_dim, latent_dim)
        
        # Decoder Linear Layer
        #The input is everything glued together: latent vector z (64 numbers) + action embedding (16 numbers) + CNN context features (flat_size numbers). 
        #The output is flat_size — the same size as one full map - channels × height × width. 
        #We are about to reshape back into a 3D grid
        self.dec_fc = nn.Linear(latent_dim + action_emb_dim + self.flat_size, self.flat_size)
        
        # ====================
        # DECODER RECONSTRUCTION
        # ====================
        #Rebuilding the grid
        #nn.ConvTranspose2d is the reverse of Conv2d. Where Conv2d shrinks spatial dimensions, ConvTranspose2d grows them
        #stride = 2 means we double the spacial size
        self.dec_deconv = nn.Sequential(
            #takes 32x5x5 and outputs 16x10x10 (stride = 2 here means you expand to twice the size)
            nn.ConvTranspose2d(self.conv_out_shape[0], 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            #from 16x10x10 to 39x10x10
            nn.Conv2d(16, self.layers, kernel_size=3, stride=1, padding=1),
            # Output is NO LONGER real continuous values for MSE Loss 
            #now it is binary cross entropy
            #sigmoid squshes everything to between 0 and 1 on the 39x10x10
            nn.Sigmoid()
            
        )

    #thing you call to run the encoder; including the 4 inputs that are passed when you call it
    def encode(self, h_s_t, r_s_t, h_s_t_1, h_a_t):
        
        #embed the action integer into a dense embedding vector using the lookup table we defined earlier
        a_emb = self.action_emb(h_a_t) 
        
        # Stack inputs along the channel dimension; sequentially so 3 things is 3*39 layers
        x = torch.cat([h_s_t, r_s_t, h_s_t_1], dim=1) 
        
        #run the stacked grid through the encoder network
        x = self.enc_conv(x)
        
        #flattens the 3d output into a 1d vector per batch item
        #x.size(0) = the batch size (how many examples at once) and -1 means figure this dimension automatically ( height×width×channels)
        x = x.view(x.size(0), -1) 
        
        #combine the cnn output (spacial features) with action embedding 
        combined = torch.cat([x, a_emb], dim=1)
        #calculate the mean and the log variance; passing in the combination
        mu = self.enc_fc_mu(combined)
        logvar = self.enc_fc_logvar(combined)
        return mu, logvar

    #sampling from distribution
    def reparameterize(self, mu, logvar):
        #convers logvar into std 
        std = torch.exp(0.5 * logvar)
        
        #randn_like means "same shape as std, but filled with random numbers
        #this is the noice
        eps = torch.randn_like(std)
        
        #returns a random sample from distribution
        return mu + eps * std

    #takes in latent, robot view, past human view, action
    def decode(self, z, r_s_t, h_s_t_1, h_a_t):
        
        #embed the action integer into a dense embedding vector using the lookup table we defined earlier
        a_emb = self.action_emb(h_a_t)
        
        #stack both robot view and partial view from last timestep into one grid
        cond = torch.cat([r_s_t, h_s_t_1], dim=1)
        
        #runs the stacked grid through the decoder's own CNN
        cond_features = self.dec_cond_conv(cond)
        #flatten the output into a 1D vector
        cond_features = cond_features.view(cond_features.size(0), -1)
        
        #combine everything
        combined = torch.cat([z, a_emb, cond_features], dim=1)
        #self.dec_fc(combined) runs this through the decoder's linear layer, outputting flat_size numbers
        #now we have a flat vector the same size as the CNN's internal representation — ready to be reshaped back into a 3D grid.
        x = self.dec_fc(combined)
        
        # Reshape back to 3D for the Deconvolution
        x = x.view(x.size(0), *self.conv_out_shape)
        #runs through deconvolution layers; unsampling layer by layer
        h_s_t_pred = self.dec_deconv(x)
        
        # Force exact match to original height/width
        #if the shape doesnt match, , the output might be 1 pixel too big. If so, slice it down: [:, :, :self.height, :self.width] takes only the exact rows and columns we need. 
        if h_s_t_pred.shape[2:] != (self.height, self.width):
            h_s_t_pred = h_s_t_pred[:, :, :self.height, :self.width]
            
        return h_s_t_pred

    #takes everything into account
    def forward(self, h_s_t, r_s_t, h_s_t_1, h_a_t):
        #runs the encoder; gets mu and logvar
        mu, logvar = self.encode(h_s_t, r_s_t, h_s_t_1, h_a_t)
        #samples from distribution
        z = self.reparameterize(mu, logvar)
        #decode with z + context
        h_s_t_pred = self.decode(z, r_s_t, h_s_t_1, h_a_t)
        
        #loss function during training needs mu and logvar to compute KL divergence too
        #KL divergence meansures against standard mu and logvar
        return h_s_t_pred, mu, logvar