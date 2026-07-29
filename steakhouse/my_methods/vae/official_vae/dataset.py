
"""
1. snipe the data so the no needed layers are removed
    we add them manually back later
2. return the data needed to train (we are only predicting the observation mask itself for VAE not full image reconstruction)
    how we do this: 
    Predicted_Belief = (mask * r_s_t) + ((1 - mask) * h_s_t_1)
    So at each timestep we only predict the masked spots (notice that although only stay in view 3 timesteps belief updates in a mask that is just first 2 timesteps it is not in spotlight yet)
3. new visualization = masked map at each timestep
4. we pad to a max size of 15x15 for any layout, but loss is only conditioned on the actual map
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SteakhouseVAEDataset(Dataset):
    def __init__(self, pkl_paths, human_player_idx=0):
        """
        Loads the Overcooked/Steakhouse pickle file and pairs up 
        timesteps t-1 and t to train a forward dynamics model of human belief.
        
        human_player_idx: 0 if p0 is the human (LimitVision), 1 if p1 is the robot.
        """
        super().__init__()
        
        self.human_idx = human_player_idx

        # 1. Load the massive dataframe(s)
        if isinstance(pkl_paths, str):
            pkl_paths = [pkl_paths]

        dfs = []
        
        for file_idx, path in enumerate(pkl_paths): # <-- NEW: Using enumerate to get a unique file index
            print(f"Loading data from {path}...")
            temp_df = pd.read_pickle(path)
            
            # 👇 NEW: Prepend a unique file prefix to the string! 👇
            # If episode was "1", it now becomes "file_0_1", "file_1_1", etc.
            if 'episode' in temp_df.columns:
                temp_df['episode'] = f"file_{file_idx}_" + temp_df['episode'].astype(str)
            
            dfs.append(temp_df)
            
        # Glue them all together!
        df = pd.concat(dfs, ignore_index=True)
        
        # Ensure it's sorted just in case
        df = df.sort_values(by=['episode', 'timestep']).reset_index(drop=True)
        
        
        self.data_pairs = []
        # 2. Group by episode so we never pair t-1 from Episode 1 with t from Episode 2
        print("Pairing up timesteps...")
        grouped = df.groupby('episode')
        
        for ep_id, group in grouped:
            
            # If an episode is too short to have a t and t-1, skip it
            if len(group) < 2:
                continue
            
            # Convert group to a list of dicts for faster indexing
            records = group.to_dict('records')
            
            # Slide a window of size 2 across the episode
            for i in range(1, len(records)):
                row_t_minus_1 = records[i - 1]
                row_t = records[i]
                self.data_pairs.append((row_t_minus_1, row_t))
                
        print(f"Dataset ready! Total paired transitions: {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """
        Needed layers: 24 total (Processed by VAE)
            - player_features: 0-9 (10 layers)
            - variable_map_features: 20-33 (14 layers, including 4 timers)
        
        Not needed layers: 14 total (Bypassed, added at end)
            - base_map_features: 10-19 (10 layers)
            - dish_order_type: 35-38 (4 layers)
            
        Dropped completely: 1 total
            - urgency_feature: 34 (1 layer)
        """
        row_t_1, row_t = self.data_pairs[idx]
        
        # --- 1. Load arrays (Shape: Width, Height, Channels) ---
        # Target (X): Human's partial observation at time t (h_s_t)
        X = np.array(row_t['partial_obs_3d'], dtype=np.float32)
        # Input 1 (r_s_t): Robot's observation / Full world state at time t
        r_s_t = np.array(row_t['full_obs_3d'], dtype=np.float32)
        # Input 2 (h_s_t_1): Human's partial observation at time t-1
        h_s_t_1 = np.array(row_t_1['partial_obs_3d'], dtype=np.float32)
        
        # --- 2. NORMALIZE CONTINUOUS DATA AND CLIP ---
        timer_indices = [26, 28, 30, 32] # Added 26!
        
        for grid in [X, r_s_t, h_s_t_1]:
            # Normalize timers
            # TODO: Add the correct max tick value for chicken cook time (layer 26)
            grid[:, :, 26] /= 1.0  # <--- REPLACE 1.0 WITH YOUR MAX CHICKEN TIME
            grid[:, :, 28] /= 29.0
            grid[:, :, 30] /= 3.0
            grid[:, :, 32] /= 2.0
            
            # Binary safety clip
            all_indices = np.arange(grid.shape[2])
            binary_indices = np.setdiff1d(all_indices, timer_indices)
            grid[:, :, binary_indices] = np.clip(grid[:, :, binary_indices], 0, 1)
        
        # --- 3. TRANSPOSE TO (Channels, Height, Width) ---
        # MUST happen before we slice by layer index!
        X = X.transpose(2, 0, 1)
        r_s_t = r_s_t.transpose(2, 0, 1)
        h_s_t_1 = h_s_t_1.transpose(2, 0, 1)

        # --- 4. THE 15x15 ZERO-PADDING HACK ---
        # We grab the current actual dimensions of the map (e.g. 5x5, 7x7)
        _, current_h, current_w = X.shape
        MAX_SIZE = 15

        def pad_grid(grid_array):
            # Creates a blank canvas of Zeros
            padded = np.zeros((grid_array.shape[0], MAX_SIZE, MAX_SIZE), dtype=np.float32)
            # Pastes the real map in the top-left corner
            padded[:, :current_h, :current_w] = grid_array
            return padded

        X = pad_grid(X)
        r_s_t = pad_grid(r_s_t)
        h_s_t_1 = pad_grid(h_s_t_1)

        # Create the Padding Mask (1 = Real Game, 0 = Void)
        pad_mask = np.zeros((1, MAX_SIZE, MAX_SIZE), dtype=np.float32)
        pad_mask[0, :current_h, :current_w] = 1.0

        # --- 5. CREATE INDICES FOR NEEDED/NOT NEEDED DATA ---
        
        # Later During your training step:
        # weights[target != 0] = 50.0  # Scale up the importance of the non-zero data
        given_idx = list(range(10, 20)) + list(range(35, 39))
        
        predict_cont_idx = [26, 28, 30, 32] 
        predict_bin_idx = list(range(0, 10)) + [i for i in range(20, 34) if i not in predict_cont_idx] 
        
        # all valid indices
        vae_input_idx = sorted(predict_bin_idx + predict_cont_idx)

        # --- 6. SLICE THE DATA ---
        # The 14 bypassed layers (we can glue these back on in the VAE later)
        given_layers = r_s_t[given_idx, :, :]
        
        # Filter ALL grids to just the 24 dynamic layers
        r_s_t_vae = r_s_t[vae_input_idx, :, :]
        h_s_t_1_vae = h_s_t_1[vae_input_idx, :, :]
        h_s_t_vae = X[vae_input_idx, :, :] # <--- NEW: Keep the Answer Key!
        
        # --- 7. ACTION ---
        h_action_col = f'p{self.human_idx}_action'
        h_a_t = row_t[h_action_col]
        
        # --- 8. EPISODE & TIMESTEP TRACKING ---
        episode_id = row_t['episode']
        timestep = row_t['timestep']

        return {
            # --- INPUTS ---
            'r_s_t': torch.tensor(r_s_t_vae),
            'h_s_t_1': torch.tensor(h_s_t_1_vae),
            'h_a_t': torch.tensor(h_a_t, dtype=torch.long),
            
            # --- TARGET (The Answer Key) ---
            'h_s_t': torch.tensor(h_s_t_vae), 
            
            # --- BYPASSED STATIC DATA ---
            'given_layers': torch.tensor(given_layers),
            
            # --- PADDING MASK (For Void-Aware Loss) ---
            'pad_mask': torch.tensor(pad_mask),
            
            # --- AUTOREGRESSIVE TRACKERS ---
            'episode': episode_id,
            'timestep': timestep
        }