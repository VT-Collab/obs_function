
"""
Dataset class to extract partial and complete obs information for eventual training.
Build tensors from the partial and complete obs at each timestep
Outputs in a pair of tensors: t and t-1 (not complete history)

Robot = full_obs_3d (input)
human = partial_obs_3d (what we want to reconstruct later)

"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SteakhouseVAEDataset(Dataset):
    def __init__(self, pkl_path, human_player_idx=0):
        """
        Loads the Overcooked/Steakhouse pickle file and pairs up 
        timesteps t-1 and t to train a forward dynamics model of human belief.
        
        human_player_idx: 0 if p0 is the human (LimitVision), 1 if p1 is the robot.
        """
        super().__init__()
        self.human_idx = human_player_idx

        # 1. Load the massive dataframe
        print(f"Loading data from {pkl_path}...")
        df = pd.read_pickle(pkl_path)
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
        row_t_1, row_t = self.data_pairs[idx]
        
        # 1. Target (X): Human's partial observation at time t (h_s_t)
        X = np.array(row_t['partial_obs_3d'], dtype=np.float32)
        
        # 2. Input 1 (r_s_t): Robot's observation / Full world state at time t
        r_s_t = np.array(row_t['full_obs_3d'], dtype=np.float32)
        
        # 3. Input 2 (h_s_t_1): Human's partial observation at time t-1
        h_s_t_1 = np.array(row_t_1['partial_obs_3d'], dtype=np.float32)
        
        
        timer_indices = [28, 30, 32]
        for grid in [X, r_s_t, h_s_t_1]:
            grid[:, :, 28] /= 29.0
            grid[:, :, 30] /= 3.0
            grid[:, :, 32] /= 2.0
            
            # --- 2. NEW: BINARY SAFETY CLIP ---
            # Identify every layer that ISN'T a timer
            all_indices = np.arange(grid.shape[2])
            binary_indices = np.setdiff1d(all_indices, timer_indices)
            
            # Force everything else to be strictly 0 or 1
            # This fixes the "urgency" or any other stray values
            grid[:, :, binary_indices] = np.clip(grid[:, :, binary_indices], 0, 1)
        
        
        # 4. Input 3 (h_a_t): Human's action at time t
        h_action_col = f'p{self.human_idx}_action'
        h_a_t = row_t[h_action_col]
        
        # Note: If your 3D arrays are saved as (Width, Height, Channels), 
        # you will need to transpose them to (Channels, Height, Width) for PyTorch CNNs:
        X = X.transpose(2, 0, 1)
        r_s_t = r_s_t.transpose(2, 0, 1)
        h_s_t_1 = h_s_t_1.transpose(2, 0, 1)

        return {
            'X': torch.tensor(X),
            'r_s_t': torch.tensor(r_s_t),
            'h_s_t_1': torch.tensor(h_s_t_1),
            'h_a_t': torch.tensor(h_a_t, dtype=torch.long)
        }