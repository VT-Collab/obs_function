"""
small_vae_dataset.py
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SteakhouseVAEDataset(Dataset):
    def __init__(self, pkl_paths, human_player_idx=0):
        super().__init__()
        self.human_idx = human_player_idx
        
        # 1. Load multiple dataframes (for SS-1 and SS-2)
        if isinstance(pkl_paths, str):
            pkl_paths = [pkl_paths]
            
        dfs = []
        for path in pkl_paths:
            print(f"Loading data from {path}...")
            df = pd.read_pickle(path)
            dfs.append(df)
            
        massive_df = pd.concat(dfs, ignore_index=True)
        massive_df = massive_df.sort_values(by=['episode', 'timestep']).reset_index(drop=True)
        
        self.data_pairs = []
        grouped = massive_df.groupby('episode')
        
        for ep_id, group in grouped:
            if len(group) < 2: continue
            records = group.to_dict('records')
            for i in range(1, len(records)):
                self.data_pairs.append((records[i - 1], records[i]))
                
        print(f"Dataset ready! Total paired transitions: {len(self.data_pairs)}")

        # THE 21 ACTIVE LAYERS
        self.active_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 22, 23, 24, 28, 29, 30, 31, 32, 33, 37]
        # In this 21-layer array, the original 28, 30, 32 become indices 14, 16, 18
        self.new_timer_indices = [14, 16, 18]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        row_t_1, row_t = self.data_pairs[idx]
        
        # Original format is (H, W, C) -> (5, 5, 39)
        X = np.array(row_t['partial_obs_3d'], dtype=np.float32)
        r_s_t = np.array(row_t['full_obs_3d'], dtype=np.float32)
        h_s_t_1 = np.array(row_t_1['partial_obs_3d'], dtype=np.float32)
        
        # --- PRUNING THE DEAD WEIGHT ---
        X = X[:, :, self.active_layers]
        r_s_t = r_s_t[:, :, self.active_layers]
        h_s_t_1 = h_s_t_1[:, :, self.active_layers]
        
        for grid in [X, r_s_t, h_s_t_1]:
            # --- UPDATED NORMALIZATION BASED ON YOUR SCAN ---
            grid[:, :, 14] /= 9.0  # Original layer 28 maxes at 9.0
            grid[:, :, 16] /= 3.0  # Original layer 30 maxes at 3.0
            grid[:, :, 18] /= 2.0  # Original layer 32 maxes at 2.0
            
            # Binary Clip
            all_indices = np.arange(grid.shape[2])
            binary_indices = np.setdiff1d(all_indices, self.new_timer_indices)
            grid[:, :, binary_indices] = np.clip(grid[:, :, binary_indices], 0, 1)
        
        h_action_col = f'p{self.human_idx}_action'
        h_a_t = row_t[h_action_col]
        
        # Transpose to (Channels, Height, Width) for PyTorch -> (21, 5, 5)
        return {
            'X': torch.tensor(X.transpose(2, 0, 1)),
            'r_s_t': torch.tensor(r_s_t.transpose(2, 0, 1)),
            'h_s_t_1': torch.tensor(h_s_t_1.transpose(2, 0, 1)),
            'h_a_t': torch.tensor(h_a_t, dtype=torch.long)
        }