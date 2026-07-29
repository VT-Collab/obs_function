"""
Pure functions file for bayesian inference



    "Overcooked2_2-4"


    os.path.join(PKL_DIR, "log_90_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_90_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_90_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_1_1-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_1-2_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-4_dedup.pkl"),
    os.path.join(PKL_DIR, "log_90_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_120_2_2-5_dedup.pkl"),
    os.path.join(PKL_DIR, "log_179_2_2-5_dedup.pkl"),
"""

import os, pickle, sys
import numpy as np
import pandas as pd
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


# Add project root (one level up) so we can import your dataset helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Your project-specific vectorizer:
#   obs_to_list -> obs_list_to_1D_vec

from dataset import obs_to_list, obs_list_to_1D_vec
def obs_to_vec_default(obs: Any) -> np.ndarray:
    return np.asarray(obs_list_to_1D_vec(obs_to_list(obs)), dtype=float)


#select files with different layout
def select_diff_layout(layout, filepath):
    
    #grabs "2-4" or "2-5" or "1-2"
    layout_suffix = layout.split('_')[-1]
    
    filtered_files = []
    for file_path in filepath:
        
        # Get just the filename from the path
        filename = os.path.basename(file_path)
        
        # Check if the layout suffix is NOT in the filename
        if layout_suffix not in filename:
            filtered_files.append(file_path)
    
    return filtered_files


def loadPickle(filepath_list):
    all_data = []
    for filepath in filepath_list:
        file = open(filepath, 'rb')
        data = pickle.load(file)
        all_data.append(data)
        file.close()
    return pd.concat(all_data, ignore_index=True)


#returns #rows in 90, 120, 179 from a list of path of pkl files
#can make more general
def estimate_only_three_prior_counts(pkl_list):
    num90 = 0
    num120 = 0
    num179 = 0
    total = 0
    
    for path in pkl_list:
        
        file_name = os.path.basename(str(path))
        
        with open(path, "rb") as f:
            content = pickle.load(f)
            rows = content.shape[0] if content.ndim >= 1 else 1
            total += rows

        if "90" in file_name:
            num90 += rows
        elif "120" in file_name:
            num120 += rows
        elif "179" in file_name:
            num179 += rows
    return num90, num120, num179, total

#add new vector equivalent version of the obs in new column called obs_vec
def add_vec_to_combined_data(combined_data):
    df = combined_data[['fov', 'p0_action', 'obs']].copy()
    #add new column that is the vector equivalent
    df['obs_vec'] = df['obs'].apply(lambda obs: obs_list_to_1D_vec(obs_to_list(obs)))
    return df


#somehow this below version is making the fov 90 alwyas the highest probabilty, fov 120 alwys the lowest probablity, and fov 179 alwyas the middle probability
#posterior = bayes_inference(prior, evidence, likelihood)
def bayes_inference(prior, evidence, likelihood):
    
    #calculate posterior through soft update
    posterior = prior*0.8 + 0.2*(((likelihood*prior)/evidence)/np.sum((likelihood*prior)/evidence))

    #nomalize posterior
    posterior = posterior/np.sum(posterior)
    #make the really smaller number a constant numer
    posterior[posterior < 0.000001] = 0.000001
    return posterior



#evidence, likelihood = process_bayes(obs_vec, Action.ACTION_TO_INDEX[player_1_action], full_collected_data, num_fov_types)
#not used;use the knn one below
def process_bayes(obs_world_vec, curr_action, collected_data, prior_len, cosine):
    obs_world_vec = np.array(obs_world_vec)
    
    # Use DataFrame column access, not NumPy indexing
    obs_world_matches = collected_data['obs_vec'].apply(
        lambda x: np.array_equal(x, obs_world_vec)
    ).to_numpy()
    
    
    human_action_matches = (collected_data['p0_action'] == curr_action).to_numpy()

    evidence = np.sum(obs_world_matches & human_action_matches)
    
    # Initialize likelihood and action probability dictionaries
    ll = np.zeros((prior_len, 2))
    action_prob = np.zeros((prior_len, 2))

    # Calculate mask_id matches and update ll and action_prob in a vectorized manner
    mask_ids = collected_data['fov'].to_numpy().astype(int)


    for i in range(prior_len):
        mask_idx = mask_ids == i

        # Update ll: given human action and obs world
        ll[i, 0] = np.sum(mask_idx & obs_world_matches & human_action_matches)
        ll[i, 1] = np.sum(mask_idx)

        # Update action_prob: human action and obs world
        action_prob[i, 0] = np.sum(mask_idx & obs_world_matches & human_action_matches)
        action_prob[i, 1] = np.sum(mask_idx & obs_world_matches)

    # Likelihood and traj_prob calculations
    likelihood = np.where(ll[:, 1] > 0, ll[:, 0] / ll[:, 1], 0.000001)
    traj_prob = np.where(action_prob[:, 1] > 0, action_prob[:, 0] / action_prob[:, 1], 0.000001)

    # Normalize traj_prob
    traj_prob /= np.sum(traj_prob)

    return evidence, likelihood



#TODO: change to KB=NN similarity score calculation with all the current data instead of pure matching
#so there are two options. cosine_similarity or euclidean_distances

def process_bayes_knn(obs_world_vec, curr_action, collected_data, prior_len, k=10, use_cosine=True, eps=1e-6):
    
    obs_world_vec = np.array(obs_world_vec).reshape(1, -1) # (1,d) aka turn column vector into row vector
    
    #calculate its similarity as compared to each observation individually
    similarities = []
    #loop through each recorded obs state
    for obs_vec in collected_data['obs_vec']:
        #flatten vector
        obs_vec = np.array(obs_vec).reshape(1, -1)
        #sim = cosine_similarity(obs_world_vec, obs_vec)
        dist = euclidean_distances(obs_world_vec, obs_vec)
        sim = 1 / (dist + eps)
        similarities.append(sim.item())
        
    # DEBUG: Check similarities
    new_similarities = np.array(similarities)

    print(f"Similarities stats: min={new_similarities.min()}, max={new_similarities.max()}, mean={new_similarities.mean()}")
    print(f"Num similarities: {len(new_similarities)}")
    
    # human_action_matches = (collected_data['p0_action'] == curr_action).to_numpy()
    # print(f"Human action matches: {np.sum(human_action_matches)} out of {len(human_action_matches)}")

    #evidence = np.sum(similarities * human_action_matches)
    evidence = np.sum(similarities)

    print(f"Evidence: {evidence}")
    
    ll = np.zeros((prior_len, 2))
    action_prob = np.zeros((prior_len, 2))

    unique_fovs = sorted(collected_data['fov'].unique())
    mask_ids = collected_data['fov'].to_numpy().astype(int)
    print(f"Mask IDs: {np.unique(unique_fovs)}")

    for i, fov_val in enumerate(unique_fovs):
        mask_idx = (mask_ids == fov_val)
        print(f"FOV {i}: {np.sum(mask_idx)} samples")

        ll[i, 0] = np.sum(mask_idx * similarities)
        ll[i, 1] = np.sum(mask_idx)

        action_prob[i, 0] = np.sum(mask_idx * similarities)
        action_prob[i, 1] = np.sum(mask_idx * similarities)
        
        print(f"  ll[{i}] = {ll[i, 0]} / {ll[i, 1]}")
        print(f"  action_prob[{i}] = {action_prob[i, 0]} / {action_prob[i, 1]}")


    # no handle 0 Likelihood and traj_prob calculations
    # likelihood = np.where(ll[:, 1] > 0, ll[:, 0] / ll[:, 1], 0.000001)
    # traj_prob = np.where(action_prob[:, 1] > 0, action_prob[:, 0] / action_prob[:, 1], 0.000001)
    
    # Likelihood and traj_prob calculations - simpler fix
    with np.errstate(divide='ignore', invalid='ignore'):
        likelihood = ll[:, 0] / ll[:, 1]
        likelihood = np.where(np.isfinite(likelihood), likelihood, 0.000001)
        
        traj_prob = action_prob[:, 0] / action_prob[:, 1]
        traj_prob = np.where(np.isfinite(traj_prob), traj_prob, 0.000001)

    # Normalize traj_prob
    traj_prob /= np.sum(traj_prob)


    return evidence, likelihood

