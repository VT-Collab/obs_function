"""
Pure functions file for bayesian inference
"""

import os, pickle, sys
import numpy as np
import pandas as pd
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


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


#change to KB=NN similarity score calculation with all the current data instead of pure matching
#so there are two options. cosine_similarity or euclidean_distances
#TODO: add action in header. this one checks without requiring action match
def process_bayes_knn(obs_world_vec, collected_data, prior_len, k=10, use_cosine=True, eps=1e-6):
    
    obs_world_vec = np.array(obs_world_vec).reshape(1, -1) # (1,d) aka turn column vector into row vector
    
    #calculate its similarity as compared to each observation individually
    similarities = []
    #loop through each recorded obs state
    for obs_vec in collected_data['obs_vec']:
        #flatten vector
        obs_vec = np.array(obs_vec).reshape(1, -1)
        #sim = cosine_similarity(obs_world_vec, obs_vec)
        dist = euclidean_distances(obs_world_vec, obs_vec) #change to cosine similarity if needed
        sim = 1 / (dist + eps)
        similarities.append(sim.item())
        
    # DEBUG: Check similarities
    # new_similarities = np.array(similarities)
    # print(f"Similarities stats: min={new_similarities.min()}, max={new_similarities.max()}, mean={new_similarities.mean()}")
    # print(f"Num similarities: {len(new_similarities)}")
    
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

