import pickle
import numpy as np

def loadPickle(filepath):
    file = open(filepath, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def bayes_inference(prior, evidence, likelihood):
    
    posterior = prior*0.8 + 0.2*(((likelihood*prior)/evidence)/np.sum((likelihood*prior)/evidence))

    posterior = posterior/np.sum(posterior)
    posterior[posterior < 0.000001] = 0.000001
    return posterior

"""
    obs_world = specific observation querying
    P(fov|obs, human action)
    match the obs, human action with the observed data 
    prior_len = number of classes we are trying to predict. in this case, 3
    
    Bayes rule:
    P(fov|obs, human action) = 
            Nominator:
            Denominator: how many times we have seen this exact (obs + action)
            
            ll[i, 0] = how many times hypotehsis i occured WITH the observed (obs + action)
            ll[i , 1] = how many times hypothesis i occurred
"""
def process_bayes(obs_world, human_action, collected_data, prior_len):

    #Vectorized condition checks for (x[1] == obs_world).all() and x[2] == human_action
    
    obs_world_matches = collected_data[:, 2] == obs_world
    human_action_matches = collected_data[:, 1] == human_action
    # p(a|data) - evidence calculation
    evidence = np.sum(obs_world_matches & human_action_matches)

    # Initialize likelihood and action probability dictionaries
    ll = np.zeros((prior_len, 2))
    action_prob = np.zeros((prior_len, 2))

    # Calculate mask_id matches and update ll and action_prob in a vectorized manner
    mask_ids = collected_data[:, 0].astype(int)

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
