"""
It crawls through a folder of JSON log files produced by your Logger, 
extracts per-timestep data (positions, orientations, held objects, actions, kitchen object states, 
and the flattened obs vector), and writes everything into a clean CSV 
and pickle DataFrame (.pkl) under data/fov_traj/.

cd /Users/mishafu/Desktop/steakhouse/Steakhouse-AI
/Users/mishafu/miniconda3/envs/steakhouse-ai/bin/python src/scripts/combine_json.py


That will crawl /Users/mishafu/Desktop/steakhouse/Steakhouse-AI/user_study/log/**/**/*.json (including your .../1/90.json) 
and write the outputs to:
src/my_methods/data/fov_traj/log.csv
src/my_methods/data/fov_traj/log.pkl

Make sure to specify whether it is T/F bayes to include the bayes in the file name and the extra columns like estimated_fov in the final csv
"""


import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import json
import pickle
import pandas as pd
from utils import flatten_obs_data
DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../my_methods/data/fov_traj')
import numpy as np


#IMPORTANT MANUAL TOGGLE
bayes = False

#new Misha Edit
import uuid
os.makedirs(DATA_FOLDER, exist_ok=True)
RUN_ID = uuid.uuid4().hex[:8]
#end new Misha Edit


# Path to the folder containing the log subfolders
folder_name = 'log'
base_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'user_study', folder_name)
base_folder = os.path.abspath(base_folder)

# Initialize an empty list to hold all the data
all_data = []
ep = 0


# Walk through each subfolder and process each JSON file
for subdir, dirs, files in os.walk(base_folder):
    #print('1HUIHSDOFIHDSLFJLDSKFJLKSDFJ')
    print(f"[debug] walking: {subdir}")
    print(f"[debug] dirs={dirs}, files={files}")

    for file in files:
        #print('2HUIHSDOFIHDSLFJLDSKFJLKSDFJ')

        if file.endswith('.json'):
            #print('3HUIHSDOFIHDSLFJLDSKFJLKSDFJ')

            file_path = os.path.join(subdir, file)
            
            # Open and read the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
                #print('4HUIHSDOFIHDSLFJLDSKFJLKSDFJ')

                # Process the data in the same way as described earlier
                
                
                # --- Handle participant_id and episodes key safely ---
                pid = data.get('participant_id')
                episodes = data.get('episode') or data.get('episodes') or []

                if not episodes:
                    print(f"[skip] {file_path}: no episodes found.")
                    continue

                # Optional debug print:
                if pid:
                    print(f"[read] {file_path}: participant_id={pid}, {len(episodes)} frames")
                # --------------------------------------------------------
                
                
                #if episodes[-1]['done']: #Misha edit
                #misha edit
                if not episodes:
                    continue
                
                episode_id = os.path.basename(os.path.dirname(file_path))
                if episodes:
                #end Misha edit
                
                    #loop through each episode in the data and build the data 
                    for episode in episodes:
                        #print('5HUIHSDOFIHDSLFJLDSKFJLKSDFJ')
                        row = {
                            'episode': str(ep),
                            'timestep': episode['timestep'],
                            'p0_position': episode['p0_position'],
                            'p0_orientation': episode['p0_orientation'],
                            'p0_held_object': episode['p0_held_object'],
                            'p0_action': episode['p0_action'],
                            'p1_position': episode['p1_position'],
                            'p1_orientation': episode['p1_orientation'],
                            'p1_held_object': episode['p1_held_object'],
                            'p1_action': episode['p1_action'],
                            'chopping_board_state': episode['chopping_board_state'],
                            'grill_states_state': episode['grill_states_state'],
                            'sink_state': episode['sink_state'],
                            'pot_states_state': episode['pot_states_state'],
                            'obs': flatten_obs_data(episode),
                            
                            #new edit
                            'partial_obs_3d': np.array(episode.get('partial_obs_3d'), dtype=np.float32),
                            'full_obs_3d': np.array(episode.get('full_obs_3d'), dtype=np.float32),

                            
                            #new Misha edit
                            'p0_subtask': episode.get('p0_subtask'),
                            
                            #'p1_subtask': episode.get('p1_subtask'),
                            # 'p0_intent_id': episode.get('p0_intent_id'),
                            # 'p1_intent_id': episode.get('p1_intent_id'),
                            #new Misha edit
                            


                            
                        }
                        
                        if bayes:
                            row.update({
                            'estimated_subtask': episode['estimated_subtask'],
                            'estimated_fov': episode['estimated_fov'],
                            'bayes_prob_90': episode['bayes_prob_90'],
                            'bayes_prob_120': episode['bayes_prob_120'],
                            'bayes_prob_179': episode['bayes_prob_179']
                        })
                        
                                
                        all_data.append(row)

                ep += 1

# Convert the collected data into a Pandas DataFrame
df = pd.DataFrame(all_data)





#new Misha edit to filter redundency
# We filter per-episode, keeping only "meaningful" state changes.
# Signature ignores orientation to be more aggressive (keeps net movement / object & env changes).
# If you want to consider orientation as meaningful, add it into _sig().

def _sig(row):
    # Build a compact, hashable signature of the *state* we care about.
    # - positions (p0, p1)
    # - held objects (p0, p1)
    # - kitchen object states (chopping, grill, sink, pot)
    # - subtask (if present)
    p0_pos = tuple(row['p0_position']) if isinstance(row['p0_position'], (list, tuple)) else row['p0_position']
    p1_pos = tuple(row['p1_position']) if isinstance(row['p1_position'], (list, tuple)) else row['p1_position']
    return (
        p0_pos,
        row.get('p0_held_object') or '',
        p1_pos,
        row.get('p1_held_object') or '',
        str(row.get('chopping_board_state')),
        str(row.get('grill_states_state')),
        str(row.get('sink_state')),
        str(row.get('pot_states_state')),
        row.get('p0_subtask')  # keep if present; remove if you don't care
    )

def _reduce_group(g):
    # Assumes 'timestep' is increasing; ensure sort just in case
    g = g.sort_values('timestep')
    kept_idx = []
    sigs = []  # signatures of kept rows in order

    # iterate rows; drop if identical to last (exact repeat)
    # or equal to second-last (2-state ABAB loop)
    for idx, row in g.iterrows():
        s = _sig(row)
        if len(sigs) >= 1 and s == sigs[-1]:
            continue  # consecutive duplicate
        if len(sigs) >= 2 and s == sigs[-2]:
            continue  # A,B,A -> drop this A to collapse ABAB oscillations
        sigs.append(s)
        kept_idx.append(idx)

    return g.loc[kept_idx]

_before = len(df)

print(f"[debug] collected {len(all_data)} rows")
print(f"[debug] df columns: {list(df.columns)}")
print(df.head())

df = df.groupby('episode', group_keys=False).apply(_reduce_group).reset_index(drop=True)
_after = len(df)
print(f"[dedupe] removed {_before - _after} / {_before} rows ({((_before - _after)/_before)*100:.1f}%)", flush=True)
#end new Misha edit to filter redundency






# Flatten the position and orientation columns into separate x, y columns
df[['p0_position_x', 'p0_position_y']] = pd.DataFrame(df['p0_position'].tolist(), index=df.index)
df[['p0_orientation_x', 'p0_orientation_y']] = pd.DataFrame(df['p0_orientation'].tolist(), index=df.index)
df[['p1_position_x', 'p1_position_y']] = pd.DataFrame(df['p1_position'].tolist(), index=df.index)
df[['p1_orientation_x', 'p1_orientation_y']] = pd.DataFrame(df['p1_orientation'].tolist(), index=df.index)

# Drop the original position and orientation list columns
df.drop(columns=['p0_position', 'p0_orientation', 'p1_position', 'p1_orientation'], inplace=True)

# Save the DataFrame to a CSV file

# Modify save directory and filename if Bayes mode is on
if bayes:
    save_folder = os.path.join(DATA_FOLDER, 'bayes')
else:
    save_folder = DATA_FOLDER

os.makedirs(save_folder, exist_ok=True)

output_csv_path = os.path.join(save_folder, f"{folder_name}_{RUN_ID}.csv")
output_pkl_path = os.path.join(save_folder, f"{folder_name}_{RUN_ID}.pkl")


df.to_csv(output_csv_path, index=False)
with open(output_pkl_path, 'wb') as f:
    pickle.dump(df, f)

print(f"Data successfully saved to {output_csv_path}")
