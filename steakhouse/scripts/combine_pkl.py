#combines 3 fov pkls into one big one

import os
import pandas as pd

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../data/fov_traj')

# Load the two pickle files
file1 = 'fov-90.pkl'
file2 = 'fov-120.pkl'
file3 = 'fov-179.pkl'

df1 = pd.read_pickle(os.path.join(DATA_FOLDER, file1))
df2 = pd.read_pickle(os.path.join(DATA_FOLDER, file2))
df3 = pd.read_pickle(os.path.join(DATA_FOLDER, file3))

# Add a new column to indicate the source file
df1['fov'] = 0 # 90
df2['fov'] = 1 # 120
df3['fov'] = 2 # 179

# Combine the two DataFrames
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save the combined DataFrame to a new pickle file
combined_df.to_pickle(os.path.join(DATA_FOLDER, 'fov_traj_data.pkl'))

# Display the combined DataFrame
print(combined_df.head())
