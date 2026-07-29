import torch
import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_path="expert_trajectories_multi_layout.npz", sequence_length=10):
        # Load the saved trajectories
        data = np.load(trajectory_path, allow_pickle=True)
        self.metadata = data['metadata'].item()
        self.sequence_length = sequence_length
        
        # Initialize lists to store sequences
        self.sequences = []
        self.times = []
        self.ml_actions = []
        self.next_sequences = []
        self.next_times = []
        self.next_ml_actions = []
        
        # Get all layout names
        self.layout_names = self.metadata['layout_names']
        
        # Process each trajectory
        for key in data.files:
            if key.startswith(tuple(layout_name + "_trajectory_" for layout_name in self.layout_names)):
                traj = data[key].item()
                states = traj['states']
                times = traj['timesteps']
                human_ml_actions = traj['human_ml_actions']

                # Append initial states
                state_history = np.array([states[0]]*sequence_length)
                human_ml_action_history = np.array([human_ml_actions[0]]*sequence_length)
                next_state_history = np.array(states[sequence_length:sequence_length*2])
                next_human_ml_action_history = np.array(human_ml_actions[sequence_length:sequence_length*2])
                
                for i in range(sequence_length):
                    self.sequences.append(state_history)
                    self.times.append(i)
                    self.ml_actions.append(human_ml_action_history)

                    self.next_sequences.append(next_state_history)
                    self.next_times.append(sequence_length+i)
                    self.next_ml_actions.append(next_human_ml_action_history)

                    state_history = np.concatenate((state_history[1:], states[i+1].reshape(1, -1)))
                    human_ml_action_history = np.concatenate((human_ml_action_history[1:], np.array([human_ml_actions[i+1]])))
                    next_state_history = np.concatenate((next_state_history[1:], states[sequence_length+i+1].reshape(1, -1)))
                    next_human_ml_action_history = np.concatenate((next_human_ml_action_history[1:], np.array([human_ml_actions[sequence_length+i+1]])))

                # Create sequences of consecutive states and their next states 
                for i in range(len(states) - sequence_length*2 + 1):
                    self.sequences.append(states[i:i + sequence_length])
                    self.times.append(times[i + sequence_length - 1])  # Use the last timestamp of sequence
                    self.ml_actions.append(human_ml_actions[i:i + sequence_length])

                    self.next_sequences.append(states[i + sequence_length:i + sequence_length*2])
                    self.next_times.append(times[i + sequence_length*2-1])
                    self.next_ml_actions.append(human_ml_actions[i + sequence_length:i + sequence_length*2])
        
        # Convert to tensors
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.times = torch.FloatTensor(np.array(self.times)) / self.metadata['max_steps']
        self.ml_actions = torch.FloatTensor(np.array(self.ml_actions))

        self.next_sequences = torch.FloatTensor(np.array(self.next_sequences))
        self.next_times = torch.FloatTensor(np.array(self.next_times)) / self.metadata['max_steps']
        self.next_ml_actions = torch.FloatTensor(np.array(self.next_ml_actions))

        # Set dimensions
        self.num_samples = len(self.sequences)
        self.input_dim = self.sequences.shape[2]  # shape: [num_samples, sequence_length, feature_dim]
        self.latent_dim = 32

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.times[idx], self.ml_actions[idx], self.next_sequences[idx], self.next_times[idx], self.next_ml_actions[idx]
