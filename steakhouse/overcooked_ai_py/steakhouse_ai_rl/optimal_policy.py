import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from rl.dataset import TrajectoryDataset
from rl.e_obs_d_subtask import Encoder, Decoder

class OptimalPolicy(nn.Module):
    def __init__(self, latent_dim=128):
        super(OptimalPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),    
            nn.ReLU(),
            nn.Linear(32, 64),    
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # Output: predicted next latent
        )

    def forward(self, z):
        return self.model(z)


# # Updated OptimalPolicy network using an LSTM.
# class OptimalPolicy(nn.Module):
#     def __init__(self, latent_dim=128, lstm_hidden_size=64, num_layers=1):
#         """
#         Args:
#             latent_dim: Dimension of the latent vector produced by the encoder.
#             lstm_hidden_size: Hidden size for the LSTM.
#             num_layers: Number of LSTM layers.
#         """
#         super(OptimalPolicy, self).__init__()
#         # The LSTM processes a sequence of latent vectors. Here, we expect a single-step sequence.
#         self.lstm = nn.LSTM(input_size=latent_dim,
#                             hidden_size=lstm_hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True)
#         # Map the LSTM output to the predicted next latent vector.
#         self.fc = nn.Linear(lstm_hidden_size, latent_dim)

#     def forward(self, z):
#         """
#         Args:
#             z: Tensor of shape (batch_size, latent_dim)
        
#         Returns:
#             pred: Tensor of shape (batch_size, latent_dim) representing the predicted next latent.
#         """
#         # Unsqueeze to create a sequence dimension: (B, 1, latent_dim)
#         z_seq = z.unsqueeze(1)
#         out, (hn, cn) = self.lstm(z_seq)  # out shape: (B, 1, lstm_hidden_size)
#         # Use the output of the final time step (only one time step here)
#         pred = self.fc(out[:, -1, :])
#         return pred

def train_optimal_policy(num_epochs=1000, batch_size=128, learning_rate=1e-4):
    # Load the training dataset
    trajectories_path = os.path.join(os.path.dirname(__file__), "../..", "expert_trajectories_layout.npz")
    train_dataset = TrajectoryDataset(trajectory_path=trajectories_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model hyperparameters.
    hidden_dim = 256
    latent_dim = 128
    ml_action_dim = 12
    sequence_length = 10

    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the models.
    encoder = Encoder(latent_dim=latent_dim).to(device) 
    decoder = Decoder(latent_dim, hidden_dim, num_actions=ml_action_dim, max_decode_len=sequence_length).to(device)
    policy_net = OptimalPolicy(latent_dim=latent_dim).to(device)

    # Load the autoencoder.
    encoder.load_state_dict(torch.load("models/lstm_obs_seq_encoder.pth", weights_only=True))
    decoder.load_state_dict(torch.load("models/lstm_decoder.pth", weights_only=True))
    encoder.eval()
    decoder.eval()

    # Loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(policy_net.parameters()), lr=learning_rate)

    # Training loop.
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # sample 
        for _, (sequences, _, _, _, _, next_ml_actions) in enumerate(train_loader):
            z = encoder(sequences.reshape(-1, 15, 10, 39).to(device)).detach()
            pred_next_z = policy_net(z)
            pred_next_subtask = decoder(pred_next_z)
            loss = criterion(pred_next_subtask.view(-1, decoder.num_actions), next_ml_actions.view(-1).long().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * sequences.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # torch.save(policy_net.state_dict(), "models/unfroze_ED_optimal_policy.pth")
    # torch.save(encoder.state_dict(), "models/opt_encoder.pth")
    # torch.save(decoder.state_dict(), "models/opt_decoder.pth")
    torch.save(policy_net.state_dict(), "models/optimal_policy.pth")
    print("Optimal policy saved to optimal_policy.pth")

if __name__ == "__main__":
    train_optimal_policy()