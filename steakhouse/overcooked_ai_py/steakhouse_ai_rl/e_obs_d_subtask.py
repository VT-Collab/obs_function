import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rl.dataset import TrajectoryDataset


class Encoder(nn.Module):
    def __init__(self, input_shape=(39, 15, 10), latent_dim=128, sequence_length=10):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=39, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Compute the flattened size after conv layers
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            conv_out = self.conv2d(dummy_input)
        self.single_flat_dim = conv_out.view(1, -1).shape[1]
        self.flat_dim = self.single_flat_dim * sequence_length

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim)
        )

        self.lstm = nn.LSTM(input_size=self.single_flat_dim,
                    hidden_size=latent_dim,
                    num_layers=1,
                    batch_first=True)

    def forward(self, x):
        B_T, H, W, C = x.shape
        B = B_T // self.sequence_length
        x = x.permute(0, 3, 1, 2) # (B, 15, 10, 39) -> (B, 39, 15, 10)
        x = self.conv2d(x)
        x = x.reshape(B * self.sequence_length, -1)
        x = x.reshape(B, self.sequence_length, -1)
        output, (h_n, c_n) = self.lstm(x)
        z = h_n[-1]
        return z


class Decoder(nn.Module):
    """
    A feedforward (non-autoregressive) decoder that takes an encoder latent vector
    and outputs a sequence of action logits, where each action is one of 12 categories.
    
    The decoder outputs a tensor of shape:
        (batch_size, max_decode_len, num_actions)
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 128, 
                 num_actions: int = 12, max_decode_len: int = 10):
        """
        Args:
            latent_dim: Dimension of the encoder's latent output.
            hidden_dim: Hidden dimension for intermediate representations.
            num_actions: Number of possible actions (default: 12).
            max_decode_len: The length of the decoded action sequence.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.max_decode_len = max_decode_len
        
        # Project the latent vector to a sequence representation.
        # The output dimension is max_decode_len * hidden_dim.
        self.fc1 = nn.Linear(latent_dim, max_decode_len * hidden_dim)
        
        # A second linear layer to map the hidden representation at each time step to logits over actions.
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_vector: Tensor of shape (batch_size, latent_dim)
            
        Returns:
            logits_seq: Tensor of shape (batch_size, max_decode_len, num_actions)
                        Each slice logits_seq[:, t, :] contains the unnormalized logit scores
                        for the 12 possible actions at time step t.
        """
        batch_size = latent_vector.size(0)
        
        # Map the latent vector to a flattened sequence representation
        x = F.relu(self.fc1(latent_vector))  # Shape: (batch_size, max_decode_len * hidden_dim)
        
        # Reshape into sequence format: (batch_size, max_decode_len, hidden_dim)
        x = x.view(batch_size, self.max_decode_len, self.hidden_dim)
        
        # Apply the final layer to each time step (applied on the last dimension automatically)
        logits_seq = self.fc2(x)  # Shape: (batch_size, max_decode_len, num_actions)
        
        return logits_seq
        
# Autoencoder that bundles the Encoder and Decoder.
class Autoencoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, sequence_length=10, ml_action_dim=12):
        super(Autoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.ml_action_dim = ml_action_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, num_actions=ml_action_dim, max_decode_len=sequence_length)
        
    def forward(self, x):
        latent = self.encoder(x)
        subtask = self.decoder(latent)
        return subtask
    
def evaluate_autoencoder(model, test_loader, criterion, device='cpu'):
    model = model.to(device)
    model.load_state_dict(torch.load("models/autoencoder.pth"))
    model.eval()

    eval_loss = 0.0
    with torch.no_grad():
        for _, (sequences, _, ml_actions, _, _, _) in enumerate(test_loader):
            subtask = model(sequences.reshape(-1, 15, 10, 39).to(device).detach())
            loss = criterion(
                subtask.view(-1, model.ml_action_dim),  # (B*T, 12)
                ml_actions.detach().to(device).view(-1).long()   # (B*T,)
            )
            eval_loss += loss.item() * sequences.size(0)
        eval_loss /= len(test_loader.dataset)
        print(f'Eval Loss: {eval_loss:.4f}')

def train_autoencoder(num_epochs=10, batch_size=128, learning_rate=1e-4):
    # Load the training dataset
    trajectories_path = os.path.join(os.path.dirname(__file__), "../..", "expert_trajectories_layout.npz")
    dataset = TrajectoryDataset(trajectory_path=trajectories_path)
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    # Model hyperparameters.
    hidden_dim = 256
    latent_dim = 128
    
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(hidden_dim, latent_dim).to(device)
    
    # Loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop.
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for _, (sequences, _, ml_actions, _, _, _) in enumerate(train_loader):
            subtask = model(sequences.reshape(-1, 15, 10, 39).to(device))
            # Cross-entropy loss at each time step
            loss = criterion(
                subtask.view(-1, model.ml_action_dim),  # (B*T, 12)
                ml_actions.to(device).view(-1).long()   # (B*T,)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * sequences.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), "models/autoencoder.pth")
        #     torch.save(model.encoder.state_dict(), "models/lstm_obs_seq_encoder.pth")
        #     torch.save(model.decoder.state_dict(), "models/lstm_decoder.pth")
        #     evaluate_autoencoder(Autoencoder(hidden_dim, latent_dim).to(device), eval_loader, criterion, device)
    
    # Save the trained model.
    torch.save(model.state_dict(), "models/autoencoder.pth")
    torch.save(model.encoder.state_dict(), "models/lstm_obs_seq_encoder.pth")
    torch.save(model.decoder.state_dict(), "models/lstm_decoder.pth")

    print("Training complete. Model saved as 'autoencoder.pth'")
    return model

if __name__ == "__main__":
    trained_model = train_autoencoder(num_epochs=500)
