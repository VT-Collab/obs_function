#we want simple MLP model here 

import torch
import torch.nn as nn #nn is big library of NN functions, classes, etc. 

class MLP(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        #Sequential allows bundling
        self.mlp = nn.Sequential(
            #nn.Flatten() takes nput of shape (batch_size, seq_len, input_dim) and reshapes it to (batch_size, seq_len * input_dim)
            #E.g. (8, 10, 65) → (8, 650).
            nn.Flatten(),          # <-- flatten
            
            #first fully connected layer of say 10*65= 650 or seq_length times vector dimension -> hidden layer
            nn.Linear(seq_len * input_dim, hidden_dim),
            #ReLu activation function introduces non‑linearity
            nn.ReLU(),
            
            #second fully connected layer: hidden layer -> output, for our purpose, output_dim = 1, or classification problem
            nn.Linear(hidden_dim, output_dim)
    )
        
    def forward(self, x):
        return self.mlp(x)
    

    