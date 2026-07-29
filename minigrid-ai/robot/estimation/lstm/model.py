
import torch
import torch.nn as nn #nn is big library of NN functions, classes, etc. 


#Start with the simplest least-layered LSTM Model that takes in input sequence of 10 dim-65 vecs, and output one number: the next p0_action, 
#which is the 29th number in the 65-dim vector

#DONE test out with sin model thing
#grab one file and a few episode and train to see if it overfits

#print out hidden to see what it represents 
#then try the whole file and see accuracy
#then try all 3 files, and see accuracy 



#nn.Module is the abstract base class that our subclass inherits from
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__() #modern python nothing inside (): super()
        
        #store hidden and layer dimensions only b/c we need to use them later in forward
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        #pass into sub-modules, names lstm and fc are our choice, .LSTM and .Linear are set
        #input_dim = 65 for my 65-dim vector
        # seq_len = 10 or 32
        # hidden_dim = my choice of how big I want LSTM hidden state to be; at each time step the output vector will be of size hidden_dim
        # layer_dim = my choice, how many layers I want LSTM to have. ******Num Layers depend on overfitting (too many layers) or underfitting (too few layers)******* 
        #output_dim = 1 b/c I want single number (p0_action) as output
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) 
        #another submodule fc, fully connected, just to get final result of prediction from LSTM
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
    #standard forward method for LSTM Model: (method) def forward(
        # self: Self@LSTMModel,
        # x: Any, -> this is input tensor typically (batch_size, seq_length, input_size) that we pass into the LSTM
        # h0: Any | None = None, -> this is the starting hidden state for LSTM at timestep 0, Shape = (num_layers * num_directions, batch_size, hidden_size)
        # c0: Any | None = None -> this is the The starting cell state for the LSTM at timestep, Shape: (num_layers * num_directions, batch_size, hidden_size)
        # Note: num_directions means like either only left to right or both directions (latter is 2 LSTM in parallel)
        # Also note: Each LSTM layer has its own separate hidden and cell states! They're NOT shared between layers.
        # ) -> None 
        # x.size(0) is the batch_size
    def forward(self, x, h0=None, c0 = None):
        #initialize h0 and c0 to 0 if there are none
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        #first out = all hidden states from all timesteps combined into one
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        #second out (reassign) is the prediction, in our case, a single int po_action 
        #it does so by taking ONLY the last time steps's output, or out[:, -1, :], and feed that into the final linear layer to get a prediction
        out = self.fc(out[:, -1, :])
        return out, hn, cn
        