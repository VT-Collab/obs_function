sns.kdeplot(points, bw_method=0.65)
sns.histplot(points, bins=2, stat="density")





# import matplotlib.pyplot as plt
# #generak format
# plt.plotting_function(x_values, y_valuues)

# plt.xlabel("x-axis hehe")
# plt.title("HEHE")


# continents = df["Continent"].value_counts() #return key, value
# plt.bar(continents.index, continents.values) #x, y where x is the key value and y is the value itself

# continents.value_counts().plot(kind="bar")


# import seaborn as sns
# #general format
# sns.plotting_function(data=df, x="x_col", y="y_col") #x="x_col" → use the column named "x_col" in df for the x-axis, y="y_col" → use the column named "y_col" for the y-axis

# #both gives a good idea of where each variable should be 
# sns.boxplot(data=df, y="y_variable") #percentile
# sns.violinplot(data=df, hue="Group 1", y="y_variable") #density, hue allows to determine group, and you can overlap it, and the hue names will be in legend














# #testing using LSTM for sine-wave prediction (predict next value of the sine wave)
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# #!!!!!!!! = think about what I need to do differently or not differently for our data 

# #Model source: https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-networks-using-pytorch/

# #-----------------------------PART ONE: PREPARE DATA--------------------------------------------------------------------------
# #seed controls the randomness of the code
# #aka making sure that we get the same random numbers every time we run the code
# np.random.seed(0)
# torch.manual_seed(0)

# #np.linspace() generates x points for sin 
# t = np.linspace(0, 100, 1000)
# #np.sin() creates sin values || create y or output points for sin
# data = np.sin(t)

# #create_sequences() makes input-output pairs, in this case it does look at a sequence but I guess for our case thats not really necessary
# #!!!!!!!!!or like seq_length = how many numbers to group together as input, NOT SURE YET ABOUT MEANING as to SEQUENTIAL DATA
# #eg. if data = [10, 20, 30, 40, 50, 60, 70] and seq_length = 3
# #then function output will be 
# # xs = [
# #   [10, 20, 30],
# #   [20, 30, 40],
# #   [30, 40, 50],
# #   [40, 50, 60]
# # ]

# # ys = [
# #   40,
# #   50,
# #   60,
# #   70
# # ]
# def create_sequences(data, seq_length):
#     #hold input sequence
#     xs = []
#     #hold answers
#     ys = []
#     #loops through data, stops early to not go out of bounds
#     for i in range(len(data) - seq_length):
#         #takes data from i to i+seq_length
#         x = data[i : i+seq_length]
#         #takes the next number right after the x sequence
#         y = data[i+seq_length]
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

# seq_length = 10
# X, y = create_sequences(data, seq_length)

# #adds 3rd parameter so x becomes (batch size, sequence length, input dimension (1) )
# trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
# #adds 2nd parameter so y becomes (batch size, output size)
# trainY = torch.tensor(y[:, None], dtype=torch.float32)
        
# #questions to answer
# #what exactly does the hidden state store
#     #stores short term memory, gets updated at each time step, captures features. EACH LAYER HAS ITS OWN HIDDEN STATE, and hidden state of lower layer along with input = input to higher layer 
# #dont really understand input dimension isnt it like a sequence of length 10 as defined above?
#     #sequence length = how many TIME STEPS per input (10) (amount of each type); input dimention = how many features (data type) per TIME STEP
# #whats the difference between hidden state and cell state
#     #hidden state = short term memory, outputted at each time step
#     #cell state = long term memory, passed with minimal modification, 
#     #both changed at each time step, just that cell state way slower
# #out[:, -1, :] why this is out an array what is this syntax also why last time step 
#     # Assume out has shape: (batch_size, sequence_length, hidden_dim)
#     # This means:
#     # : → take all batches
#     # -1 → take the last time step (which has the most complete info)
#     # : → take all hidden units


# #give me overall structure and answer those in detail

# #-----------------------------PART TWO: DEFINE LSTM MODEL--------------------------------------------------------------------------
# #LSTMModel class inherits from nn.Module
# #!!!!!!!!initialize LSTM layer and fully connected layer
# #the LSTM layer processes the sequences and
# #the fully connected layer maps the hidden state to the output 
# #output of the LSTM layer is passed through fully connected layer which produces final prediction

# class LSTMModel(nn.Module):
    
#     # input_dim: number of features or size of input per time step (e.g., 1 for a sine wave)
#     # hidden_dim: size of the hidden state in the LSTM (how much memory LSTM has)
#     # layer_dim: how many LSTM stacked layers
#     # output_dim: size of the final output (e.g., 1 for predicting a single number)

#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(LSTMModel, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.layer_dim = layer_dim
        
#         #creates actual LSTM layer, which process sequence. batch_first=True: input will be shaped like (batch, seq_len, input_dim) instead of default (seq_len, batch, input_dim)
#         self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
#         #Fully Connected layer = simple linear layer 
#         #It takes the final LSTM output (shape: hidden_dim) and converts it to output_dim (e.g., 1 number).
#         self.fc = nn.Linear(hidden_dim, output_dim)
    
#     #forward() function = we check if hidden states (h0 and c0) are provided. If not they are initialized to zeros.
#     def forward(self, x, h0=None, c0 =None):
#         if h0 is None or c0 is None:
#             #torch.zero() typical syntax = torch.zeros(#of layers, batch size, hidden state size)
#             h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
#             c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

#         #runs lstm on input x using hidden states h0 and c0 
#         #standard documentation of output:
#         # out = output of hidden states (collects for each timestep)
#         # hn = final hidden state (for all layers)
#         # cn = final cell state??
#         out, (hn, cn) = self.lstm(x, (h0, c0))
        
#         #takes ONLY the last time steps's output, or out[:, -1, :], and feed that into the final linear layer to get a prediction
#         out = self.fc(out[:, -1, :])
        
#         return out, hn, cn #out = prediction
    
#        #basically input shape = (batch_size, seq_len, input_dim) -> out from LSTM (batch_size, seq_len, hidden_dim) -> out[:, -1, :] 加工out： (batch_size, hidden_dim) -> self.fc(out[:, -1, :]): (batch_size, 1)
        
# #-----------------------------PART THREE: Initializing Model, LOSS FUNCTION and OPTIMIZER--------------------------------------------------------------------------
# model = LSTMModel(input_dim=1, hidden_dim = 100, layer_dim = 1, output_dim=1) #!!!!!!! Why those numbers
# criterion = nn.MSELoss() #loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #use the polular adam optimizer

# #-----------------------------PART FOUR: TRAIN the LSTM MODEL--------------------------------------------------------------------------
# #In each epoch we set the model to training mode, perform a forward pass, compute the loss and then backpropagate the error to update the weights.
# #We detach the hidden states (h0 and c0) after each iteration to prevent backpropagating through the entire sequence in the past (which already been backpropagated before) 
# #Every 10 epochs we print the current loss value to monitor the model's progress.
#     #!!!!!!!!!use wandb


# num_epochs = 100 # # iterations
# h0, c0 = None, None

# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()

#     outputs, h0, c0 = model(trainX, h0, c0)

#     loss = criterion(outputs, trainY)
#     loss.backward()
#     optimizer.step()

#     h0 = h0.detach()
#     c0 = c0.detach()

#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# #-----------------------------PART FIVE: EVALUATION AND PLOTTING--------------------------------------------------------------------------
# #.eval method turns on evaluation mode for consistency
# model.eval()

# #calls model with input trainX, h0, c0 as defined above, returns 3 values: output, hidden state, and cell state. _ is used to ignore the last two.
# predicted, _, _ = model(trainX, h0, c0)

# #Slices the array data from index seq_length to the end, cuz If you used seq_length=10, then your model started predicting at data[10].
# original = data[seq_length:]

# #np.arange(start, stop) creates a NumPy array of numbers from start to stop - 1.
# #!!!!!!!! can't really just use this? cuz compute it by episodes etc. 
# time_steps = np.arange(seq_length, len(data))

# #adds some distortion to every 30th and 70th value
# predicted[::30] += 0.2 
# predicted[::70] -= 0.2

# plt.figure(figsize=(12, 6))
# plt.plot(time_steps, original, label='Original Data')
# plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
# plt.title('LSTM Model Predictions vs. Original Data')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()