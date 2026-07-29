import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import pyro.contrib.bnn as bnn

# Define the Bayesian Neural Network
class BayesianNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianNetwork, self).__init__()
        # Bayesian Linear layers using Pyro's bnn module
        self.fc1 = bnn.BayesianLinear(input_size, hidden_size)
        self.fc2 = bnn.BayesianLinear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# Instantiate the model
input_size = len(combined_data.columns) - 1  # minus agent_type column
hidden_size = 64
output_size = len(combined_data['agent_type'].unique())  # number of agent types

model = BayesianNetwork(input_size, hidden_size, output_size)

# Define the Pyro model for SVI
def pyro_model(x_data, y_data=None):
    # Prior distributions for the weights
    priors = {
        'fc1.weight': dist.Normal(0., 1.).expand([hidden_size, input_size]).to_event(2),
        'fc1.bias': dist.Normal(0., 1.).expand([hidden_size]).to_event(1),
        'fc2.weight': dist.Normal(0., 1.).expand([output_size, hidden_size]).to_event(2),
        'fc2.bias': dist.Normal(0., 1.).expand([output_size]).to_event(1)
    }
    lifted_module = pyro.random_module("module", model, priors)
    lifted_reg_model = lifted_module()

    # forward pass
    logits = lifted_reg_model(x_data)
    
    with pyro.plate("data", x_data.shape[0]):
        if y_data is None:
            return pyro.sample("obs", dist.Categorical(logits=logits))
        else:
            return pyro.sample("obs", dist.Categorical(logits=logits), obs=y_data)

# Variational inference setup
optimizer = Adam({"lr": 0.01})
svi = SVI(pyro_model, pyro.infer.autoguide.AutoDiagonalNormal(pyro_model), optimizer, loss=Trace_ELBO())

