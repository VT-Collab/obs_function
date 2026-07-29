
#a practice training vae script based on
#https://www.codecademy.com/article/variational-autoencoder-tutorial-vaes-explained


import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader  
import torch.nn.functional as F 
import numpy as np  
import matplotlib.pyplot as plt  
 
# Use GPU if available  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset; which is handwritten digits dataset
transform = transforms.ToTensor()   
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)  
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True) 

#Encoder class; input -> hidden -> latent
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Encoder, self).__init__()
        #flattens image 28x28 = 784 pixels
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        The encoder first flattens the image (28×28 pixels = 784 inputs).
        It then maps this to a hidden layer of size 400 using a ReLU activation.
        From the hidden layer, it produces two outputs:
            mu: the center of the latent distribution
            logvar: the log of the variance (used for sampling and stability)
 
        """
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
       
#Decoder class; latent -> hidden -> output (same as encoder input)
class Decoder(nn.Module):
    def __init__(self, latent_dim = 20, hidden_dim=400, output_dim = 784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        """
        Takes a 20-dimensional latent vector z (sampled from the encoded distribution).
        Maps it back to the original data dimensions through a hidden layer.
        The final sigmoid ensures pixel values are between 0 and 1.
        """
        h = torch.relu(self.fc1(z))
        output = torch.sigmoid(self.fc2(h))
        return output
    
#Main VAE class
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        The complete model takes an image, encodes it to a latent distribution, samples from it, and reconstructs the image.
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    # natural log = log base e!!!!! Meaning log <-> e!!!!!!
    def reparameterize(self, mu, logvar):
        """
        The reparameterize() function enables differentiable sampling.
        logvar is exponentiated and scaled to get the standard deviation.
        eps is random noise from a standard normal distribution.
        The final sampled z has the desired mean and variance but is differentiable with respect to mu and logvar.
        """
        
        #log var = log (std^2) = 2 log(std) 
        #std = e^ [1/2*logvar]
        std = torch.exp(0.5*logvar) #standard deviation
        eps = torch.randn_like(std) #random noice that is same shape as std!
        
        return mu + eps * std    #mu and eps and std are all the same shape, aka the latent dim shape
    
    def forward(self, x):
        mu, logvar = self.encoder(x)  
        z = self.reparameterize(mu, logvar)  
        reconstructed = self.decoder(z)  
        return reconstructed, mu, logvar 
    
def loss_function(recon_x, x, mu, logvar):
    """
    Training a VAE involves optimizing a composite loss function that balances two goals:

    Reconstruction loss: Ensures the output is close to the original input.

    KL divergence: Encourages the learned latent distribution to be close to a standard normal distribution.

    The total loss is often referred to as the ELBO (Evidence Lower Bound), and we aim to maximize it (or equivalently, minimize its negative).
    """
    
    
    #Reconstruction loss: Binary Cross Entropy is used for normalized pixel values (0-1), it measures how accurately the output matches the input.
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    #KL divergence: Encourages mu and logvar to represent a distribution close to N(0, 1), helping the latent space generalize and avoid overfitting.
    #KL difvergence between 2 gaussians: N(mu, std) and N (0,1) is 1/2(mu^2 + e^logvar - log var - 1) and multiply by -1 everywhere and we get
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  
    return recon_loss + kl_loss
    
#actual training
import matplotlib.pyplot as plt

#set training parameters 
epochs = 10
learning_rate = 1e-3 #the e here stands for exponent, not natural log e

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []

model.train()


for epoch in range(epochs):
    total_loss = 0
    #gives batches, batch data (x= images, _ = labels)
    for batch_idx, (x, _) in enumerate(train_loader):
        #-1 means figure it out automatically 
        #so (-1, 784) becomes (64, 784)
        x = x.view(-1, 784).to(device) 
        optimizer.zero_grad()
        
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward() #backpropagation
        optimizer.step() #actual update
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}") 
    
#actual plot
plt.plot(train_losses) 
plt.title("VAE Training Loss") 
plt.xlabel("Epoch") 
plt.ylabel("Loss") 
plt.grid(True) 
plt.show() 


#evaluation 
#write from scratch

#TODO: 1. Sampling from the latent space

#TODO: 2. Latent space interpolation




    