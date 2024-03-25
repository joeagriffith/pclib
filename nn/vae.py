import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, sizes, z_dim):
        super().__init__()
        self.encoder = []
        for i in range(len(sizes) - 1):
            self.encoder.append(nn.Linear(sizes[i], sizes[i+1]))
            self.encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*self.encoder)

        self.mu = nn.Linear(sizes[-1], z_dim)
        self.logvar = nn.Linear(sizes[-1], z_dim)
        self.z2hidden = nn.Linear(z_dim, sizes[-1])

        self.decoder = []
        for i in reversed(range(len(sizes) - 1)):
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Linear(sizes[i+1], sizes[i]))
        self.decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.decoder)

        self.register_buffer('epochs_trained', torch.tensor(0, dtype=torch.long))
        self.register_buffer('min_val_loss', torch.tensor(float('inf')))

        self.z_dim = z_dim
    
    def inc_epochs(self):
        self.epochs_trained += 1
    
    def is_best(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            return True
        return False
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):     
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(self.z2hidden(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, n, device='cpu'):
        z = torch.randn(n, self.z_dim, device=device)
        return self.decode(z)
    
    def loss(self, x_hat, x, mu, logvar, beta=1.0):
        BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD