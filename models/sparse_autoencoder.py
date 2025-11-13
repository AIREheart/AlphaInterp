import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sparsity_lambda=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        recon_loss = torch.mean((x - x_hat) ** 2)
        return x_hat, recon_loss + sparsity_loss
