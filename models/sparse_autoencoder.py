# models/sparse_autoencoder.py
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=256, sparsity_lambda=1e-3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # L1 on latent to encourage sparsity (can add KL or other)
        sparsity_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        recon_loss = torch.mean((x - x_hat) ** 2)
        return x_hat, recon_loss + sparsity_loss, z
