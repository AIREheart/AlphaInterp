# training/train_sae.py
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.sparse_autoencoder import SparseAutoencoder
from tqdm import tqdm
import os

def load_embeddings_npz(npz_paths, key="single"):
    all_emb = []
    for p in npz_paths:
        data = np.load(p)
        if key not in data:
            raise KeyError(f"{key} not in {p}, available keys: {list(data.keys())}")
        emb = data[key]  # shape (L, C) or (batch,) depending on how you saved
        # flatten residues into rows (concatenate across proteins)
        if emb.ndim == 2:
            all_emb.append(emb)  # (L, C)
        elif emb.ndim == 3:
            # If shape (1, L, C)
            all_emb.append(emb[0])
        else:
            raise ValueError("Unexpected embedding shape")
    X = np.vstack(all_emb).astype(np.float32)
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", required=True)
    parser.add_argument("--key", default="single")
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", default="outputs/sae")
    args = parser.parse_args()

    X = load_embeddings_npz(args.embeddings, key=args.key)  # shape (N_residues, C)
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    model = SparseAutoencoder(input_dim=X.shape[1], latent_dim=args.latent)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for (batch,) in tqdm(loader):
            batch = batch
            x_hat, loss, z = model(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * batch.shape[0]
        avg = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, loss: {avg:.6f}")
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.outdir, f"sae_epoch{epoch+1}.pt"))
    torch.save(model.state_dict(), os.path.join(args.outdir, "sae_final.pt"))

if __name__ == "__main__":
    main()
