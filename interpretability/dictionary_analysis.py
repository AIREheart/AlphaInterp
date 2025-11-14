# interpretability/dictionary_analysis.py
import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from models.sparse_autoencoder import SparseAutoencoder

def load_sae(model_path, input_dim, latent_dim):
    model = SparseAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def compute_latents(model, X):  # X is np.array (N, D)
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32))
        z = model.encoder(X_t).numpy()
    return z

def cluster_latents(z, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(z)
    return kmeans.labels_, kmeans.cluster_centers_

def visualize_cluster_map(labels, seq_positions=None, out_png=None):
    # labels per residue; plot cluster label along sequence
    plt.figure(figsize=(12,3))
    sns.scatterplot(x=np.arange(len(labels)), y=labels, hue=labels, palette="tab20", legend=False, s=10)
    plt.xlabel("Residue index")
    plt.ylabel("Cluster")
    if out_png:
        plt.savefig(out_png, dpi=200)
    plt.show()
