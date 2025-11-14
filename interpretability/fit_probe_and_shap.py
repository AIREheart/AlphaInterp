# interpretability/fit_probe_and_shap.py
import argparse
import numpy as np
from sklearn.linear_model import Ridge
import shap
import joblib
import os

def fit_probe(X, y, out_path):
    # X: (N_residues, C), y: (N_residues,)
    clf = Ridge(alpha=1.0)
    clf.fit(X, y)
    joblib.dump(clf, out_path)
    return clf

def compute_shap(clf, X, nsamples=100):
    # Use KernelExplainer since model is sklearn (fast)
    explainer = shap.KernelExplainer(clf.predict, shap.kmeans(X, 10))
    shap_vals = explainer.shap_values(X, nsamples=nsamples)
    return shap_vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="npz embedding file")
    parser.add_argument("--key", default="single")
    parser.add_argument("--outdir", default="outputs/shap")
    args = parser.parse_args()
    data = np.load(args.emb)
    X = data[args.key]
    # If X is (1, L, C)
    if X.ndim == 3:
        X = X[0]
    # target: predicted plddt per residue if available, else synthetic (sum of pair features)
    if "plddt" in data:
        y = data["plddt"]
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
    else:
        # fallback: use L2 norm of embedding as a proxy target
        y = np.linalg.norm(X, axis=1)
    os.makedirs(args.outdir, exist_ok=True)
    clf = fit_probe(X, y, os.path.join(args.outdir, "ridge_probe.joblib"))
    shap_vals = compute_shap(clf, X)
    np.savez_compressed(os.path.join(args.outdir, "shap_vals.npz"), shap=shap_vals, y=y, X=X)
    print("Saved SHAP results to", args.outdir)

if __name__ == "__main__":
    main()
