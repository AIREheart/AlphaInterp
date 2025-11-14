# visualization/plot_residue_scores.py
import matplotlib.pyplot as plt
import numpy as np
def plot_scores(scores, out=None, title="Residue importance"):
    plt.figure(figsize=(12,3))
    plt.plot(scores, marker="o", markersize=3)
    plt.xlabel("Residue index")
    plt.ylabel("Score")
    plt.title(title)
    if out:
        plt.savefig(out, dpi=200)
    plt.show()
