import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_matrix, title="Residue Attention"):
    plt.figure(figsize=(6,5))
    sns.heatmap(attention_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Residue j")
    plt.ylabel("Residue i")
    plt.show()
