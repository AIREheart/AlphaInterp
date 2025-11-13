import py3Dmol
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_matrix, title="Residue Attention"):
    plt.figure(figsize=(6,5))
    sns.heatmap(attention_matrix, cmap="viridis")
    plt.title(title)
    plt.xlabel("Residue j")
    plt.ylabel("Residue i")
    plt.show()



def show_structure(pdb_str, residue_importance):
    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb_str, "pdb")
    for i, score in enumerate(residue_importance):
        color = f"red" if score > 0.5 else "blue"
        view.setStyle({'resi': i+1}, {"cartoon": {"color": color}})
    view.zoomTo()
    return view
