import shap
import torch

def explain_residue_importance(model, embeddings, target):
    explainer = shap.DeepExplainer(model, embeddings)
    shap_values = explainer.shap_values(target)
    return shap_values
