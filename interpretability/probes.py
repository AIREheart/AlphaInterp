# interpretability/probes.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_secondary_structure_probe(X, ss_labels):
    # X: (N_res, C), ss_labels: (N_res,) 0/1/2
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, ss_labels)
    preds = clf.predict(X)
    return accuracy_score(ss_labels, preds), clf
