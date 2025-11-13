from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#linear probes
def probe_layer(embeddings, labels):
    clf = LogisticRegression(max_iter=500)
    clf.fit(embeddings, labels)
    preds = clf.predict(embeddings)
    return accuracy_score(labels, preds)
