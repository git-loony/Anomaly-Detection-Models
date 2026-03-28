import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


class ClusteredIForest:
    def __init__(self, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.models = {}

    def fit(self, X):
        self.clusters = self.kmeans.fit_predict(X)

        for c in np.unique(self.clusters):
            X_c = X[self.clusters == c]
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(X_c)
            self.models[c] = model

    def score(self, X):
        clusters = self.kmeans.predict(X)
        scores = []

        for i, x in enumerate(X):
            model = self.models[clusters[i]]
            score = -model.decision_function([x])[0]
            scores.append(score)

        scores = np.array(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())