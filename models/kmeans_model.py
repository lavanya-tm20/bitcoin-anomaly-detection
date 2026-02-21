from sklearn.cluster import KMeans
import numpy as np
import joblib
import os

class KMeansAnomaly:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.model.fit(X)
        return self

    def distances(self, X):
        labels = self.model.predict(X)
        centers = self.model.cluster_centers_
        d = np.linalg.norm(X - centers[labels], axis=1)
        return d

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
