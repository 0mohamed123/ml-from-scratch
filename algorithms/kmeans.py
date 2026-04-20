import numpy as np


class KMeans:
    def __init__(self, k=3, n_iterations=100, random_state=42):
        self.k = k
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def fit(self, X):
        X = np.array(X)
        np.random.seed(self.random_state)

        # KMeans++ initialization
        centroids = [X[np.random.randint(0, len(X))]]
        for _ in range(self.k - 1):
            distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X])
            probs = distances / distances.sum()
            centroids.append(X[np.random.choice(len(X), p=probs)])

        self.centroids = np.array(centroids)

        for _ in range(self.n_iterations):
            self.labels = self._assign_clusters(X)
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.inertia_ = sum(
            np.linalg.norm(X[i] - self.centroids[self.labels[i]]) ** 2
            for i in range(len(X))
        )
        return self

    def _assign_clusters(self, X):
        distances = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(np.array(X))

    def fit_predict(self, X):
        return self.fit(X).labels