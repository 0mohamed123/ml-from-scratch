import numpy as np
from utils.metrics import accuracy_score


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = len(X_c) / len(y)

        return self

    def _gaussian_likelihood(self, c, x):
        mean = self.mean[c]
        var = self.var[c] + 1e-9
        exponent = -((x - mean) ** 2) / (2 * var)
        return -0.5 * np.sum(np.log(2 * np.pi * var)) + np.sum(exponent)

    def _predict_single(self, x):
        posteriors = {}
        for c in self.classes:
            posteriors[c] = np.log(self.priors[c]) + self._gaussian_likelihood(c, x)
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in np.array(X)])

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))