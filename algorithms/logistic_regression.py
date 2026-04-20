import numpy as np
from utils.metrics import accuracy_score, precision_score, recall_score, f1_score


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

        return self

    def _compute_loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def predict_proba(self, X):
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))