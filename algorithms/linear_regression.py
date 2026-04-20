import numpy as np
from utils.metrics import r2_score, mean_squared_error


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = self._predict(X)
            error = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = mean_squared_error(y, y_pred)
            self.loss_history.append(loss)

        return self

    def _predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        X = np.array(X)
        return self._predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def normal_equation(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]
        return self