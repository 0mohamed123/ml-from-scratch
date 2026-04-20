import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLR
from algorithms.logistic_regression import LogisticRegression


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    return X, y


def test_accuracy(binary_data):
    X, y = binary_data
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    assert model.score(X, y) > 0.80


def test_matches_sklearn(binary_data):
    X, y = binary_data
    our_model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    our_model.fit(X, y)

    sklearn_model = SklearnLR(max_iter=1000)
    sklearn_model.fit(X, y)

    assert abs(our_model.score(X, y) - sklearn_model.score(X, y)) < 0.05


def test_predict_proba_range(binary_data):
    X, y = binary_data
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_loss_decreases(binary_data):
    X, y = binary_data
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    assert model.loss_history[0] > model.loss_history[-1]


def test_predict_binary(binary_data):
    X, y = binary_data
    model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})