import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.neighbors import KNeighborsClassifier
from algorithms.knn import KNN


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    return X, y


def test_knn_accuracy(classification_data):
    X, y = classification_data
    model = KNN(k=3)
    model.fit(X, y)
    assert model.score(X, y) > 0.85


def test_knn_matches_sklearn(classification_data):
    X, y = classification_data
    our_model = KNN(k=3)
    our_model.fit(X, y)

    sklearn_model = KNeighborsClassifier(n_neighbors=3)
    sklearn_model.fit(X, y)

    assert abs(our_model.score(X, y) - sklearn_model.score(X, y)) < 0.02


def test_knn_different_k(classification_data):
    X, y = classification_data
    for k in [1, 3, 5, 7]:
        model = KNN(k=k)
        model.fit(X, y)
        assert model.score(X, y) > 0.75


def test_knn_predict_shape(classification_data):
    X, y = classification_data
    model = KNN(k=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_knn_predict_classes(classification_data):
    X, y = classification_data
    model = KNN(k=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))