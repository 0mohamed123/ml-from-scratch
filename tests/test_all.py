import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from algorithms.knn import KNN
from algorithms.decision_tree import DecisionTree


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    return X, y


# ===== KNN Tests =====
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
    assert model.predict(X).shape == y.shape


def test_knn_predict_classes(classification_data):
    X, y = classification_data
    model = KNN(k=3)
    model.fit(X, y)
    assert set(np.unique(model.predict(X))).issubset(set(np.unique(y)))


# ===== Decision Tree Tests =====
def test_decision_tree_accuracy(classification_data):
    X, y = classification_data
    model = DecisionTree(max_depth=5)
    model.fit(X, y)
    assert model.score(X, y) > 0.90


def test_decision_tree_matches_sklearn(classification_data):
    X, y = classification_data
    our_model = DecisionTree(max_depth=5)
    our_model.fit(X, y)
    sklearn_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    sklearn_model.fit(X, y)
    assert abs(our_model.score(X, y) - sklearn_model.score(X, y)) < 0.05


def test_decision_tree_predict_shape(classification_data):
    X, y = classification_data
    model = DecisionTree(max_depth=5)
    model.fit(X, y)
    assert model.predict(X).shape == y.shape


def test_decision_tree_depth(classification_data):
    X, y = classification_data
    model_shallow = DecisionTree(max_depth=2)
    model_deep = DecisionTree(max_depth=10)
    model_shallow.fit(X, y)
    model_deep.fit(X, y)
    assert model_deep.score(X, y) >= model_shallow.score(X, y)


def test_decision_tree_entropy(classification_data):
    X, y = classification_data
    model = DecisionTree(max_depth=5, criterion="entropy")
    model.fit(X, y)
    assert model.score(X, y) > 0.90