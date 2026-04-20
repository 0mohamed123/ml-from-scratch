import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLR
from algorithms.linear_regression import LinearRegression


@pytest.fixture
def simple_data():
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def multi_feature_data():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2*X[:,0] + 3*X[:,1] - X[:,2] + 1 + np.random.randn(100) * 0.1
    return X, y


def test_r2_score_simple(simple_data):
    X, y = simple_data
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    assert model.score(X, y) > 0.99


def test_r2_score_multi_feature(multi_feature_data):
    X, y = multi_feature_data
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    assert model.score(X, y) > 0.99


def test_matches_sklearn(simple_data):
    X, y = simple_data

    our_model = LinearRegression(learning_rate=0.1, n_iterations=2000)
    our_model.fit(X, y)

    sklearn_model = SklearnLR()
    sklearn_model.fit(X, y)

    our_r2 = our_model.score(X, y)
    sklearn_r2 = sklearn_model.score(X, y)

    assert abs(our_r2 - sklearn_r2) < 0.01


def test_normal_equation(simple_data):
    X, y = simple_data

    model = LinearRegression()
    model.normal_equation(X, y)

    sklearn_model = SklearnLR()
    sklearn_model.fit(X, y)

    assert abs(model.weights[0] - sklearn_model.coef_[0]) < 0.01
    assert abs(model.bias - sklearn_model.intercept_) < 0.01


def test_loss_decreases(simple_data):
    X, y = simple_data
    model = LinearRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    assert model.loss_history[0] > model.loss_history[-1]


def test_predict_shape(simple_data):
    X, y = simple_data
    model = LinearRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape