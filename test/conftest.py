"""Shared fixtures for grove tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def binary_model_path(tmp_path):
    """Train a small binary classification XGBoost model, save as JSON."""
    xgb = pytest.importorskip("xgboost")
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    y_bin = (y == 2).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.3,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y_bin)

    path = tmp_path / "binary_model.json"
    model.save_model(str(path))
    return path


@pytest.fixture
def regression_model_path(tmp_path):
    """Train a small regression XGBoost model, save as JSON."""
    xgb = pytest.importorskip("xgboost")
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    model = xgb.XGBRegressor(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.3,
    )
    model.fit(X, y.astype(float))

    path = tmp_path / "regression_model.json"
    model.save_model(str(path))
    return path


@pytest.fixture
def iris_data():
    """Return Iris dataset as (X, y)."""
    pytest.importorskip("sklearn")
    from sklearn.datasets import load_iris

    return load_iris(return_X_y=True)
