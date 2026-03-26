"""End-to-end roundtrip tests: train -> compile -> simulate -> verify."""

from __future__ import annotations

import numpy as np
import pytest

from grove.compile import compile as grove_compile, simulate_quantized_forest
from grove.ir import QuantConfig
from grove.parse import parse_model
from grove.quantize import (
    dequantize_output,
    quantize_feature,
    quantize_forest,
)


def _get_xgb_raw_prediction(model, X):
    """Get raw margin predictions from XGBoost (before sigmoid)."""
    import xgboost as xgb

    dmat = xgb.DMatrix(X)
    return model.get_booster().predict(dmat, output_margin=True)


def test_roundtrip_binary_classification(tmp_path):
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

    path = tmp_path / "model.json"
    model.save_model(str(path))

    X_test = X[:20]
    raw_pred = _get_xgb_raw_prediction(model, X_test)

    forest = parse_model(path)
    config = QuantConfig(
        feature_width=16,
        output_width=32,
        fractional_bit=10,
        feature_min=X.min(axis=0).tolist(),
        feature_max=X.max(axis=0).tolist(),
    )
    qf = quantize_forest(forest, config)

    max_error = 0.0
    for i, sample in enumerate(X_test):
        q_feat = [
            quantize_feature(float(sample[j]), j, config)
            for j in range(forest.num_feature)
        ]

        hw_result_int = simulate_quantized_forest(qf, q_feat)
        hw_result_float = dequantize_output(hw_result_int, config)

        error = abs(hw_result_float - raw_pred[i])
        max_error = max(max_error, error)

    assert max_error < 0.1, f"Max quantization error {max_error:.4f} exceeds tolerance"


def test_roundtrip_regression(tmp_path):
    xgb = pytest.importorskip("xgboost")
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)

    model = xgb.XGBRegressor(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.3,
    )
    model.fit(X, y.astype(float))

    path = tmp_path / "model.json"
    model.save_model(str(path))

    X_test = X[:20]
    raw_pred = _get_xgb_raw_prediction(model, X_test)

    forest = parse_model(path)
    config = QuantConfig(
        feature_width=16,
        output_width=32,
        fractional_bit=10,
        feature_min=X.min(axis=0).tolist(),
        feature_max=X.max(axis=0).tolist(),
    )
    qf = quantize_forest(forest, config)

    max_error = 0.0
    for i, sample in enumerate(X_test):
        q_feat = [
            quantize_feature(float(sample[j]), j, config)
            for j in range(forest.num_feature)
        ]

        hw_result_int = simulate_quantized_forest(qf, q_feat)
        hw_result_float = dequantize_output(hw_result_int, config)

        error = abs(hw_result_float - raw_pred[i])
        max_error = max(max_error, error)

    assert max_error < 0.2, f"Max quantization error {max_error:.4f} exceeds tolerance"


def test_roundtrip_deep_tree(tmp_path):
    xgb = pytest.importorskip("xgboost")
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    y_bin = (y == 2).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=3,
        max_depth=6,
        learning_rate=0.3,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y_bin)

    path = tmp_path / "model.json"
    model.save_model(str(path))

    X_test = X[:10]
    raw_pred = _get_xgb_raw_prediction(model, X_test)

    forest = parse_model(path)
    config = QuantConfig(
        feature_width=16,
        output_width=32,
        fractional_bit=10,
        feature_min=X.min(axis=0).tolist(),
        feature_max=X.max(axis=0).tolist(),
    )
    qf = quantize_forest(forest, config)

    max_error = 0.0
    for i, sample in enumerate(X_test):
        q_feat = [
            quantize_feature(float(sample[j]), j, config)
            for j in range(forest.num_feature)
        ]

        hw_result_int = simulate_quantized_forest(qf, q_feat)
        hw_result_float = dequantize_output(hw_result_int, config)

        error = abs(hw_result_float - raw_pred[i])
        max_error = max(max_error, error)

    assert max_error < 0.1, f"Max quantization error {max_error:.4f} exceeds tolerance"


def test_roundtrip_verilog_output(tmp_path):
    """Verify compile() produces valid Verilog files."""
    xgb = pytest.importorskip("xgboost")
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    y_bin = (y == 2).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=3,
        max_depth=3,
        learning_rate=0.3,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y_bin)

    model_path = tmp_path / "model.json"
    model.save_model(str(model_path))

    out_dir = tmp_path / "output"
    result = grove_compile(
        model_path=model_path,
        output_dir=str(out_dir),
        feature_min=X.min(axis=0).tolist(),
        feature_max=X.max(axis=0).tolist(),
    )

    assert len(result) == 4  # 3 trees + 1 top

    for filename in result:
        file_path = out_dir / filename
        assert file_path.exists()
        content = file_path.read_text()
        assert "module" in content
        assert "endmodule" in content
