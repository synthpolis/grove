"""Tests for fixed-point quantization."""

from __future__ import annotations

import math

import pytest

from grove.ir import Forest, Node, QuantConfig, Tree
from grove.quantize import (
    dequantize_output,
    infer_feature_range,
    quantize_feature,
    quantize_forest,
    quantize_leaf,
    quantize_threshold,
)


@pytest.fixture
def simple_config():
    return QuantConfig(
        feature_width=16,
        output_width=32,
        fractional_bit=10,
        feature_min=[0.0, 0.0],
        feature_max=[1.0, 1.0],
    )


def test_quantize_threshold_midpoint(simple_config):
    result = quantize_threshold(0.5, 0, simple_config)
    expected = (1 << 16) // 2
    assert abs(result - expected) <= 1


def test_quantize_threshold_zero(simple_config):
    result = quantize_threshold(0.0, 0, simple_config)
    assert result == 0


def test_quantize_threshold_max(simple_config):
    result = quantize_threshold(1.0, 0, simple_config)
    assert result == (1 << 16) - 1


def test_quantize_leaf_positive():
    config = QuantConfig(fractional_bit=10)
    result = quantize_leaf(0.5, config)
    assert result == 512  # 0.5 * 2^10


def test_quantize_leaf_negative():
    config = QuantConfig(fractional_bit=10)
    result = quantize_leaf(-0.25, config)
    assert result == -256  # -0.25 * 2^10


def test_quantize_leaf_zero():
    config = QuantConfig(fractional_bit=10)
    result = quantize_leaf(0.0, config)
    assert result == 0


def test_dequantize_roundtrip():
    config = QuantConfig(fractional_bit=10)
    original = 0.123
    quantized = quantize_leaf(original, config)
    recovered = dequantize_output(quantized, config)
    assert abs(recovered - original) < 1.0 / (1 << 10)


def test_infer_feature_range():
    nodes = [
        Node(0, False, feature_index=0, threshold=1.0, left=1, right=2),
        Node(1, True, value=0.1),
        Node(2, False, feature_index=1, threshold=5.0, left=3, right=4),
        Node(3, True, value=0.2),
        Node(4, True, value=0.3),
    ]
    tree = Tree(tree_id=0, node=nodes, num_feature=2)
    forest = Forest(
        tree=[tree], num_feature=2, base_score=0.5, objective="reg:squarederror"
    )

    f_min, f_max = infer_feature_range(forest, padding=0.1)

    assert len(f_min) == 2
    assert len(f_max) == 2
    assert f_min[0] < 1.0
    assert f_max[0] > 1.0
    assert f_min[1] < 5.0
    assert f_max[1] > 5.0


def test_infer_feature_range_unused_feature():
    """Feature never split on should get default range."""
    nodes = [
        Node(0, False, feature_index=0, threshold=2.0, left=1, right=2),
        Node(1, True, value=0.1),
        Node(2, True, value=0.2),
    ]
    tree = Tree(tree_id=0, node=nodes, num_feature=3)
    forest = Forest(
        tree=[tree], num_feature=3, base_score=0.5, objective="reg:squarederror"
    )

    f_min, f_max = infer_feature_range(forest)

    assert f_min[1] == 0.0
    assert f_max[1] == 1.0
    assert f_min[2] == 0.0
    assert f_max[2] == 1.0


def test_quantize_forest_preserves_structure(binary_model_path):
    from grove.parse import parse_model

    forest = parse_model(binary_model_path)
    config = QuantConfig(feature_width=16, output_width=32, fractional_bit=10)
    qf = quantize_forest(forest, config)

    assert len(qf.tree) == len(forest.tree)
    for qt, ft in zip(qf.tree, forest.tree):
        assert len(qt.node) == len(ft.node)
        for qn, fn in zip(qt.node, ft.node):
            assert qn.is_leaf == fn.is_leaf
            if not qn.is_leaf:
                assert qn.feature_index == fn.feature_index
                assert qn.left == fn.left
                assert qn.right == fn.right


def test_quantize_feature_matches_threshold(simple_config):
    """Feature and threshold quantized the same way."""
    feat_val = 0.75
    thresh_val = 0.75
    f = quantize_feature(feat_val, 0, simple_config)
    t = quantize_threshold(thresh_val, 0, simple_config)
    assert f == t
