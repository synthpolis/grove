"""Tests for XGBoost JSON model parsing."""

from __future__ import annotations

import json

import pytest

from grove.parse import parse_model


def test_parse_binary_model(binary_model_path):
    forest = parse_model(binary_model_path)

    assert len(forest.tree) == 5
    assert forest.num_feature == 4
    assert forest.objective == "binary:logistic"
    assert isinstance(forest.base_score, float)


def test_parse_regression_model(regression_model_path):
    forest = parse_model(regression_model_path)

    assert len(forest.tree) == 5
    assert forest.num_feature == 4
    assert "reg:" in forest.objective or "multi:" in forest.objective


def test_parse_tree_structure(binary_model_path):
    forest = parse_model(binary_model_path)
    tree = forest.tree[0]

    root = tree.node[0]
    assert not root.is_leaf
    assert root.feature_index is not None
    assert root.threshold is not None
    assert root.left is not None
    assert root.right is not None


def test_parse_leaf_node(binary_model_path):
    forest = parse_model(binary_model_path)

    found_leaf = False
    for tree in forest.tree:
        for node in tree.node:
            if node.is_leaf:
                assert node.value is not None
                assert node.feature_index is None
                assert node.left is None
                assert node.right is None
                found_leaf = True
                break
        if found_leaf:
            break

    assert found_leaf, "No leaf nodes found"


def test_parse_base_score_scientific_notation(tmp_path):
    """Verify scientific notation like '5E-1' parses correctly."""
    model_json = {
        "learner": {
            "learner_model_param": {
                "base_score": "5E-1",
                "num_feature": "2",
            },
            "objective": {"name": "reg:squarederror"},
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "trees": [
                        {
                            "left_children": [-1],
                            "right_children": [-1],
                            "split_indices": [0],
                            "split_conditions": [0.0],
                            "base_weights": [0.1],
                        }
                    ],
                },
            },
        }
    }

    path = tmp_path / "sci_model.json"
    path.write_text(json.dumps(model_json))

    forest = parse_model(path)
    assert abs(forest.base_score - 0.5) < 1e-9


def test_parse_stump_tree(tmp_path):
    """Single-node tree (leaf only)."""
    model_json = {
        "learner": {
            "learner_model_param": {
                "base_score": "0.5",
                "num_feature": "2",
            },
            "objective": {"name": "reg:squarederror"},
            "gradient_booster": {
                "name": "gbtree",
                "model": {
                    "trees": [
                        {
                            "left_children": [-1],
                            "right_children": [-1],
                            "split_indices": [0],
                            "split_conditions": [0.0],
                            "base_weights": [0.25],
                        }
                    ],
                },
            },
        }
    }

    path = tmp_path / "stump_model.json"
    path.write_text(json.dumps(model_json))

    forest = parse_model(path)
    assert len(forest.tree) == 1
    assert len(forest.tree[0].node) == 1
    assert forest.tree[0].node[0].is_leaf
    assert abs(forest.tree[0].node[0].value - 0.25) < 1e-9


def test_parse_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"not": "xgboost"}')

    with pytest.raises(ValueError, match="missing 'learner'"):
        parse_model(path)


def test_parse_no_trees(tmp_path):
    model_json = {
        "learner": {
            "learner_model_param": {"base_score": "0.5", "num_feature": "2"},
            "objective": {"name": "reg:squarederror"},
            "gradient_booster": {
                "name": "gbtree",
                "model": {"trees": []},
            },
        }
    }

    path = tmp_path / "empty.json"
    path.write_text(json.dumps(model_json))

    with pytest.raises(ValueError, match="no trees"):
        parse_model(path)
