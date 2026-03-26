"""Tests for Verilog code generation."""

from __future__ import annotations

import re

from grove.ir import QuantForest, QuantNode, QuantTree
from grove.emit import emit_forest, emit_top, emit_tree


def _make_simple_quant_forest() -> QuantForest:
    """A 2-tree forest with 2 features for testing."""
    tree0 = QuantTree(
        tree_id=0,
        num_feature=2,
        node=[
            QuantNode(0, False, feature_index=0, threshold_int=500, left=1, right=2),
            QuantNode(1, True, value_int=100),
            QuantNode(2, True, value_int=-50),
        ],
    )
    tree1 = QuantTree(
        tree_id=1,
        num_feature=2,
        node=[
            QuantNode(0, False, feature_index=1, threshold_int=300, left=1, right=2),
            QuantNode(1, True, value_int=75),
            QuantNode(2, True, value_int=-25),
        ],
    )
    return QuantForest(
        tree=[tree0, tree1],
        num_feature=2,
        base_score_int=512,
        feature_width=16,
        output_width=32,
        fractional_bit=10,
    )


def test_emit_forest_file_count():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest)
    # 2 tree files + 1 top module
    assert len(result) == 3
    assert "tree_0.v" in result
    assert "tree_1.v" in result
    assert "grove_forest.v" in result


def test_emit_tree_module_name():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest)
    assert "module tree_0" in result["tree_0.v"]
    assert "module tree_1" in result["tree_1.v"]


def test_emit_tree_feature_extraction():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest)
    src = result["tree_0.v"]
    assert "wire [15:0] f0 = feature_pack[15:0];" in src
    assert "wire [15:0] f1 = feature_pack[31:16];" in src


def test_emit_tree_ternary_structure():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest)
    src = result["tree_0.v"]
    assert "f0 < 16'd500" in src
    assert "32'sd100" in src
    assert "(-32'sd50)" in src


def test_emit_top_instantiation():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest)
    src = result["grove_forest.v"]
    assert "tree_0 t0" in src
    assert "tree_1 t1" in src


def test_emit_top_sum():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest)
    src = result["grove_forest.v"]
    assert "32'sd512" in src
    assert "t0_out" in src
    assert "t1_out" in src


def test_emit_endmodule():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest)
    for filename, src in result.items():
        assert src.strip().endswith("endmodule"), f"{filename} missing endmodule"


def test_emit_custom_module_name():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest, module_name="my_xgb")
    assert "my_xgb.v" in result
    assert "module my_xgb" in result["my_xgb.v"]


def test_emit_tree_leaf_only():
    """Single leaf node tree."""
    tree = QuantTree(
        tree_id=0,
        num_feature=2,
        node=[QuantNode(0, True, value_int=42)],
    )
    src = emit_tree(tree, feature_width=16, output_width=32)
    assert "32'sd42" in src
    assert "module tree_0" in src


def test_emit_deeper_tree():
    """Depth-2 tree with 7 nodes."""
    tree = QuantTree(
        tree_id=0,
        num_feature=3,
        node=[
            QuantNode(0, False, feature_index=0, threshold_int=100, left=1, right=2),
            QuantNode(1, False, feature_index=1, threshold_int=200, left=3, right=4),
            QuantNode(2, False, feature_index=2, threshold_int=300, left=5, right=6),
            QuantNode(3, True, value_int=10),
            QuantNode(4, True, value_int=20),
            QuantNode(5, True, value_int=30),
            QuantNode(6, True, value_int=40),
        ],
    )
    src = emit_tree(tree, feature_width=16, output_width=32)

    assert "f0 < 16'd100" in src
    assert "f1 < 16'd200" in src
    assert "f2 < 16'd300" in src
    assert "32'sd10" in src
    assert "32'sd20" in src
    assert "32'sd30" in src
    assert "32'sd40" in src


def test_emit_verilog_balanced_parens():
    forest = _make_simple_quant_forest()
    result = emit_forest(forest)
    for filename, src in result.items():
        opens = src.count("(")
        closes = src.count(")")
        assert opens == closes, (
            f"{filename}: unbalanced parens ({opens} open, {closes} close)"
        )
