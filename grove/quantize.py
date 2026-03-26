"""Float-to-fixed-point quantization for decision tree ensembles."""

from __future__ import annotations

import math

from .ir import (
    Forest,
    Node,
    QuantConfig,
    QuantForest,
    QuantNode,
    QuantTree,
    Tree,
)


def quantize_forest(forest: Forest, config: QuantConfig) -> QuantForest:
    """Convert a float Forest to a QuantForest with integer values.

    Feature quantization (per-feature linear scaling):
        int_val = round((float_val - min) / (max - min) * (2^width - 1))

    Leaf value quantization (fixed-point):
        int_val = round(float_val * 2^fractional_bit)
    """
    if not config.feature_min or not config.feature_max:
        config.feature_min, config.feature_max = infer_feature_range(forest)

    quant_tree = []
    for tree in forest.tree:
        qt = _quantize_tree(tree, config)
        quant_tree.append(qt)

    base_score_int = quantize_leaf(forest.base_score, config)

    return QuantForest(
        tree=quant_tree,
        num_feature=forest.num_feature,
        base_score_int=base_score_int,
        feature_width=config.feature_width,
        output_width=config.output_width,
        fractional_bit=config.fractional_bit,
    )


def quantize_threshold(
    value: float,
    feature_index: int,
    config: QuantConfig,
) -> int:
    """Quantize a split threshold to the same integer space as features."""
    f_min = config.feature_min[feature_index]
    f_max = config.feature_max[feature_index]

    if f_max == f_min:
        return 0

    max_int = (1 << config.feature_width) - 1
    scaled = (value - f_min) / (f_max - f_min) * max_int
    return _clamp(round(scaled), 0, max_int)


def quantize_feature(
    value: float,
    feature_index: int,
    config: QuantConfig,
) -> int:
    """Quantize a feature value to fixed-point integer."""
    return quantize_threshold(value, feature_index, config)


def quantize_leaf(value: float, config: QuantConfig) -> int:
    """Quantize a leaf value to signed fixed-point integer."""
    return round(value * (1 << config.fractional_bit))


def dequantize_output(value_int: int, config: QuantConfig) -> float:
    """Convert a quantized output sum back to float."""
    return value_int / (1 << config.fractional_bit)


def infer_feature_range(
    forest: Forest,
    padding: float = 0.1,
) -> tuple[list[float], list[float]]:
    """Infer per-feature min/max from split thresholds in the forest.

    Adds padding (default 10%) to each side of the range.
    """
    f_min = [math.inf] * forest.num_feature
    f_max = [-math.inf] * forest.num_feature

    for tree in forest.tree:
        for node in tree.node:
            if not node.is_leaf and node.feature_index is not None:
                idx = node.feature_index
                val = node.threshold
                f_min[idx] = min(f_min[idx], val)
                f_max[idx] = max(f_max[idx], val)

    for i in range(forest.num_feature):
        if f_min[i] == math.inf:
            f_min[i] = 0.0
            f_max[i] = 1.0
        else:
            span = f_max[i] - f_min[i]
            if span == 0:
                span = abs(f_min[i]) if f_min[i] != 0 else 1.0
            pad = span * padding
            f_min[i] -= pad
            f_max[i] += pad

    return f_min, f_max


def _quantize_tree(tree: Tree, config: QuantConfig) -> QuantTree:
    """Quantize a single tree."""
    quant_node = []
    for node in tree.node:
        qn = _quantize_node(node, config)
        quant_node.append(qn)

    return QuantTree(
        tree_id=tree.tree_id,
        node=quant_node,
        num_feature=tree.num_feature,
    )


def _quantize_node(node: Node, config: QuantConfig) -> QuantNode:
    """Quantize a single node."""
    if node.is_leaf:
        return QuantNode(
            index=node.index,
            is_leaf=True,
            value_int=quantize_leaf(node.value, config),
        )

    return QuantNode(
        index=node.index,
        is_leaf=False,
        feature_index=node.feature_index,
        threshold_int=quantize_threshold(
            node.threshold, node.feature_index, config
        ),
        left=node.left,
        right=node.right,
    )


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))
