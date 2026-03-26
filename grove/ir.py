"""Internal representation for parsed and quantized decision tree ensembles."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    """A single node in a decision tree (split or leaf)."""

    index: int
    is_leaf: bool
    feature_index: int | None = None
    threshold: float | None = None
    left: int | None = None
    right: int | None = None
    value: float | None = None


@dataclass
class Tree:
    """A single decision tree."""

    tree_id: int
    node: list[Node]
    num_feature: int


@dataclass
class Forest:
    """Complete parsed XGBoost model."""

    tree: list[Tree]
    num_feature: int
    base_score: float
    objective: str


@dataclass
class QuantConfig:
    """Configuration for fixed-point quantization."""

    feature_width: int = 16
    output_width: int = 32
    fractional_bit: int = 10
    feature_min: list[float] = field(default_factory=list)
    feature_max: list[float] = field(default_factory=list)


@dataclass
class QuantNode:
    """Quantized version of a node with integer values."""

    index: int
    is_leaf: bool
    feature_index: int | None = None
    threshold_int: int | None = None
    left: int | None = None
    right: int | None = None
    value_int: int | None = None


@dataclass
class QuantTree:
    """Quantized tree ready for Verilog emission."""

    tree_id: int
    node: list[QuantNode]
    num_feature: int


@dataclass
class QuantForest:
    """Quantized forest ready for Verilog emission."""

    tree: list[QuantTree]
    num_feature: int
    base_score_int: int
    feature_width: int
    output_width: int
    fractional_bit: int
