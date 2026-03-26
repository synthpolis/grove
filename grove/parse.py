"""Parse XGBoost JSON model files into Forest IR."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .ir import Forest, Node, Tree


def parse_model(path: str | Path) -> Forest:
    """Parse an XGBoost JSON model file into a Forest IR.

    Args:
        path: Path to model.json (from xgboost model.save_model()).

    Returns:
        Forest with all trees, base_score, and objective.

    Raises:
        ValueError: If the JSON structure is not a recognized XGBoost format.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if "learner" not in data:
        raise ValueError("Not a valid XGBoost JSON model: missing 'learner' key")

    learner = data["learner"]
    model_param = learner.get("learner_model_param", {})

    base_score_raw = float(model_param.get("base_score", "0.5"))
    num_feature = int(model_param.get("num_feature", "0"))

    objective_obj = learner.get("objective", {})
    objective = objective_obj.get("name", "reg:squarederror")

    # For classification, XGBoost stores base_score as probability
    # but uses log-odds internally for the raw margin.
    if "binary:logistic" in objective:
        p = max(1e-7, min(1 - 1e-7, base_score_raw))
        base_score = math.log(p / (1 - p))
    else:
        base_score = base_score_raw

    booster = learner.get("gradient_booster", {})
    booster_name = booster.get("name", "gbtree")
    if booster_name != "gbtree":
        raise ValueError(f"Unsupported booster type: {booster_name}")

    model = booster.get("model", {})
    raw_trees = model.get("trees", [])

    if not raw_trees:
        raise ValueError("Model contains no trees")

    trees = []
    for i, raw_tree in enumerate(raw_trees):
        tree = _parse_tree(i, raw_tree, num_feature)
        trees.append(tree)

    return Forest(
        tree=trees,
        num_feature=num_feature,
        base_score=base_score,
        objective=objective,
    )


def _parse_tree(tree_id: int, raw: dict, num_feature: int) -> Tree:
    """Parse a single tree from XGBoost JSON format."""
    left_children = raw.get("left_children", [])
    right_children = raw.get("right_children", [])
    split_indices = raw.get("split_indices", [])
    split_conditions = raw.get("split_conditions", [])
    base_weights = raw.get("base_weights", [])

    num_nodes = len(left_children)
    nodes = []

    for i in range(num_nodes):
        is_leaf = left_children[i] == -1

        if is_leaf:
            node = Node(
                index=i,
                is_leaf=True,
                value=base_weights[i],
            )
        else:
            node = Node(
                index=i,
                is_leaf=False,
                feature_index=split_indices[i],
                threshold=split_conditions[i],
                left=left_children[i],
                right=right_children[i],
            )

        nodes.append(node)

    return Tree(tree_id=tree_id, node=nodes, num_feature=num_feature)
