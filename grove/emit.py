"""Generate synthesizable Verilog from quantized decision tree ensembles."""

from __future__ import annotations

from .ir import QuantForest, QuantTree


def emit_forest(
    forest: QuantForest,
    module_name: str = "grove_forest",
) -> dict[str, str]:
    """Generate Verilog source files for the quantized forest.

    Returns:
        Dict mapping filename to Verilog source code.
    """
    result = {}

    for tree in forest.tree:
        filename = f"tree_{tree.tree_id}.v"
        source = emit_tree(tree, forest.feature_width, forest.output_width)
        result[filename] = source

    top_filename = f"{module_name}.v"
    result[top_filename] = emit_top(forest, module_name)

    return result


def emit_tree(
    tree: QuantTree,
    feature_width: int,
    output_width: int,
) -> str:
    """Generate Verilog for a single tree module."""
    total_width = tree.num_feature * feature_width
    lines = []

    lines.append(f"module tree_{tree.tree_id} (")
    lines.append(f"    input wire [{total_width - 1}:0] feature_pack,")
    lines.append(
        f"    output wire signed [{output_width - 1}:0] prediction"
    )
    lines.append(");")

    for i in range(tree.num_feature):
        lo = i * feature_width
        hi = lo + feature_width - 1
        lines.append(
            f"    wire [{feature_width - 1}:0] "
            f"f{i} = feature_pack[{hi}:{lo}];"
        )

    lines.append("")

    expr = _emit_node(tree, 0, feature_width, output_width)
    lines.append(f"    assign prediction =")
    lines.append(f"        {expr};")

    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines) + "\n"


def emit_top(
    forest: QuantForest,
    module_name: str,
) -> str:
    """Generate the forest top module."""
    total_width = forest.num_feature * forest.feature_width
    ow = forest.output_width
    n = len(forest.tree)

    lines = []
    lines.append(f"module {module_name} (")
    lines.append(f"    input wire [{total_width - 1}:0] feature_pack,")
    lines.append(f"    output wire signed [{ow - 1}:0] prediction")
    lines.append(");")

    for i in range(n):
        lines.append(
            f"    wire signed [{ow - 1}:0] t{i}_out;"
        )

    lines.append("")

    for i in range(n):
        lines.append(
            f"    tree_{i} t{i} ("
            f".feature_pack(feature_pack), "
            f".prediction(t{i}_out));"
        )

    lines.append("")

    bs = forest.base_score_int
    if bs < 0:
        parts = [f"(-{ow}'sd{abs(bs)})"]
    else:
        parts = [f"{ow}'sd{bs}"]
    for i in range(n):
        parts.append(f"t{i}_out")

    sum_expr = " + ".join(parts)
    lines.append(f"    assign prediction = {sum_expr};")

    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines) + "\n"


def _emit_node(
    tree: QuantTree,
    node_idx: int,
    feature_width: int,
    output_width: int,
    depth: int = 0,
) -> str:
    """Recursively emit a nested ternary expression for a tree node."""
    node = tree.node[node_idx]
    indent = "        " + "    " * depth

    if node.is_leaf:
        val = node.value_int
        if val < 0:
            return f"(-{output_width}'sd{abs(val)})"
        return f"{output_width}'sd{val}"

    feat = f"f{node.feature_index}"
    thresh = f"{feature_width}'d{node.threshold_int}"

    left_expr = _emit_node(
        tree, node.left, feature_width, output_width, depth + 1
    )
    right_expr = _emit_node(
        tree, node.right, feature_width, output_width, depth + 1
    )

    return (
        f"({feat} < {thresh}) ?\n"
        f"{indent}{left_expr} :\n"
        f"{indent}{right_expr}"
    )
