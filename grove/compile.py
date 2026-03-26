"""Pipeline orchestrator: parse -> quantize -> emit Verilog."""

from __future__ import annotations

from pathlib import Path

from .emit import emit_forest
from .ir import QuantConfig, QuantForest
from .parse import parse_model
from .quantize import (
    dequantize_output,
    quantize_feature,
    quantize_forest,
)
from .testbench import emit_testbench


def compile(
    model_path: str | Path,
    output_dir: str | Path | None = None,
    feature_min: list[float] | None = None,
    feature_max: list[float] | None = None,
    feature_width: int = 16,
    output_width: int = 32,
    fractional_bit: int = 10,
    module_name: str = "grove_forest",
    testbench: bool = False,
    testbench_sample: list[list[float]] | None = None,
) -> dict[str, str]:
    """Full pipeline: parse -> quantize -> emit Verilog.

    Args:
        model_path: Path to XGBoost JSON model.
        output_dir: If provided, write .v files here.
        feature_min: Per-feature minimums for quantization.
        feature_max: Per-feature maximums for quantization.
        feature_width: Bit width per feature (default 16).
        output_width: Bit width for outputs (default 32).
        fractional_bit: Fractional bits for leaf values (default 10).
        module_name: Name of the top-level Verilog module.
        testbench: Whether to generate a testbench file.
        testbench_sample: Float feature vectors for testbench.

    Returns:
        Dict mapping filename to Verilog source string.
    """
    forest = parse_model(model_path)

    config = QuantConfig(
        feature_width=feature_width,
        output_width=output_width,
        fractional_bit=fractional_bit,
        feature_min=feature_min or [],
        feature_max=feature_max or [],
    )

    qf = quantize_forest(forest, config)
    result = emit_forest(qf, module_name)

    if testbench and testbench_sample:
        tb_quant, tb_expected = _prepare_testbench_data(
            qf, config, testbench_sample
        )
        tb_src = emit_testbench(qf, tb_quant, tb_expected, module_name)
        result["tb.v"] = tb_src

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for filename, source in result.items():
            (out / filename).write_text(source)

    return result


def _prepare_testbench_data(
    qf: QuantForest,
    config: QuantConfig,
    sample: list[list[float]],
) -> tuple[list[list[int]], list[int]]:
    """Quantize samples and compute expected outputs."""
    quant_sample = []
    expected = []

    for feat_vec in sample:
        q_vec = [
            quantize_feature(feat_vec[i], i, config)
            for i in range(qf.num_feature)
        ]
        quant_sample.append(q_vec)
        exp = simulate_quantized_forest(qf, q_vec)
        expected.append(exp)

    return quant_sample, expected


def simulate_quantized_forest(
    qf: QuantForest,
    feature_int: list[int],
) -> int:
    """Simulate the hardware behavior in Python for verification.

    Walks each tree using integer comparisons and sums the results.
    This produces the exact same output the Verilog would.
    """
    total = qf.base_score_int

    for tree in qf.tree:
        node_idx = 0
        while not tree.node[node_idx].is_leaf:
            node = tree.node[node_idx]
            feat_val = feature_int[node.feature_index]
            if feat_val < node.threshold_int:
                node_idx = node.left
            else:
                node_idx = node.right
        total += tree.node[node_idx].value_int

    return total
