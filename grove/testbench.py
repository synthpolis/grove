"""Generate Verilog testbench for verifying compiled forests."""

from __future__ import annotations

from .ir import QuantForest


def emit_testbench(
    forest: QuantForest,
    sample: list[list[int]],
    expected: list[int],
    module_name: str = "grove_forest",
) -> str:
    """Generate a Verilog testbench that drives samples and checks outputs.

    Args:
        forest: Quantized forest (for dimension info).
        sample: List of quantized feature vectors (list of int per sample).
        expected: Expected raw integer output per sample.
        module_name: Name of the DUT module.

    Returns:
        Verilog testbench source code.
    """
    total_width = forest.num_feature * forest.feature_width
    ow = forest.output_width
    fw = forest.feature_width
    n_feat = forest.num_feature

    lines = []
    lines.append("`timescale 1ns / 1ps")
    lines.append("")
    lines.append("module tb;")
    lines.append(f"    reg [{total_width - 1}:0] feature_pack;")
    lines.append(f"    wire signed [{ow - 1}:0] prediction;")
    lines.append("")
    lines.append(f"    {module_name} dut (")
    lines.append("        .feature_pack(feature_pack),")
    lines.append("        .prediction(prediction)")
    lines.append("    );")
    lines.append("")
    lines.append("    integer pass_count;")
    lines.append("    integer fail_count;")
    lines.append("")
    lines.append("    initial begin")
    lines.append("        pass_count = 0;")
    lines.append("        fail_count = 0;")

    for i, (feat, exp) in enumerate(zip(sample, expected)):
        packed = _pack_feature(feat, fw, n_feat)
        lines.append("")
        lines.append(f"        // Sample {i}")
        lines.append(f"        feature_pack = {total_width}'d{packed};")
        lines.append("        #10;")
        if exp < 0:
            exp_lit = f"(-{ow}'sd{abs(exp)})"
        else:
            exp_lit = f"{ow}'sd{exp}"
        lines.append(
            f"        if (prediction === {exp_lit}) begin"
        )
        lines.append(
            f'            $display("PASS sample {i}: %d", prediction);'
        )
        lines.append("            pass_count = pass_count + 1;")
        lines.append("        end else begin")
        lines.append(
            f'            $display("FAIL sample {i}: '
            f'expected {exp}, got %d", prediction);'
        )
        lines.append("            fail_count = fail_count + 1;")
        lines.append("        end")

    lines.append("")
    lines.append(
        '        $display("%0d passed, %0d failed", pass_count, fail_count);'
    )
    lines.append("        $finish;")
    lines.append("    end")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines) + "\n"


def _pack_feature(
    feature_int: list[int],
    feature_width: int,
    num_feature: int,
) -> int:
    """Pack quantized features into a single integer for the Verilog bus."""
    packed = 0
    mask = (1 << feature_width) - 1
    for i in range(num_feature):
        val = feature_int[i] & mask
        packed |= val << (i * feature_width)
    return packed
