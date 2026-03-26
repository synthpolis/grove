"""Command-line interface for grove."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .compile import compile as grove_compile


@click.group()
@click.version_option(package_name="grove")
def main():
    """Grove - compile XGBoost models to Verilog."""


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output", "output_dir",
    type=click.Path(), default="output",
    help="Output directory for Verilog files.",
)
@click.option(
    "--feature-width", type=int, default=16,
    help="Bit width per feature (default: 16).",
)
@click.option(
    "--output-width", type=int, default=32,
    help="Bit width for accumulator (default: 32).",
)
@click.option(
    "--fractional-bit", type=int, default=10,
    help="Fractional bits for leaf values (default: 10).",
)
@click.option(
    "--module-name", default="grove_forest",
    help="Top-level Verilog module name.",
)
@click.option(
    "--testbench", is_flag=True,
    help="Generate a Verilog testbench.",
)
@click.option(
    "--feature-range", type=click.Path(exists=True), default=None,
    help="JSON file with feature_min and feature_max arrays.",
)
def compile(
    model_path,
    output_dir,
    feature_width,
    output_width,
    fractional_bit,
    module_name,
    testbench,
    feature_range,
):
    """Compile an XGBoost JSON model to Verilog HDL."""
    feature_min = None
    feature_max = None

    if feature_range:
        with open(feature_range) as f:
            ranges = json.load(f)
        feature_min = ranges.get("feature_min")
        feature_max = ranges.get("feature_max")

    result = grove_compile(
        model_path=model_path,
        output_dir=output_dir,
        feature_min=feature_min,
        feature_max=feature_max,
        feature_width=feature_width,
        output_width=output_width,
        fractional_bit=fractional_bit,
        module_name=module_name,
        testbench=testbench,
    )

    click.echo(f"Generated {len(result)} file(s) in {output_dir}/")
    for name in sorted(result.keys()):
        click.echo(f"  {name}")
