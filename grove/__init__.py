"""Grove - compile XGBoost models to synthesizable Verilog HDL."""

__version__ = "0.1.0"

from .compile import compile, simulate_quantized_forest
from .ir import Forest, QuantConfig, QuantForest
from .parse import parse_model
from .quantize import dequantize_output, quantize_feature, quantize_forest
