# Grove

Compile XGBoost models into synthesizable Verilog HDL. Every decision tree becomes a combinational circuit. All trees evaluate in parallel. One clock cycle per prediction.

## How it works

```
model.json --> grove --> Verilog HDL --> FPGA bitstream
```

1. **Parse** - Read XGBoost model (JSON format)
2. **Quantize** - Convert float thresholds to fixed-point integers
3. **Emit** - Generate Verilog where each tree is a hardwired comparator chain

Each tree becomes purely combinational logic (no clock needed). The forest module instantiates all trees in parallel and sums the outputs. At 100 MHz on a modest FPGA, that's 100 million predictions per second.

## Install

```bash
pip install git+https://github.com/synthpolis/grove.git
```

For development:

```bash
git clone https://github.com/synthpolis/grove.git
cd grove
pip install -e ".[dev]"
```

## Quick start

### 1. Train and save a model

```python
import xgboost as xgb
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = xgb.XGBClassifier(n_estimators=10, max_depth=4)
model.fit(X, (y == 2).astype(int))
model.save_model("model.json")
```

### 2. Compile to Verilog

**CLI:**

```bash
grove compile model.json -o output/
```

**Python API:**

```python
import grove

result = grove.compile(
    model_path="model.json",
    output_dir="output/",
    feature_min=[4.3, 2.0, 1.0, 0.1],  # from training data
    feature_max=[7.9, 4.4, 6.9, 2.5],
)
```

### 3. Use the generated Verilog

```
output/
  tree_0.v          # First tree (combinational logic)
  tree_1.v          # Second tree
  ...
  grove_forest.v    # Top module (all trees + sum)
  tb.v              # Testbench (optional, with --testbench flag)
```

Load these into your FPGA toolchain (Vivado, Quartus, etc.) or simulate with Icarus Verilog:

```bash
cd output/
iverilog -o sim tree_*.v grove_forest.v tb.v && vvp sim
```

## CLI reference

```bash
# Basic compilation
grove compile model.json -o output/

# Custom bit widths
grove compile model.json -o output/ --feature-width 16 --output-width 32 --fractional-bit 10

# Custom module name
grove compile model.json -o output/ --module-name my_predictor

# Generate testbench
grove compile model.json -o output/ --testbench

# Provide feature ranges from a JSON file
grove compile model.json -o output/ --feature-range range.json
```

Feature range JSON format:

```json
{
    "feature_min": [4.3, 2.0, 1.0, 0.1],
    "feature_max": [7.9, 4.4, 6.9, 2.5]
}
```

## Python API reference

```python
import grove

# Full pipeline: parse, quantize, emit
result = grove.compile(
    model_path="model.json",
    output_dir="output/",           # optional - omit to just get the dict
    feature_min=[4.3, 2.0, 1.0],   # from X_train.min(axis=0)
    feature_max=[7.9, 4.4, 6.9],   # from X_train.max(axis=0)
    feature_width=16,               # bits per feature (default: 16)
    output_width=32,                # bits for accumulator (default: 32)
    fractional_bit=10,              # fractional bits for leaf values (default: 10)
    module_name="grove_forest",     # top module name (default: grove_forest)
    testbench=True,                 # generate testbench (default: False)
    testbench_sample=X[:10].tolist(),  # samples for testbench
)
# result is a dict: {"tree_0.v": "...", "grove_forest.v": "...", ...}

# Step-by-step for more control
forest = grove.parse_model("model.json")
config = grove.QuantConfig(
    feature_width=16,
    output_width=32,
    fractional_bit=10,
    feature_min=X.min(axis=0).tolist(),
    feature_max=X.max(axis=0).tolist(),
)
qf = grove.quantize_forest(forest, config)
```

## Verification

Grove includes a Python simulator that executes the quantized trees using integer arithmetic - exactly matching what the Verilog hardware computes.

```python
import grove
from grove.quantize import quantize_feature, dequantize_output

# Compile
forest = grove.parse_model("model.json")
config = grove.QuantConfig(
    feature_width=16,
    fractional_bit=10,
    feature_min=X.min(axis=0).tolist(),
    feature_max=X.max(axis=0).tolist(),
)
qf = grove.quantize_forest(forest, config)

# Simulate one sample
sample = X[0]
q_feat = [quantize_feature(float(sample[j]), j, config) for j in range(len(sample))]
result_int = grove.simulate_quantized_forest(qf, q_feat)
result_float = dequantize_output(result_int, config)

print(f"Hardware prediction: {result_float}")
```

Tested on Iris (150 samples, 4 features) and Breast Cancer (569 samples, 30 features) with 100% classification match against XGBoost float predictions. Typical quantization error is < 0.005.

## What the Verilog looks like

Each tree is a single combinational `assign` with nested ternary operators:

```verilog
module tree_0 (
    input wire [63:0] feature_pack,
    output wire signed [31:0] prediction
);
    wire [15:0] f0 = feature_pack[15:0];
    wire [15:0] f1 = feature_pack[31:16];
    wire [15:0] f2 = feature_pack[47:32];
    wire [15:0] f3 = feature_pack[63:48];

    assign prediction =
        (f2 < 16'd627) ?
            (-32'sd25) :
            (f3 < 16'd448) ?
                (32'sd24) :
                (-32'sd22);
endmodule
```

The forest top module instantiates all trees and sums:

```verilog
module grove_forest (
    input wire [63:0] feature_pack,
    output wire signed [31:0] prediction
);
    wire signed [31:0] t0_out, t1_out, t2_out;

    tree_0 t0 (.feature_pack(feature_pack), .prediction(t0_out));
    tree_1 t1 (.feature_pack(feature_pack), .prediction(t1_out));
    tree_2 t2 (.feature_pack(feature_pack), .prediction(t2_out));

    assign prediction = 32'sd512 + t0_out + t1_out + t2_out;
endmodule
```

## Quantization

Features and thresholds are quantized to fixed-point integers using per-feature linear scaling. Provide `feature_min` and `feature_max` from your training data for best accuracy. If not provided, ranges are inferred from the split thresholds in the model (less accurate).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_width` | 16 | Bits per feature |
| `output_width` | 32 | Bits for accumulator |
| `fractional_bit` | 10 | Fractional bits for leaf values |

For binary classification models, Grove automatically converts the stored probability base_score to log-odds, matching XGBoost's internal representation.

## Supported model type

- XGBoost binary classification (`binary:logistic`)
- XGBoost regression (`reg:squarederror`)
- XGBoost multiclass (`multi:softprob`) - generates trees for all classes

Models must be saved with `model.save_model("model.json")` (JSON format).

## FPGA resource estimate

| Trees | Depth | Feature count | LUT (approx) |
|-------|-------|---------------|---------------|
| 10 | 4 | 4 | ~2,500 |
| 50 | 6 | 10 | ~75,000 |
| 100 | 6 | 10 | ~150,000 |

Scales linearly with tree count. At 100 MHz, all configurations produce 100M prediction/sec.

## Accuracy

Tested against XGBoost float predictions:

| Dataset | Sample count | Feature count | Tree count | HW == XGB classification |
|---------|-------------|---------------|-----------|--------------------------|
| Iris | 150 | 4 | 10 | 100% |
| Breast Cancer | 569 | 30 | 20 | 100% |

Typical quantization error: < 0.005 absolute on raw margin output.

## License

MIT
