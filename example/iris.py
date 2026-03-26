"""Example: train XGBoost on Iris, compile to Verilog."""

import xgboost as xgb
from sklearn.datasets import load_iris

import grove


def main():
    # Train
    X, y = load_iris(return_X_y=True)
    y_bin = (y == 2).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=4,
        learning_rate=0.3,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X, y_bin)
    model.save_model("model.json")

    # Compile to Verilog
    result = grove.compile(
        model_path="model.json",
        output_dir="output",
        feature_min=X.min(axis=0).tolist(),
        feature_max=X.max(axis=0).tolist(),
        feature_width=16,
        output_width=32,
        fractional_bit=10,
        testbench=True,
        testbench_sample=X[:5].tolist(),
    )

    print(f"Generated {len(result)} files:")
    for name in sorted(result):
        print(f"  {name}")

    # Verify a prediction
    forest = grove.parse_model("model.json")
    config = grove.QuantConfig(
        feature_width=16,
        output_width=32,
        fractional_bit=10,
        feature_min=X.min(axis=0).tolist(),
        feature_max=X.max(axis=0).tolist(),
    )
    qf = grove.quantize_forest(forest, config)

    sample = X[0]
    q_feat = [
        grove.quantize_feature(float(sample[j]), j, config)
        for j in range(forest.num_feature)
    ]

    hw_result = grove.simulate_quantized_forest(qf, q_feat)
    hw_float = grove.dequantize_output(hw_result, config)

    raw_pred = model.get_booster().predict(
        xgb.DMatrix(X[:1]), output_margin=True
    )[0]

    print(f"\nSample 0 verification:")
    print(f"  XGBoost raw prediction: {raw_pred:.6f}")
    print(f"  Hardware prediction:    {hw_float:.6f}")
    print(f"  Error:                  {abs(hw_float - raw_pred):.6f}")


if __name__ == "__main__":
    main()
