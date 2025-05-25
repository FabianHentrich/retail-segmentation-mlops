"""train_xgb.py
================================================================================
Train an XGBoost regression model to predict customer revenue (Monetary) – or an
alternative target – from engineered features.

Run:
    python -m src.models.train_xgb \
        --input data/interim/rfm_clustered.csv \
        --target Monetary \
        --model-out models/xgb_revenue.pkl \
        --metrics-out reports/xgb_metrics.json

Arguments:
    --input        Path to the CSV with customer-level features.
    --target       Column to predict (`Monetary` or `Profit`).
    --test-size    Fraction for test split (default 0.3).
    --random-state Random seed (default 42).
    --model-out    Where to save trained model (.pkl).
    --metrics-out  Where to dump eval metrics (.json).
    --shap         If given, save SHAP summary plot PNG next to metrics.

Outputs:
    * Fitted XGBRegressor saved via joblib.
    * JSON file with MAE, RMSE, R^2.
    * (Optional) SHAP PNG for interpretability.

Note:
    - Expects that `input` CSV contains `Recency`, `Frequency`, `Monetary`,
      and optionally `Cluster`, categorical columns `Region`, `Segment`, ...
    - Numeric columns are kept raw (XGBoost handles scaling internally).
================================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import matplotlib as plt
import shap

# Optional SHAP import is heavy – load lazily
try:
    import shap  # noqa: F401
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select & one‑hot‑encode features for the model.

    Adjust this list as your feature set grows.
    """
    numeric_cols = ["Recency", "Frequency", "Monetary"]

    cat_cols = []
    for col in ["Region", "Segment", "Cluster"]:
        if col in df.columns:
            cat_cols.append(col)

    df_num = df[numeric_cols].copy()
    df_cat = pd.get_dummies(df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame()

    X = pd.concat([df_num, df_cat], axis=1)
    return X


def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> XGBRegressor:
    """Train a baseline XGBRegressor."""
    params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        reg_lambda=1.0,
        random_state=random_state,
    )
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model


def evaluate(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    preds = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGB revenue model")
    p.add_argument("--input", type=str, default="data/interim/rfm_clustered.csv")
    p.add_argument("--target", type=str, default="Monetary", choices=["Monetary", "Profit"])
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-out", type=str, default="models/xgb_revenue.pkl")
    p.add_argument("--metrics-out", type=str, default="reports/xgb_metrics.json")
    p.add_argument("--shap", action="store_true", help="Save SHAP summary plot next to metrics JSON")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in {args.input}")

    y = df[args.target]
    X = build_feature_matrix(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Train model
    model = train_model(X_train, y_train, random_state=args.random_state)

    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    print("Evaluation metrics:", metrics)

    # Persist outputs
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # SHAP summary plot (optional)
    if args.shap:
        if not _HAS_SHAP:
            raise ImportError("shap is not installed. Run 'pip install shap' and retry.")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False, plot_size=(10, 6))

        # Pfad: <metrics_stem>_shap.png  (reports/xgb_metrics_shap.png)
        metrics_path = Path(args.metrics_out)
        shap_path = metrics_path.with_name(metrics_path.stem + "_shap.png")

        plt.gcf().savefig(shap_path, bbox_inches="tight")
        print("SHAP plot saved to", shap_path)


if __name__ == "__main__":
    main()
