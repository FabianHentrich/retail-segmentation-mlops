"""
train_xgb.py – Train and evaluate an XGBoost regression model with Optuna optimization

This script trains an XGBoost regressor on customer-level RFM features to predict revenue (or another target),
using Optuna for automated hyperparameter tuning. It also supports SHAP analysis and exports interactive plots.

Usage:
    python -m src.models.train_xgb \
        --input data/interim/rfm_clustered.csv \
        --target Monetary \
        --model-out models/xgb_revenue.pkl \
        --metrics-out reports/xgb_metrics.json \
        --n-trials 50 \
        [--shap]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler
import optuna.visualization as vis
import matplotlib.pyplot as plt
import plotly.io as pio

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()  # Optional: auch Ausgabe im Terminal
    ]
)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a feature matrix from a given DataFrame by selecting RFM (Recency, Frequency, Monetary) features
    and optionally encoding categorical variables (Region, Segment, Cluster) if they are present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that contains raw customer-related features, including RFM and potentially categorical variables.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the final feature matrix `X`, where:
        - RFM features are included, with 'Recency' log-transformed.
        - Categorical variables are one-hot encoded (excluding the first category to avoid multicollinearity).
    """
    numeric_cols = ["Recency", "Frequency", "Monetary"]
    cat_cols = [col for col in ["Region", "Segment", "Cluster"] if col in df.columns]
    df_num = df[numeric_cols].copy()
    df_num["Recency"] = np.log1p(df_num["Recency"])
    df_cat = pd.get_dummies(df[cat_cols], drop_first=True) if cat_cols else pd.DataFrame()
    return pd.concat([df_num, df_cat], axis=1)


def train_model_with_optimized_params(X: pd.DataFrame, y: pd.Series, n_trials: int, valid_size: float,
                                      random_state: int) -> tuple[XGBRegressor, dict]:
    """
    Train an XGBoost regression model using Optuna to optimize hyperparameters.

    This function performs hyperparameter tuning using Optuna's TPE sampler.
    It evaluates each trial on a holdout validation set using RMSE as the metric.
    After optimization, it fits the final model on the full dataset using the best parameters found.
    Additionally, it generates and saves Optuna visualization reports.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_trials : int
        Number of optimization trials to run.
    valid_size : float
        Proportion of the dataset to use for validation (between 0 and 1).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    model : XGBRegressor
        Trained XGBoost regression model with optimized hyperparameters.
    best_params : dict
        Dictionary of the best hyperparameters found during optimization.
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna hyperparameter tuning.

        The function trains and evaluates a model using parameters suggested by Optuna,
        and returns the validation RMSE as the optimization target.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object for suggesting parameters.

        Returns
        -------
        float
            Root Mean Squared Error (RMSE) on the validation set.
        """
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "reg_lambda": trial.suggest_float("reg_lambda", 10.0, 30.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 10.0),
            "random_state": random_state,
            "objective": "reg:squarederror",
        }

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=random_state)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)

        return float(np.sqrt(mean_squared_error(y_valid, preds)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    logging.info(f"Best parameters found: {best_params}")

    model = XGBRegressor(**best_params)
    model.fit(X, y)

    # Save the study for later analysis
    logging.info("Saving Optuna visualizations...")
    Path("reports").mkdir(parents=True, exist_ok=True)
    pio.write_html(vis.plot_optimization_history(study), "reports/optuna_optimization_history.html")
    pio.write_html(vis.plot_param_importances(study), "reports/optuna_param_importance.html")
    pio.write_html(vis.plot_parallel_coordinate(study), "reports/optuna_parallel_coords.html")
    pio.write_html(vis.plot_contour(study), "reports/optuna_contour_plot.html")

    return model, study.best_params


def evaluate(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a regression model's performance on log training and log test data,
    and check for potential overfitting or underfitting based on error margins.

    Parameters
    ----------
    model : fitted regressor object
        A trained regression model that supports `.predict()`.

    X_train : pd.DataFrame
        Feature matrix for training data.
    y_train : pd.Series
        Target values for training data.
    X_test : pd.DataFrame
        Feature matrix for test data.
    y_test : pd.Series
        Target values for test data.

    Returns
    -------
    dict
        Dictionary containing:
        - Train and test metrics (MAE, RMSE, R²)
        - Flags for overfitting and underfitting.
    """

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_metrics = {
        "MAE": float(mean_absolute_error(y_train, y_train_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "R2": float(r2_score(y_train, y_train_pred)),
    }

    test_metrics = {
        "MAE": float(mean_absolute_error(y_test, y_test_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "R2": float(r2_score(y_test, y_test_pred)),
    }

    # Differences
    mae_gap = abs(train_metrics["MAE"] - test_metrics["MAE"])
    rmse_gap = abs(train_metrics["RMSE"] - test_metrics["RMSE"])
    r2_gap = abs(train_metrics["R2"] - test_metrics["R2"])

    # Thresholds
    overfit = (
            mae_gap > 0.05 and  # 5% error margin log scale
            rmse_gap > 0.10 and  # 10% error margin log scale
            r2_gap > 0.03  # 3 pp difference
    )

    underfit = (
            train_metrics["R2"] < 0.85 and
            test_metrics["R2"] < 0.85
    )

    # Logging
    logging.info(f"Train metrics: {train_metrics}")
    logging.info(f"Test  metrics: {test_metrics}")
    logging.info(f"Overfitting detected: {overfit}")
    logging.info(f"Underfitting detected: {underfit}")

    if overfit:
        logging.warning("⚠️ Potential overfitting detected – train performance significantly better than test.")
    elif underfit:
        logging.warning("⚠️ Potential underfitting – both train and test scores are low.")
    else:
        logging.info("✅ No overfitting or underfitting detected – model generalizes well.")

    return {
        "train": train_metrics,
        "test": test_metrics,
        "overfitting": overfit,
        "underfitting": underfit
    }


def plot_learning_curve(model, X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Plot a learning curve showing how training and validation RMSE evolve with increasing training size.

    This function uses 5-fold cross-validation to compute RMSE for different training set sizes
    and plots both training and validation performance to diagnose learning behavior.

    Parameters
    ----------
    model : XGBRegressor
        The model instance (must implement fit and predict) to evaluate.

    X : pd.DataFrame
        Full feature matrix containing training features.

    y : pd.Series
        Target variable.

    random_state : int, default=42
        Random seed for reproducibility in cross-validation splits.

    Saves
    -----
    - reports/learning_curve.png : PNG image of the learning curve plot.

    Notes
    -----
    - Uses negative RMSE (neg_root_mean_squared_error) as scoring, so results are negated before plotting.
    - Creates the 'reports/' folder if it doesn't exist.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mean = -np.mean(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Train RMSE", marker="o")
    plt.plot(train_sizes, val_mean, label="Validation RMSE", marker="s")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/learning_curve.png", bbox_inches="tight")
    logging.info("Learning curve saved to reports/learning_curve.png")


def main():
    """
    Entry point for training an XGBoost regression model with Optuna hyperparameter optimization.

    This script performs the following steps:
    1. Parses command-line arguments.
    2. Loads and preprocesses data.
    3. Builds a feature matrix and log-transforms the target.
    4. Splits the dataset into training and test sets.
    5. Optimizes hyperparameters using Optuna and trains the model.
    6. Evaluates model performance and detects overfitting/underfitting.
    7. Saves the trained model and metrics to disk.
    8. Optionally generates a SHAP feature importance summary plot.

    Command-Line Arguments
    ----------------------
    --input : str (required)
        Path to the CSV input file.
    --target : str (default="Monetary")
        Target column name for regression.
    --test-size : float (default=0.2)
        Proportion of the dataset to allocate to the test set.
    --optuna-valid-size : float (default=0.4)
        Validation set size for Optuna during hyperparameter tuning (from training split).
    --random-state : int (default=42)
        Random seed for reproducibility.
    --n-trials : int (default=200)
        Number of Optuna trials to run.
    --model-out : str (default="models/xgb_model.pkl")
        Path to save the trained model (includes column names).
    --metrics-out : str (default="reports/xgb_metrics.json")
        Path to save evaluation metrics as JSON.
    --shap : flag
        If set, generates a SHAP summary plot for model interpretability.

    Notes
    -----
    - Log-transform is applied to the target variable for variance stabilization.
    - Saves visualizations, metrics, and the final model in specified paths.
    - Requires SHAP if --shap is enabled.
    """
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--target", type=str, default="Monetary")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--optuna-valid-size", type=float, default=0.4,
                        help="Validation size for Optuna optimization split")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--model-out", type=str, default="models/xgb_model.pkl")
    parser.add_argument("--metrics-out", type=str, default="reports/xgb_metrics.json")
    parser.add_argument("--shap", action="store_true")
    args = parser.parse_args()

    logging.info("Starting XGBoost model training...")
    logging.info(f"Arguments: {args}")

    # Load data
    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not in dataframe.")

    # Target transformation
    y = np.log1p(df[args.target])  # Log-transform target to stabilize variance
    X = build_feature_matrix(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    logging.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # Train model with Optuna
    model, best_params = train_model_with_optimized_params(
        X_train, y_train,
        n_trials=args.n_trials,
        valid_size=args.optuna_valid_size,
        random_state=args.random_state
    )

    # Evaluation metrics
    plot_learning_curve(model, X_train, y_train, random_state=args.random_state)
    metrics = evaluate(model, X_train, y_train, X_test, y_test)

    # Save model and metrics
    logging.info(f"Final model parameters: {best_params}")
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, X_train.columns.tolist()), args.model_out)

    logging.info(f"Model saved to {args.model_out}")
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    # SHAP explainability (optional)
    if args.shap:
        if not _HAS_SHAP:
            logging.error("SHAP not available. Run 'pip install shap'")
            raise ImportError("Install SHAP: pip install shap")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, show=False, plot_size=(10, 6))
        plt.gcf().savefig(Path(args.metrics_out).with_name("xgb_metrics_shap.png"))
        logging.info("SHAP summary plot saved.")


if __name__ == "__main__":
    main()
