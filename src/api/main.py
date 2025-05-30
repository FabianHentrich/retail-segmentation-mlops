# src/api/main.py

from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from pathlib import Path

# Load trained model
# Assumes model was saved with: joblib.dump((model, feature_columns), ...)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "xgb_revenue.pkl"
model = joblib.load(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Revenue Prediction API",
    description="Predict customer revenue using RFM features and optional categorical variables (Region, Segment, Cluster).",
    version="1.0.0"
)


# Define input data schema using Pydantic
class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    Region: str = None
    Segment: str = None
    Cluster: int = None


@app.get("/")
def read_root():
    """
    Health check endpoint to confirm API is running.
    """
    return {"message": "Retail Prediction API is running!"}


@app.post("/predict")
def predict(features: CustomerFeatures):
    """
    Predict customer revenue based on input features.

    Input
    -----
    JSON object with the following fields:
      - Recency (float)
      - Frequency (float)
      - Monetary (float)
      - Region (str, optional)
      - Segment (str, optional)
      - Cluster (int, optional)

    Processing
    ----------
    - Builds feature vector.
    - Applies one-hot encoding to categorical variables if present.
    - Ensures feature alignment with model input.
    - Applies inverse log transformation to get revenue in original scale.

    Returns
    -------
    dict
        A dictionary with the predicted revenue.
    """
    # Convert input to DataFrame
    data = pd.DataFrame([features.dict()])

    # Split numerical and categorical columns
    numeric_cols = ["Recency", "Frequency", "Monetary"]
    cat_cols = [col for col in ["Region", "Segment", "Cluster"] if data[col].notna().all()]

    # Apply log transform (model expects log-scale)
    df_num = data[numeric_cols]
    df_cat = pd.get_dummies(data[cat_cols], drop_first=True) if cat_cols else pd.DataFrame()

    # Combine into final feature vector
    X = pd.concat([df_num, df_cat], axis=1)

    # Align columns with training features (zero-fill missing, re-order)
    for col in model.get_booster().feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[model.get_booster().feature_names]

    # Predict and inverse log1p to original scale
    prediction = np.expm1(model.predict(X)[0])

    return {"prediction": float(prediction)}
