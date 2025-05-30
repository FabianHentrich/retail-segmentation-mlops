# src/api/main.py
from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
from pathlib import Path

# Load trained model
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "xgb_revenue.pkl"
model = joblib.load(MODEL_PATH)

# FastAPI instance
app = FastAPI(title="Customer Revenue Prediction API",
              description="Predict customer revenue using RFM features and optional categoricals.",
              version="1.0.0")

# Input schema
class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    Region: str = None
    Segment: str = None
    Cluster: int = None

@app.get("/")
def read_root():
    return {"message": "Retail Prediction API is running!"}

@app.post("/predict")
def predict(features: CustomerFeatures):
    # Build feature vector
    data = pd.DataFrame([features.dict()])
    numeric_cols = ["Recency", "Frequency", "Monetary"]
    cat_cols = [col for col in ["Region", "Segment", "Cluster"] if data[col].notna().all()]

    df_num = data[numeric_cols]
    df_cat = pd.get_dummies(data[cat_cols], drop_first=True) if cat_cols else pd.DataFrame()
    X = pd.concat([df_num, df_cat], axis=1)

    # Ensure same columns as training
    for col in model.get_booster().feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[model.get_booster().feature_names]

    prediction = model.predict(X)[0]
    return {"prediction": float(prediction)}

