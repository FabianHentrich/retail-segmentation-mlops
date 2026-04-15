import numpy as np
import pandas as pd

from src.models.train_xgb import build_feature_matrix


def test_build_feature_matrix_numeric_only():
    df = pd.DataFrame({
        "Recency": [0, 9, 99],
        "Frequency": [1, 2, 3],
        "Monetary": [10.0, 20.0, 30.0],
    })

    X = build_feature_matrix(df)

    assert list(X.columns) == ["Recency", "Frequency", "Monetary"]
    assert len(X) == 3
    np.testing.assert_allclose(X["Recency"], np.log1p(df["Recency"]))


def test_build_feature_matrix_with_categoricals():
    df = pd.DataFrame({
        "Recency": [1, 2],
        "Frequency": [3, 4],
        "Monetary": [5.0, 6.0],
        "Region": ["EU", "US"],
        "Segment": ["A", "B"],
        "Cluster": [0, 1],
    })

    X = build_feature_matrix(df)

    assert {"Recency", "Frequency", "Monetary"}.issubset(X.columns)
    assert any(col.startswith("Region_") for col in X.columns)
    assert any(col.startswith("Segment_") for col in X.columns)
    assert len(X) == 2
