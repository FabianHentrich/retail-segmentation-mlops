from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Retail Prediction API is running!"}


def test_predict_returns_float():
    payload = {"Recency": 10.0, "Frequency": 5.0, "Monetary": 200.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
