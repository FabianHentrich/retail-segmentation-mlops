# 🛍️ retail-segmentation-mlops  
**End-to-End ML Workflow for RFM-Based Revenue Prediction**

> This project demonstrates a complete machine learning workflow for customer segmentation and revenue prediction based on RFM features (Recency, Frequency, Monetary). The solution integrates model training, hyperparameter optimization, explainability, and deployment via API – structured for maintainability and production-readiness.

---

## 🚀 Project Highlights

| Component               | Technologies / Tools                                     | Description                                                                 |
|------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------|
| **EDA & RFM Clustering** | `pandas`, `seaborn`, `matplotlib`                        | Exploratory analysis and segmentation using RFM logic.                      |
| **Model Training**       | `XGBoost`, `Optuna`, `scikit-learn`, `joblib`            | Revenue prediction using tuned XGBoost with Optuna hyperparameter search.  |
| **Evaluation & Logging** | `matplotlib`, `logging`, `json`                          | Auto-generated reports: RMSE curves, overfitting checks, SHAP plots.       |
| **API Deployment**       | `FastAPI`, `pydantic`, `uvicorn`, `Docker`              | Ready-to-use REST API with real-time prediction endpoint.                  |
| **Model Explainability** | `SHAP`, `plotly`, `Optuna.visualization`                | Transparent model behavior via SHAP & parameter importance charts.         |

---

## 🧠 Use Case

This project simulates a real-world **retail analytics** scenario:

- **Input**: Customer-level RFM features and optional group identifiers (e.g., Region, Segment, Cluster).
- **Output**: Predicted customer revenue (log-transformed regression).
- **Objective**: Enable **targeted marketing** or **budget allocation** via revenue forecasts per segment.

---

## 📁 Repository Structure

```
retail-segmentation-mlops/
├── data/                 # Input & intermediate CSVs (not versioned)
├── models/               # Serialized trained models (.pkl)
├── reports/              # Evaluation metrics, SHAP plots, learning curves
├── src/
│   ├── api/              # FastAPI app for real-time prediction
│   └── models/           # Training pipeline incl. Optuna tuning & SHAP
├── notebooks/            # EDA & RFM clustering (Jupyter)
├── Dockerfile            # Containerization config
├── requirements.txt      # Dependency list
└── README.md             # 📄 You are here
```

---

## 📦 How to Run

### 1. Train the Model

```bash
python -m src.models.train_xgb \
    --input data/interim/rfm_clustered.csv \
    --target Monetary \
    --model-out models/xgb_revenue.pkl \
    --metrics-out reports/xgb_metrics.json \
    --n-trials 100 \
    --shap
```

### 2. Start the API

```bash
uvicorn src.api.main:app --reload
```

Then test via:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"Recency": 15, "Frequency": 8, "Monetary": 230, "Region": "West", "Segment": "Gold", "Cluster": 2}'
```

---

## 🐳 Docker Deployment

To run the API in a containerized environment:

### 1. Build the Docker image

```bash
docker build -t retail-api .
```

### 2. Run the container

```bash
docker run -p 8000:8000 retail-api
```

This starts the FastAPI app at `http://localhost:8000`.

---

### 📝 Notes

- The model file `models/xgb_revenue.pkl` must exist before building the image.
- If you're running training separately, make sure to **mount the model and data folders** or copy them into the image during build.
- Example Dockerfile assumes `uvicorn` is used as the API server:

```dockerfile
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- ✅ To access the interactive Swagger UI, open your browser at:  
  `http://localhost:8000/docs`

- ReDoc UI is also available at:  
  `http://localhost:8000/redoc`

---

### 🔁 Test the API in Docker

Once running, test it using:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"Recency": 15, "Frequency": 8, "Monetary": 230, "Region": "West", "Segment": "Gold", "Cluster": 2}'
```


### 📦 Optional: Mount model on container start

If you don't want to bake the model into the image:

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models retail-api
```

---

## 🛠️ Skills Demonstrated

✅ Clean machine learning pipeline architecture  
✅ Feature engineering & log transformation  
✅ Hyperparameter tuning with Optuna & TPE  
✅ Model generalization checks (under/overfitting diagnostics)  
✅ SHAP-based model interpretation  
✅ FastAPI integration & request validation with Pydantic  
✅ Container-ready deployment (Dockerfile included)

---

## 📊 Example Outputs

- `reports/xgb_metrics.json` – RMSE, MAE, R², over/underfitting flags  
- `reports/learning_curve.png` – Training vs. validation curve  
- `reports/xgb_metrics_shap.png` – Global SHAP summary plot  
- `reports/optuna_*.html` – Interactive optimization visuals

---

## 🧪 Dependencies

See [`requirements.txt`](requirements.txt) for full list (Python ≥ 3.8):

```bash
pip install -r requirements.txt
```

---

## 📌 Author Notes

This project was designed to reflect **real-world data science workflows** beyond notebooks:
- Emphasizes modularity, reproducibility, and automation.
- Focus on bridging model development and production via APIs.

Ideal for roles involving **machine learning engineering**, **model deployment**, or **data-driven product development**.

---

## ⚠️ Disclaimer

This project is for demonstration and educational purposes only.  
The revenue predictions and segmentation results are based on synthetic or pre-processed data and should **not** be used for real business decisions without proper validation in a production environment.

The author does not guarantee the accuracy or suitability of the model for any specific commercial use case.

