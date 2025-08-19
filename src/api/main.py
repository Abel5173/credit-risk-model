from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictRequest, PredictResponse
import mlflow
import mlflow.sklearn
import pandas as pd

app = FastAPI()

# Load model from MLflow registry (update model name and stage as needed)
MODEL_NAME = "CreditRiskModel_LogisticRegression"
MODEL_STAGE = "Production"

model = None


def load_model_from_registry():
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        # Load native sklearn pipeline so we can call predict_proba
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        model = None
        print(f"Failed to load model: {e}")


load_model_from_registry()


@app.get("/health")
def health():
    return {"model_loaded": model is not None, "model_name": MODEL_NAME, "stage": MODEL_STAGE}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    # Convert request to DataFrame
    input_df = pd.DataFrame([request.dict()])
    try:
        if hasattr(model, 'predict_proba'):
            risk_prob = float(model.predict_proba(input_df)[:, 1][0])
        else:
            # Fallback to predict; treat as probability directly if model outputs proba-like
            pred = model.predict(input_df)
            risk_prob = float(pred[0]) if hasattr(
                pred, "__len__") else float(pred)
        is_high_risk = int(risk_prob >= 0.5)
        return PredictResponse(
            risk_probability=risk_prob, is_high_risk=is_high_risk
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
