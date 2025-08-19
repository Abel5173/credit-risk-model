from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictRequest, PredictResponse
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load model from MLflow registry (update model name and stage as needed)
MODEL_NAME = "CreditRiskModel_LogisticRegression"
MODEL_STAGE = "Production"

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    # Convert request to DataFrame
    input_df = pd.DataFrame([request.dict()])
    try:
        pred_proba = model.predict(input_df)
        # If model returns probability, use it; else, set to 0/1
        if hasattr(model, 'predict_proba'):
            risk_prob = float(model.predict_proba(input_df)[:, 1][0])
        else:
            risk_prob = float(pred_proba[0])
        is_high_risk = int(risk_prob >= 0.5)
        return PredictResponse(
            risk_probability=risk_prob, is_high_risk=is_high_risk
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
