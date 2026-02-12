# Create a Flask or FastAPI application.
# Implement a /health endpoint to check API status.
# Implement a /predict endpoint that:
# Loads the registered model and scaler from MLflow (e.g., mlflow.pyfunc.load_model("models:/ClassificationModel/Production")). Model and scaler should be loaded only once at API startup.
# Expects a JSON payload with features.
# Preprocesses the input data using the loaded scaler.
# Makes predictions and returns them as JSON.
# Handles input validation and errors with appropriate HTTP status codes.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import joblib
import os
import pandas as pd
from contextlib import asynccontextmanager

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "ClassificationModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

model = None
scaler = None

class Features(BaseModel):
    features: list

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    scaler_path = mlflow.artifacts.download_artifacts(model_uri, artifact_path="scaler.pkl")
    scaler = joblib.load(scaler_path)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Features):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")
    try:
        X = pd.DataFrame([payload.features])
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        return {"prediction": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)