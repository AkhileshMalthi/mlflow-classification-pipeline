"""
Model inference API using FastAPI.

This module provides a REST API for serving predictions from MLflow-registered models.
The API loads the production model and scaler at startup and exposes endpoints for
health checks and predictions.

Endpoints:
    GET /health: Health check endpoint
    POST /predict: Prediction endpoint accepting feature arrays

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow tracking server URL (default: http://mlflow_server:5000)
    REGISTERED_MODEL_NAME: Name of the registered model (default: ClassificationModel)
    MODEL_STAGE: Model stage to load (default: Production)
"""

import os
from contextlib import asynccontextmanager

import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "ClassificationModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

model = None
scaler = None


class Features(BaseModel):
    """Request model for prediction endpoint."""

    features: list


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for loading model and scaler at startup.

    Loads the production model and preprocessing scaler from MLflow when the
    application starts, avoiding repeated loading on each prediction request.
    """
    global model, scaler
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"

    # Load the sklearn model directly instead of via pyfunc wrapper
    model = mlflow.sklearn.load_model(model_uri)

    # Get the run ID from the model registry to download artifacts
    client = mlflow.MlflowClient()
    model_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[MODEL_STAGE])[0]
    run_id = model_version.run_id

    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessing_artifacts/scaler.pkl"
    )
    scaler = joblib.load(scaler_path)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
        dict: Status message indicating API is operational.

    Example:
        >>> GET /health
        {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Features):
    """
    Prediction endpoint for classification inference.

    Accepts feature arrays, applies preprocessing using the loaded scaler,
    and returns both predicted class labels and probability distributions.

    Args:
        payload (Features): JSON payload containing:
            - features (list): Array of feature values matching model input schema

    Returns:
        dict: Prediction results containing:
            - prediction (list): Predicted class labels
            - probabilities (list): Class probability distributions

    Raises:
        HTTPException (503): If model or scaler failed to load at startup
        HTTPException (400): If input features are invalid or prediction fails

    Example:
        >>> POST /predict
        >>> {"features": [5.1, 3.5, 1.4, 0.2]}
        {"prediction": [0], "probabilities": [[0.9127, 0.0583, 0.0290]]}
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")
    try:
        X = pd.DataFrame([payload.features])  # noqa: N806
        X_scaled = scaler.transform(X)  # noqa: N806
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)

        return {"prediction": preds.tolist(), "probabilities": probs.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
