# Write unit tests for inference_api.py functions using pytest, covering:
# Successful prediction with valid input.
# Error handling for invalid JSON payload and missing features.
# Health check endpoint.

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import numpy as np

@pytest.fixture
def mock_model():
    """Mock MLflow model"""
    model = Mock()
    model.predict = Mock(return_value=np.array([0]))
    model.predict_proba = Mock(return_value=np.array([[0.9, 0.05, 0.05]]))
    return model

@pytest.fixture
def mock_scaler():
    """Mock StandardScaler"""
    scaler = Mock()
    scaler.transform = Mock(return_value=np.array([[5.1, 3.5, 1.4, 0.2]]))
    return scaler

@pytest.fixture
def client(mock_model, mock_scaler):
    """Create test client with mocked dependencies"""
    with patch("src.inference_api.mlflow.pyfunc.load_model", return_value=mock_model), \
         patch("src.inference_api.mlflow.artifacts.download_artifacts", return_value="dummy_path"), \
         patch("src.inference_api.joblib.load", return_value=mock_scaler):
        from src.inference_api import app
        with TestClient(app) as test_client:
            yield test_client

def test_health_endpoint_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_input(client):
    # Example features, adjust length/type to match your model
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probabilities" in response.json()
    assert isinstance(response.json()["prediction"], list)
    assert isinstance(response.json()["probabilities"], list)

def test_predict_missing_features(client):
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_invalid_json(client):
    response = client.post("/predict", data="not a json")
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_invalid_feature_type(client):
    """Test with invalid feature types"""
    payload = {"features": ["invalid", "data"]}
    response = client.post("/predict", json=payload)
    # Should return 400 due to transformation error
    assert response.status_code in [400, 422]

def test_predict_empty_features(client):
    """Test with empty features list"""
    payload = {"features": []}
    response = client.post("/predict", json=payload)
    # Should fail due to shape mismatch
    assert response.status_code in [400, 422]
