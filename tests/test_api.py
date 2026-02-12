# Write unit tests for inference_api.py functions using pytest, covering:
# Successful prediction with valid input.
# Error handling for invalid JSON payload and missing features.
# Health check endpoint.

import pytest
from fastapi.testclient import TestClient
from src.inference_api import app

@pytest.fixture
def client():
    return TestClient(app)

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
    assert isinstance(response.json()["prediction"], list)

def test_predict_missing_features(client):
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_invalid_json(client):
    response = client.post("/predict", data="not a json")
    assert response.status_code == 422  # Unprocessable Entity
