from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from qsf.api.main import app, _forecast_cache

client = TestClient(app)

MOCK_FORECAST = {
    "ticker": "IONQ",
    "last_updated": "2026-02-17T00:00:00",
    "data_points": 504,
    "n_features": 18,
    "feature_names": ["ionq_rsi_14", "news_sentiment"],
    "models": [
        {"name": "LogisticRegression", "type": "classifier", "metrics": {}, "trading": {}},
    ],
    "best_model": {
        "name": "LogisticRegression",
        "sharpe": 1.2,
        "directional_accuracy": 0.55,
        "total_return_pct": 8.0,
        "max_drawdown_pct": -12.0,
    },
}


@pytest.fixture(autouse=True)
def clear_cache():
    _forecast_cache.clear()
    yield
    _forecast_cache.clear()


@patch("qsf.api.main.pipeline")
def test_forecast_valid_ticker(mock_pipeline):
    mock_pipeline.invoke.return_value = {"forecast": MOCK_FORECAST}

    response = client.post("/forecast", json={"ticker": "IONQ"})
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "IONQ"
    assert "best_model" in data
    assert data["best_model"]["name"] == "LogisticRegression"


@patch("qsf.api.main.pipeline")
def test_forecast_enables_forecast_flag(mock_pipeline):
    mock_pipeline.invoke.return_value = {"forecast": MOCK_FORECAST}

    client.post("/forecast", json={"ticker": "ionq"})
    call_args = mock_pipeline.invoke.call_args[0][0]
    assert call_args["ticker"] == "IONQ"
    assert call_args["forecast_enabled"] is True


@patch("qsf.api.main.pipeline")
def test_forecast_returns_404_on_error(mock_pipeline):
    mock_pipeline.invoke.return_value = {"error": "No market data found for 'FAKE123'"}

    response = client.post("/forecast", json={"ticker": "FAKE123"})
    assert response.status_code == 404
    assert "FAKE123" in response.json()["detail"]


def test_forecast_missing_ticker():
    response = client.post("/forecast", json={})
    assert response.status_code == 422


@patch("qsf.api.main.pipeline")
def test_forecast_is_cached_per_ticker(mock_pipeline):
    mock_pipeline.invoke.return_value = {"forecast": MOCK_FORECAST}

    client.post("/forecast", json={"ticker": "IONQ"})
    client.post("/forecast", json={"ticker": "IONQ"})
    # Second call should be served from cache, not recomputed.
    assert mock_pipeline.invoke.call_count == 1
