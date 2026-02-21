from unittest.mock import patch

from fastapi.testclient import TestClient

from qsf.api.main import app

client = TestClient(app)

MOCK_RESULT = {
    "ticker": "IONQ",
    "last_updated": "2026-02-17T00:00:00",
    "sentiment_score": 0.5,
    "data_points": 10,
    "breakdown": {"news_sentiment": 0.6, "social_sentiment": 0.4, "trend": "stable"},
    "daily_data": [
        {"date": "2026-01-10", "close": 48.0, "volume": 1000000, "ror": None,
         "news_sentiment": 0.5, "social_sentiment": 0.4},
    ],
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("qsf.api.main.pipeline")
def test_analyze_valid_ticker(mock_pipeline):
    mock_pipeline.invoke.return_value = {"result": MOCK_RESULT}

    response = client.post("/analyze", json={"ticker": "IONQ"})
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "IONQ"
    assert "sentiment_score" in data
    assert "breakdown" in data
    assert "daily_data" in data
    assert data["breakdown"]["trend"] in ("increasing", "decreasing", "stable")


@patch("qsf.api.main.pipeline")
def test_analyze_returns_404_on_error(mock_pipeline):
    mock_pipeline.invoke.return_value = {"error": "No price data found for 'FAKE123'"}

    response = client.post("/analyze", json={"ticker": "FAKE123"})
    assert response.status_code == 404
    assert "FAKE123" in response.json()["detail"]


def test_analyze_missing_ticker():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


@patch("qsf.api.main.pipeline")
def test_analyze_ticker_is_uppercased(mock_pipeline):
    mock_pipeline.invoke.return_value = {"result": MOCK_RESULT}

    client.post("/analyze", json={"ticker": "ionq"})
    call_args = mock_pipeline.invoke.call_args[0][0]
    assert call_args["ticker"] == "IONQ"
