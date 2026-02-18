from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from qsf.api.main import app

client = TestClient(app)

# Shared mock data
MOCK_NEWS = [
    {"text": "IonQ announces breakthrough", "date": "2025-01-10", "source": "news"},
    {"text": "Quantum computing stocks rise", "date": "2025-01-11", "source": "news"},
]
MOCK_REDDIT = [
    {"text": "IONQ looking bullish", "date": "2025-01-10", "source": "social"},
]
MOCK_SCORES = [0.8, 0.6, 0.7]
MOCK_HIST = pd.DataFrame(
    {
        "Close": [10.0, 10.5, 11.0],
        "Volume": [1000000, 1200000, 1100000],
    },
    index=pd.to_datetime(["2025-01-10", "2025-01-11", "2025-01-12"]),
)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("qsf.api.main._score_sentiment", return_value=MOCK_SCORES)
@patch("qsf.api.main._get_reddit", return_value=MOCK_REDDIT)
@patch("qsf.api.main._get_news", return_value=MOCK_NEWS)
@patch("yfinance.Ticker")
def test_analyze_valid_ticker(mock_yf, mock_news, mock_reddit, mock_score):
    mock_yf.return_value.history.return_value = MOCK_HIST

    response = client.post("/analyze", json={"ticker": "IONQ"})
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "IONQ"
    assert "sentiment_score" in data
    assert "breakdown" in data
    assert "daily_data" in data
    assert len(data["daily_data"]) == 3
    assert data["breakdown"]["trend"] in ("increasing", "decreasing", "stable")


@patch("qsf.api.main._score_sentiment", return_value=[])
@patch("qsf.api.main._get_reddit", return_value=[])
@patch("qsf.api.main._get_news", return_value=[])
@patch("yfinance.Ticker")
def test_analyze_no_text_data(mock_yf, mock_news, mock_reddit, mock_score):
    mock_yf.return_value.history.return_value = MOCK_HIST

    response = client.post("/analyze", json={"ticker": "IONQ"})
    assert response.status_code == 404


@patch("yfinance.Ticker")
def test_analyze_invalid_ticker(mock_yf):
    mock_yf.return_value.history.return_value = pd.DataFrame()

    response = client.post("/analyze", json={"ticker": "FAKE123XYZ"})
    assert response.status_code == 404


def test_analyze_missing_ticker():
    response = client.post("/analyze", json={})
    assert response.status_code == 422
