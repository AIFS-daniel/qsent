"""
Integration tests for the /diagnostics/news-comparison endpoint.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from qsf.api.main import app

client = TestClient(app)

MOCK_COMPARISON_RESULT = {
    "tickers": ["IONQ"],
    "results": [
        {
            "ticker": "IONQ",
            "company": "IonQ",
            "providers": [
                {"name": "NewsAPI", "total": 20, "relevant": 15, "precision": 0.75, "free_tier": "1,000 req/day (30-day history)", "error": None},
                {"name": "Massive", "total": 10, "relevant": 8, "precision": 0.8, "free_tier": "5 req/min, unlimited/day (2yr history)", "error": None},
            ],
        }
    ],
}


@patch("qsf.api.main.run_news_comparison")
def test_valid_request_returns_200_with_correct_shape(mock_runner):
    mock_runner.return_value = MOCK_COMPARISON_RESULT
    response = client.post("/diagnostics/news-comparison", json={"tickers": ["IONQ"]})
    assert response.status_code == 200
    data = response.json()
    assert "tickers" in data
    assert "results" in data
    assert data["results"][0]["ticker"] == "IONQ"
    assert "providers" in data["results"][0]


@patch("qsf.api.main.run_news_comparison")
def test_tickers_are_uppercased(mock_runner):
    mock_runner.return_value = MOCK_COMPARISON_RESULT
    client.post("/diagnostics/news-comparison", json={"tickers": ["ionq", "form"]})
    call_args = mock_runner.call_args[0][0]
    assert "IONQ" in call_args
    assert "FORM" in call_args


@patch("qsf.api.main.run_news_comparison")
def test_whitespace_tickers_filtered_out(mock_runner):
    mock_runner.return_value = MOCK_COMPARISON_RESULT
    client.post("/diagnostics/news-comparison", json={"tickers": ["IONQ", "  ", "FORM"]})
    call_args = mock_runner.call_args[0][0]
    assert "  " not in call_args
    assert len(call_args) == 2


def test_all_empty_tickers_returns_422():
    response = client.post("/diagnostics/news-comparison", json={"tickers": ["  ", ""]})
    assert response.status_code == 422


def test_missing_tickers_field_returns_422():
    response = client.post("/diagnostics/news-comparison", json={})
    assert response.status_code == 422


@patch("qsf.api.main.run_news_comparison")
def test_provider_error_returns_200_with_error_in_payload(mock_runner):
    """Provider-level errors are captured in the payload, not as HTTP errors."""
    result_with_error = {
        "tickers": ["IONQ"],
        "results": [
            {
                "ticker": "IONQ",
                "company": "IonQ",
                "providers": [
                    {"name": "NewsAPI", "total": 0, "relevant": 0, "precision": 0.0,
                     "free_tier": "1,000 req/day (30-day history)", "error": "ValueError: NEWS_API_KEY not set"},
                ],
            }
        ],
    }
    mock_runner.return_value = result_with_error
    response = client.post("/diagnostics/news-comparison", json={"tickers": ["IONQ"]})
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["providers"][0]["error"] is not None
