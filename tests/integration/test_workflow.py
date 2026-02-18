"""
Integration tests for the LangGraph pipeline.
All external API calls are mocked. These tests verify that nodes are wired
correctly and state flows through the full graph as expected.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from qsf.agents.workflow import pipeline

MOCK_HIST = pd.DataFrame(
    {"Close": [10.0, 10.5, 11.0], "Volume": [1000000, 1200000, 1100000]},
    index=pd.to_datetime(["2026-02-10", "2026-02-11", "2026-02-12"]),
)
MOCK_NEWS = [
    {"text": "IonQ beats earnings", "date": "2026-02-10", "source": "news"},
    {"text": "Quantum stocks surge", "date": "2026-02-11", "source": "news"},
]
MOCK_REDDIT = [
    {"text": "IONQ looking bullish", "date": "2026-02-10", "source": "social"},
]
MOCK_SCORES = [0.8, 0.6, 0.7]


@patch("qsf.agents.nodes._score_sentiment", return_value=MOCK_SCORES)
@patch("qsf.agents.nodes._get_reddit", return_value=MOCK_REDDIT)
@patch("qsf.agents.nodes._get_news", return_value=MOCK_NEWS)
@patch("qsf.agents.nodes.yf.Ticker")
def test_happy_path_produces_result(mock_ticker, mock_news, mock_reddit, mock_score):
    mock_ticker.return_value.history.return_value = MOCK_HIST

    state = pipeline.invoke({"ticker": "IONQ"})

    assert state.get("error") is None
    assert state.get("result") is not None
    assert state["result"]["ticker"] == "IONQ"
    assert len(state["result"]["daily_data"]) == 3
    assert state["result"]["sentiment_score"] is not None


@patch("qsf.agents.nodes.yf.Ticker")
def test_invalid_ticker_sets_error(mock_ticker):
    mock_ticker.return_value.history.return_value = pd.DataFrame()

    state = pipeline.invoke({"ticker": "FAKE"})

    assert "error" in state
    assert state.get("result") is None


@patch("qsf.agents.nodes._score_sentiment", return_value=[])
@patch("qsf.agents.nodes._get_reddit", return_value=[])
@patch("qsf.agents.nodes._get_news", return_value=[])
@patch("qsf.agents.nodes.yf.Ticker")
def test_no_text_data_sets_error(mock_ticker, mock_news, mock_reddit, mock_score):
    mock_ticker.return_value.history.return_value = MOCK_HIST

    state = pipeline.invoke({"ticker": "IONQ"})

    assert "error" in state
    assert state.get("result") is None


@patch("qsf.agents.nodes._score_sentiment", return_value=MOCK_SCORES)
@patch("qsf.agents.nodes._get_reddit", return_value=MOCK_REDDIT)
@patch("qsf.agents.nodes._get_news", return_value=MOCK_NEWS)
@patch("qsf.agents.nodes.yf.Ticker")
def test_ticker_is_available_throughout_pipeline(mock_ticker, mock_news, mock_reddit, mock_score):
    """Verify ticker is preserved in state from start to finish."""
    mock_ticker.return_value.history.return_value = MOCK_HIST

    state = pipeline.invoke({"ticker": "IONQ"})

    assert state["ticker"] == "IONQ"
    mock_news.assert_called_once_with("IONQ")
    mock_reddit.assert_called_once_with("IONQ")
