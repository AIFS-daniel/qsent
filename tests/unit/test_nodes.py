"""
Unit tests for individual LangGraph node functions.
Nodes are tested by passing mock state dicts directly — no LangGraph involved.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qsf.agents.nodes import (
    aggregate,
    fetch_market_data,
    fetch_news,
    fetch_reddit,
    score_sentiment,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_STOCK_DF = pd.DataFrame(
    {"Close": [10.0, 10.5, 11.0], "Volume": [1000000, 1200000, 1100000]},
    index=pd.to_datetime(["2026-02-10", "2026-02-11", "2026-02-12"]).date,
)
MOCK_STOCK_DF["ror"] = MOCK_STOCK_DF["Close"].pct_change() * 100

MOCK_NEWS = [
    {"text": "IonQ beats earnings", "date": "2026-02-10", "source": "news"},
    {"text": "Quantum stocks surge", "date": "2026-02-11", "source": "news"},
]
MOCK_REDDIT = [
    {"text": "IONQ looking bullish", "date": "2026-02-10", "source": "social"},
]
MOCK_SCORED = [
    {**item, "weighted_sentiment": score}
    for item, score in zip(
        MOCK_NEWS + MOCK_REDDIT, [0.8, 0.6, 0.7]
    )
]


# ---------------------------------------------------------------------------
# fetch_market_data
# ---------------------------------------------------------------------------

class TestFetchMarketData:
    @patch("qsf.agents.nodes.yf.Ticker")
    def test_returns_stock_df_on_success(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame(
            {"Close": [10.0], "Volume": [1000000]},
            index=pd.to_datetime(["2026-02-10"]),
        )
        result = fetch_market_data({"ticker": "IONQ"})
        assert "stock_df" in result
        assert "error" not in result
        assert "Close" in result["stock_df"].columns
        assert "ror" in result["stock_df"].columns

    @patch("qsf.agents.nodes.yf.Ticker")
    def test_returns_error_on_empty_history(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        result = fetch_market_data({"ticker": "FAKE"})
        assert "error" in result
        assert "FAKE" in result["error"]

    @patch("qsf.agents.nodes.yf.Ticker")
    def test_index_is_date_objects(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame(
            {"Close": [10.0], "Volume": [1000000]},
            index=pd.to_datetime(["2026-02-10"]),
        )
        result = fetch_market_data({"ticker": "IONQ"})
        import datetime as dt
        assert isinstance(result["stock_df"].index[0], dt.date)


# ---------------------------------------------------------------------------
# fetch_news
# ---------------------------------------------------------------------------

class TestFetchNews:
    @patch("qsf.agents.nodes._get_news", return_value=MOCK_NEWS)
    def test_returns_news_items(self, mock_get_news):
        result = fetch_news({"ticker": "IONQ"})
        assert result == {"news_items": MOCK_NEWS}
        mock_get_news.assert_called_once_with("IONQ")

    def test_skips_on_error(self):
        result = fetch_news({"ticker": "IONQ", "error": "upstream failure"})
        assert result == {}

    @patch("qsf.agents.nodes._get_news", return_value=[])
    def test_returns_empty_list_when_no_articles(self, _):
        result = fetch_news({"ticker": "IONQ"})
        assert result == {"news_items": []}


# ---------------------------------------------------------------------------
# fetch_reddit
# ---------------------------------------------------------------------------

class TestFetchReddit:
    @patch("qsf.agents.nodes._get_reddit", return_value=MOCK_REDDIT)
    def test_returns_reddit_items(self, mock_get_reddit):
        result = fetch_reddit({"ticker": "IONQ"})
        assert result == {"reddit_items": MOCK_REDDIT}
        mock_get_reddit.assert_called_once_with("IONQ")

    def test_skips_on_error(self):
        result = fetch_reddit({"ticker": "IONQ", "error": "upstream failure"})
        assert result == {}

    @patch("qsf.agents.nodes._get_reddit", return_value=[])
    def test_returns_empty_list_when_no_posts(self, _):
        result = fetch_reddit({"ticker": "IONQ"})
        assert result == {"reddit_items": []}


# ---------------------------------------------------------------------------
# score_sentiment
# ---------------------------------------------------------------------------

class TestScoreSentimentNode:
    @patch("qsf.agents.nodes._score_sentiment", return_value=[0.8, 0.6, 0.7])
    def test_scores_combined_items(self, mock_score):
        state = {"ticker": "IONQ", "news_items": MOCK_NEWS, "reddit_items": MOCK_REDDIT}
        result = score_sentiment(state)
        assert "scored_items" in result
        assert len(result["scored_items"]) == 3
        assert result["scored_items"][0]["weighted_sentiment"] == 0.8

    @patch("qsf.agents.nodes._score_sentiment", return_value=[0.8, 0.6, 0.7])
    def test_preserves_original_item_fields(self, mock_score):
        state = {"ticker": "IONQ", "news_items": MOCK_NEWS, "reddit_items": MOCK_REDDIT}
        result = score_sentiment(state)
        first = result["scored_items"][0]
        assert first["text"] == "IonQ beats earnings"
        assert first["source"] == "news"
        assert first["date"] == "2026-02-10"

    def test_returns_error_when_no_items(self):
        state = {"ticker": "IONQ", "news_items": [], "reddit_items": []}
        result = score_sentiment(state)
        assert "error" in result
        assert "IONQ" in result["error"]

    def test_skips_on_error(self):
        state = {"ticker": "IONQ", "error": "upstream failure",
                 "news_items": [], "reddit_items": []}
        result = score_sentiment(state)
        assert result == {}


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def _base_state(self):
        return {
            "ticker": "IONQ",
            "stock_df": MOCK_STOCK_DF.copy(),
            "scored_items": MOCK_SCORED,
        }

    def test_returns_result_with_expected_keys(self):
        result = aggregate(self._base_state())
        assert "result" in result
        r = result["result"]
        assert r["ticker"] == "IONQ"
        assert "sentiment_score" in r
        assert "breakdown" in r
        assert "daily_data" in r
        assert "last_updated" in r

    def test_daily_data_length_matches_stock_df(self):
        result = aggregate(self._base_state())
        assert len(result["result"]["daily_data"]) == len(MOCK_STOCK_DF)

    def test_daily_data_contains_expected_fields(self):
        result = aggregate(self._base_state())
        day = result["result"]["daily_data"][0]
        assert "date" in day
        assert "close" in day
        assert "volume" in day
        assert "ror" in day
        assert "news_sentiment" in day
        assert "social_sentiment" in day

    def test_trend_is_valid_value(self):
        result = aggregate(self._base_state())
        assert result["result"]["breakdown"]["trend"] in ("increasing", "decreasing", "stable")

    def test_skips_on_error(self):
        state = {**self._base_state(), "error": "upstream failure"}
        result = aggregate(state)
        assert result == {}

    def test_overall_sentiment_is_average_of_news_and_social(self):
        result = aggregate(self._base_state())
        r = result["result"]
        expected = round(
            (r["breakdown"]["news_sentiment"] + r["breakdown"]["social_sentiment"]) / 2, 4
        )
        assert r["sentiment_score"] == expected
