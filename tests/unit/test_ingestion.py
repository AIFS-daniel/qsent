"""
Unit tests for ingestion providers (YFinanceMarketData, NewsAPIProvider, RedditProvider).
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from qsf.ingestion.market import YFinanceMarketData
from qsf.ingestion.news import NewsAPIProvider
from qsf.ingestion.social import RedditProvider


# ---------------------------------------------------------------------------
# YFinanceMarketData.get_history()
# ---------------------------------------------------------------------------

class TestYFinanceMarketData:
    @patch("qsf.ingestion.market.yf.Ticker")
    def test_returns_dataframe_from_yfinance(self, mock_ticker):
        expected = pd.DataFrame(
            {"Close": [10.0], "Volume": [1000000]},
            index=pd.to_datetime(["2026-02-10"]),
        )
        mock_ticker.return_value.history.return_value = expected
        result = YFinanceMarketData().get_history("IONQ", "30d")
        assert result.equals(expected)

    @patch("qsf.ingestion.market.yf.Ticker")
    def test_passes_ticker_to_yfinance(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        YFinanceMarketData().get_history("IONQ", "30d")
        mock_ticker.assert_called_once_with("IONQ")

    @patch("qsf.ingestion.market.yf.Ticker")
    def test_passes_period_to_history(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        YFinanceMarketData().get_history("IONQ", "90d")
        mock_ticker.return_value.history.assert_called_once_with(period="90d")

    @patch("qsf.ingestion.market.yf.Ticker")
    def test_returns_empty_dataframe_for_unknown_ticker(self, mock_ticker):
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        result = YFinanceMarketData().get_history("FAKE", "30d")
        assert result.empty


# ---------------------------------------------------------------------------
# NewsAPIProvider.get_articles()
# ---------------------------------------------------------------------------

class TestNewsAPIProvider:
    @patch("qsf.ingestion.news.NewsApiClient")
    def test_returns_correctly_shaped_items(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {
            "articles": [
                {"title": "IonQ beats earnings", "publishedAt": "2026-02-10T10:00:00Z"},
                {"title": "Quantum stocks rise", "publishedAt": "2026-02-11T14:00:00Z"},
            ]
        }
        items = NewsAPIProvider().get_articles("IONQ")
        assert len(items) == 2
        assert items[0]["text"] == "IonQ beats earnings"
        assert items[0]["date"] == "2026-02-10"
        assert items[0]["source"] == "news"

    @patch("qsf.ingestion.news.NewsApiClient")
    def test_skips_articles_with_no_title(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {
            "articles": [
                {"title": None, "publishedAt": "2026-02-10T10:00:00Z"},
                {"title": "Valid article", "publishedAt": "2026-02-10T10:00:00Z"},
            ]
        }
        items = NewsAPIProvider().get_articles("IONQ")
        assert len(items) == 1
        assert items[0]["text"] == "Valid article"

    @patch("qsf.ingestion.news.NewsApiClient")
    def test_empty_response(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {"articles": []}
        items = NewsAPIProvider().get_articles("IONQ")
        assert items == []


# ---------------------------------------------------------------------------
# RedditProvider.get_posts()
# ---------------------------------------------------------------------------

class TestRedditProvider:
    def _make_post(self, title, days_ago):
        post = MagicMock()
        post.title = title
        post.created_utc = (datetime.now() - timedelta(days=days_ago)).timestamp()
        return post

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_returns_recent_posts(self, mock_reddit_class):
        mock_reddit_class.return_value.subreddit.return_value.search.return_value = [
            self._make_post("IONQ hits new high", days_ago=5),
            self._make_post("Quantum computing outlook", days_ago=10),
        ]
        items = RedditProvider().get_posts("IONQ")
        assert len(items) == 6  # 2 posts × 3 subreddits
        assert items[0]["source"] == "social"
        assert items[0]["text"] == "IONQ hits new high"

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_filters_out_old_posts(self, mock_reddit_class):
        mock_reddit_class.return_value.subreddit.return_value.search.return_value = [
            self._make_post("Old post", days_ago=31),
        ]
        items = RedditProvider().get_posts("IONQ")
        assert items == []

    @patch("qsf.ingestion.social.praw.Reddit")
    def test_searches_all_three_subreddits(self, mock_reddit_class):
        mock_subreddit = mock_reddit_class.return_value.subreddit.return_value
        mock_subreddit.search.return_value = [
            self._make_post("IONQ post", days_ago=3),
        ]
        RedditProvider().get_posts("IONQ")
        assert mock_reddit_class.return_value.subreddit.call_count == 3
