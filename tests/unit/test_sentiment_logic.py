"""
Unit tests for the core logic functions in the analyze endpoint.
All external API calls (Reddit, NewsAPI, HuggingFace) are mocked.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from qsf.agents.nodes import (
    SENTIMENT_MAP,
    _get_news,
    _get_reddit,
    _safe,
    _score_sentiment,
)


# ---------------------------------------------------------------------------
# _safe() - NaN/inf sanitizer
# ---------------------------------------------------------------------------

class TestSafe:
    def test_normal_float(self):
        assert _safe(0.12345) == 0.1235

    def test_rounding(self):
        assert _safe(0.123456789) == 0.1235

    def test_custom_decimals(self):
        assert _safe(1.23456, 2) == 1.23

    def test_nan_returns_none(self):
        import math
        assert _safe(float("nan")) is None

    def test_inf_returns_none(self):
        assert _safe(float("inf")) is None

    def test_negative_inf_returns_none(self):
        assert _safe(float("-inf")) is None

    def test_none_returns_none(self):
        assert _safe(None) is None

    def test_zero(self):
        assert _safe(0.0) == 0.0

    def test_negative(self):
        assert _safe(-0.75) == -0.75


# ---------------------------------------------------------------------------
# _score_sentiment() - HuggingFace FinBERT scoring
# ---------------------------------------------------------------------------

class TestScoreSentiment:
    def _make_hf_response(self, label, score):
        """Simulate HuggingFace response for one text: list of label dicts."""
        others = [l for l in SENTIMENT_MAP if l != label]
        remaining = (1 - score) / 2
        return [
            {"label": label, "score": score},
            {"label": others[0], "score": remaining},
            {"label": others[1], "score": remaining},
        ]

    @patch("qsf.agents.nodes.requests.post")
    def test_positive_sentiment(self, mock_post):
        mock_post.return_value.json.return_value = [
            self._make_hf_response("positive", 0.9)
        ]
        scores = _score_sentiment(["Great earnings report"])
        assert len(scores) == 1
        assert scores[0] == pytest.approx(0.9)

    @patch("qsf.agents.nodes.requests.post")
    def test_negative_sentiment(self, mock_post):
        mock_post.return_value.json.return_value = [
            self._make_hf_response("negative", 0.8)
        ]
        scores = _score_sentiment(["Company files for bankruptcy"])
        assert scores[0] == pytest.approx(-0.8)

    @patch("qsf.agents.nodes.requests.post")
    def test_neutral_sentiment(self, mock_post):
        mock_post.return_value.json.return_value = [
            self._make_hf_response("neutral", 0.7)
        ]
        scores = _score_sentiment(["Company releases quarterly report"])
        assert scores[0] == pytest.approx(0.0)

    @patch("qsf.agents.nodes.requests.post")
    def test_one_call_per_text(self, mock_post):
        """Each text must trigger exactly one API call and produce one score.

        This guards against batching bugs where the HF API only scores the
        first item in a batch, silently dropping the rest.
        """
        mock_post.return_value.json.return_value = [self._make_hf_response("positive", 0.9)]
        scores = _score_sentiment(["text"] * 5)
        assert mock_post.call_count == 5
        assert len(scores) == 5

    @patch("qsf.agents.nodes.requests.post")
    def test_failed_item_does_not_stop_others(self, mock_post):
        """A single failed API call should not prevent the rest from scoring."""
        good = MagicMock()
        good.json.return_value = [self._make_hf_response("positive", 0.9)]

        bad = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        bad.raise_for_status.side_effect = requests.HTTPError(response=mock_response)

        mock_post.side_effect = [good, bad, good]
        scores = _score_sentiment(["text1", "text2", "text3"])
        assert mock_post.call_count == 3
        assert len(scores) == 2

    @patch("qsf.agents.nodes.requests.post")
    def test_unexpected_response_format_is_skipped(self, mock_post):
        """A non-list API response (e.g. error dict) should be skipped gracefully."""
        mock_post.return_value.json.return_value = {"error": "Model is loading"}
        scores = _score_sentiment(["text"])
        assert scores == []

    @patch("qsf.agents.nodes.requests.post")
    def test_empty_input(self, mock_post):
        scores = _score_sentiment([])
        assert scores == []
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# _get_news() - NewsAPI fetching
# ---------------------------------------------------------------------------

class TestGetNews:
    @patch("qsf.agents.nodes.NewsApiClient")
    def test_returns_correctly_shaped_items(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {
            "articles": [
                {"title": "IonQ beats earnings", "publishedAt": "2026-02-10T10:00:00Z"},
                {"title": "Quantum stocks rise", "publishedAt": "2026-02-11T14:00:00Z"},
            ]
        }
        items = _get_news("IONQ")
        assert len(items) == 2
        assert items[0]["text"] == "IonQ beats earnings"
        assert items[0]["date"] == "2026-02-10"
        assert items[0]["source"] == "news"

    @patch("qsf.agents.nodes.NewsApiClient")
    def test_skips_articles_with_no_title(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {
            "articles": [
                {"title": None, "publishedAt": "2026-02-10T10:00:00Z"},
                {"title": "Valid article", "publishedAt": "2026-02-10T10:00:00Z"},
            ]
        }
        items = _get_news("IONQ")
        assert len(items) == 1
        assert items[0]["text"] == "Valid article"

    @patch("qsf.agents.nodes.NewsApiClient")
    def test_empty_response(self, mock_client_class):
        mock_client_class.return_value.get_everything.return_value = {"articles": []}
        items = _get_news("IONQ")
        assert items == []


# ---------------------------------------------------------------------------
# _get_reddit() - Reddit post fetching
# ---------------------------------------------------------------------------

class TestGetReddit:
    def _make_post(self, title, days_ago):
        post = MagicMock()
        post.title = title
        post.created_utc = (datetime.now() - timedelta(days=days_ago)).timestamp()
        return post

    @patch("qsf.agents.nodes.praw.Reddit")
    def test_returns_recent_posts(self, mock_reddit_class):
        mock_reddit_class.return_value.subreddit.return_value.search.return_value = [
            self._make_post("IONQ hits new high", days_ago=5),
            self._make_post("Quantum computing outlook", days_ago=10),
        ]
        items = _get_reddit("IONQ")
        assert len(items) == 6  # 2 posts × 3 subreddits
        assert items[0]["source"] == "social"
        assert items[0]["text"] == "IONQ hits new high"

    @patch("qsf.agents.nodes.praw.Reddit")
    def test_filters_out_old_posts(self, mock_reddit_class):
        mock_reddit_class.return_value.subreddit.return_value.search.return_value = [
            self._make_post("Old post", days_ago=31),
        ]
        items = _get_reddit("IONQ")
        assert items == []

    @patch("qsf.agents.nodes.praw.Reddit")
    def test_searches_all_three_subreddits(self, mock_reddit_class):
        mock_subreddit = mock_reddit_class.return_value.subreddit.return_value
        mock_subreddit.search.return_value = [
            self._make_post("IONQ post", days_ago=3),
        ]
        _get_reddit("IONQ")
        assert mock_reddit_class.return_value.subreddit.call_count == 3
