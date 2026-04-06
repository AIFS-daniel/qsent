"""
Unit tests for the RelevanceClassifier.
"""
from unittest.mock import MagicMock, patch

import pytest

from qsf.nlp.relevance import MAX_ARTICLES, RelevanceClassifier


def _make_openai_response(content: str):
    """Build a minimal mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_articles(n: int) -> list[dict]:
    return [{"text": f"Article {i}", "date": "2026-02-10", "source": "news"} for i in range(n)]


class TestRelevanceClassifier:
    @patch("qsf.nlp.relevance.OpenAI")
    def test_yes_returns_true(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")
        result = RelevanceClassifier().classify("IONQ", "IonQ", [{"text": "IonQ Q4 results", "date": "2026-02-10", "source": "news"}])
        assert result == [True]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_no_returns_false(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("no")
        result = RelevanceClassifier().classify("IONQ", "IonQ", [{"text": "Unrelated article", "date": "2026-02-10", "source": "news"}])
        assert result == [False]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_case_insensitive(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("Yes")
        result = RelevanceClassifier().classify("IONQ", "IonQ", [{"text": "IonQ article", "date": "2026-02-10", "source": "news"}])
        assert result == [True]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_failed_api_call_returns_false_no_raise(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        result = RelevanceClassifier().classify("IONQ", "IonQ", [{"text": "Some article", "date": "2026-02-10", "source": "news"}])
        assert result == [False]

    @patch("qsf.nlp.relevance.OpenAI")
    def test_returns_same_length_as_input(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")
        articles = _make_articles(5)
        result = RelevanceClassifier().classify("IONQ", "IonQ", articles)
        assert len(result) == 5

    @patch("qsf.nlp.relevance.OpenAI")
    def test_caps_at_max_articles(self, mock_openai_cls, monkeypatch):
        """Only MAX_ARTICLES API calls should be made; positions beyond cap default to False."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response("yes")

        articles = _make_articles(MAX_ARTICLES + 5)
        result = RelevanceClassifier().classify("IONQ", "IonQ", articles)

        # Only MAX_ARTICLES API calls should have been made
        assert mock_client.chat.completions.create.call_count == MAX_ARTICLES
        # Result length matches input length
        assert len(result) == MAX_ARTICLES + 5
        # Positions beyond cap are False
        assert all(r is False for r in result[MAX_ARTICLES:])

    @patch("qsf.nlp.relevance.OpenAI")
    def test_empty_input_returns_empty_list(self, mock_openai_cls, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        result = RelevanceClassifier().classify("IONQ", "IonQ", [])
        assert result == []
        mock_client.chat.completions.create.assert_not_called()
