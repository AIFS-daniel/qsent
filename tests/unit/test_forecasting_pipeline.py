"""
Unit tests for ForecastingPipeline.run(), which now receives its market
history from the caller (via the injected provider) instead of fetching it
internally with yfinance.
"""
import numpy as np
import pandas as pd

from qsf.forecasting.pipeline import ForecastingPipeline


def synth_hist(n: int = 520, start: str = "2024-01-02", tz: bool = False) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame large enough to train the models.

    A mild zig-zag keeps both up and down days present so the direction
    classifiers always see two classes.
    """
    idx = pd.bdate_range(start=start, periods=n)
    if tz:
        idx = idx.tz_localize("America/New_York")
    price = 10.0
    closes = []
    for i in range(n):
        price *= 1 + 0.004 * (((i * 7) % 11) - 5) / 5.0
        closes.append(price)
    close = pd.Series(closes, index=idx)
    return pd.DataFrame(
        {
            "Open": close.values * 0.99,
            "High": close.values * 1.02,
            "Low": close.values * 0.98,
            "Close": close.values,
            "Volume": [1_000_000 + (i % 50) * 1000 for i in range(n)],
        },
        index=idx,
    )


class TestForecastingPipelineRun:
    def test_returns_best_model_on_valid_history(self):
        result = ForecastingPipeline(ticker="IONQ").run(synth_hist())
        assert "error" not in result
        assert result["ticker"] == "IONQ"
        assert result["models"], "expected at least one trained model"
        assert "best_model" in result
        assert "sharpe" in result["best_model"]
        assert "directional_accuracy" in result["best_model"]

    def test_includes_next_day_prediction(self):
        result = ForecastingPipeline(ticker="IONQ").run(synth_hist())
        nd = result["next_day"]
        assert nd is not None
        assert nd["direction"] in ("up", "down")
        assert nd["model"] == result["best_model"]["name"]
        assert nd["horizon"] == "next_trading_day"
        # as_of must be the most recent market date, target a later date.
        assert nd["as_of"] < nd["target_date"]
        # Exactly one of confidence / predicted_return_pct is populated,
        # depending on whether the best model is a classifier or regressor.
        assert (nd["confidence"] is not None) or (nd["predicted_return_pct"] is not None)

    def test_next_day_is_json_serialisable(self):
        import json

        result = ForecastingPipeline(ticker="IONQ").run(synth_hist())
        # The whole payload (incl. models list) must serialise — no model objects leak.
        json.dumps(result)
        assert all("model" not in m for m in result["models"])

    def test_returns_error_on_empty_history(self):
        result = ForecastingPipeline(ticker="FAKE").run(pd.DataFrame())
        assert "error" in result
        assert "FAKE" in result["error"]

    def test_returns_error_on_none_history(self):
        result = ForecastingPipeline(ticker="FAKE").run(None)
        assert "error" in result

    def test_strips_timezone_from_history(self):
        # A tz-aware index must not raise; it should be handled transparently.
        result = ForecastingPipeline(ticker="IONQ").run(synth_hist(tz=True))
        assert "error" not in result

    def test_sentiment_overlay_adds_features(self):
        hist = synth_hist()
        sentiment = pd.DataFrame(
            {
                "news_sentiment": np.linspace(-0.2, 0.2, len(hist)),
                "social_sentiment": np.linspace(0.1, -0.1, len(hist)),
            },
            index=hist.index,
        )
        result = ForecastingPipeline(ticker="IONQ").run(hist, sentiment_daily=sentiment)
        assert "error" not in result
        feature_names = result["feature_names"]
        assert any("news_sentiment" in f for f in feature_names)
        assert any("social_sentiment" in f for f in feature_names)
