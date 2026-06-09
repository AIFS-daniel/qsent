"""
End-to-end forecasting pipeline.

Combines the QSent sentiment pipeline output with the Phase 2 feature
engineering and classical ML models to produce price movement forecasts.
"""
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from qsf.features.engineering import engineer_features
from qsf.features.targets import create_targets
from qsf.features.preparation import prepare_data_bundle
from qsf.forecasting.models import (
    LogisticRegressionModel,
    RandomForestModel,
    RidgeRegressionModel,
    ModelResult,
)

logger = logging.getLogger(__name__)

ANNUALIZATION_FACTOR = 252.0


def _compute_metrics(
    y_true: np.ndarray, predictions: np.ndarray, model_type: str
) -> dict[str, float]:
    """Compute evaluation metrics for a model's predictions."""
    metrics: dict[str, float] = {}

    if model_type == "classifier":
        metrics["accuracy"] = float(np.mean(y_true == predictions))
        direction_true = (y_true > 0).astype(int) if y_true.dtype != int else y_true
        metrics["directional_accuracy"] = float(np.mean(direction_true == predictions))
    else:
        residuals = y_true - predictions
        metrics["rmse"] = float(np.sqrt(np.mean(residuals**2)))
        metrics["mae"] = float(np.mean(np.abs(residuals)))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        dir_correct = np.sign(predictions) == np.sign(y_true)
        metrics["directional_accuracy"] = float(np.mean(dir_correct))

    return metrics


def _compute_trading_metrics(
    returns: np.ndarray,
    predictions: np.ndarray,
    cost_bps: float = 20.0,
) -> dict[str, float]:
    """Simulate a sign-based strategy and compute trading metrics."""
    position = (predictions > 0).astype(float)
    turnover = np.abs(np.diff(position, prepend=position[0]))
    cost_per_side = (cost_bps / 2.0) / 10_000.0
    lagged_pos = np.roll(position, 1)
    lagged_pos[0] = 0.0

    strategy_returns = lagged_pos * returns - turnover * cost_per_side
    equity = np.cumprod(1.0 + strategy_returns)

    mean_daily = np.mean(strategy_returns)
    std_daily = np.std(strategy_returns, ddof=0)
    sharpe = 0.0
    if std_daily > 0:
        sharpe = float(
            (mean_daily * ANNUALIZATION_FACTOR)
            / (std_daily * np.sqrt(ANNUALIZATION_FACTOR))
        )

    cummax = np.maximum.accumulate(equity)
    drawdowns = (equity - cummax) / cummax
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "total_return_pct": float((equity[-1] - 1.0) * 100),
        "sharpe": sharpe,
        "max_drawdown_pct": float(max_drawdown * 100),
        "trade_fraction_pct": float(np.mean(position) * 100),
    }


def _predict_next_day(
    best_model: Any,
    model_type: str,
    ticker: str,
    df_features: pd.DataFrame,
    feature_cols: list[str],
    scaler: Any,
    X_all: np.ndarray,
    y_all_dir: np.ndarray,
    y_all_return: np.ndarray,
) -> dict[str, Any] | None:
    """Forecast the direction of the next trading day.

    The most recent engineered row has no realised forward return yet (it is
    exactly the row ``create_targets`` drops), so its features are what we feed
    the model to predict the upcoming session. The best model is refit on all
    labelled rows first so the live call benefits from the full history, then
    the live row is scaled with the same (leak-free) scaler used for backtest.
    """
    try:
        live = df_features.iloc[[-1]][feature_cols].fillna(0.0)
        X_live = scaler.transform(live.values)
        as_of = df_features.index[-1]
        target_date = (as_of + pd.tseries.offsets.BDay(1)).date()

        out: dict[str, Any] = {
            "as_of": str(as_of.date()),
            "target_date": str(target_date),
            "horizon": "next_trading_day",
            "model": getattr(best_model, "name", "?"),
            "confidence": None,
            "predicted_return_pct": None,
        }

        if model_type == "classifier":
            best_model.fit(X_all, y_all_dir)
            pred = int(best_model.predict(X_live)[0])
            out["direction"] = "up" if pred == 1 else "down"
            try:
                proba = best_model.predict_proba(X_live)[0]
                out["confidence"] = float(np.max(proba))
            except Exception:  # model may not expose probabilities
                pass
        else:
            best_model.fit(X_all, y_all_return)
            predicted_return = float(best_model.predict(X_live)[0])
            out["direction"] = "up" if predicted_return > 0 else "down"
            out["predicted_return_pct"] = predicted_return * 100.0

        return out
    except Exception as exc:  # never let a live-prediction failure sink the run
        logger.warning("[%s] next-day prediction failed: %s", ticker, exc)
        return None


class ForecastingPipeline:
    """Orchestrates feature engineering, model training, and evaluation."""

    def __init__(
        self,
        ticker: str = "IONQ",
        period: str = "2y",
        cost_bps: float = 20.0,
    ):
        self.ticker = ticker
        self.period = period
        self.cost_bps = cost_bps

    def run(
        self,
        hist: pd.DataFrame,
        sentiment_daily: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Execute the full forecasting pipeline.

        Parameters
        ----------
        hist : pd.DataFrame
            Historical OHLCV market data with a DatetimeIndex and at minimum
            'Close' and 'Volume' columns. The caller fetches this through the
            injected market provider (yfinance today, Supabase later), so this
            method stays agnostic to where the data comes from.
        sentiment_daily : pd.DataFrame, optional
            Daily sentiment scores from the QSent sentiment pipeline.
            Expected columns: news_sentiment, social_sentiment.

        Returns
        -------
        dict
            Pipeline results including model comparisons and best forecast.
        """
        # 1. Validate market data (fetched by the caller via the market provider)
        if hist is None or hist.empty:
            return {"error": f"No market data for {self.ticker}"}
        hist = hist.copy()
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)

        # The sentiment overlay only covers a recent slice of the (much longer)
        # market window. Reindex it onto the full history with a neutral 0 fill
        # so rows outside the sentiment window aren't dropped for missing
        # features downstream — "no sentiment" is treated as neutral, matching
        # how the sentiment aggregation already fills gaps.
        if sentiment_daily is not None and not sentiment_daily.empty:
            sentiment_daily = sentiment_daily.reindex(hist.index).fillna(0.0)

        # 2. Feature engineering
        logger.info("[%s] Engineering features", self.ticker)
        df_features, feature_cols = engineer_features(
            hist, sentiment_daily=sentiment_daily
        )

        # 3. Create targets
        df_targets, tau = create_targets(df_features)

        # 4. Prepare data bundle
        bundle = prepare_data_bundle(
            df_targets, feature_cols=feature_cols, primary_target="target_return"
        )

        X_train = bundle["X_train"]
        y_train = bundle["y_train"]
        X_test = bundle["X_test"]
        y_test = bundle["y_test"]
        X_val = bundle["X_val"]
        y_val = bundle["y_val"]

        # Binary direction targets for classifiers
        y_train_dir = (y_train > 0).astype(int)
        y_test_dir = (y_test > 0).astype(int)
        y_val_dir = (y_val > 0).astype(int)

        # 5. Train models
        logger.info("[%s] Training models", self.ticker)
        classifiers = [
            LogisticRegressionModel(),
            RandomForestModel(),
        ]
        regressors = [
            RidgeRegressionModel(),
        ]

        # Try to add XGBoost if available
        try:
            from qsf.forecasting.models import (
                XGBoostClassifierModel,
                XGBoostRegressorModel,
            )
            classifiers.append(XGBoostClassifierModel())
            regressors.append(XGBoostRegressorModel())
        except ImportError:
            logger.info("xgboost not available, skipping XGBoost models")

        model_results: list[dict] = []

        for model in classifiers:
            result = model.run(X_train, y_train_dir, X_test)
            metrics = _compute_metrics(y_test_dir, result.predictions, "classifier")

            # Trading metrics: convert classifier predictions to return-signed
            pred_returns = np.where(result.predictions == 1, 1.0, -1.0)
            trading = _compute_trading_metrics(y_test, pred_returns, self.cost_bps)

            model_results.append({
                "name": result.name,
                "type": "classifier",
                "metrics": metrics,
                "trading": trading,
                "model": model,
            })

        for model in regressors:
            result = model.run(X_train, y_train, X_test)
            metrics = _compute_metrics(y_test, result.predictions, "regressor")
            trading = _compute_trading_metrics(
                y_test, result.predictions, self.cost_bps
            )

            model_results.append({
                "name": result.name,
                "type": "regressor",
                "metrics": metrics,
                "trading": trading,
                "model": model,
            })

        # 6. Select best model by Sharpe ratio
        best = max(model_results, key=lambda r: r["trading"]["sharpe"])

        # 7. Forecast the next trading day with the best model, refit on all
        #    labelled data (train+val+test) so the live call uses full history.
        X_all = np.vstack([X_train, X_val, X_test])
        y_all_dir = np.concatenate([y_train_dir, y_val_dir, y_test_dir])
        y_all_return = np.concatenate([y_train, y_val, y_test])
        next_day = _predict_next_day(
            best["model"], best["type"], self.ticker,
            df_features, feature_cols, bundle["scaler"],
            X_all, y_all_dir, y_all_return,
        )

        # Model instances are not JSON-serialisable — drop them from the output.
        for r in model_results:
            r.pop("model", None)

        return {
            "ticker": self.ticker,
            "last_updated": datetime.now().isoformat(),
            "data_points": len(hist),
            "train_samples": len(bundle["train_df"]),
            "val_samples": len(bundle["val_df"]),
            "test_samples": len(bundle["test_df"]),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
            "dead_zone_tau": float(tau),
            "models": model_results,
            "best_model": {
                "name": best["name"],
                "sharpe": best["trading"]["sharpe"],
                "directional_accuracy": best["metrics"]["directional_accuracy"],
                "total_return_pct": best["trading"]["total_return_pct"],
                "max_drawdown_pct": best["trading"]["max_drawdown_pct"],
            },
            "next_day": next_day,
        }
