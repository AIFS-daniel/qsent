# QSent — Project Context for Claude

## What This Project Is

QSent is a sentiment analysis and forecasting pipeline for **quantum computing stocks**. The goal is to generate daily stock price movement predictions by combining market data, news sentiment, social media sentiment, and technical indicators into a unified forecasting system.

## Current State

The core sentiment pipeline is functional, exposed via a FastAPI REST API. A first-pass forecasting layer has been integrated, porting the broad strokes from the Phase 2 notebook into production-ready modules.

## Architecture

**Sentiment Pipeline (LangGraph, sequential):**
1. `fetch_market_data` — 30-day OHLCV + company name via Yahoo Finance
2. `fetch_news` — Articles from NewsAPI (company name or ticker query)
3. `fetch_reddit` — Posts/comments from 6 investing subreddits via PRAW
4. `score_sentiment` — FinBERT (ProsusAI/finbert via HuggingFace Inference API)
5. `aggregate` — Daily sentiment time series aligned to trading days

**Forecasting Pipeline (new):**
1. Feature engineering — Technical indicators (RSI, volatility, ROC, MA), cross-asset spreads, macro transforms, sentiment features. Grand Shift enforces causality.
2. Target creation — Next-day return (regression), direction (binary classification), ternary with dead zone.
3. Data preparation — Chronological train/val/test splits, StandardScaler fit on train only.
4. Model training — LogisticRegression, RandomForest, XGBoost (classifiers); Ridge, XGBoost (regressors).
5. Evaluation — Directional accuracy, Sharpe ratio, max drawdown, profit factor.
6. Walk-forward validation — Rolling-window backtest framework.

**Key source files:**
- `src/qsf/api/main.py` — FastAPI app (sentiment + forecasting endpoints)
- `src/qsf/agents/workflow.py` — LangGraph sentiment pipeline
- `src/qsf/features/engineering.py` — Feature engineering with Grand Shift
- `src/qsf/features/targets.py` — Target creation
- `src/qsf/features/preparation.py` — Data splits and scaling
- `src/qsf/forecasting/models.py` — Classical ML models
- `src/qsf/forecasting/pipeline.py` — End-to-end forecasting orchestration
- `src/qsf/backtesting/walk_forward.py` — Walk-forward validation
- `src/qsf/backtesting/metrics.py` — Trading and risk metrics

**Frontend:** `index.html` — browser prototype using Chart.js, calls `/analyze`

## API Endpoints

- `GET /health` — Health check
- `POST /analyze` — Sentiment analysis. Input: `{"ticker": "IONQ"}`. Returns sentiment score, news/social breakdown, trend, and daily data series.
- `POST /forecast` — Forecasting pipeline. Input: `{"ticker": "IONQ", "period": "2y", "include_sentiment": false}`. Returns model comparison, best model selection, and trading metrics.
- `POST /diagnostics/news-comparison` — Multi-provider news comparison
- `GET /diagnostics/news-comparison/stream` — Streaming news comparison

## Data Sources

| Source | Library | Window |
|--------|---------|--------|
| Market data | yfinance | 30 days (sentiment) / 2 years (forecasting) |
| News | newsapi-python | 28 days (free plan limit) |
| Social | praw (Reddit) | 30 days |
| Sentiment model | HuggingFace FinBERT | per-text scoring |

## Anti-Leakage Design

The "Grand Shift" is the core anti-leakage mechanism: all features at row t are shifted so they use only data from t-1 or earlier. This is verified programmatically. The scaler is fit on training data only and applied to val/test splits.

## Future Work (Not Yet Implemented)

- Deep learning sequence models (LSTM, GRU, BiLSTM, CNN1D)
- NeuralForecast models (N-BEATS, N-HiTS, TFT)
- Macro data integration (VIX, Treasury rates via FRED)
- Hybrid ensemble strategies
- Trading simulation with more realistic cost models
- Docker/Terraform infrastructure
