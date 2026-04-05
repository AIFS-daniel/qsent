# QSent — Project Context for Claude

## What This Project Is

QSent is a sentiment analysis pipeline for **quantum computing stocks**. The goal is to generate daily stock price movement predictions by combining market data, news sentiment, and social media sentiment into a unified signal.

The longer-term vision is a full AI forecasting system — using sentiment as a feature for predicting price movement, with backtesting capabilities.

## Current State

Early stage. The core ingestion and scoring pipeline is functional, exposed via a FastAPI REST API. Development is focused on validating data quality and tuning the sentiment pipeline.

## Architecture

**Pipeline flow (LangGraph, sequential):**
1. `fetch_market_data` — 30-day OHLCV + company name via Yahoo Finance
2. `fetch_news` — Articles from NewsAPI (company name or ticker query)
3. `fetch_reddit` — Posts/comments from 6 investing subreddits via PRAW
4. `score_sentiment` — FinBERT (ProsusAI/finbert via HuggingFace Inference API)
5. `aggregate` — Daily sentiment time series aligned to trading days

**Key source files:**
- `src/qsf/api/main.py` — FastAPI app
- `src/qsf/agents/workflow.py` — LangGraph pipeline
- `src/qsf/ingestion/market.py` — Yahoo Finance provider
- `src/qsf/ingestion/news.py` — NewsAPI provider
- `src/qsf/ingestion/social.py` — Reddit provider
- `src/qsf/nlp/sentiment.py` — FinBERT sentiment model
- `src/qsf/common/utils.py` — Company name normalization, helpers

**Frontend:** `index.html` — browser prototype using Chart.js, calls `/analyze`

## API Endpoints

- `GET /health` — Health check
- `POST /analyze` — Main endpoint. Input: `{"ticker": "IONQ"}`. Returns sentiment score, news/social breakdown, trend, and daily data series.
- Diagnostics endpoint — Used during development to inspect raw data fetching and scoring behavior for troubleshooting.

## Data Sources

| Source | Library | Window |
|--------|---------|--------|
| Market data | yfinance | 30 days |
| News | newsapi-python | 28 days (free plan limit) |
| Social | praw (Reddit) | 30 days |
| Sentiment model | HuggingFace FinBERT | per-text scoring |

Reddit subreddits: `stocks`, `investing`, `wallstreetbets`, `Superstonk`, `StockMarket`, `QuantumComputing`

## Sentiment Scoring

- FinBERT classifies each text as positive (+1), neutral (0), or negative (-1), weighted by confidence
- News: title + description combined
- Reddit: title + body + top 5 comments by upvotes, capped at 1800 chars
- Daily aggregation: average sentiment per day by source
- Trend: last 7 vs first 7 items, ±0.05 threshold → increasing / decreasing / stable

## Placeholder Modules (Not Yet Implemented)

- `src/qsf/features/` — Feature engineering
- `src/qsf/forecasting/` — Price movement prediction
- `src/qsf/backtesting/` — Historical validation
- `src/qsf/pipelines/` — End-to-end pipeline orchestration
