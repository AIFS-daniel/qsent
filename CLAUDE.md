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

## Agent Workflow

When implementing a feature or fixing a bug, invoke agents in this order:
1. `test-writer` — write failing tests that describe the intended behavior (before any implementation)
2. `code-builder` — implement the code until the tests pass
3. `test-runner` — verify the full suite is green
4. `code-reviewer` — review before finalizing or creating a PR

Always invoke `test-writer` before `code-builder`. Tests come first.

The `qa` agent starts the webapp (if not already running) and uses Playwright to verify each behavior in the live browser, including screenshots. It runs after `code-reviewer`. Server start command: `TEST_MODE=true .venv/bin/uvicorn qsf.api.main:app --reload`. Auth bypass: `GET /auth/test-login` (only available when `TEST_MODE=true`).

## /tdd Command

`/tdd` runs a strict red-green-refactor TDD cycle, orchestrating `test-writer`, `test-runner`, `code-builder`, and `code-reviewer` in sequence for each behavior.

**Start a new session:**
```
/tdd "add a normalize_ticker function to src/qsf/common/utils.py"
```
Claude will enumerate the behaviors to implement, confirm the list with you, then loop through each one: write one failing test → confirm it's red → implement minimum code → confirm green → move to next behavior. After all behaviors are green, it runs a refactor pass then a code review.

**Resume an interrupted session:**
```
/tdd
```
Running `/tdd` with no arguments checks for `.claude/tdd-state.json`. If an in-progress session exists, Claude will offer to resume from exactly where it left off (the specific behavior and phase that was interrupted). Choose fresh to discard the previous session and start over.

**State file:** `.claude/tdd-state.json` — written after every phase transition. Do not edit manually. Delete it to clear a completed or abandoned session.

## Placeholder Modules (Not Yet Implemented)

- `src/qsf/features/` — Feature engineering
- `src/qsf/forecasting/` — Price movement prediction
- `src/qsf/backtesting/` — Historical validation
- `src/qsf/pipelines/` — End-to-end pipeline orchestration
