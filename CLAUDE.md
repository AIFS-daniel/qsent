# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

QSent is a sentiment analysis pipeline for **quantum computing stocks**. It combines market data (Yahoo Finance), news sentiment (NewsAPI), and social sentiment (Reddit) into a unified daily score, exposed via a FastAPI REST API and a browser frontend.

The longer-term vision is a full AI forecasting system with backtesting. A `ForecastingPipeline` is partially implemented in `src/qsf/forecasting/` but not yet wired into the API.

## Commands

```bash
# Install (first time)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run server (normal — requires Google OAuth credentials in .env)
.venv/bin/uvicorn qsf.api.main:app --reload

# Run server (test mode — bypasses Google auth)
TEST_MODE=true .venv/bin/uvicorn qsf.api.main:app --reload

# Run all fast tests
.venv/bin/pytest tests/unit/ tests/integration/

# Run a single test file
.venv/bin/pytest tests/unit/test_nodes.py

# Run a single test by name
.venv/bin/pytest tests/unit/test_nodes.py::TestFetchMarketData::test_returns_error_on_empty_history

# Run E2E tests (requires server running in TEST_MODE)
.venv/bin/playwright install chromium  # first time only
.venv/bin/pytest tests/e2e/

# Optional: start Langfuse tracing UI (Docker required)
docker compose -f docker-compose.langfuse.yml up -d
```

No lint or formatter is configured yet. There is no `Makefile` with useful targets.

## Architecture

**Pipeline flow (LangGraph, sequential):**
`fetch_market_data` → `fetch_news` → `fetch_reddit` → `score_sentiment` → `aggregate`

Each node returns a partial `PipelineState` dict. If any node sets `state["error"]`, all downstream nodes skip execution by checking `state.get("error")` at entry.

**Key source files:**
- `src/qsf/api/main.py` — FastAPI app, `TraceMiddleware`, endpoint handlers
- `src/qsf/api/auth.py` — Google OAuth 2.0 flow, JWT session cookies, `get_current_user` dependency
- `src/qsf/agents/workflow.py` — `build_pipeline()` factory + module-level `pipeline` instance
- `src/qsf/agents/news_comparison.py` — separate LangGraph pipeline for the diagnostics endpoint
- `src/qsf/common/providers.py` — `typing.Protocol` interfaces for all four external dependencies
- `src/qsf/common/logging.py` — structlog setup, Langfuse singleton, per-request trace context
- `src/qsf/common/utils.py` — `safe()` (numpy → JSON float), `company_search_name()` (strips legal suffixes)
- `src/qsf/ingestion/market.py` — `YFinanceMarketData`
- `src/qsf/ingestion/news.py` — `NewsAPIProvider`
- `src/qsf/ingestion/social.py` — `RedditProvider`
- `src/qsf/nlp/sentiment.py` — `FinBERTModel` (HuggingFace Inference API)

**Frontend:** `index.html` and `login.html` are served directly by FastAPI. Chart.js from CDN — no build step.

## Provider Pattern (Critical for Testing)

The four external dependencies are abstracted as `typing.Protocol` classes in `src/qsf/common/providers.py`:

```python
MarketDataProvider  →  get_history(ticker, period) -> DataFrame
                        get_company_name(ticker) -> str
NewsProvider        →  get_articles(ticker, company_name, days) -> list[dict]
SocialProvider      →  get_posts(ticker, company_name, days) -> list[dict]
SentimentModel      →  score(texts) -> list[float | None]
```

Providers are injected via `build_pipeline(market, news, social, model)`. The module-level `pipeline` in `workflow.py` wires up the production providers.

**Tests use direct injection, not `@patch`:**
```python
market, news, social, model = make_providers()  # MagicMocks
pipeline = build_pipeline(market, news, social, model)
state = pipeline.invoke({"ticker": "IONQ"})
```

Never use `@patch("qsf.agents.workflow....")` for node tests — inject via `build_pipeline()` instead. See `tests/unit/test_nodes.py` for the canonical pattern including the `make_providers()` helper.

Provider item dict shapes:
- News/Social items: `{"text": str, "date": "YYYY-MM-DD", "source": "news"|"social"}`
- `SentimentModel.score()` always returns a list the same length as input. `None` at an index = that item failed.

## API Endpoints

- `GET /health` — unauthenticated health check
- `POST /analyze` — main endpoint; input `{"ticker": "IONQ"}`; returns sentiment score, breakdown, trend, daily data series
- `POST /diagnostics/news-comparison` — compare news across multiple tickers
- `GET /diagnostics/news-comparison/stream` — SSE streaming version of the above

All endpoints except `/health` require a valid session cookie (`qsent_session`).

## Authentication

Google OAuth 2.0 Authorization Code flow. Sessions stored as signed HttpOnly JWT cookies, 8-hour expiry.

**Key auth files:** `src/qsf/api/auth.py` — handles `/auth/login`, `/auth/callback`, `/auth/logout`, `/auth/me`.

**Integration test auth bypass:**
```python
# tests/integration/conftest.py — applied autouse to all integration tests
app.dependency_overrides[get_current_user] = lambda: MOCK_USER
```

**QA/E2E bypass (TEST_MODE only):**
```bash
TEST_MODE=true .venv/bin/uvicorn qsf.api.main:app --reload
# Then visit http://localhost:8000/auth/test-login once to get a session cookie
```

The `TEST_MODE` route is never registered unless `TEST_MODE=true`. Never enable it in production.

**PII rule:** `user_id` in logs and Langfuse traces is always the Google `sub` (an opaque permanent ID like `"116234567890"`), never the user's email. The JWT stores both `sub` (email) and `google_sub` (Google's opaque ID) for legacy reasons, but observability code uses `google_sub` exclusively.

## Observability

Two correlated systems per request:

**structlog** — structured logs with `trace_id` and `user_id` bound per-request by `TraceMiddleware`. Every log line across the entire stack (including third-party libraries) carries these fields. Set `LOG_FORMAT=json` for machine-readable output in production.

**Langfuse v2** — agentic trace UI. One trace per `/analyze` request, five child spans (one per pipeline node). Optional: if `LANGFUSE_PUBLIC_KEY` is not set, tracing silently disables and the pipeline runs normally. Pinned to `langfuse>=2.0,<3.0` — v3 requires ClickHouse, which is not set up.

The same `trace_id` links stdout logs to the Langfuse UI. The `X-Trace-ID` response header carries it for client-side correlation.

## Sentiment Scoring

- FinBERT classifies each text as positive (+1), neutral (0), or negative (-1), weighted by confidence
- News: title + description combined
- Reddit: title + body + top 5 comments by upvotes, capped at 1800 chars
- Daily aggregation: average sentiment per day per source, forward-filled across calendar days to the next trading day
- Overall score: average of `news_mean` and `social_mean`
- Trend: last 7 vs first 7 scored items, ±0.05 threshold → `increasing` / `decreasing` / `stable`

## Agent Workflow

When implementing a feature or fixing a bug, invoke agents in this order:
1. `test-writer` — write failing tests that describe the intended behavior (before any implementation)
2. `code-builder` — implement the code until the tests pass
3. `test-runner` — verify the full suite is green
4. `code-reviewer` — review before finalizing or creating a PR

Always invoke `test-writer` before `code-builder`. Tests come first.

The `qa` agent starts the webapp (if not already running) and uses Playwright to verify each behavior in the live browser, including screenshots. It runs after `code-reviewer`. Server start command: `TEST_MODE=true .venv/bin/uvicorn qsf.api.main:app --reload`. Auth bypass: `GET /auth/test-login` (only available when `TEST_MODE=true`).

## Autonomous GitHub Agent

Issues can be implemented automatically without human involvement by applying labels:

- `autonomous` — triggers the implementation workflow: the agent reads the issue and follows the TDD workflow above; the workflow wrapper (not the agent) then commits, pushes, and opens the PR itself if the working tree is dirty when the session ends, using its own `opencode/issue{N}-{timestamp}` branch naming
- `autonomous-review` — triggers the review workflow: the agent checks the open PR linked to the issue and posts its findings as a PR comment

**Abort behaviour:** if `autonomous` is applied to an issue that already has an open PR, the workflow aborts and posts a comment on the issue explaining why, with a pointer to use `autonomous-review` instead.

**Key files:**
- `.github/workflows/opencode-implement.yml` — implementation workflow (CI orchestration only)
- `.github/workflows/opencode-review.yml` — review workflow (CI orchestration only)
- `.github/prompts/implement.md` — agent instructions for implementation (edit this to tune behaviour)
- `.github/prompts/review.md` — agent instructions for review (edit this to tune behaviour)

**Required GitHub secret:** `GOOGLE_GENERATIVE_AI_API_KEY` — free API key from Google AI Studio. Model: `google/gemini-2.5-flash`.

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

## Data Sources

| Source | Library | Window |
|--------|---------|--------|
| Market data | yfinance | 30 days |
| News | newsapi-python | 28 days (free plan limit) |
| Social | praw (Reddit) | 30 days |
| Sentiment model | HuggingFace FinBERT | per-text scoring |

Reddit subreddits: `stocks`, `investing`, `wallstreetbets`, `Superstonk`, `StockMarket`, `QuantumComputing`

## Placeholder Modules (Not Yet Implemented)

- `src/qsf/features/` — Feature engineering
- `src/qsf/forecasting/` — Price movement prediction (partially implemented, not wired into API)
- `src/qsf/backtesting/` — Historical validation
- `src/qsf/pipelines/` — End-to-end pipeline orchestration

## Architecture Decision Records

Key decisions are documented in `docs/decisions/`:
- `001` — Protocol-based provider abstractions (why `typing.Protocol` over ABC or DI framework)
- `002` — NewsAPI 28-day date window
- `003` — FinBERT scoring reliability
- `004` — Reddit text enrichment (comments + char cap)
- `005` — Company name search query (why `company_search_name()` strips legal suffixes)
- `006` — Observability: structlog + Langfuse v2 (why not LangSmith, Phoenix, or v3)
- `007` — Google SSO (why not Auth0, username/password, or implicit flow)
