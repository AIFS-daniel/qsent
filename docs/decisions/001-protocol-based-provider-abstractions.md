# ADR 001: Protocol-Based Provider Abstractions

**Date:** 2026-02-21
**Status:** Accepted

## Context

The sentiment analysis pipeline depends on four external services: Yahoo Finance (market data), NewsAPI (news articles), Reddit via PRAW (social posts), and HuggingFace FinBERT (sentiment scoring). 

## Decision

Introduce `typing.Protocol` abstractions for each external dependency:

- `MarketDataProvider` — `get_history(ticker, period) -> DataFrame`
- `NewsProvider` — `get_articles(ticker, days) -> list[dict]`
- `SocialProvider` — `get_posts(ticker, days) -> list[dict]`
- `SentimentModel` — `score(texts) -> list[float]`

Concrete implementations (`YFinanceMarketData`, `NewsAPIProvider`, `RedditProvider`, `FinBERTModel`) live in `qsf.ingestion` and `qsf.nlp`. They are injected into `build_pipeline()` as typed parameters. The module-level `pipeline` instance wires up the production providers in one place (`agents/workflow.py`).

All protocols are decorated with `@runtime_checkable` so `isinstance()` checks work in tests and startup validation.

## Alternatives Considered

**Abstract Base Classes (`ABC`)** — rejected. ABCs imply shared implementation via inheritance, which is inappropriate here; these providers are fully independent. ABCs also require providers to explicitly subclass them, creating unnecessary coupling to the abstraction layer.

**Dependency injection framework** — rejected. The pipeline has exactly four dependencies. A framework would add complexity and a new dependency for no practical gain at this scale.

**Leave hard-coded** — rejected. Made testing require module-level patching (`@patch("qsf.agents.nodes.yf.Ticker")`), which is brittle and ties tests to internal import paths rather than behaviour.

## Consequences

- Swapping a provider means writing one class and changing one argument in `build_pipeline(...)` — nothing else.
- Tests use direct injection (`build_pipeline(mock_market, mock_news, ...)`) instead of `@patch`, making them independent of internal import structure.
- No new runtime dependencies — `typing.Protocol` is in the standard library from Python 3.8.
