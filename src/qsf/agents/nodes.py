"""
LangGraph node functions for the sentiment analysis pipeline.

Each node receives the full pipeline state and returns a dict of
keys to update in the state. Nodes short-circuit on error.
"""
import logging
import math
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd
import praw
import requests
import yfinance as yf
from newsapi import NewsApiClient

FINBERT_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
SENTIMENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}
REDDIT_SUBREDDITS = ["stocks", "investing", "wallstreetbets"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val: Any, decimals: int = 4) -> float | None:
    """Convert numpy/pandas numerics to JSON-safe Python floats."""
    if val is None:
        return None
    f = float(val)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, decimals)


def _get_news(ticker: str, days: int = 30) -> list[dict]:
    client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    response = client.get_everything(
        q=ticker,
        from_param=from_date,
        language="en",
        sort_by="publishedAt",
        page_size=100,
    )
    return [
        {"text": a["title"], "date": a["publishedAt"][:10], "source": "news"}
        for a in response.get("articles", [])
        if a.get("title")
    ]


def _get_reddit(ticker: str, days: int = 30) -> list[dict]:
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "qsent/0.1"),
    )
    cutoff = datetime.now() - timedelta(days=days)
    posts = []
    for subreddit in REDDIT_SUBREDDITS:
        for post in reddit.subreddit(subreddit).search(
            ticker, sort="new", time_filter="month", limit=50
        ):
            created = datetime.fromtimestamp(post.created_utc)
            if created >= cutoff:
                posts.append({
                    "text": post.title,
                    "date": created.strftime("%Y-%m-%d"),
                    "source": "social",
                })
    return posts


def _score_sentiment(texts: list[str]) -> list[float]:
    """Score sentiment for each text individually.

    The HuggingFace Inference API for FinBERT only processes the first item
    when given a batch, so we send one text per request.
    """
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    scores = []
    failed = 0

    for idx, text in enumerate(texts):
        try:
            response = requests.post(FINBERT_URL, headers=headers, json={"inputs": text})
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, list) or not result:
                logger.warning(
                    "_score_sentiment: item %d/%d returned unexpected response: %s",
                    idx + 1, len(texts), str(result)[:200],
                )
                failed += 1
                continue
            # Single-text requests return [[{label_dicts}]] — unwrap the outer list
            label_scores = result[0] if isinstance(result[0], list) else result
            top = max(label_scores, key=lambda x: x["score"])
            scores.append(SENTIMENT_MAP.get(top["label"].lower(), 0) * top["score"])
        except requests.HTTPError as e:
            logger.warning(
                "_score_sentiment: item %d/%d failed — HTTP %s: %s",
                idx + 1, len(texts), e.response.status_code, e.response.text[:200],
            )
            failed += 1
        except Exception as e:
            logger.warning(
                "_score_sentiment: item %d/%d failed — %s: %s",
                idx + 1, len(texts), type(e).__name__, e,
            )
            failed += 1

    if failed:
        logger.warning(
            "_score_sentiment: %d/%d items failed to score",
            failed, len(texts),
        )

    return scores


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def fetch_market_data(state: dict) -> dict:
    ticker = state["ticker"]
    hist = yf.Ticker(ticker).history(period="30d")
    if hist.empty:
        return {"error": f"No price data found for '{ticker}'"}
    stock_df = hist[["Close", "Volume"]].copy()
    stock_df.index = stock_df.index.date
    stock_df["ror"] = stock_df["Close"].pct_change() * 100
    days = len(stock_df)
    logger.info("[%s] fetch_market_data: %d trading days returned", ticker, days)
    if days < 15:
        logger.warning("[%s] fetch_market_data: only %d trading days returned, expected ~21", ticker, days)
    return {"stock_df": stock_df}


def fetch_news(state: dict) -> dict:
    if state.get("error"):
        return {}
    ticker = state["ticker"]
    news_items = _get_news(ticker)
    count = len(news_items)
    logger.info("[%s] fetch_news: %d articles returned", ticker, count)
    if count == 0:
        logger.warning("[%s] fetch_news: 0 articles returned — NewsAPI may be unavailable or quota exceeded", ticker)
    elif count < 5:
        logger.warning("[%s] fetch_news: only %d articles returned, sentiment coverage may be sparse", ticker, count)
    return {"news_items": news_items}


def fetch_reddit(state: dict) -> dict:
    if state.get("error"):
        return {}
    ticker = state["ticker"]
    reddit_items = _get_reddit(ticker)
    count = len(reddit_items)
    logger.info("[%s] fetch_reddit: %d posts returned", ticker, count)
    if count == 0:
        logger.warning("[%s] fetch_reddit: 0 posts returned — Reddit API may be unavailable or ticker has low social coverage", ticker)
    return {"reddit_items": reddit_items}


def score_sentiment(state: dict) -> dict:
    if state.get("error"):
        return {}
    ticker = state["ticker"]
    news_items = state.get("news_items", [])
    reddit_items = state.get("reddit_items", [])
    items = news_items + reddit_items
    logger.info(
        "[%s] score_sentiment: received %d news + %d reddit = %d items from state",
        ticker, len(news_items), len(reddit_items), len(items),
    )
    if not items:
        return {"error": f"No news or social data found for '{ticker}'"}
    scores = _score_sentiment([item["text"] for item in items])
    scored = [
        {**item, "weighted_sentiment": score}
        for item, score in zip(items, scores)
    ]
    mean_score = sum(scores) / len(scores)
    logger.info(
        "[%s] score_sentiment: %d items scored — mean=%.3f, min=%.3f, max=%.3f",
        ticker, len(scores), mean_score, min(scores), max(scores),
    )
    out_of_range = [s for s in scores if not -1.0 <= s <= 1.0]
    if out_of_range:
        logger.warning("[%s] score_sentiment: %d scores outside [-1, 1]: %s", ticker, len(out_of_range), out_of_range)
    if abs(mean_score) < 0.02:
        logger.warning("[%s] score_sentiment: mean score near zero (%.3f) — model may be returning all-neutral", ticker, mean_score)
    return {"scored_items": scored}


def aggregate(state: dict) -> dict:
    if state.get("error"):
        return {}

    ticker = state["ticker"]
    stock_df: pd.DataFrame = state["stock_df"]
    scored_items: list[dict] = state["scored_items"]

    df = pd.DataFrame(scored_items)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    news_daily = (
        df[df["source"] == "news"]
        .groupby("date")["weighted_sentiment"]
        .mean()
        .rename("news_sentiment")
    )
    social_daily = (
        df[df["source"] == "social"]
        .groupby("date")["weighted_sentiment"]
        .mean()
        .rename("social_sentiment")
    )

    # Forward-fill across calendar range so weekend/holiday articles
    # carry forward to the next trading day
    all_dates = pd.date_range(
        start=pd.Timestamp(min(stock_df.index)),
        end=pd.Timestamp(max(stock_df.index)),
    ).map(lambda d: d.date())

    total_days = len(stock_df)
    news_coverage = news_daily.reindex(stock_df.index).notna().sum()
    social_coverage = social_daily.reindex(stock_df.index).notna().sum()
    logger.info(
        "[%s] aggregate: news coverage %d/%d trading days, social coverage %d/%d trading days",
        ticker, news_coverage, total_days, social_coverage, total_days,
    )
    if news_coverage == 0:
        logger.warning("[%s] aggregate: no news sentiment aligned to any trading day", ticker)
    if social_coverage == 0:
        logger.warning("[%s] aggregate: no social sentiment aligned to any trading day", ticker)

    news_aligned = news_daily.reindex(all_dates).ffill().reindex(stock_df.index).fillna(0)
    news_aligned.name = "news_sentiment"
    social_aligned = social_daily.reindex(all_dates).ffill().reindex(stock_df.index).fillna(0)
    social_aligned.name = "social_sentiment"

    merged = stock_df.copy()
    merged["news_sentiment"] = news_aligned
    merged["social_sentiment"] = social_aligned

    # Overall scores
    news_mean = news_daily.mean()
    social_mean = social_daily.mean()
    overall_news = float(news_mean) if (not news_daily.empty and pd.notna(news_mean)) else 0.0
    overall_social = float(social_mean) if (not social_daily.empty and pd.notna(social_mean)) else 0.0
    overall_sentiment = round((overall_news + overall_social) / 2, 4)

    # Trend
    sorted_df = df.sort_values("date")
    recent_mean = sorted_df.tail(7)["weighted_sentiment"].mean()
    older_mean = sorted_df.head(7)["weighted_sentiment"].mean()
    if recent_mean > older_mean + 0.05:
        trend = "increasing"
    elif recent_mean < older_mean - 0.05:
        trend = "decreasing"
    else:
        trend = "stable"

    result = {
        "ticker": ticker,
        "last_updated": datetime.now().isoformat(),
        "sentiment_score": _safe(overall_sentiment),
        "data_points": len(scored_items),
        "breakdown": {
            "news_sentiment": _safe(overall_news),
            "social_sentiment": _safe(overall_social),
            "trend": trend,
        },
        "daily_data": [
            {
                "date": str(idx),
                "close": _safe(row["Close"], 2),
                "volume": int(row["Volume"]),
                "ror": _safe(row["ror"]),
                "news_sentiment": _safe(row["news_sentiment"]),
                "social_sentiment": _safe(row["social_sentiment"]),
            }
            for idx, row in merged.iterrows()
        ],
    }
    return {"result": result}
