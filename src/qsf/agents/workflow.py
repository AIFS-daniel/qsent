"""
LangGraph pipeline for sentiment analysis.

Graph:
    fetch_market_data → fetch_news → fetch_reddit → score_sentiment → aggregate
"""
import logging
from datetime import datetime
from typing import Any, Optional
from typing_extensions import TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from qsf.common.providers import MarketDataProvider, NewsProvider, SentimentModel, SocialProvider
from qsf.common.utils import safe
from qsf.ingestion import YFinanceMarketData, NewsAPIProvider, RedditProvider
from qsf.nlp import FinBERTModel

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    ticker: str
    stock_df: Any
    news_items: list
    reddit_items: list
    scored_items: list
    result: Optional[dict]
    error: Optional[str]


def build_pipeline(
    market: MarketDataProvider,
    news: NewsProvider,
    social: SocialProvider,
    model: SentimentModel,
):
    def _fetch_market_data(state: dict) -> dict:
        ticker = state["ticker"]
        hist = market.get_history(ticker, "30d")
        if hist.empty:
            return {"error": f"No price data found for '{ticker}'"}
        stock_df = hist[["Close", "Volume"]].copy()
        stock_df.index = stock_df.index.date
        stock_df["ror"] = stock_df["Close"].pct_change() * 100
        days = len(stock_df)
        logger.info("[%s] fetch_market_data: %d trading days returned", ticker, days)
        if days < 15:
            logger.warning(
                "[%s] fetch_market_data: only %d trading days returned, expected ~21",
                ticker, days,
            )
        return {"stock_df": stock_df}

    def _fetch_news(state: dict) -> dict:
        if state.get("error"):
            return {}
        ticker = state["ticker"]
        news_items = news.get_articles(ticker)
        count = len(news_items)
        logger.info("[%s] fetch_news: %d articles returned", ticker, count)
        if count == 0:
            logger.warning(
                "[%s] fetch_news: 0 articles returned — NewsAPI may be unavailable or quota exceeded",
                ticker,
            )
        elif count < 5:
            logger.warning(
                "[%s] fetch_news: only %d articles returned, sentiment coverage may be sparse",
                ticker, count,
            )
        return {"news_items": news_items}

    def _fetch_reddit(state: dict) -> dict:
        if state.get("error"):
            return {}
        ticker = state["ticker"]
        reddit_items = social.get_posts(ticker)
        count = len(reddit_items)
        logger.info("[%s] fetch_reddit: %d posts returned", ticker, count)
        if count == 0:
            logger.warning(
                "[%s] fetch_reddit: 0 posts returned — Reddit API may be unavailable or ticker has low social coverage",
                ticker,
            )
        return {"reddit_items": reddit_items}

    def _score_sentiment(state: dict) -> dict:
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
        logger.info("[%s] score_sentiment: calling model.score on %d items", ticker, len(items))
        scores = model.score([item["text"] for item in items])
        logger.info("[%s] score_sentiment: model.score returned", ticker)
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
            logger.warning(
                "[%s] score_sentiment: %d scores outside [-1, 1]: %s",
                ticker, len(out_of_range), out_of_range,
            )
        if abs(mean_score) < 0.02:
            logger.warning(
                "[%s] score_sentiment: mean score near zero (%.3f) — model may be returning all-neutral",
                ticker, mean_score,
            )
        return {"scored_items": scored}

    def _aggregate(state: dict) -> dict:
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
            "sentiment_score": safe(overall_sentiment),
            "data_points": len(scored_items),
            "breakdown": {
                "news_sentiment": safe(overall_news),
                "social_sentiment": safe(overall_social),
                "trend": trend,
            },
            "daily_data": [
                {
                    "date": str(idx),
                    "close": safe(row["Close"], 2),
                    "volume": int(row["Volume"]),
                    "ror": safe(row["ror"]),
                    "news_sentiment": safe(row["news_sentiment"]),
                    "social_sentiment": safe(row["social_sentiment"]),
                }
                for idx, row in merged.iterrows()
            ],
        }
        return {"result": result}

    graph = StateGraph(PipelineState)

    graph.add_node("fetch_market_data", _fetch_market_data)
    graph.add_node("fetch_news", _fetch_news)
    graph.add_node("fetch_reddit", _fetch_reddit)
    graph.add_node("score_sentiment", _score_sentiment)
    graph.add_node("aggregate", _aggregate)

    graph.set_entry_point("fetch_market_data")
    graph.add_edge("fetch_market_data", "fetch_news")
    graph.add_edge("fetch_news", "fetch_reddit")
    graph.add_edge("fetch_reddit", "score_sentiment")
    graph.add_edge("score_sentiment", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


pipeline = build_pipeline(
    market=YFinanceMarketData(),
    news=NewsAPIProvider(),
    social=RedditProvider(),
    model=FinBERTModel(),
)
