import math
import os
from datetime import datetime, timedelta

import pandas as pd
import praw
import requests
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from newsapi import NewsApiClient
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

FINBERT_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
SENTIMENT_MAP = {"positive": 1, "negative": -1, "neutral": 0}
REDDIT_SUBREDDITS = ["stocks", "investing", "wallstreetbets"]


def _safe(val, decimals=4):
    """Convert numpy/pandas numerics to JSON-safe Python floats."""
    if val is None:
        return None
    f = float(val)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, decimals)


class AnalyzeRequest(BaseModel):
    ticker: str


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
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    scores = []
    for i in range(0, len(texts), 10):
        batch = texts[i:i + 10]
        response = requests.post(FINBERT_URL, headers=headers, json={"inputs": batch})
        response.raise_for_status()
        for result in response.json():
            top = max(result, key=lambda x: x["score"])
            scores.append(SENTIMENT_MAP.get(top["label"].lower(), 0) * top["score"])
    return scores


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    ticker = request.ticker.upper().strip()

    # Stock data
    hist = yf.Ticker(ticker).history(period="30d")
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for '{ticker}'")
    stock_df = hist[["Close", "Volume"]].copy()
    stock_df.index = stock_df.index.date
    stock_df["ror"] = stock_df["Close"].pct_change() * 100

    # Text data
    items = _get_news(ticker) + _get_reddit(ticker)
    if not items:
        raise HTTPException(status_code=404, detail=f"No news or social data found for '{ticker}'")

    # Sentiment scoring
    scores = _score_sentiment([item["text"] for item in items])
    for item, score in zip(items, scores):
        item["weighted_sentiment"] = score

    # Aggregate daily scores by source
    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    news_daily = df[df["source"] == "news"].groupby("date")["weighted_sentiment"].mean().rename("news_sentiment")
    social_daily = df[df["source"] == "social"].groupby("date")["weighted_sentiment"].mean().rename("social_sentiment")

    # Forward-fill sentiment across full calendar range so weekend/holiday
    # articles carry forward to the next trading day before joining
    all_dates = pd.date_range(
        start=pd.Timestamp(min(stock_df.index)),
        end=pd.Timestamp(max(stock_df.index)),
    ).map(lambda d: d.date())
    news_aligned = news_daily.reindex(all_dates).ffill().reindex(stock_df.index).fillna(0)
    news_aligned.name = "news_sentiment"
    social_aligned = social_daily.reindex(all_dates).ffill().reindex(stock_df.index).fillna(0)
    social_aligned.name = "social_sentiment"
    merged = stock_df.copy()
    merged["news_sentiment"] = news_aligned
    merged["social_sentiment"] = social_aligned

    # Overall scores — guard against NaN from empty or all-NaN series
    news_mean = news_daily.mean()
    social_mean = social_daily.mean()
    overall_news = float(news_mean) if (not news_daily.empty and pd.notna(news_mean)) else 0.0
    overall_social = float(social_mean) if (not social_daily.empty and pd.notna(social_mean)) else 0.0
    overall_sentiment = round((overall_news + overall_social) / 2, 4)

    # Trend: compare most recent 7 days vs earliest 7 days
    sorted_df = df.sort_values("date")
    recent_mean = sorted_df.tail(7)["weighted_sentiment"].mean()
    older_mean = sorted_df.head(7)["weighted_sentiment"].mean()
    if recent_mean > older_mean + 0.05:
        trend = "increasing"
    elif recent_mean < older_mean - 0.05:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "ticker": ticker,
        "last_updated": datetime.now().isoformat(),
        "sentiment_score": _safe(overall_sentiment),
        "data_points": len(items),
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
