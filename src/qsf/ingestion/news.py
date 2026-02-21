"""
NewsAPI news provider.
"""
import os
from datetime import datetime, timedelta

from newsapi import NewsApiClient


class NewsAPIProvider:
    def get_articles(self, ticker: str, days: int = 30) -> list[dict]:
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
