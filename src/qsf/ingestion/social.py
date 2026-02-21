"""
Reddit social media provider.
"""
import os
from datetime import datetime, timedelta

import praw

REDDIT_SUBREDDITS = ["stocks", "investing", "wallstreetbets"]


class RedditProvider:
    def get_posts(self, ticker: str, days: int = 30) -> list[dict]:
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
