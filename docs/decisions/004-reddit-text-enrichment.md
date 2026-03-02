# ADR 004: Reddit Text Enrichment — Title + Selftext + Top Comments by Upvote

**Date:** 2026-03-01
**Status:** Accepted

## Context

Reddit posts fetched via PRAW were originally scored using the post title alone. Titles are typically 5–15 words — far too short for FinBERT to produce a non-neutral score. In practice, nearly all Reddit items scored 0.0 (neutral), making the social sentiment signal useless.

Two additional problems were present:

1. **`MoreComments` placeholders.** PRAW's `post.comments` list contains `MoreComments` objects (lazy-load stubs for collapsed threads) alongside real `Comment` objects. Iterating without handling these raises exceptions or silently processes stub objects.

2. **No comment quality ranking.** Comments fetched in default order (PRAW's "best" sort, which is opaque) mix high-signal, community-validated opinions with low-effort noise.

## Decision

Build a single text block per post using `_post_text()`:

1. Start with the post title.
2. Append `selftext` (the post body) if non-empty.
3. Call `post.comments.replace_more(limit=0)` to discard `MoreComments` placeholders and iterate only real comments.
4. Sort comments by `comment.score` (upvote count) descending and take the top 5.
5. Append comment bodies in upvote order.
6. If the combined text exceeds 1800 characters, drop the lowest-upvoted comment and repeat until it fits.

```python
REDDIT_MAX_COMMENTS = 5
REDDIT_MAX_CHARS = 1800  # ~450 BERT tokens — soft cap before retry kicks in

def _post_text(post) -> str:
    parts = [post.title]
    if post.selftext and post.selftext.strip():
        parts.append(post.selftext.strip())
    post.comments.replace_more(limit=0)
    top_comments = sorted(post.comments, key=lambda c: getattr(c, "score", 0), reverse=True)[:REDDIT_MAX_COMMENTS]
    for comment in top_comments:
        body = getattr(comment, "body", "").strip()
        if body:
            parts.append(body)
    text = ". ".join(parts)
    while len(text) > REDDIT_MAX_CHARS and len(parts) > 1:
        parts.pop()
        text = ". ".join(parts)
    return text
```

The 1800-character limit (~450 BERT tokens) acts as a soft pre-filter. Items that still exceed FinBERT's 512-token hard limit are handled by the trim-and-retry logic in `FinBERTModel.score()` (see ADR 003).

## Alternatives Considered

**Title only** — rejected. Too short for FinBERT to produce non-neutral scores in practice; produced 0.0 for nearly every Reddit post regardless of content.

**One FinBERT call per comment** — rejected. Multiplies API calls (5 comments = 5 calls per post, vs 1), dramatically increases latency for 30 posts × 3 subreddits, and loses the contextual relationship between title and comments.

**Sort comments by recency instead of upvotes** — rejected. Recency favors new, unvetted opinions. Upvote ranking surfaces the comments the community has validated as most insightful or representative.

**`replace_more(limit=None)` (fetch all collapsed comments)** — rejected. Triggers additional PRAW API requests to expand collapsed threads, significantly increasing latency and Reddit API quota usage. `limit=0` discards stubs without extra requests, which is sufficient for the top-5 use case.

**Include all comments up to character limit** — rejected. Without a count cap, a post with 50 short comments would still include many low-quality or off-topic replies before hitting the character limit. Capping at 5 by upvote keeps only the highest-signal content.

## Consequences

- Social sentiment coverage improved from 0/30 to 12/30 trading days for TSLA, with 24 social items scored vs 0 previously.
- Posts with only a title (no selftext or comments) still score neutral. This is a data quality limitation of low-activity tickers, not a code bug.
- Reddit search surfaces posts that mention a ticker incidentally (not as the primary topic). These score neutral correctly — FinBERT is responding to the actual financial content, which in these cases is not ticker-specific.
- The single combined text preserves context across title, body, and comments, which is more appropriate for FinBERT's sequence classification than scoring each part independently.
