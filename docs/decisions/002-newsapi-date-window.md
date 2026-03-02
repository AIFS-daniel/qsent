# ADR 002: NewsAPI Date Window — 28 Days with ISO Datetime Format

**Date:** 2026-03-01
**Status:** Accepted

## Context

The NewsAPI free plan enforces a rolling 30-day window: articles older than 30 days are not returned and the API returns a `parameterInvalid` error if `from` falls outside that window.

The pipeline originally requested `timedelta(days=30)` back from the current date. On March 1 this produces January 30 — which is 31 days ago (30 intervals, 31 boundaries), causing the API to reject the request. Switching to `days=29` also failed in practice: the API server's clock differs slightly from the local clock, so the boundary can shift by a day depending on when the request is made, making 29 days unreliable near the limit.

The `from` parameter was also passed as a date string (`YYYY-MM-DD`), which the API interprets as midnight UTC. When the local clock is ahead of UTC, "yesterday" in local time is still "today" in UTC, pushing the effective window one day shorter than intended.

## Decision

Request 28 days of history (not 30 or 29) and pass `from` as an ISO 8601 datetime string (`YYYY-MM-DDTHH:MM:SS`) anchored to the current local time.

```python
def news_from_date(now: datetime, days: int = 28) -> str:
    return (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
```

28 days provides a two-day buffer against API server clock skew while still covering the full month of meaningful news history. The datetime format avoids the UTC-midnight truncation issue.

## Alternatives Considered

**`days=30`** — rejected. Off-by-one: 30 intervals back from March 1 is January 30, which is 31 days ago and outside the API window.

**`days=29`** — rejected. Worked on some days but failed when the API server clock was ahead of local time; the boundary shifted and the request was rejected intermittently.

**`days=30` with `timedelta(days=days - 1)` inside the function** — rejected. Obscures intent; 28 days as the explicit default is clearer and requires no mental subtraction.

**Keep date-only format** — rejected. `YYYY-MM-DD` is interpreted as midnight UTC, which truncates the window when the local timezone is UTC+N and introduces a latent off-by-one on those days.

## Consequences

- 28 days of news coverage instead of the theoretical 30-day maximum, accepting a two-day reduction as a reliability buffer.
- The `news_from_date(now, days)` helper is independently testable and the 28-day default is explicit in the function signature.
- If NewsAPI upgrades the free plan window, the buffer can be adjusted in one place.
