# ADR 005: Use Company Name for Search Queries Instead of Raw Ticker Symbol

**Date:** 2026-03-03
**Status:** Accepted

## Context

Search queries using the raw ticker symbol produced poor results for tickers that share their name with common English words. For example, searching `FORM` (FormFactor Inc.) returned articles and Reddit posts containing the word "form" in any context — chemistry papers, tax guides, movie reviews, squirrel behaviour articles — none related to the stock. The same problem applies to any ticker that is also a real word: `OPEN`, `WORK`, `RIDE`, `HOOD`, etc.

The `$TICKER` cashtag convention was considered as a fix. It works on social platforms (Reddit, Twitter/X) where cashtags are a deliberate tagging system, but NewsAPI treats `$` as punctuation and strips it, making `$FORM` identical to `FORM`.

## Decision

Fetch the company's legal name from yfinance (already a dependency for market data) and strip legal suffixes to produce a search-friendly brand name:

```python
def company_search_name(full_name: str) -> str:
    name = full_name.split(",")[0]
    name = re.sub(r'\s+(Inc\.?|Corp\.?|Corporation|Ltd\.?|LLC|Co\.)$', '', name, flags=re.IGNORECASE)
    return name.strip()
```

This runs once per pipeline invocation in `_fetch_market_data` via `market.get_company_name(ticker)` and is stored in pipeline state.

The two providers use different query strategies because they have different search semantics:

- **NewsAPI**: `q='"FormFactor"'` — quoted exact phrase only. Cashtags are not supported; adding `OR "$FORM"` would reintroduce the noise problem since `$` is stripped.
- **Reddit**: `query='"FormFactor" OR "$FORM"'` — company name plus cashtag fallback, since Reddit natively understands cashtag notation.

## Alternatives Considered

**Raw ticker as search query** — rejected. Produces irrelevant results for any ticker that is a common English word. 100 articles returned for `FORM`, none about FormFactor.

**`$TICKER` cashtag only** — rejected for NewsAPI (cashtag stripped, no improvement). Kept for Reddit as a fallback alongside company name.

**Quoted ticker (`"FORM"`)** — rejected. Still matches any article quoting the word "form", just with slightly fewer false positives. Does not help with body-text matches.

**External ticker→company API (Alpha Vantage, FMP, etc.)** — rejected. Adds a new dependency and API key. yfinance already provides `longName` and `shortName` via `.info` with no additional cost or dependency.

## Consequences

**Improvements:**
- Unambiguous tickers (`IONQ`, `TSLA`, `NVDA`) are unaffected — their names don't appear in everyday prose.
- For ambiguous tickers, relevant articles surface that were previously buried under noise. For `FORM`, the two actual FormFactor news articles (earnings beat, price target raise) now appear in results.

**Known Limitations:**

1. **Company name doubles as a common term.** `FormFactor` is also the industry term for the physical size/shape of hardware. Searches still return camera reviews, NAS reviews, and phone articles that use "form factor" as a noun. There is no keyword-only solution to this disambiguation.

2. **Paywalled article descriptions.** NewsAPI free plan does not provide full article body. For wire service articles (Seeking Alpha, TD Cowen research), the description is often `"See the rest of the story"`. FinBERT receives only the headline and that stub, which is insufficient context to score confidently — these articles score neutral (0.000) even when the headline clearly signals positive or negative sentiment (e.g. "FormFactor price target raised by $30", "FormFactor reports Q4 EPS beat").

3. **Low-coverage tickers.** Small-cap stocks like `FORM` are rarely discussed on Reddit. Even with the correct search query, Reddit returns general investing posts that happen to mention "form factor" or include the text `$FORM` incidentally. The social sentiment signal for low-coverage tickers is noise, not signal.

4. **yfinance `.info` latency.** `get_company_name()` makes a separate yfinance API call on each pipeline invocation. For well-known tickers this is fast (~100ms), but adds latency for the first cold request.
