# ADR 003: FinBERT Scoring Reliability — One Request Per Item, Sentinel Returns, Token Retry

**Date:** 2026-03-01
**Status:** Accepted

## Context

The pipeline scores sentiment using the HuggingFace Inference API for ProsusAI/finbert. Three reliability problems emerged during development:

1. **Batching silently drops items.** The HuggingFace Inference API for FinBERT only scores the first element when given a list of inputs. Subsequent items are silently discarded with no error. The original code sent all texts in one call and received one result back.

2. **Failed items shorten the result list, causing zip truncation.** The original `score()` appended to a list only on success, so a timeout on item 3 of 10 produced a 9-element list. When zipped with the 10-element input list, items at the tail (typically Reddit posts appended after news items) were silently dropped and never scored.

3. **Long texts exceed FinBERT's 512-token hard limit.** Reddit posts with selftext and multiple comments can exceed 512 tokens. The HuggingFace API returns HTTP 400 with `"size of tensor a (N) must match the size of tensor b (512)"` and the item fails entirely.

## Decision

**One HTTP request per text item.** Each text is scored in its own `POST` request. This is slower but correct — batching is not available on the free inference tier.

**Sentinel return type (`list[float | None]`).** `score()` pre-fills a list of `None` values with the same length as the input and writes results at their original index. Failed items remain `None` rather than being omitted. The list is always the same length as the input, making it safe to `zip()` without truncation.

```python
scores: list[float | None] = [None] * len(texts)
# ... score each item at scores[idx] ...
# caller: zip(items, scores) is always safe
```

**Character-based trim-and-retry for token overflow.** When the API returns HTTP 400 with `"size of tensor"` in the response body, the text is trimmed by 200 characters and the request is retried in a loop until it succeeds or the text is exhausted.

```python
while True:
    response = requests.post(...)
    if response.status_code == 400 and "size of tensor" in response.text:
        trimmed_len = len(current_text) - 200
        if trimmed_len <= 0:
            response.raise_for_status()
        current_text = current_text[:trimmed_len]
        continue
    response.raise_for_status()
    break
```

## Alternatives Considered

**Batch all texts in one request** — rejected. The HuggingFace Inference API for FinBERT silently returns only one result per batch. This is an undocumented API limitation discovered empirically.

**Raise on first failure, abort the pipeline** — rejected. Individual item failures (timeouts, transient errors) should not discard all successfully scored items. Graceful degradation with `None` sentinels is preferable.

**Precise token counting with the `transformers` tokenizer** — rejected. Using `AutoTokenizer.from_pretrained("ProsusAI/finbert")` would add `transformers` and `torch` (or `tensorflow`) as runtime dependencies — several gigabytes of packages for a single pre-filter. Character-based trimming (200 chars ≈ 50 tokens) is a good enough approximation: the retry loop handles cases where the first trim is insufficient, and the 1800-char pre-filter in `_post_text()` means most texts never hit the limit at all.

**Fixed character cap with no retry** — rejected. A static cap is always a guess; some posts with many short words fit in 512 tokens at 2000 chars while others do not. Retry-on-failure adapts to the actual token count without requiring an offline estimate.

## Consequences

- Scoring 124 items requires 125 HTTP round-trips (1 health check + 124 item requests). At ~3s per request this takes several minutes; acceptable for an async analysis endpoint, not for real-time use.
- The `SentimentModel` protocol return type is `list[float | None]`. Callers must filter `None` values before computing statistics.
- The retry loop adds latency only for items that actually exceed the token limit, which is rare given the 1800-char pre-filter.
