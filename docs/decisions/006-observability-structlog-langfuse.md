# ADR 006: Structured Logging with structlog and Agentic Tracing with Langfuse

**Date:** 2026-05-18
**Status:** Accepted

## Context

The pipeline runs five LangGraph nodes per `/analyze` request, each making external API calls to yfinance, NewsAPI, Reddit, and HuggingFace. Before this change, logs were unstructured stdlib output with no correlation IDs. There was no way to answer "what happened when user X analyzed IONQ at 2pm" without grepping raw stdout, and no visibility into which pipeline node failed or how long each external call took.

Two categories of tooling were needed:

1. **Request/response logging** — a structured record of every HTTP request with enough context to reconstruct what happened
2. **Agentic tracing** — a UI showing each pipeline node's inputs, outputs, and child spans (analogous to LangSmith or Splunk for agentic workflows)

A key constraint: **no PII in logs**. Email addresses must never appear in log output. Google's OAuth `sub` field (a permanent, opaque numeric ID like `"116234567890"`) is the right identifier — it can be looked up against the OAuth provider if needed for incident response, without storing PII at rest in logs.

## Decision

### structlog for structured logging

Replace stdlib `logging.basicConfig` with [structlog](https://www.structlog.org/). structlog's `contextvars` integration injects per-request context (`trace_id`, `user_id`) into every log line automatically — no manual threading through function signatures.

`TraceMiddleware` in `main.py` generates a `uuid4().hex` trace ID per request, binds it to structlog's contextvars, and sets it as the `X-Trace-ID` response header. Every log line emitted anywhere in the stack (including third-party libraries via `ProcessorFormatter`) carries that ID.

`LOG_FORMAT=json` switches to machine-readable output for production. Default is the pretty console renderer.

### Langfuse v2 (self-hosted) for agentic tracing

[Langfuse](https://langfuse.com) is an open-source LangSmith equivalent. It persists traces to Postgres, provides a filterable web UI, and supports spans and generations natively.

The v2 server is pinned specifically because v3+ requires ClickHouse as a separate analytics store, which adds significant infrastructure overhead for local development. The SDK is pinned to `langfuse>=2.0,<3.0` accordingly.

Each `/analyze` request creates a Langfuse trace keyed by the same `trace_id` used in structured logs, so a single ID links stdout logs to the Langfuse UI. The five pipeline nodes each create a child span. External API calls (yfinance, NewsAPI, Reddit searches) create grandchild spans. Each FinBERT call creates a `generation` entry.

`user_id` in Langfuse traces is always the Google `sub` — never email.

### Google `sub` as the user identifier

The JWT previously stored email in both `sub` and the payload. The OAuth `sub` (Google's permanent opaque user ID) was discarded. This change captures it at the OAuth callback, stores it in the JWT payload as `google_sub`, and uses it everywhere in observability output.

To trace a specific user's activity: ask them to open `/auth/me` in their browser — it returns their own `google_sub`. Search Langfuse or grep logs by that value.

## Alternatives Considered

**LangSmith** — Free tier (5K traces/month) exists but data leaves the machine and the vendor is Anthropic's competitor LangChain. Rejected in favor of a self-hosted option.

**Langfuse Cloud** — Available and free up to 50K observations/month. Chosen not to use for initial setup to keep all data local. Switching is a three-env-var change with no code modifications.

**Langfuse v3/v4** — Requires ClickHouse. Tested and rejected; the server crashes immediately without it. v2 covers all needed functionality (spans, generations, user/session filtering) with only Postgres.

**Phoenix (Arize)** — Excellent notebook-level debugging tool but in-memory by default. Does not persist traces across restarts. Better suited to offline evaluation than production request tracing.

**OpenTelemetry directly** — Would work but requires more plumbing and a separate collector. Langfuse v2 provides the right abstraction level without OTEL complexity.

**Logging email** — Rejected. Email is PII. Google's `sub` provides the same traceability for incident response without creating a PII liability in log storage.

## Consequences

**Improvements:**
- Every log line carries `trace_id` and `user_id` — requests are fully reconstructable from stdout
- Langfuse UI shows the full chain: request → LangGraph node → external API call → result, with durations
- Failed FinBERT calls surface individually in the trace with the input text and error status
- `X-Trace-ID` response header lets clients (or support staff) reference a specific request
- No PII in any log output

**Operational notes:**
- Langfuse requires Docker to run locally (`docker compose -f docker-compose.langfuse.yml up -d`)
- If `LANGFUSE_PUBLIC_KEY` is not set, tracing is silently disabled — the pipeline runs normally without it
- `langfuse.flush()` is called after each pipeline invocation to ensure traces are sent before the HTTP response returns
- Switching to Langfuse Cloud: change `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in `.env`. No code changes required.
