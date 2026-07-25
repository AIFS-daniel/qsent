# ADR 008: Batch Test-Writing and Implementation in implement.md for Free-Tier Rate Limits

**Date:** 2026-07-18
**Status:** Accepted

## Context

`.github/prompts/implement.md` drives the autonomous `opencode-implement.yml`
workflow, which runs on Gemini 2.5 Flash via the free API tier
(`GOOGLE_GENERATIVE_AI_API_KEY`, see CLAUDE.md). The free tier caps at 10
requests/minute (also 250k TPM and 1,500 RPD as of this writing — free-tier
limits shift often enough that these should be re-verified against Google AI
Studio at execution time; one report notes a 50-80% cut to free-tier limits
in December 2025 alone).

The workflow's TDD loop, per `docs/TDD_AUTONOMOUS_POLICY.md`, previously ran
fully per-behavior: for each behavior, write one test (a model call), confirm
it's red, implement it (another call), confirm it's green. A 3-behavior
feature is 12+ sequential model calls before the refactor, review, and QA
phases even start. RPM, not tokens or daily request count, is almost
certainly the binding constraint here: a loop making 12+ sequential turns for
a 3-behavior feature hits the 10/min ceiling well before any token cap, and
each 429 typically forces a 60-second-or-longer backoff.

This was not a hypothetical concern. PR #54 documented a prior run that spent
approximately 5.5 hours in 429 backoff retries against the model provider
before completing in under 30 seconds of actual work. PR #55 raised the job
timeout to 35 minutes as a stopgap to give a legitimate multi-behavior
session enough headroom to complete without being cut off mid-run, but a
higher timeout only bounds the damage from RPM exhaustion; it does nothing to
reduce the number of calls that trigger it in the first place.

## Decision

Rewrite the TDD workflow section of `.github/prompts/implement.md` to batch
test-writing and implementation instead of looping per-behavior:

1. Understand the feature and produce the full behavior list (unchanged from
   before, one pass).
2. Validate the behavior list against the issue's actual requirements before
   any test or code exists, checking for scope mismatches, missing edge
   cases, or a misunderstood approach. This is cheap here, since revising a
   plan costs nothing before tests and code are written against it.
3. Write tests for all behaviors in a single batch call, with a size guard:
   if the behavior count exceeds roughly 5-6, split test-writing across
   multiple batches of at most 5-6 behaviors each, kept roughly even rather
   than one large batch plus a small remainder. A single call covering too
   many behaviors risks response truncation or a malformed multi-file diff,
   and recovering from a failed large-batch call is more expensive than
   recovering from a failed single-test call.
4. Run the full suite once to confirm every new test is red. This step is
   kept because its value isn't confirming the obvious (no implementation
   exists yet) — it catches tautological assertions, tests that
   accidentally exercise existing code, wrong-target copy-paste errors, and
   silent early-return fixture issues, before implementation and refactor
   get built on top of a broken test.
5. Implement all behaviors in a single batch call, with the same size guard
   as step 3.
6. Run the full suite once. If everything is green, proceed directly to the
   refactor phase. If something is still red, identify exactly which
   behavior(s) failed from the suite output and retry only those, one
   targeted implement-and-recheck cycle per failing behavior, using the
   existing single-test retry cap from `docs/TDD_AUTONOMOUS_POLICY.md`.
   Behaviors that already passed are not touched again.
7. Refactor, review, QA, and CLAUDE.md reconciliation gating are unchanged —
   these already run once per session regardless of behavior count, so
   batching test-writing and implementation doesn't affect them.
8. The Failure Protocol and all existing retry caps in
   `docs/TDD_AUTONOMOUS_POLICY.md` are unchanged.

This applies only to `.github/prompts/implement.md`, the OpenCode/Gemini
free-tier path. It does not change `.claude/commands/tdd.md` or
`.claude/commands/tdd-autonomous.md`, which keep their existing per-behavior
discipline for Claude Code CLI use, where the rate-limit pressure that
motivates this change does not apply.

## Alternatives Considered

**Keep per-behavior looping, raise the timeout further**

Rejected. PR #55 already raised the timeout to 35 minutes as a stopgap. That
bounds the damage from a runaway 429 retry storm but does nothing to reduce
the number of model calls that cause it. A larger feature would still risk
exhausting the timeout, just at a higher behavior count.

**Switch to a paid API key or a different provider**

Rejected for now, not because it wouldn't help, but because it's an
infrastructure/cost decision outside the scope of this change, and it
doesn't preclude batching later even if adopted. If the free-tier constraint
is later removed (a paid key, a higher-limit provider, or the gated
Claude-personal path once set up), this tradeoff should be revisited; see
Consequences.

**Batch everything, including refactor/review/QA into fewer calls**

Rejected. Those phases already run once per session regardless of behavior
count (see `docs/TDD_AUTONOMOUS_POLICY.md`, "Full-Suite Run Cadence" and the
Review/QA phase caps), so they weren't contributing to the O(behaviors)
call growth that per-behavior test-writing and implementation was. There was
no equivalent problem there to fix.

## Consequences

**Reduced isolation within a batch.** If implementing behavior 3 breaks
something behavior 1's test depends on, that surfaces at the single
end-of-batch suite run (step 6) rather than immediately, the way
per-behavior TDD would catch it. This is a real reduction in rigor compared
to strict per-behavior TDD, not a claim that batched verification is
equally thorough.

**Why this is an acceptable tradeoff here.** Issues run through this
autonomous path are scoped to be small, and every resulting PR is
human-reviewed before merge (per CLAUDE.md's autonomous workflow). The
batch-then-targeted-retry approach in step 6 still isolates and re-verifies
individual failing behaviors when the end-of-batch run does catch a
regression; only the fully-green case skips per-behavior confirmation
entirely.

**This is a constraint-driven compromise, not a permanent judgment.** It
exists to make autonomous runs viable at all under the current Gemini
free-tier RPM ceiling, and is scoped to `.github/prompts/implement.md`
only — `.claude/commands/tdd.md` and `.claude/commands/tdd-autonomous.md`
keep strict per-behavior TDD for Claude Code CLI use, since the rate-limit
pressure driving this tradeoff doesn't apply there. If the free-tier
constraint changes (a paid key, a higher-limit provider, or the gated
Claude-personal path once set up), this decision should be revisited rather
than assumed to still hold.

**Not a measured fix for the 5.5-hour incident.** The PR #54/#55 timeout
increase is a hypothesis about where time was lost, not a confirmed
root-cause measurement. There's no direct evidence yet that RPM backoff
during the TDD loop specifically (as opposed to QA/Playwright startup,
Doppler, or checkout steps) was the dominant time sink in that run. This
change will very likely reduce the number of model calls and therefore the
429 backoff exposure, but confirming it closes the gap would require
grepping a prior timed-out run's Actions log for 429s and retry-after gaps
to see where the time actually went.
