Implement the feature described in this GitHub issue.

Follow the TDD workflow strictly. The retry caps, failure behavior, and
gating rules below are the shared autonomous-mode policy — see
docs/TDD_AUTONOMOUS_POLICY.md for the full definitions; only the specific
sections cited below apply to this single-pass run.

This workflow runs on the free-tier Gemini API key, which caps at 10
requests/minute. A full per-behavior loop (one call to write a test, one to
confirm red, one to implement, one to confirm green — repeated per behavior)
costs 12+ calls before refactor/review/QA even start, and burns through that
RPM cap fast enough to stall the whole run in 429 backoff (see ADR 008 in
docs/decisions/). So test-writing and implementation are batched across all
behaviors instead of looped one at a time, with a size guard to keep any
single call from covering too much at once, and a plan-validation pass up
front since revising an unwritten plan is cheap.

1. Understand the feature and produce the full list of behaviors to
   implement, same as before — one pass, no looping here.
2. Validate that behavior list against the issue's actual requirements
   before writing any test or code: check for scope mismatches, missing
   edge cases, or a misunderstood approach. Do this now, while revising the
   plan is still free of any test/code sunk cost.
3. Write tests for all behaviors in a single batch call. If the behavior
   count is more than 5-6, split test-writing across multiple batch calls
   of at most 5-6 behaviors each instead of one call covering all of them
   (e.g. 12 behaviors → three batches of 4, or four of 3; keep batches
   roughly even rather than one large batch plus a small remainder). A
   single call covering too many behaviors risks response truncation or a
   malformed multi-file diff, and recovering from a failed large-batch call
   is more expensive than recovering from a failed single-test call.
4. Run the full test suite exactly once to confirm every new test is
   red/failing. Do not skip this: it doesn't just confirm the obvious (no
   implementation exists yet) — it also catches tautological assertions,
   tests that accidentally exercise existing code, wrong-target copy-paste
   errors, and silent early-return fixture issues, before implementation
   and refactor get built on top of a broken test.
5. Implement all behaviors in a single batch call, applying the same size
   guard as step 3: batches of at most 5-6 behaviors each once the count
   exceeds that threshold, using the same batch groupings produced for
   test-writing in step 3 so each implementation batch lines up with the
   tests it needs to turn green.
6. Run the full test suite exactly once — see "Full-Suite Run Cadence" in
   docs/TDD_AUTONOMOUS_POLICY.md.
   - If everything is green, proceed directly to the refactor phase (step
     7 below).
   - If something is still red, identify exactly which behavior(s) failed
     from the suite output and retry only those, individually — one
     targeted implement-and-recheck cycle per failing behavior, using the
     single-test retry cap ("Bounded Retry Caps" in
     docs/TDD_AUTONOMOUS_POLICY.md). Do not re-touch behaviors that already
     passed.
7. Refactor for clarity if needed, then re-run the full suite once more —
   this is a separate run from step 6, per "Full-Suite Run Cadence" in
   docs/TDD_AUTONOMOUS_POLICY.md.
8. Review the implementation for correctness, security, and test quality.
   If critical issues are found, fix and recheck up to the Review Phase cap
   in docs/TDD_AUTONOMOUS_POLICY.md.
9. Verify the feature per the "QA Gating Rule" in
   docs/TDD_AUTONOMOUS_POLICY.md — browser/screenshot verification if UI
   files changed, otherwise a pass/fail smoke test per behavior with no
   narration. If a behavior fails, fix and reverify up to the QA Phase cap
   in that doc.
10. Reconcile CLAUDE.md only if required — see "CLAUDE.md Reconciliation
    Gating Rule" in docs/TDD_AUTONOMOUS_POLICY.md; otherwise skip this step
    entirely.

Do not create a branch, commit, push, or open a PR yourself. The workflow
wrapper that runs this session owns all of that: it pre-creates a branch
before this session starts, and once this session ends, it checks whether
the working tree is dirty and, if so, commits, pushes, and opens the PR
automatically. Just implement the change and leave the changes uncommitted
in the working tree.

Note: the wrapper's PR creation has no documented way to customize the PR
title or body, so the resulting PR body will be minimal and auto-generated
by the wrapper rather than following the "PR Body Template Structure" in
docs/TDD_AUTONOMOUS_POLICY.md (Plan, Behaviors Implemented, Files Changed,
QA Results, Test Plan). That richer structure no longer applies to this
workflow's PRs unless a different mechanism to feed the wrapper a custom
body is found later.

If any retry cap above is exhausted, or a precondition can't be resolved
automatically (e.g. a test passes when it should still be failing), follow
the Failure Protocol in docs/TDD_AUTONOMOUS_POLICY.md: stop, print full
diagnostics, and do not proceed to any later step. This run has no state
file and does not resume — here, following the Failure Protocol just means
terminating the workflow instead of continuing.

Your final summary should state "Closes #$ISSUE_NUMBER" so the intent is
visible in the run output and logs, but note that you are not the one
opening the PR — the wrapper is — so this text will not automatically
appear in the PR body unless the wrapper is later configured to source it
from your output.

Project context is in CLAUDE.md. Tests live in tests/unit/ and
tests/integration/. Run tests with:
  .venv/bin/pytest tests/unit/ tests/integration/
