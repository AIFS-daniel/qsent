Implement the feature described in this GitHub issue.

Follow the TDD workflow strictly. The retry caps, failure behavior, and
gating rules below are the shared autonomous-mode policy — see
docs/TDD_AUTONOMOUS_POLICY.md for the full definitions; only the specific
sections cited below apply to this single-pass run.

1. Write failing tests first that describe the intended behavior.
2. Implement the minimum code to make each test pass. If a test doesn't turn
   green, retry up to the single-test retry cap ("Bounded Retry Caps" in
   docs/TDD_AUTONOMOUS_POLICY.md).
3. Once every behavior's test is green, run the full test suite exactly
   once — see "Full-Suite Run Cadence" in docs/TDD_AUTONOMOUS_POLICY.md. If
   there are regressions, retry up to the full-suite regression cap in that
   doc.
4. Refactor for clarity if needed, then re-run the full suite once more —
   this is a separate run from step 3, per "Full-Suite Run Cadence" in
   docs/TDD_AUTONOMOUS_POLICY.md.
5. Review the implementation for correctness, security, and test quality.
   If critical issues are found, fix and recheck up to the Review Phase cap
   in docs/TDD_AUTONOMOUS_POLICY.md.
6. Verify the feature per the "QA Gating Rule" in
   docs/TDD_AUTONOMOUS_POLICY.md — browser/screenshot verification if UI
   files changed, otherwise a pass/fail smoke test per behavior with no
   narration. If a behavior fails, fix and reverify up to the QA Phase cap
   in that doc.
7. Reconcile CLAUDE.md only if required — see "CLAUDE.md Reconciliation
   Gating Rule" in docs/TDD_AUTONOMOUS_POLICY.md; otherwise skip this step
   entirely.
8. Open a PR linked to this issue.

If any retry cap above is exhausted, or a precondition can't be resolved
automatically (e.g. a test passes when it should still be failing), follow
the Failure Protocol in docs/TDD_AUTONOMOUS_POLICY.md: stop, print full
diagnostics, and do not proceed to any later step. This run has no state
file and does not resume — here, following the Failure Protocol just means
terminating the workflow instead of continuing.

Branch naming: create a branch named auto/issue-$ISSUE_NUMBER.

PR body must include "Closes #$ISSUE_NUMBER" so the issue closes
automatically on merge and the PR can be detected by other workflows.

Project context is in CLAUDE.md. Tests live in tests/unit/ and
tests/integration/. Run tests with:
  .venv/bin/pytest tests/unit/ tests/integration/
