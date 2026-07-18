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
