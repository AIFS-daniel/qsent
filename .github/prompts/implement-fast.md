Implement the change described in this GitHub issue.

This is the lighter, non-TDD autonomous lane, triggered by the `autonomous`
label (as opposed to `autonomous-tdd`, which runs the full red-green-refactor
workflow in .github/prompts/implement.md). Use this flow for tasks where
full TDD discipline isn't the right fit — doc updates, config
clarifications, dependency cleanup, small refactors, or small features where
less process is an acceptable tradeoff for speed. There is no TDD loop, no
red/green cycle, no per-behavior looping, and no forced classification of
what "counts" for this lane.

1. Understand the issue and produce a short plan: what will change, which
   files, and why. Explicitly state whether the change needs new test
   coverage to be verified properly, and if so, roughly what that test would
   check. If existing tests already cover the change adequately, say so and
   note that no new test is needed.
2. Self-review the plan for soundness before touching anything. In
   particular, re-check the test-coverage call from step 1 — if the change
   clearly alters behavior, judging that it needs no test is exactly the
   kind of mistake worth catching here, and it's cheap to catch now since
   nothing has been touched yet.
3. Make the change, including writing any new test identified in the plan.
   There is no confirm-red step and no TDD ceremony — implement the change
   and its test coverage together.
4. Run the full unit test suite once to confirm everything passes,
   including any new tests added in step 3:
     .venv/bin/pytest tests/unit/
5. If tests fail: fix and re-run, up to 4 attempts total. If still failing
   after 4 attempts, follow the Failure Protocol in
   docs/TDD_AUTONOMOUS_POLICY.md — stop, print full diagnostics (which
   test(s) failed, the full output, and how many attempts were made), and
   terminate immediately. Do not proceed to any later step. This flow has
   its own retry cap; the TDD-specific caps in that doc (single-test retry,
   full-suite regression retry, etc.) do not apply here.

Do not create a branch, commit, push, or open a PR yourself. The workflow
wrapper that runs this session owns all of that: it pre-creates a branch
before this session starts, and once this session ends, it checks whether
the working tree is dirty and, if so, commits, pushes, and opens the PR
automatically. Just implement the change and leave the changes uncommitted
in the working tree.

Your final summary should state "Closes #$ISSUE_NUMBER" so the intent is
visible in the run output and logs, but note that you are not the one
opening the PR — the wrapper is — so this text will not automatically
appear in the PR body unless the wrapper is later configured to source it
from your output.

Project context is in CLAUDE.md. Tests live in tests/unit/. Run tests with:
  .venv/bin/pytest tests/unit/
