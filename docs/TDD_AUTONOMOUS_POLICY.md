# Autonomous TDD Policy

This document is the shared autonomous-mode policy that both the Claude Code
`tdd-autonomous` command and the OpenCode `implement`/`review` prompts follow.
It is not owned by either implementation — it describes the rules an
unattended red-green-refactor session must obey regardless of which tool is
running it. Each tool's own instructions (`.claude/commands/tdd-autonomous.md`
for Claude Code, `.github/prompts/implement.md` and `.github/prompts/review.md`
for OpenCode) cover the tool-specific execution mechanics — how steps are
invoked, what agents or prompts do the work, CLI syntax, etc. This document
covers the policy those mechanics must implement.

## Failure Protocol

An autonomous session runs unattended, so no step may pause waiting for a
human. The Failure Protocol fires whenever either of these occurs:

- A bounded retry cap (see below) is exhausted without success.
- A precondition can't be safely resolved automatically (e.g. uncommitted
  changes are found where a fresh session was expected, or a test that should
  fail instead passes).

When it fires:

1. Set the session status to `"failed"`. If the failure is specific to one
   behavior, also mark that behavior's phase as `"failed"`.
2. Print a full diagnostic report, including:
   - Which phase/step failed and why
   - The current behavior (id + description), if applicable
   - The exact failing test name and file, if applicable
   - The full test output or error message from the last attempt
   - How many retries were attempted and the cap that was hit
3. Terminate immediately. Do not proceed to any later phase (commit,
   refactor, review, QA, or PR creation) for the remainder of the run.

Do not loop silently, and do not continue as if a failed step had succeeded.
A later invocation should find the failed status and resume from it, but
resuming does not fix the underlying problem by itself — the diagnostics
printed at failure time are what make the problem visible to whoever is
watching the run's output/logs.

## Bounded Retry Caps

These caps are fixed and must not be raised or invented anew per phase:

| Loop | Cap |
|------|-----|
| Single-test retry (implementation fails to turn a test green) | 2 |
| Full-suite regression retry (after all behaviors are green) | 1 |
| Review Phase critical-issue fix-and-recheck loop | 3 cycles |
| QA Phase per-behavior failure fix-and-reverify loop | 2 cycles |

When a cap is hit, follow the Failure Protocol immediately — do not retry
silently, do not continue to the next phase, and do not leave session state
in a status that could be mistaken for success.

## Full-Suite Run Cadence

Run the full test suite exactly once — after every behavior in the session
has gone green, and before the refactor phase. Do not run the full suite
after each individual behavior; only the single test for that behavior is
confirmed green in the loop itself. Running the full suite once instead of
per-behavior keeps an autonomous session's total test time bounded.

The refactor phase performs its own separate full-suite run afterward, to
confirm the refactor itself didn't regress anything. That run is distinct
from — and in addition to — the one described here.

## QA Gating Rule

Whether QA needs a browser is decided by what changed, not asked:

- **UI files changed** (any of the project's HTML templates, `.js`, or `.css`
  files) — run full browser verification: start the app, exercise each
  behavior, capture screenshots.
- **No UI files changed** — run a lighter check: start the app if needed,
  then for each behavior make one pass/fail request against the relevant
  API endpoint(s) (e.g. via `curl` or an HTTP client library) and confirm
  the expected response. Report exactly one pass/fail line per behavior.
  Skip browser automation and screenshots entirely — there is nothing to
  narrate in this path.

## CLAUDE.md Reconciliation Gating Rule

Skipped by default. Only run this step if, for the current session, at least
one of the following is true:

- A new top-level module file was added under `src/qsf/` (not an edit to an
  existing file).
- A new route decorator (`@app.get`, `@app.post`, `@app.put`, `@app.delete`,
  or equivalent router registration) was added to `src/qsf/api/main.py` —
  not a change to an existing route's contract.

If neither condition holds, skip the step entirely — do not open or re-read
the project's CLAUDE.md at all. If a condition holds, update only the
sections that are now stale; do not rewrite sections the session didn't
affect.

## State File Schema

`.claude/tdd-state.json` is specific to the Claude Code `tdd-autonomous`
command today — it is not a shared file that OpenCode's `implement`/`review`
prompts read or write. As of this writing, neither
`.github/prompts/implement.md` nor `.github/prompts/review.md` references
any state file: an OpenCode run is a single pass within one CI job (write
tests, implement, verify the suite, open or review a PR) with no
resume-across-invocations model, so there is nothing for it to persist
between runs. If OpenCode's prompts later grow a resumable, multi-invocation
workflow like Claude Code's, that should get its own state file rather than
reading or writing `.claude/tdd-state.json` — the schema below (particularly
`branch`, `current_behavior_index`, and per-behavior `phase`) encodes
assumptions about Claude Code's specific phase sequence and retry loops that
shouldn't be treated as a stable cross-tool contract.

Session state is tracked in a JSON file (`.claude/tdd-state.json`) with these
top-level fields:

- `session_id`
- `feature`
- `created_at`
- `status` (includes `"failed"` as a value, per the Failure Protocol, in
  addition to the in-progress/complete values used by the non-autonomous
  flow)
- `branch`
- `behaviors` — list of objects, each with `id`, `description`, `test_name`,
  `test_file`, `phase` (also includes `"failed"` as a value)
- `current_behavior_index`
- `test_files_modified`
- `implementation_files_modified`
- `pr_url`

No fields are added beyond the `"failed"` status/phase values described
above. See `.claude/commands/tdd-autonomous.md` ("Plan Behaviors" section)
for the canonical example of this schema as a fully worked JSON block.

## PR Body Template Structure

The PR opened at the end of a successful session must include, in order:

1. **Plan** — goal (one or two sentences), affected files (with why each is
   touched), the full behavior list, and an assumptions/decisions section
   capturing every judgment call made automatically during the session
   (resume-vs-fresh, ambiguous-requirement interpretations, a
   programmatically-derived branch name, etc.) — only "None" if the session
   genuinely required zero judgment calls.
2. **Behaviors Implemented** — numbered list, each with its test name.
3. **Files Changed** — implementation files and test files.
4. **QA Results** — a table of behavior → pass/fail → one-sentence
   description of what was observed (screenshot-based if UI testing ran,
   API-response-based otherwise).
5. **Test Plan** — a checklist a reviewer can follow to verify manually,
   plus the commands to re-run the full suite (and E2E suite, if relevant).

## Commit Message Formats

Two commit shapes are used during an autonomous session:

```
test: <behavior description> (red-green)
```
— one per behavior, once its single test is confirmed green.

```
refactor: <feature description>
```
— once after the refactor phase, covering all implementation files touched
in that phase plus any regenerated docs.

Both should carry a trailer identifying the automation that authored the
commit (e.g. `Co-Authored-By: <tool> <noreply-address>`); the exact
attribution varies by which tool produced the commit and is not part of
this shared policy.
