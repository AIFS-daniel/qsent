# Autonomous TDD Orchestration Loop

You are the autonomous TDD orchestrator for QSent. Your job is to run a strict red-green-refactor cycle using the existing agents: `test-writer`, `test-runner`, `code-builder`, and `code-reviewer`.

This command is the headless sibling of `/tdd`. Every checkpoint where `/tdd` would ask a human and wait is instead resolved automatically here, using the most reasonable interpretation available and logged as an assumption in the eventual PR body (not as a new state file field), or treated as a hard failure that terminates the run. It exists for future unattended/CI use. It is not currently wired into any GitHub Actions workflow.

## Arguments

$ARGUMENTS

## Failure Protocol

This command runs unattended, so no step may pause waiting for a human. When a retry limit is exhausted, a precondition is violated (e.g. uncommitted changes before a fresh session), or an ambiguous state can't be safely resolved automatically, do not loop silently and do not continue as if the step succeeded. Instead:

1. Update `.claude/tdd-state.json`: set `"status"` to `"failed"`. This is an additional value on the existing `status` field, not a new field. If the failure is specific to one behavior, also set that behavior's `"phase"` to `"failed"` (an additional value on the existing per-behavior `phase` field).
2. Print a diagnostic report to the command output, including:
   - Which phase/step failed and why
   - The current behavior (id + description), if applicable
   - The exact failing test name and file, if applicable
   - The full test output or error message from the last attempt
   - How many retries were attempted and the limit that was hit
3. Terminate execution of this command immediately. Do not proceed to any later phase (commit, refactor, review, QA, or PR creation) for the remainder of this run.

A later invocation will find `status: "failed"` and, per **On Invocation** below, resume automatically. Resuming does not fix the underlying problem by itself — the printed diagnostics from the failed run are what makes the problem visible to whoever is watching the job's output/logs.

## On Invocation

1. Check whether `.claude/tdd-state.json` exists.
   - If it exists and `status` is not `"complete"`, read it and resume automatically from the recorded `current_behavior_index` and `status` — do not ask, and never discard it silently. Log to console output that an in-progress session for `[feature]` was found and is being resumed. Skip **Understand the Feature** and **Branch Setup** if `"branch"` is already set in the state file; resume directly into whichever phase `status` indicates (**The TDD Loop**, **Full Suite Check**, **Refactor Phase**, **Review Phase**, or **QA Phase**).
   - If it does not exist, or `status` is `"complete"`, proceed to **Understand the Feature** for a new session.

## Understand the Feature

Before writing any code or tests, work out what needs to be built. There is no one to confirm this with, so resolve ambiguity with the best available evidence and be explicit about every judgment call.

1. Read `$ARGUMENTS` carefully. If `$ARGUMENTS` is empty and no state file exists, follow the **Failure Protocol** — there is nothing to build.
2. Search the codebase for relevant existing code — find the module(s) the feature touches, related tests, and any interfaces or patterns it must follow. Use Glob and Grep to explore.
3. Compose the same summary `/tdd` would present to a human, covering:
   - **What you understand the goal to be** — one or two sentences on the problem being solved
   - **Where the change lives** — which files and modules will be affected
   - **What you're planning to implement** — a numbered list of the behaviors you intend to build, each as a single sentence (e.g. "normalize_ticker strips leading and trailing whitespace")
   - **Any assumptions or open questions** — anything ambiguous in the request that could affect the approach, and the interpretation you're proceeding with
4. Do not ask a question and do not wait. Proceed immediately using the most reasonable interpretation of `$ARGUMENTS`, informed by existing patterns in the codebase. Log the summary to console output.
5. Carry the **Any assumptions or open questions** bullets forward — they become the **Assumptions / decisions made** section of the PR body in **Create PR** below. They are not written to `.claude/tdd-state.json`; the schema does not gain new fields for this.

## Branch Setup

Before planning behaviors, ensure you are on a fresh feature branch:

1. Check the current branch with `git branch --show-current`
2. If already on a branch other than `main` and the state file confirms this is a resume, skip — the branch was set up in a previous session.
3. Otherwise:
   - If there are uncommitted changes, follow the **Failure Protocol** — do not guess whether to stash or commit someone else's uncommitted work.
   - Run `git checkout main && git pull` to fetch the latest.
   - Derive a branch name programmatically. No prompt. Slugify the feature description: lowercase it, drop filler words (a, an, the, to, for, with, of), keep roughly the first five significant words, replace remaining whitespace/punctuation with single hyphens, and trim to about 40 characters. Prefix with `auto/`, mirroring the `auto/issue-{N}` convention the GitHub Actions autonomous workflow already uses (this command takes a freeform feature description rather than an issue number, so the slug takes the place of `issue-{N}`). Example: "add a normalize_ticker function that strips whitespace and uppercases the ticker" derives `auto/normalize-ticker`.
   - If a local branch with that exact name already exists, append `-2`, `-3`, etc. until the name is free.
   - Run `git checkout -b <branch-name>`.
   - Record the branch name in the state file under `"branch"`. Carry forward, as an assumption for **Create PR**, that the branch name was derived automatically rather than chosen by a human.

## Plan Behaviors

Use the behavior list produced in the **Understand the Feature** phase — do not re-derive from `$ARGUMENTS`. Each behavior maps to one item in the state file.

Write the initial state file to `.claude/tdd-state.json`. The schema is identical to `/tdd`'s — no fields are added or removed:

```json
{
  "session_id": "<feature-slug>-<YYYYMMDD-HHMM>",
  "feature": "<feature description from $ARGUMENTS>",
  "created_at": "<ISO 8601 timestamp>",
  "status": "in_progress | refactoring | reviewing | qa | complete | failed",
  "branch": "<feature branch name>",
  "behaviors": [
    {
      "id": 1,
      "description": "<one sentence describing the behavior>",
      "test_name": "",
      "test_file": "",
      "phase": "not_started"
    }
  ],
  "current_behavior_index": 0,
  "test_files_modified": [],
  "implementation_files_modified": [],
  "pr_url": ""
}
```

The only difference from `/tdd`'s schema is the additional `"failed"` value available on `status` (and, per-behavior, on `phase`) — used by the **Failure Protocol**. No new fields.

Log the behavior list to console output, then proceed immediately into the loop. Do not pause for confirmation.

## The TDD Loop

Repeat for each behavior (starting at `current_behavior_index`):

### Step 1 — Write one failing test

Invoke `test-writer` with:
- The behavior description
- The intended public interface (function/method signature, expected inputs and outputs)
- The test tier (unit/integration/e2e) and target file
- Instruction: write exactly one test for this behavior using Given-When-Then naming; the implementation does not exist yet so the test must fail

After the agent completes, update state: set `test_name`, `test_file`, and `phase` → `"writing"` for this behavior.

### Step 2 — Confirm red

Invoke `test-runner` with:
- Instruction: run **only this one test** with `.venv/bin/pytest <test_file>::<test_name> -x` and report the exact failure output

Evaluate the result:
- **ImportError or AttributeError** — correct. The code doesn't exist yet. Proceed.
- **AssertionError** — correct. The behavior exists but produces wrong output. Proceed.
- **SyntaxError** — the test has a syntax error. Send back to `test-writer` to fix. Do not advance.
- **Test passes** — the behavior is already implemented or the test is not exercising unwritten code. Follow the **Failure Protocol** — there is no safe way to guess intent here, so terminate rather than silently skipping or overwriting existing behavior.

Update state: `phase` → `"red"`.

### Step 3 — Implement

Invoke `code-builder` with:
- The behavior description
- The failing test name and file
- The exact failure output from Step 2
- Instruction: write the minimum code to make this one test pass — nothing more

Update state: add any modified files to `implementation_files_modified`.

### Step 4 — Confirm green

Invoke `test-runner` with:
- Instruction: run **only this one test**: `.venv/bin/pytest <test_file>::<test_name> -x`

If still red:
- Return to `code-builder` with the new failure output. Maximum 2 retries.
- If still failing after 2 retries, follow the **Failure Protocol**. Do not loop silently.

Do not run the full suite here. The full suite is checked exactly once, after every behavior in the loop is green — see **Full Suite Check** below — not per behavior.

Update state: `phase` → `"green"`. Advance `current_behavior_index`. Write updated state file.

### Step 5 — Commit

Stage only the files written or modified for this behavior (the test file and any implementation files touched in Steps 3–4 — both tracked in state). Then commit with a message in this format:

```
test: <behavior description> (red-green)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Example: `test: normalize_ticker uppercases ticker symbols (red-green)`

Do not stage unrelated files. Do not push.

### Repeat

Continue the loop for the next behavior.

## Full Suite Check

Once every behavior has reached `phase: "green"` (the loop above has completed for all entries in `behaviors`):

Invoke `test-runner` once with:
- Instruction: run the full suite `.venv/bin/pytest` and report any failures

If there are regressions:
- Return to `code-builder` with the regression output. Maximum 1 retry.
- If regressions persist, follow the **Failure Protocol**.

This is the only full-suite run before refactor. It replaces the per-behavior full-suite check `/tdd` performs after every individual behavior — running it once here instead keeps an autonomous session's total test time bounded rather than re-running the whole suite after every behavior.

Once the full suite is green, proceed to **Refactor Phase**.

## Refactor Phase

Once the **Full Suite Check** above has passed:

Update state: `status` → `"refactoring"`.

Invoke `code-builder` with:
- The list of all implementation files modified
- Instruction: refactor for clarity and design — eliminate duplication, improve naming, simplify structure. Do not change behavior. Do not add features.

Invoke `test-runner` with:
- Instruction: run the full suite `.venv/bin/pytest`
- If any tests fail, return to `code-builder` once. If still failing, follow the **Failure Protocol**.

(This is a second, separate full-suite run — it checks the refactor itself, distinct from the **Full Suite Check** above.)

Regenerate living feature docs (test names are stable after refactor):
```
python scripts/generate_features.py
```
If `docs/FEATURES.md` was written or changed, stage it alongside the implementation files.

**CLAUDE.md reconciliation is skipped by default.** Only run this step if at least one of the following is true for this session:
- A new top-level module file was added under `src/qsf/` — check with `git diff --diff-filter=A --name-only main...<branch> -- src/qsf/` and confirm it's a genuinely new file, not an edit to an existing one.
- A new API endpoint was introduced — check the diff of `src/qsf/api/main.py` and confirm it adds a new route decorator (`@app.get`, `@app.post`, `@app.put`, `@app.delete`, or equivalent router registration) rather than only modifying an existing one.

If neither condition holds, skip this step entirely — do not open or re-read `CLAUDE.md`.

If a condition holds, check `implementation_files_modified` against the current `CLAUDE.md` content and update only the sections that are now stale or incomplete. Sections to check:

- **Key source files** — if a new module or file was added to `src/qsf/`, add it with a one-line description
- **API Endpoints** — if a new endpoint was added or an existing one changed its contract, update the list
- **Provider Pattern** — if a new provider protocol or concrete implementation was added, reflect it
- **Pipeline flow** — if the LangGraph graph structure changed (new node, reordered edges), update the flow description
- **Placeholder Modules** — if a placeholder was implemented (moved from placeholder to real), remove it from that section
- **Data Sources table** — if a new external data source was introduced, add a row

Do not rewrite sections unaffected by this session. Do not change the Agent Workflow, /tdd Command, /tdd-autonomous Command, or Architecture Decision Records sections. If, after checking, nothing in `CLAUDE.md` is actually stale, skip staging it.

If `CLAUDE.md` was changed, stage it alongside the implementation files.

Commit the refactor:
```
refactor: <feature description>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

Stage all implementation files modified during the session plus docs/FEATURES.md if it changed. Do not push yet.

## Review Phase

Update state: `status` → `"reviewing"`.

Invoke `code-reviewer` with:
- All implementation files and test files modified during the session
- Instruction: review for correctness, security, consistency, and test quality

If the reviewer flags any **Critical** issues:
- Return to `code-builder` with the specific findings
- Re-run `test-runner` to confirm the suite is still green after fixes
- Re-run `code-reviewer` to confirm the critical issues are resolved
- Maximum 3 cycles of this fix-and-recheck loop. If Critical issues still remain after 3 cycles, follow the **Failure Protocol** — findings surviving 3 rounds usually mean the model is stuck on something a 4th attempt won't crack (an ambiguous requirement or a design issue beyond "fix this bug"), not a transient miss.
- Do not proceed to QA until no Critical issues remain

## QA Phase

Update state: `status` → `"qa"`.

**Check whether any UI files were modified** (look in `implementation_files_modified` for changes to `index.html`, `login.html`, or any `.js`/`.css` files):

- **UI files changed** — invoke `qa` with the full browser testing instruction: start the server, write a Playwright script per behavior, take screenshots, describe them, run the full E2E suite.
- **No UI files changed** — invoke `qa` with a lighter instruction: start the server if needed, then for each behavior run a single check against the relevant API endpoint(s) with `curl` or `httpx` and confirm the expected response. Report exactly one pass/fail line per behavior with minimal commentary — no per-screenshot narration, since there are no screenshots in this path. Skip Playwright and screenshots entirely.

Invoke `qa` with:
- The feature description
- The session ID (from `session_id` in the state file — used as the screenshot directory name)
- The list of behaviors implemented (descriptions + test names)
- The implementation files modified
- Whether UI testing is needed (based on the check above)

If any behavior fails QA:
- Return to `code-builder` with the failure description and screenshot context
- Re-run `test-runner` to confirm tests still pass after the fix
- Re-run `qa` to verify
- Maximum 2 cycles of this fix-and-reverify loop per behavior (matching the single-test retry cap used elsewhere in this command, rather than inventing a new number). If the behavior still fails QA after 2 cycles, follow the **Failure Protocol**.

If the E2E suite has regressions, follow the **Failure Protocol** before marking complete.

## Create PR

Before marking complete, commit any screenshots and open a pull request.

The **Assumptions / decisions made** bullet below must capture every judgment call made automatically during this session: the resume-vs-fresh decision (if applicable), the interpretation adopted for any ambiguous part of the feature description, and the programmatically-derived branch name. Do not leave it as "None" unless the session genuinely required zero judgment calls.

**1. Push the branch:**
```bash
git push -u origin <branch>
```

**2. Create the PR** using `gh pr create`. The body must include:

- **Plan** — the full understanding from the start of the session: the problem being solved, affected files and modules, the complete behavior list, and any assumptions or open questions that were resolved
- **Behaviors implemented** — numbered list, each with its test name
- **Files changed** — implementation files and test files
- **QA results** — for each behavior: pass/fail and a one-sentence description of what was observed. If no UI changes were made, describe the API response verified instead.
- **Test plan checklist** — steps a reviewer can follow to verify the feature manually

Use this format:
```
gh pr create --title "<feature description>" --body "$(cat <<'EOF'
## Plan

**Goal:** <one or two sentences describing the problem being solved and why>

**Affected files:**
- `<file>` — <why it's touched>

**Behaviors agreed with the author:**
1. <behavior description>
2. ...

**Assumptions / decisions made:**
- <any open questions that were resolved during the understanding phase, or "None" if there were none>

## Behaviors Implemented

1. <behavior> — `<test_name>`
2. ...

## Files Changed

- `<implementation file>`
- `<test file>`

## QA Results

| Behavior | Result | Notes |
|----------|--------|-------|
| <behavior> | Pass / Fail | <one sentence description of what was observed in the screenshot> |

## Test Plan

- [ ] <manual verification step>
- [ ] Run `.venv/bin/pytest` — all tests pass
- [ ] Run `.venv/bin/pytest tests/e2e/` — E2E suite passes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Record the PR URL in the state file under `"pr_url"`.

## Complete

Update state: `status` → `"complete"`.

Print a summary:
- PR URL
- Behaviors implemented
- QA result
- Any issues flagged by the reviewer

This printed summary is the primary record of what happened in an unattended run — make sure it's complete. The state file is left in place as a record; delete `.claude/tdd-state.json` manually (or via a separate cleanup step) once the PR is merged.

## General Rules

- Update `.claude/tdd-state.json` after every phase transition — not just at the end of each behavior.
- Never skip a step. Confirm red before implementing. Confirm single-test green before advancing to the next behavior.
- Run the full suite exactly once, after every behavior is green and before the Refactor phase — not after each individual behavior. The Refactor phase's own full-suite run afterward is separate and stays as-is.
- Never loop more than the specified retry limit. When a limit is reached, follow the **Failure Protocol** immediately — do not retry silently, do not continue to the next phase, and do not leave the state file in a status that could be mistaken for success.
- This command never pauses to ask a question. Every checkpoint that `/tdd` would ask a human about is instead resolved automatically (see the relevant phase above) and logged as an assumption for the PR body, or treated as a hard failure per the **Failure Protocol**.
- If `$ARGUMENTS` is empty and no state file exists, there is nothing to build — follow the **Failure Protocol** immediately rather than waiting for input.
