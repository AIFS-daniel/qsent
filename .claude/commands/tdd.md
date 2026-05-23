# TDD Orchestration Loop

You are the TDD orchestrator for QSent. Your job is to run a strict red-green-refactor cycle using the existing agents: `test-writer`, `test-runner`, `code-builder`, and `code-reviewer`.

## Arguments

$ARGUMENTS

## On Invocation

1. Check whether `.claude/tdd-state.json` exists.
   - If it exists and `status` is not `complete`, read it and ask the user: "Found an in-progress TDD session for: [feature]. Resume, or start fresh?" Wait for their answer before proceeding.
   - If it does not exist (or user chooses fresh), proceed to **Plan Behaviors**.

## Branch Setup

Before planning behaviors, ensure you are on a fresh feature branch:

1. Check the current branch with `git branch --show-current`
2. If already on a branch other than `main` and the state file confirms this is a resume, skip — the branch was set up in the previous session.
3. Otherwise:
   - If there are uncommitted changes, stop and tell the user to stash or commit them first.
   - Run `git checkout main && git pull` to fetch the latest.
   - Ask the user for a branch name, suggesting one derived from the feature description (e.g. `feature/normalize-ticker`).
   - Run `git checkout -b <branch-name>`.
   - Record the branch name in the state file under `"branch"`.

## Plan Behaviors

From the feature description in `$ARGUMENTS`, enumerate the distinct observable behaviors to implement. Each behavior must map to exactly one public-interface action and its expected outcome. Aim for 2–6 behaviors — if you think of more, consider whether the feature scope is too large.

Write the initial state file to `.claude/tdd-state.json`:

```json
{
  "session_id": "<feature-slug>-<YYYYMMDD-HHMM>",
  "feature": "<feature description from $ARGUMENTS>",
  "created_at": "<ISO 8601 timestamp>",
  "status": "in_progress",
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
  "implementation_files_modified": []
}
```

Show the behavior list to the user and confirm before starting the loop.

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
- **Test passes** — the behavior is already implemented or the test is not exercising unwritten code. Surface this to the user and stop. Do not proceed until the user clarifies.

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
- Instruction: run **only this one test** first: `.venv/bin/pytest <test_file>::<test_name> -x`

If still red:
- Return to `code-builder` with the new failure output. Maximum 2 retries.
- If still failing after 2 retries, surface to the user with full context and pause. Do not loop silently.

Once the single test passes, invoke `test-runner` again with:
- Instruction: run the full suite `.venv/bin/pytest` and report any failures

If there are regressions:
- Return to `code-builder` with the regression output. Maximum 1 retry.
- If regressions persist, surface to the user and pause.

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

## Refactor Phase

Once all behaviors are `"green"`:

Update state: `status` → `"refactoring"`.

Invoke `code-builder` with:
- The list of all implementation files modified
- Instruction: refactor for clarity and design — eliminate duplication, improve naming, simplify structure. Do not change behavior. Do not add features.

Invoke `test-runner` with:
- Instruction: run the full suite `.venv/bin/pytest`
- If any tests fail, return to `code-builder` once. If still failing, surface to the user.

## Review Phase

Update state: `status` → `"reviewing"`.

Invoke `code-reviewer` with:
- All implementation files and test files modified during the session
- Instruction: review for correctness, security, consistency, and test quality

## Complete

Update state: `status` → `"complete"`.

Print a summary:
- Feature implemented
- Behaviors covered (list each with test name)
- Files changed (implementation + tests)
- Any issues flagged by the reviewer

The state file is left in place as a record. The user can delete `.claude/tdd-state.json` manually.

## General Rules

- Update `.claude/tdd-state.json` after every phase transition — not just at the end of each behavior.
- Never skip a step. Confirm red before implementing. Confirm single-test green before running the full suite.
- Never loop more than the specified retry limit without surfacing to the user.
- If `$ARGUMENTS` is empty and no state file exists, ask the user what feature to implement before proceeding.
