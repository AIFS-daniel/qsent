Implement the feature described in this GitHub issue.

Follow the TDD workflow strictly:
1. Write failing tests first that describe the intended behavior
2. Implement the minimum code to make the tests pass
3. Verify the full test suite is green
4. Open a PR linked to this issue

Branch naming: create a branch named auto/issue-$ISSUE_NUMBER.

PR body must include "Closes #$ISSUE_NUMBER" so the issue closes
automatically on merge and the PR can be detected by other workflows.

Project context is in CLAUDE.md. Tests live in tests/unit/ and
tests/integration/. Run tests with:
  .venv/bin/pytest tests/unit/ tests/integration/
