Review the open PR linked to issue #$ISSUE_NUMBER.

Check that:
1. The implementation matches the issue requirements
2. Tests exist and cover the new behaviour
3. The full test suite passes

For each finding, assign exactly one severity:

- **Critical** — the implementation is incorrect, insecure, or doesn't
  actually satisfy the issue's requirements. Examples: a security
  vulnerability (injection, auth bypass, secret exposure), a behavior
  that contradicts what the issue asked for, a test that doesn't
  actually exercise the behavior it claims to (false green), or the
  full suite failing.
- **Medium** — the implementation is correct and satisfies the issue,
  but has a real gap that should be fixed before merge: missing
  edge-case test coverage, an error path that isn't handled,
  inconsistent behavior with similar existing code in the repo.
- **Small** — correct and complete, but has minor quality issues that
  don't block merging: naming that could be clearer, minor
  duplication, a docstring/comment that's missing or stale.
- **Nice-to-have** — optional suggestions with no bearing on
  correctness or quality bar: alternative approaches, possible future
  refactors, style preferences not enforced by the repo's existing
  conventions.

Post a comment on the PR summarising your findings, grouped under these
four headings in this order: Critical, Medium, Small, Nice-to-have. If
a category has no findings, write "None" under it rather than omitting
the heading.
