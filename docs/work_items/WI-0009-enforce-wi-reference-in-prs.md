# WI-0009 Enforce Work Item reference in pull requests

Status: Done
Owner: You
Created:
Updated:

## Objective
Ensure every pull request references a Work Item ID (WI-XXXX) in the PR title or PR description so changes remain traceable to the docs/work_items system of record.

## Acceptance criteria
- [ ] CI includes a pull_request check that fails if neither the PR title nor PR body contains a pattern like WI-0000 (WI- followed by 4 digits).
- [ ] The check runs on pull_request events and reliably re-evaluates after PR updates (for example after a new commit).
- [ ] The PR template prompts contributors to include a Work Item ID.
- [ ] A PR without a WI reference fails with a clear, actionable error message.
- [ ] A PR with a WI reference passes.

## Out of scope
- Enforcing that the WI exists as a file (only pattern enforcement is required here).
- Enforcing branch naming conventions.

## Plan
1. Add a minimal CI job that validates PR title/body.
2. Add a PR template section for Work Item ID.
3. Validate with one failing and one passing PR.

## Verification
- Create a PR without any WI reference and confirm CI fails.
- Add WI-0000 to PR title or body (or push a new commit) and confirm CI passes.

## Notes
- If PR edits do not trigger a new run reliably, a new commit will always trigger re-evaluation via pull_request synchronize.

## Links
- PR:
- Related docs:
  - docs/hub/PIPELINE.md
