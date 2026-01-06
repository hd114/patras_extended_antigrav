# WI-0003 Pre-push quality gates

Status: Done
Owner: You
Created:
Updated:

## Objective
Provide a local, blocking quality gate for pushes that validates the snapshot contract and detects snapshot drift without mutating the working tree.

## Acceptance criteria
- [ ] tools/pre_push_check.sh exists and runs make validate and make snapshot-check.
- [ ] The pre-push hook calls tools/pre_push_check.sh from the repository root.
- [ ] tools/install_git_hooks.sh exists and installs the pre-push hook locally.
- [ ] A push triggers the hook and blocks on failure.
- [ ] The drift check is non-mutating and ignores volatile snapshot metadata (for example Generated at).

## Out of scope
- GitHub-enforced branch protection rules.

## Plan
1. Implement a non-mutating snapshot drift check.
2. Wire it into tools/pre_push_check.sh.
3. Install the git hook locally.
4. Validate behavior for both passing and failing cases.

## Verification
- make validate
- make snapshot-check
- Simulate a failing case by changing a tracked file that affects the snapshot without regenerating it, then run:
  - bash tools/pre_push_check.sh (must fail)
  - make snapshot (then commit) and bash tools/pre_push_check.sh (must pass)
- git push (hook triggers)

## Notes
- This provides local enforcement even when branch protection is not configured.

## Links
- PR:
- Related docs:
  - docs/hub/PIPELINE.md
