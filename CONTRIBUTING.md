# Contributing

## Workflow
This repository is maintained with a lightweight GitHub Flow:
1. Create a branch per work item.
2. Keep changes small and focused.
3. Ensure quality gates are satisfied (tests, typing, docs) per the active profile.
4. Update docs/hub/CONTEXT_SNAPSHOT.md at the end of each work item.

## Commit conventions
Use concise, descriptive messages. Recommended format:
- feat: add new capability
- fix: bug fix
- refactor: internal change without behavior change
- docs: documentation changes
- test: tests only
- chore: maintenance

## Pull requests
A change is ready when:
1. Acceptance criteria are met.
2. Profile gates are satisfied.
3. Snapshot updated.
4. Risks and follow ups documented in the PR description.

## Local checks
Projects using this template should define:
- tests
- lint
- type check
- format
- snapshot generation
