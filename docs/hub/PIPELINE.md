# Pipeline

This document defines the collaboration loop for working with an LLM across multiple chats without autonomous agents.

## Inner loop (seconds to minutes)
Goal: fast, safe iteration while editing code or docs.

Prevent
- Keep changes small and focused.
- Write or update acceptance criteria before coding.
- Use the snapshot and rules as the contract.

Detect
- Validate assumptions against the codebase.
- Prefer tests for behavior and invariants.
- Run make validate frequently.

Correct
- If something is unclear, reduce scope and isolate the change.
- Roll back quickly when a change increases risk or breaks invariants.
- Keep diffs reversible.

## Middle loop (hours to days)
Goal: complete a work item end to end.

Prevent
- One branch per work item.
- Keep a clear Definition of Done in the WI file.
- Update snapshot sections as the work evolves.

Detect
- Run the full verification list for the WI.
- Ensure docs and snapshot reflect reality.

Correct
- Split oversized work items.
- Create an ADR when interfaces or architecture change.

## Outer loop (weeks to months)
Goal: keep the repository and workflow maintainable.

Prevent
- Reduce workspace confusion by keeping docs/hub authoritative.
- Keep the snapshot compact and current.
- Maintain strict zones and quality gates.

Detect
- Periodically audit documentation drift.
- Review CI failures and recurring manual steps.

Correct
- Automate repetitive snapshot sections once stable.
- Refactor documentation structure if onboarding becomes slow.

## Repository enforcement without protected branches
This repository may run on a GitHub plan where protected branch rules are not enforced for private repositories. In that case, we enforce the workflow by convention and by quality gates.

Policy
- main is updated via pull requests only. Avoid direct pushes to main.
- Every pull request must reference a Work Item ID (WI-0000 pattern) in the PR title or PR description.
- A pull request should only be merged when all CI checks are green.

Quality gates
- Local pre-push hook runs non-mutating snapshot gates:
  - make validate
  - make snapshot-check
- CI runs:
  - pr-work-item (pull_request): fails if no WI reference is present
  - snapshot-contract: runs make snapshot-check and blocks tracked .ipynb_checkpoints

Setup
- Install git hooks locally:
  - bash tools/install_git_hooks.sh
- Before pushing:
  - make snapshot-check
- Before handing off to a new chat:
  - make handoff

## Multi chat operating rule
When starting a new chat, provide:
1. docs/hub/CONTEXT_SNAPSHOT.md
2. docs/hub/RULES_AND_HANDOFF.md
3. The specific WI file you are working on
4. Only the source files required for the change