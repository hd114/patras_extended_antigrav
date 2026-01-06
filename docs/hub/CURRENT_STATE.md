# Current State

## Metadata
Purpose: Provide an operational view of what currently works, what is broken, what is in progress, and what to do next.
Audience: Contributors and reviewers who need an accurate project status before starting work.
Update cadence: Update after each meaningful workflow or repository change, and at least once per week during active development.
Source of truth:
- docs/hub/CONTEXT_SNAPSHOT.md
- docs/work_items
- .github/workflows/ci.yml
Last updated: 2026-01-04

## Status summary
Overall status: Stabilizing
Current profile: Pragmatic with strict zones
Current focus: Finalize hub documentation and keep the collaboration pipeline reproducible

## What is working
- Snapshot contract: make validate and make snapshot-check are stable
- Non-mutating local quality gate: pre-push blocks snapshot drift without modifying the working tree
- CI baseline: pull requests run snapshot quality gates and enforce a WI reference in PR title or PR body
- Deterministic handoff: make handoff produces a tracked-only handoff_bundle.zip and keeps the working tree clean
- Work Item conventions: WI template exists and aligns verification with snapshot gates

## What is not working
- Protected main branch rules may not be enforceable on the current GitHub plan for private repositories, so enforcement relies on convention and CI signals
- PR text edits may not reliably trigger a new CI run in all cases; a new commit reliably triggers re-evaluation

## In progress
- Hub documentation hardening: PROJECT_OVERVIEW, CURRENT_STATE, MODULE_OWNERSHIP, EXECUTIVE_REPORT completion and consistency
- Governance by convention: documenting and following PR-only merges and green-check merges

## Next actions
1. Complete MODULE_OWNERSHIP.md and EXECUTIVE_REPORT.md with real content and remove placeholders
2. Align NEW_CHAT_STARTER.md with the current handoff bundle and the required artifacts
3. Keep WORK ITEMS authoritative: add or update WIs for any new workflow changes and regenerate the snapshot

## Known risks
- Risk: Workflow discipline depends on contributors installing local hooks
  - Mitigation: keep CI checks strong and document the setup in hub docs
- Risk: Missing enforcement of protected branches can allow bypassing PR checks
  - Mitigation: PR-only convention, review discipline, and CI visibility; consider upgrading plan if strict enforcement becomes necessary
- Risk: Documentation drift as hub docs expand
  - Mitigation: treat docs/hub as strict process artifacts and update CURRENT_STATE routinely
