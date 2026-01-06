# Executive Report: coding_pipe

## Metadata
Purpose: Provide an up-to-date, detailed, decision-oriented overview of the repository, its collaboration pipeline, and current priorities.
Audience: Maintainers and contributors coordinating LLM-assisted work across multiple chats.
Update cadence: Update after major workflow changes, or at least monthly during active development.
Source of truth:
- docs/hub/CONTEXT_SNAPSHOT.md
- docs/hub/RULES_AND_HANDOFF.md
- docs/hub/PIPELINE.md
- docs/work_items
- docs/adr
- tools/*
- .github/workflows/ci.yml
Last updated: 2026-01-04

## Executive summary
coding_pipe is a repository-native collaboration pipeline for LLM-assisted development across multiple chat sessions, without autonomous agents. It standardizes onboarding, planning, verification, documentation, and handoff so that work remains reproducible and reviewable in Git history.

The pipeline is operational and stable:
- Snapshot contract is enforced locally and in CI via non-mutating drift checks.
- Pull requests are required by convention and must reference a Work Item ID (WI-0000 pattern) via CI enforcement.
- A deterministic handoff bundle can be generated via make handoff without polluting the working tree.
- Hub documentation is transitioning from placeholders to authoritative, maintained content.

Primary remaining focus is content hardening and governance-by-convention: completing hub documents and ensuring work items and ADRs remain the system of record.

## What the repository delivers
### 1) Onboarding and context preservation
- CONTEXT_SNAPSHOT.md acts as the single onboarding artifact for new chats.
- RULES_AND_HANDOFF.md defines collaboration rules, quality profiles, and strict zones.
- NEW_CHAT_STARTER.md defines how to start a new chat with minimal but sufficient context.
- The hub directory contains authoritative workflow documentation intended to replace ad hoc chat memory.

### 2) Planning and traceability
- Work Items in docs/work_items are the system of record for work.
- Each substantive change should map to a WI with acceptance criteria and verification steps.
- ADRs in docs/adr capture architectural or operational decisions and are indexed into the snapshot.

### 3) Deterministic generation and drift control
- Snapshot validation and drift detection are first-class gates.
- AUTO blocks are generated and must not be edited manually.
- Drift detection is non-mutating and ignores volatile snapshot metadata to remain stable in hooks and CI.

### 4) Quality gates aligned across local and CI
Local:
- Pre-push hook runs snapshot formatting validation and drift detection without modifying the working tree.

CI:
- GitHub Actions validates snapshot gates and blocks tracked notebook checkpoints.
- A pull_request-only job enforces presence of a Work Item reference in PR title or PR body.

### 5) Reproducible handoff
- make handoff produces handoff_bundle.zip deterministically.
- The bundle is tracked-only and therefore cannot inadvertently include local caches or notebook artifacts.
- Snapshot gates run before bundling, preventing distribution of inconsistent context.

## Current status
Overall status: Stable pipeline, hub content hardening in progress.

Operationally stable components:
- Snapshot contract: make validate and make snapshot-check
- Non-mutating local enforcement: pre-push checks do not modify the working tree
- CI baseline: snapshot gates in CI plus WI reference enforcement in PRs
- Deterministic handoff: make handoff produces a clean bundle and leaves git status clean
- Work Item conventions: WI template aligns verification with make snapshot-check

In progress:
- Completing hub documents and removing placeholders
- Ensuring all new workflow changes are reflected in WIs and ADRs
- Governance-by-convention where branch protection is not enforceable

## Governance and enforcement model
### Enforcement constraints
Protected branch rules may not be enforceable on the current GitHub plan for private repositories. As a result, the repository relies on a layered enforcement model:
- Convention: main is updated via pull requests only, avoid direct pushes to main
- Visibility: CI provides clear pass/fail signals for required checks
- Local gating: hooks block pushes with snapshot drift
- Documentation: PIPELINE.md documents the enforcement model explicitly

### Required signals for merging
A pull request should be merged only when:
- CI snapshot-contract is green
- CI pr-work-item is green
- The PR references a WI and includes verification steps and outcomes

## Key risks and mitigations
### Risk 1: Bypassing PR discipline due to non-enforced branch rules
Impact: High, likelihood: Medium
Mitigations:
- Document PR-only policy prominently in PIPELINE.md and snapshot
- Keep CI checks strong and highly visible
- Keep local hooks installation documented and encouraged
- Consider upgrading plan if strict enforcement becomes necessary

### Risk 2: Documentation drift as hub content grows
Impact: Medium, likelihood: Medium
Mitigations:
- Treat docs/hub as authoritative artifacts
- Keep CURRENT_STATE updated routinely
- Use snapshot as the index and regenerate it frequently
- Prefer small, reversible doc changes

### Risk 3: Local hook adoption gap for new contributors
Impact: Medium, likelihood: Medium
Mitigations:
- CI remains the minimum safety net
- NEW_CHAT_STARTER and PIPELINE emphasize hook installation
- Keep pre-push scripts robust and non-mutating

### Risk 4: Conflicts in strict zones
Impact: Medium, likelihood: Medium
Mitigations:
- Module ownership guidance for hub and tools
- Prefer short diffs and avoid parallel edits in the same strict files
- Regenerate snapshot after merges rather than manual conflict resolution

## Roadmap and priorities
### Immediate priorities
1. Finalize hub content:
   - PROJECT_OVERVIEW, CURRENT_STATE, MODULE_OWNERSHIP, EXECUTIVE_REPORT consistency
   - Remove placeholders and ensure each document has clear metadata and purpose
2. Keep traceability tight:
   - Ensure new workflow changes always have a WI
   - Ensure significant decisions are captured in ADRs
3. Maintain reproducibility:
   - Keep snapshot gates green
   - Keep handoff bundle deterministic and tracked-only

### Near-term enhancements
- Improve NEW_CHAT_STARTER to explicitly reference make handoff and the minimal artifact set
- Add a lightweight WI quality check in CI (optional) to prevent incomplete WIs
- Add a short section in the snapshot on how to resolve common CI failures (for example snapshot drift)

### Longer-term considerations
- Evaluate whether enforced branch protection is required as the number of contributors grows
- Consider additional automated generated sections once patterns stabilize (avoid premature automation)

## Operational checklist
For daily work:
- Use one branch per WI
- Keep diffs small and reversible
- Run:
  - make validate
  - make snapshot-check
- Run make snapshot only when regeneration is needed and you will commit the updated snapshot
- Before a new chat handoff:
  - make handoff

## References
- Snapshot: docs/hub/CONTEXT_SNAPSHOT.md
- Rules and handoff: docs/hub/RULES_AND_HANDOFF.md
- Pipeline: docs/hub/PIPELINE.md
- Current State: docs/hub/CURRENT_STATE.md
- Work Items: docs/work_items/
- ADRs: docs/adr/
- CI: .github/workflows/ci.yml
