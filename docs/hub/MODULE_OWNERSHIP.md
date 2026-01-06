# Module Ownership

## Metadata
Purpose: Define lightweight module ownership to reduce conflicting edits when working across multiple chats.
Audience: Contributors coordinating work items and reviews.
Update cadence: Update when responsibilities shift, or when a module becomes actively edited in parallel.
Source of truth:
- docs/work_items (active WI assignments)
- docs/hub/CONTEXT_SNAPSHOT.md (overall context)
Last updated: 2026-01-04

## Purpose
Module ownership is a lightweight coordination mechanism. It helps avoid merge conflicts and duplicated work when multiple chats touch the same areas.

Ownership is not a hard lock. It is a default reviewer and point of contact:
- Owners should be consulted before large changes in their modules.
- Owners coordinate parallel edits when multiple WIs touch the same module.
- Owners keep module-level documentation accurate.

## Ownership rules
1. One branch per Work Item. Use WI IDs in PR title or body.
2. If you need to change a module with an active owner, add a note in the relevant WI and mention the owner in the PR.
3. Prefer small, reversible diffs. Regenerate snapshot only when necessary and commit it.
4. For strict zones (docs/hub, tools, .github/workflows), coordinate changes explicitly to avoid churn.

## Module map
This repository is organized around workflow and documentation modules rather than application code:
- docs/hub: authoritative workflow documentation
- docs/work_items: work tracking and acceptance criteria
- docs/adr: architecture and process decisions
- tools: generators and quality gates
- .github/workflows: CI enforcement

## Active ownership
Format:
- Module | Owner | Work item | Since | Notes

Entries:
- docs/hub | You | - | 2026-01-04 | Hub docs are authoritative, coordinate edits if multiple chats touch them
- tools | You | - | 2026-01-04 | Snapshot gates and handoff scripts
- docs/work_items | You | - | 2026-01-04 | WI conventions and updates
- docs/adr | You | - | 2026-01-04 | ADR process and indexing into snapshot
- .github/workflows | You | - | 2026-01-04 | CI jobs, PR WI enforcement, snapshot checks

## How to claim ownership
Add an entry above with:
- a specific WI (preferred) and a short note about intended edits,
then regenerate snapshot and commit.

## How to release ownership
When a WI is done, replace the Work item field with '-' and update Notes if ongoing stewardship remains.

