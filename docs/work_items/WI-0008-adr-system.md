# WI-0008 ADR system and snapshot integration

Status: In progress
Owner: You
Created:
Updated:
Depends on: WI-0002, WI-0005

## Objective
Add a repository-native ADR system and automatically surface recent ADRs into the decisions section of docs/hub/CONTEXT_SNAPSHOT.md.

## Acceptance criteria
- [ ] docs/adr/README.md exists and documents conventions.
- [ ] docs/adr/_TEMPLATE.md exists and is usable for new ADRs.
- [ ] At least one ADR exists (example is acceptable initially).
- [ ] tools/adr_to_snapshot.py updates the decisions AUTO block deterministically.
- [ ] make snapshot updates decisions along with other snapshot blocks.
- [ ] make validate passes.

## Out of scope
- Enforcing ADR content quality gates beyond formatting and basic metadata presence.
- Auto-linking PRs or work items beyond a simple text field.

## Plan
1. Add ADR directory and template.
2. Add an example ADR.
3. Implement tools/adr_to_snapshot.py.
4. Update Makefile snapshot target.
5. Run make snapshot and make validate.
6. Commit, open PR, and merge.

## Verification
- make snapshot
- make validate

## Notes
- The snapshot should stay compact. Only the most recent ADRs are listed.
- ADR id and title come from the H1 line to keep parsing stable.

## Links
- PR:
- Related docs:
  - docs/hub/CONTEXT_SNAPSHOT.md
  - docs/adr/README.md
