# ADR-0001 ADR system and snapshot integration

Date: 2026-01-03
Status: Accepted
Deciders: Repository maintainers
Related work items: WI-0008

## Context
We need a durable, repo-native way to record architectural, interface, and operational decisions so they survive across chats and remain reviewable in Git history.
Chat summaries are helpful, but they are not versioned artifacts and they are not easily discoverable for new contributors.

The repository already relies on docs/hub/CONTEXT_SNAPSHOT.md as the primary onboarding artifact. We want important decisions to be visible there without manual duplication.

## Decision
1. Store decisions as Architecture Decision Records in docs/adr.
2. Use a lightweight, consistent template to keep authoring cost low.
3. Auto-generate a compact index of recent ADRs into the Decisions section of docs/hub/CONTEXT_SNAPSHOT.md via tools/adr_to_snapshot.py.
4. Keep the decisions index deterministic and review-friendly.

## Consequences
- New chats can discover recent decisions directly from the snapshot without searching the full history.
- ADRs add small overhead, but reduce repeated discussions and ambiguous implementation changes.
- The snapshot decisions table is generated and must not be edited manually.

## Alternatives considered
- Only using PR descriptions: too ephemeral and hard to browse.
- Only using chat summaries: not reviewable in Git history.
- A heavyweight RFC process: too costly for this repository.

## Links
- PR:
- Docs:
