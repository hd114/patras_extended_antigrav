# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADRs).

ADRs document significant architectural, interface, and operational decisions in a lightweight, reviewable format.

## Conventions
- File name: ADR-XXXX-short-title.md
- First line (H1): "# ADR-XXXX Title"
- Metadata fields near the top:
  - Date: YYYY-MM-DD
  - Status: Proposed | Accepted | Superseded | Deprecated

## Workflow
1. Copy docs/adr/_TEMPLATE.md to a new ADR file in docs/adr.
2. Fill in context, decision, and consequences. Keep it concise and specific.
3. Commit the ADR.
4. Run `make snapshot` so the Decisions section in docs/hub/CONTEXT_SNAPSHOT.md is refreshed.
5. Run `make validate`.
6. Open a PR and reference the related Work Item (if applicable).

Note: The snapshot Decisions table is auto-generated. Do not edit it manually.
