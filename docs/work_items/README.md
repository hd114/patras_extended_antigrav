# Work Items

This directory is the system of record for work tracking.

## Naming
- File name: WI-XXXX-short-title.md
- Title line: "# WI-XXXX Title"

## Required fields
Each Work Item must include these fields near the top:
- Status: Planned | In progress | Blocked | Done
- Owner:
- Created: YYYY-MM-DD
- Updated: YYYY-MM-DD
- Depends on: WI-XXXX or - (if none)

## Required sections
- Objective
- Acceptance criteria
- Out of scope
- Plan
- Verification
- Notes
- Links

## Workflow
1. Copy docs/work_items/_TEMPLATE.md to a new WI file
2. Fill in Objective and Acceptance criteria before implementation
3. Work on a dedicated branch per WI
4. Keep Updated current when scope changes
5. Run local gates before pushing:
   - make validate
   - make snapshot-check
6. Regenerate snapshot when WI or docs lists change:
   - make snapshot

## Verification guidance
- Verification must be runnable and specific.
- Prefer:
  - make validate
  - make snapshot-check
- Use make snapshot only when you intentionally regenerate snapshot content and will commit the result.
