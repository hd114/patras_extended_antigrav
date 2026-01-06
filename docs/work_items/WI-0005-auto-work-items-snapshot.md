# WI-0005 Auto-generate snapshot work items

Status: In progress
Owner: You
Created:
Updated:

## Objective
Automatically populate the work_items block in docs/hub/CONTEXT_SNAPSHOT.md by scanning docs/work_items/*.md.

## Acceptance criteria
- [ ] A script exists that reads docs/work_items/*.md and updates the snapshot work_items AUTO block.
- [ ] The script lists up to 5 active items and up to 3 recently completed items.
- [ ] Active items include Status: Planned, In progress, Blocked.
- [ ] Completed items include Status: Done.
- [ ] make snapshot updates metadata and work_items consistently.
- [ ] make validate passes.

## Out of scope
- Auto-generation of task_graph and module_graph (future WI).
- Heuristics based on git history (keep it deterministic and file-based).

## Plan
1. Add tools/work_items_to_snapshot.py.
2. Update Makefile snapshot target to call the new script after tools/snapshot.py.
3. Run make snapshot and make validate.
4. Commit and open PR.

## Verification
- make snapshot
- make validate

## Notes
- Deterministic ordering: status priority, then WI numeric ID.
- Titles are taken from the first Markdown heading line.

## Links
- PR:
- Related docs:
  - docs/work_items/README.md
  - docs/hub/CONTEXT_SNAPSHOT.md
