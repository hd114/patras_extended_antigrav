# WI-0006 Auto-generate snapshot task graph

Status: In progress
Owner: You
Created:
Updated:
Depends on: WI-0002, WI-0005

## Objective
Automatically populate the task_graph AUTO block in docs/hub/CONTEXT_SNAPSHOT.md by scanning docs/work_items/*.md and reading declared dependencies.

## Acceptance criteria
- [ ] docs/work_items/_TEMPLATE.md includes a "Depends on:" field.
- [ ] Existing work items are updated to include a valid "Depends on:" line (or "-").
- [ ] A script exists that reads work item dependencies and updates the snapshot task_graph AUTO block.
- [ ] The generated task graph is deterministic and stable across runs.
- [ ] make snapshot updates the snapshot metadata, work_items, and task_graph consistently.
- [ ] make validate passes.

## Out of scope
- Automatic generation of the module graph (future WI).
- Dependency inference from git history or PRs.
- Advanced graph layout or grouping beyond a simple Mermaid flowchart.

## Plan
1. Add "Depends on:" to docs/work_items/_TEMPLATE.md.
2. Update existing WI files to include "Depends on:" with correct WI IDs.
3. Add tools/task_graph_to_snapshot.py.
4. Update the Makefile snapshot target to call tools/task_graph_to_snapshot.py.
5. Run make snapshot and make validate.
6. Commit and open PR.

## Verification
- make snapshot
- make validate

## Notes
- Dependency format is file-based and explicit: "Depends on: WI-0002, WI-0005" or "-".
- Edges are dependency -> work item.
- Missing dependency targets should be ignored rather than causing failures.

## Links
- PR:
- Related docs:
  - docs/hub/CONTEXT_SNAPSHOT.md
  - docs/work_items/README.md
  - docs/work_items/_TEMPLATE.md
- ADR:
