# WI-0007 Auto-generate snapshot repo inventory

Status: In progress
Owner: You
Created:
Updated:
Depends on: WI-0002, WI-0005, WI-0006

## Objective
Automatically populate the repo_inventory AUTO block in docs/hub/CONTEXT_SNAPSHOT.md from a curated source of truth file.

## Acceptance criteria
- [ ] docs/hub/modules.toml exists as the source of truth.
- [ ] tools/repo_inventory_to_snapshot.py exists and updates the repo_inventory AUTO block deterministically.
- [ ] make snapshot updates snapshot metadata, work_items, task_graph, and repo_inventory consistently.
- [ ] make validate passes.

## Out of scope
- Heuristic scanning of repository structure.
- Auto-generation of module_graph (future WI).

## Plan
1. Add docs/hub/modules.toml.
2. Add tools/repo_inventory_to_snapshot.py.
3. Update Makefile snapshot target to call the new tool.
4. Run make snapshot and make validate.
5. Commit and open PR.

## Verification
- make snapshot
- make validate

## Notes
- This keeps repo navigation stable for new chats with minimal uploads.
- modules.toml can evolve as projects get more complex.

## Links
- PR:
- Related docs:
  - docs/hub/CONTEXT_SNAPSHOT.md
  - docs/hub/PIPELINE.md
