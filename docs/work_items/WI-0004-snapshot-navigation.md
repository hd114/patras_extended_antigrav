# WI-0004 Snapshot navigation baseline

Status: In progress
Owner: You
Created:
Updated:

## Objective
Make docs/hub/CONTEXT_SNAPSHOT.md a reliable navigation entry point for new chats by standardizing:
1. Work item listing rules (active and recently completed).
2. A non-example task graph that references real WI IDs.
3. A minimal “modules present in this repo” description aligned with the module graph template.

## Acceptance criteria
- [ ] This WI file exists and follows docs/work_items/_TEMPLATE.md structure.
- [ ] docs/hub/CONTEXT_SNAPSHOT.md work_items block lists:
      - up to 5 active items
      - up to 3 recently completed items
      - each entry links to a WI file path
- [ ] docs/hub/CONTEXT_SNAPSHOT.md task_graph block references real WI IDs and an execution flow.
- [ ] docs/hub/CONTEXT_SNAPSHOT.md module_graph remains a generic layer template and is complemented by a short list of modules present in this repo in repo_inventory.
- [ ] make validate passes.

## Out of scope
- Automatic extraction of work items into the snapshot (future WI).
- Any project-specific application modules (this repo is a pipeline scaffold).

## Plan
1. Create this WI file.
2. Update CONTEXT_SNAPSHOT.md work_items block to include WI-0004 and WI-0003 as Done.
3. Update CONTEXT_SNAPSHOT.md task_graph block to reflect WI flow.
4. Update CONTEXT_SNAPSHOT.md repo_inventory to list the modules present in this repo.
5. Run make snapshot and make validate.
6. Commit and open PR.

## Verification
- make snapshot
- make validate

## Notes
- The goal is stable onboarding for new chats with minimal context upload.
- Once stable, we can automate parts of work_items and repo_inventory generation.

## Links
- PR:
- Related docs:
  - docs/hub/PIPELINE.md
  - docs/hub/RULES_AND_HANDOFF.md
- ADR:
