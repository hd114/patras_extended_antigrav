# WI-0002 Work item system baseline

Status: In progress
Owner: You
Created:
Updated:

## Objective
Establish work items as repo files and connect them to the snapshot so new chats can resume work with minimal context.

## Acceptance criteria
- [ ] docs/work_items/README.md exists and documents the workflow.
- [ ] docs/work_items/_TEMPLATE.md exists for new items.
- [ ] This WI file exists and is referenced from docs/hub/CONTEXT_SNAPSHOT.md.
- [ ] CONTEXT_SNAPSHOT.md Active work items section lists WI-0002.
- [ ] CONTEXT_SNAPSHOT.md Task graph includes WI-0002.
- [ ] make validate passes.

## Out of scope
- GitHub Issues integration
- Automatic extraction of work items into the snapshot (later)

## Plan
1. Add docs/work_items/README.md and _TEMPLATE.md.
2. Add this WI file.
3. Update docs/hub/CONTEXT_SNAPSHOT.md work_items and task_graph blocks.
4. Validate and commit.

## Verification
- make validate
- make snapshot
- make validate

## Notes
- This establishes the manual baseline. Automation can follow once the structure stabilizes.
