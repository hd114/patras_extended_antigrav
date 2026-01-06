# New Chat Starter

Purpose: A ready-to-use handoff message and operating checklist for starting a new chat safely and efficiently.

## Minimum artifacts to provide
Provide these files in the new chat:
- docs/hub/CONTEXT_SNAPSHOT.md
- docs/hub/RULES_AND_HANDOFF.md
- docs/hub/PIPELINE.md
- The specific Work Item file you are working on (docs/work_items/WI-XXXX-*.md)

If you created a handoff bundle, you can upload:
- handoff_bundle.zip (generated via make handoff)

## Optional artifacts
Provide only if needed for the task:
- docs/hub/modules.toml
- docs/adr/*
- Specific source files relevant to the change

## Preflight before starting a new chat
Recommended local commands before handing off:
- make snapshot-check
- make handoff

Notes:
- make handoff exports tracked files only, to avoid leaking local artifacts.
- If snapshot drift is reported, run make snapshot and commit the changes before creating the handoff.

## First assistant response requirements
Before proposing any code changes, the assistant must:
1. Restate scope, success criteria, and constraints.
2. Identify the active collaboration profile and any strict zones for files that may be touched.
3. Propose a step-by-step plan and verification steps.

For any code changes, the assistant must:
- Always specify the module and function being changed.
- Always provide complete updated function bodies.
- Keep changes small and reversible.

## Copy-paste prompt for a new chat
Paste the text below as your first message:

---
You are working in a repository that uses a repo-native pipeline for planning, execution, verification, and multi-chat handoffs.

You are given:
- docs/hub/CONTEXT_SNAPSHOT.md
- docs/hub/RULES_AND_HANDOFF.md
- docs/hub/PIPELINE.md
- The active Work Item file (docs/work_items/WI-XXXX-*.md)
Optionally:
- handoff_bundle.zip (if provided)
- docs/adr/* (if relevant)

Task:
1) Read the snapshot and restate your understanding of the repository structure and current status.
2) Identify the active collaboration profile and strict zones for any files you will touch.
3) Propose a plan with verification steps before writing code.
4) Ask for only the minimal additional source files needed to complete the Work Item.

Important:
- Do not assume missing context.
- Do not edit generated AUTO blocks manually.
- Keep changes minimal and reversible.
- Maintain traceability to the Work Item acceptance criteria and verification steps.
---
