# coding_pipe

## What this repository is
coding_pipe is a repository-native collaboration pipeline for working with LLM chats on software projects. It standardizes planning, implementation, verification, documentation, and chat handoffs so work can continue safely across chat sessions without relying on long chat history.

This repo is intentionally project-independent. You can use it as:
- A template for new projects
- A companion repo that defines how you and an LLM collaborate across modules and repositories

## Quick start
Prerequisites:
- GNU Make
- Python 3.11+ (or a version that supports tomllib)
- zip

Core commands:
1. Regenerate the onboarding snapshot
   - make snapshot

2. Validate the snapshot contract
   - make validate

3. Create a new-chat handoff bundle
   - make handoff

## Repository navigation
Start here:
- docs/hub/CONTEXT_SNAPSHOT.md
  - Single source of truth for onboarding and current status
  - Contains machine-maintained AUTO blocks

Operational rules and quality bar:
- docs/hub/RULES_AND_HANDOFF.md
  - Collaboration profiles (pragmatic, strict)
  - Strict zones and quality gates
  - Coding rules for safe incremental changes

Workflow documentation:
- docs/hub/PIPELINE.md
  - How to plan work, implement, verify, and merge changes

Executive overview:
- docs/hub/EXECUTIVE_REPORT.md
  - Detailed explanation of how the pipeline works and how to use it

Work tracking (primary system):
- docs/work_items/
  - Work items are repo files WI-XXXX-*.md
  - Dependencies are explicit via a "Depends on:" line
  - The task graph is generated from those dependencies

Architecture decisions:
- docs/adr/
  - ADRs capture key architectural and operational decisions
  - The snapshot decisions section is generated from ADR files

Inventory source of truth:
- docs/hub/modules.toml
  - Curated repository inventory used to populate the snapshot repo_inventory section

Tooling:
- tools/
  - Deterministic generators and validators used by make snapshot and make validate

## How the snapshot works
docs/hub/CONTEXT_SNAPSHOT.md contains AUTO blocks delimited by markers:
- <!-- AUTO:BEGIN <name> -->
- <!-- AUTO:END <name> -->

Do not edit AUTO block contents manually. Use:
- make snapshot

Then validate:
- make validate

## Day-to-day workflow
1. Sync main
   - git checkout main
   - git pull --ff-only

2. Create or update a work item
   - Add or update docs/work_items/WI-XXXX-*.md
   - Maintain "Depends on:" explicitly

3. Create a branch
   - git checkout -b <area>/WI-XXXX-<short-title>

4. Implement change
   - Follow docs/hub/RULES_AND_HANDOFF.md
   - Keep changes small and reversible

5. Regenerate and validate snapshot
   - make snapshot
   - make validate

6. Commit and push
   - git add -A
   - git commit -m "<type(scope): message>"
   - git push -u origin <branch>

7. Open a PR
   - Wait for CI to be green
   - Squash merge
   - Delete the branch

8. Cleanup locally
   - git checkout main
   - git pull --ff-only
   - git branch -d <branch>
   - git fetch --prune

## Starting a new chat
Goal: transfer full situational awareness with minimal uploads.

1. Create a handoff bundle
- make handoff

This produces:
- handoff_bundle.zip in the repo root

2. Upload handoff_bundle.zip in the new chat.

3. In the new chat, instruct the assistant to:
- Read CONTEXT_SNAPSHOT.md and RULES_AND_HANDOFF.md first
- Restate scope, plan, and verification steps before coding
- Request only the minimal additional source files needed

## Contributing
- Follow the collaboration profiles and strict zones in docs/hub/RULES_AND_HANDOFF.md
- Keep PRs small, reviewable, and focused
- Prefer deterministic tooling over manual edits of generated content

## License
TBD

## Security
TBD
