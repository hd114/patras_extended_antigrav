# WI-0010 Deterministic handoff bundle export

Status: Done
Owner: You
Created:
Updated:

## Objective
Provide a deterministic, single-command handoff bundle export so a new chat can start with a minimal set of curated repository artifacts.

## Acceptance criteria
- [ ] `make handoff` exists and produces handoff_bundle.zip in the repository root.
- [ ] The export includes only curated, git-tracked files (to avoid leaking local artifacts).
- [ ] The export runs snapshot gates before bundling (make snapshot-check).
- [ ] handoff_bundle.zip is not tracked by git and is ignored via .gitignore.
- [ ] After running `make handoff`, `git status` remains clean.

## Out of scope
- Including untracked files in the bundle.
- Bundling .git history.

## Plan
1. Add tools/export_handoff_bundle.sh to create a tracked-only deterministic zip.
2. Add a Makefile target `handoff`.
3. Ensure handoff_bundle.zip is ignored and not tracked.
4. Validate content and reproducibility.

## Verification
- make snapshot-check
- make handoff
- zipinfo -1 handoff_bundle.zip | head
- git status (must be clean)

## Notes
- Tracked-only export is intentional to prevent accidental inclusion of local caches or notebook artifacts.

## Links
- PR:
- Related docs:
  - docs/hub/RULES_AND_HANDOFF.md
  - docs/hub/NEW_CHAT_STARTER.md
