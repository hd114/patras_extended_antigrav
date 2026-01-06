# Rules and Handoff

## Purpose
This document defines the working contract for collaborating with an LLM across multiple chats. It is designed to minimize repeated context uploads while keeping quality, security, and traceability high.

## Core principles
1. Do not guess missing requirements. Ask for missing information or proceed with explicitly stated assumptions.
2. Preserve existing behavior unless the change request explicitly includes behavior changes.
3. Prefer small, reversible changes. Avoid large refactors unless the work item is explicitly scoped as a refactor.
4. No facades. Do not ship placeholder logic, hardcoded outputs, or test-only implementations unless explicitly required.
5. Verify everything. Every change must be validated by tests or an explicit verification procedure.
6. Clean as you go. Include cleanup work in every change set.
7. Never introduce secrets into the repository or generated documentation.

## Work item discipline
### Scope rule
One chat works on one work item at a time. If scope grows, split into multiple work items.

### Deliverables checklist
For every work item, maintain an explicit checklist of deliverables and confirm each item was delivered to specification.

### Context economy
Prefer referencing files and structured snapshots over long chat narratives. Avoid repeating information already captured in repository documentation.

## Assistant output expectations for code changes
When proposing code changes:
1. Specify the module and the function where changes are made.
2. Provide the complete updated function body, not partial fragments.
3. Respect existing logic, interfaces, and invariants.
4. All comments and prints inside code must be in English.

## Snapshot formatting contract (CONTEXT_SNAPSHOT.md)
This section defines non-negotiable formatting rules for docs/hub/CONTEXT_SNAPSHOT.md to ensure it renders correctly and can be safely updated by an automated generator.

### Block markers
1. Every auto-generated section must be enclosed by exactly one pair of markers:
   - <!-- AUTO:BEGIN <block_name> -->
   - <!-- AUTO:END <block_name> -->
2. Markers must be on their own lines.
3. Markers must not be indented.
4. Block names must be unique within the file.
5. The generator may replace only content between matching BEGIN and END markers. Everything else is manual and must remain unchanged.
6. AUTO markers must never appear inside fenced code blocks.

### Markdown structure
1. Section headings must use Markdown headings (for example, ## Section).
2. Do not use 4 space indentation for regular content. In Markdown, 4 spaces creates a code block and can break rendering.
3. Lists must use standard Markdown list syntax:
   - Bullets: "- item"
   - Numbers: "1. item"

### Mermaid blocks
1. Mermaid diagrams must always be contained in a fenced code block:
   - ```mermaid
   - diagram content
   - ```
2. The mermaid code fence must be fully closed before the AUTO:END marker.
3. The AUTO markers must never appear inside a fenced code block.

### Tables and placeholders
1. Prefer bullet lists for placeholders to keep diffs small.
2. If tables are used, use standard Markdown tables and keep them short.
3. Do not embed large file contents into the snapshot. Prefer links and references.

### Content constraints
1. Do not include secrets, tokens, credentials, or private keys.
2. Do not include large logs or large data samples. Use short excerpts only.
3. Keep the snapshot compact. If a section grows, move details to a separate doc and link to it.

### Regeneration rules
1. Run make snapshot-check frequently. It validates formatting and detects drift without modifying the working tree.
2. Run make snapshot only when you intend to regenerate the snapshot and commit the changes.
3. Do not manually edit content inside AUTO blocks. Manual edits will be overwritten on regeneration.
4. If regeneration fails, fix the generator or the inputs instead of patching AUTO blocks by hand.
5. Every regeneration must preserve marker integrity and Markdown validity.

## Handoff bundle
### Deterministic handoff
Use make handoff to generate a deterministic handoff_bundle.zip for starting a new chat.
- The handoff bundle includes curated, git-tracked files only.
- Snapshot gates run before bundling (make snapshot-check).
- The bundle must not be tracked by git and must not dirty the working tree.

Recommended workflow:
1. make snapshot-check
2. make handoff
3. Provide the bundle and the active Work Item in the new chat

## Quality profiles

### Active profile
Set exactly one:
- Profile: pragmatic
- Profile: strict

### Strict zones
Strict zones always require the strict bar, even if the active profile is pragmatic.

Define strict zones as repository paths or module globs:
- strict_zones:
  - docs/hub/**
  - tools/**
  - .github/workflows/**
  - src/<public_api>/**
  - src/<interfaces>/**
  - src/<security_or_crypto>/**
  - src/<serialization>/**
  - src/<config_and_cli>/**
  - src/<compiler_or_ir>/**
  - src/<benchmarking_and_metrics>/**

What belongs in a strict zone:
1. Public APIs and cross module interfaces
2. Correctness critical transformations and parameter handling
3. Parsing, serialization, configuration, and boundary IO
4. Security relevant code paths
5. Anything that affects measurement validity and reproducibility
6. Collaboration contracts and generators (hub docs, tools, CI workflows)

### Quality gates by profile

#### Pragmatic profile
Typing
1. Mandatory type hints for public APIs and cross module interfaces.
2. Internal helper functions: type hints recommended, mandatory when complexity is non trivial.
3. Type checking enabled if configured by the project. New type errors must not be introduced.

Tests
1. New features: unit tests for core logic. Add an integration style test when IO or boundaries are involved.
2. Bug fixes: regression tests required when feasible. If not feasible, document why and define an alternative verification procedure.
3. Refactors: add tests if behavior can change, otherwise rely on existing suite plus smoke checks.
4. Prioritize risk bearing paths: parsing, conversion, serialization, config, boundary IO, error handling, security sensitive logic.

Docs
1. Run make snapshot-check frequently during work.
2. Regenerate the snapshot at the end of each work item only when needed, and commit it (make snapshot).
3. Add an ADR only for material architectural or interface decisions.

#### Strict profile
Typing
1. Mandatory type hints for all functions and methods, including return types.
2. Avoid Any. If unavoidable, document why and contain it to a boundary.
3. Type checking is a hard gate when configured by the project. No new warnings or errors allowed.

Tests
1. New features: unit tests plus integration tests for boundary and wiring. Core workflows should have end to end coverage where applicable.
2. Bug fixes: regression tests required. If not feasible, document why and define an alternative verification procedure.
3. Complex logic: add edge case tests and failure mode tests.
4. Avoid over mocking. Ensure at least one test hits real implementations for critical wiring.
5. If performance is a requirement, add a benchmark or a measurable regression check.

Docs
1. Run make snapshot-check frequently during work.
2. Update snapshot at the end of each work item and commit the regeneration.
3. Public APIs and cross module contracts must be documented.
4. Architecture changes require an ADR.
5. Ensure docstrings for public modules, classes, and functions.

### Override rule
If a file matches a strict zone, apply the strict profile gates regardless of the active profile.

### Assistant execution checklist
Before implementation:
1. Identify the active profile.
2. Determine whether the touched files are in a strict zone.
3. State the required gates for typing, tests, and documentation for this work item.
4. Propose a verification plan aligned to those gates.

## Coding standards (Python)

### Primary style references
1. PEP 8 for formatting and naming.
2. PEP 257 for docstrings.

### Recommended toolchain
- Formatter: Black
- Import sorting: isort
- Linting: Ruff
- Type checking: mypy or pyright
- Testing: pytest

### Naming and readability
1. Use descriptive names. Avoid single letter names except for trivial loop indices.
2. Use verbs for functions and nouns for data structures.
3. Keep functions focused. Prefer small, composable units.
4. Avoid deeply nested logic. Prefer early returns and guard clauses.
5. Prefer explicitness over cleverness.

### Project structure
1. Separate domain logic from IO and external services.
2. Separate configuration from application logic.
3. Keep entry points (CLI, scripts, app bootstrapping) thin.
4. Avoid circular imports and hidden global state.

### Docstrings
1. Public modules, classes, and functions must have docstrings.
2. Docstrings must describe purpose, parameters, return values, and side effects.
3. Comments should explain intent and rationale, not restate the code.

### Error handling and logging
1. Raise specific exceptions. Do not use bare except.
2. Convert low level exceptions into domain meaningful errors at boundaries.
3. Add context without leaking secrets.
4. Do not log secrets or sensitive payloads.

## Definition of Done
A work item is done only if all items below are satisfied:
1. Acceptance criteria satisfied and deliverables checklist completed.
2. Quality gates satisfied according to the active profile and strict zone overrides.
3. Cleanup performed and documented in the work item notes.
4. Snapshot updated when regeneration is required, and make snapshot-check passes.

## New chat startup protocol

### Inputs to provide
1. docs/hub/CONTEXT_SNAPSHOT.md
2. docs/hub/RULES_AND_HANDOFF.md
3. docs/hub/PIPELINE.md
4. The active Work Item file (docs/work_items/WI-XXXX-*.md)
5. Optional: the relevant source files for the change
6. Optional: handoff_bundle.zip (generated via make handoff)

### Startup checklist
1. Confirm the work item objective and Definition of Done.
2. Confirm scope boundaries and non goals.
3. Identify impacted modules and interfaces.
4. Identify profile and strict zone coverage for touched files.
5. Propose a step plan with verification steps.
6. Identify risks and assumptions.

## Chat closeout protocol

### Closeout checklist
1. Summarize what changed and why.
2. List files changed, added, removed.
3. List tests executed and results, or the explicit verification procedure used.
4. Record open issues and follow ups.
5. Update docs/hub/CONTEXT_SNAPSHOT.md via make snapshot if regeneration is required.
6. If architecture or interfaces changed, add or update an ADR.

## Work item brief template
Title:
Owner:
Status: Planned | In progress | Blocked | Done
Priority:
Target release:

Objective:
Non goals:
Acceptance criteria:
1.
2.
3.

Deliverables checklist:
- [ ]
- [ ]
- [ ]

Impacted modules:
Dependencies:

Profile: pragmatic | strict
Touched strict zones: Yes | No

Test plan:
1.
2.

Verification procedure:
1.
2.

Cleanup tasks:
- [ ]
- [ ]
- [ ]

Risks:
Assumptions:

## Handoff pack template
Summary of changes:
Files changed:
Files added:
Files removed:

Tests executed:
Verification procedure (if no tests):
Known issues:
Follow ups:

Snapshot updated: Yes | No
ADR updated: Yes | No
Security notes: None | Added
