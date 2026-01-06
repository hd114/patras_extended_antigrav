# Project Overview

## Metadata
Purpose: Define what this repository is, how it should be used, and what it explicitly does not do.
Audience: Contributors using an LLM to collaborate across multiple chats on code and documentation.
Update cadence: Update when collaboration rules, core workflow, or repository structure changes.
Source of truth:
- docs/hub/CONTEXT_SNAPSHOT.md
- docs/hub/RULES_AND_HANDOFF.md
Last updated: YYYY-MM-DD

## Purpose
This repository is a template for an LLM-assisted coding workflow. It is designed to enable safe, repeatable collaboration across multiple chats without autonomous agents.

It defines:
- A stable onboarding artifact for new chats
- Quality profiles and strict zones
- A handoff protocol to preserve context across chats
- A structure for planning, decisions, and generated documentation

## Primary artifacts
Start here:
- docs/hub/CONTEXT_SNAPSHOT.md (single onboarding snapshot)
- docs/hub/RULES_AND_HANDOFF.md (operational rules and handoff contract)

Supporting hub docs:
- docs/hub/PIPELINE.md (collaboration loops and enforcement policy)
- docs/hub/NEW_CHAT_STARTER.md (how to start a new chat with minimal context)
- docs/adr (Architecture Decision Records)
- docs/work_items (Work Items, system of record for work)

## Working model
Use this cycle for each Work Item:
1. Objective framing: define the outcome and acceptance criteria
2. Task decomposition: break into small reversible steps
3. Conversation: implement with explicit scope and verification steps
4. Review with care: focus on contract, strict zones, and regressions
5. Test and verify: run snapshot and project-specific checks
6. Refine and iterate: update docs and WIs as reality changes

## Repository map
- docs/hub: authoritative workflow documentation
- docs/work_items: work tracking and acceptance criteria
- docs/adr: decisions that affect interfaces, architecture, and operations
- docs/auto: generated summaries (module graph, task graph, inventory)
- tools: generators and quality gates (snapshot, validation, handoff)

## Non-goals
This repository is not intended for:
- Running autonomous multi-agent development inside the repository
- Replacing project-specific engineering judgment
- Treating the LLM as an unsupervised actor with permission to change code without review
- Using generated documentation as a substitute for verification and tests

## How to start
- If you are continuing work: open the relevant Work Item in docs/work_items and follow its Verification section.
- If you are starting a new chat: follow docs/hub/NEW_CHAT_STARTER.md and provide the snapshot, rules, the active WI, and only the relevant source files.

