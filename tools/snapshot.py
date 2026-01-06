#!/usr/bin/env python3
"""
Minimal snapshot updater for docs/hub/CONTEXT_SNAPSHOT.md.

This script intentionally keeps scope small:
- Updates Snapshot metadata fields (Generated at, Repository revision, Working tree status).
- Optionally refreshes the Commands block with Make targets if a Makefile is present.
- Refreshes the Hub navigation block if present.
- Does not attempt deep repository analysis.

All output and messages are in English.
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def run_git(repo_root: Path, args: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        return p.returncode, out if out else err
    except FileNotFoundError:
        return 1, "git not available"


def replace_metadata(text: str, generated_at: str, rev: str, status: str) -> str:
    def repl_line(prefix: str, value: str) -> str:
        return f"{prefix} {value}"

    lines = text.splitlines()
    out_lines: list[str] = []
    for line in lines:
        if line.startswith("Generated at:"):
            out_lines.append(repl_line("Generated at:", generated_at))
        elif line.startswith("Repository revision:"):
            out_lines.append(repl_line("Repository revision:", rev))
        elif line.startswith("Working tree status:"):
            out_lines.append(repl_line("Working tree status:", status))
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def update_auto_block(text: str, block_name: str, new_body: str) -> str:
    begin = f"<!-- AUTO:BEGIN {block_name} -->"
    end = f"<!-- AUTO:END {block_name} -->"
    if begin not in text or end not in text:
        return text

    pre, rest = text.split(begin, 1)
    mid, post = rest.split(end, 1)

    # Keep markers on their own lines.
    if not pre.endswith("\n"):
        pre += "\n"
    if not post.startswith("\n"):
        post = "\n" + post

    updated = pre + begin + "\n" + new_body.rstrip("\n") + "\n" + end + post
    return updated


def make_hub_navigation_block(repo_root: Path) -> str:
    # Keep stable links. This is intentionally static and curated.
    return """Authoritative hub docs:
- Project overview: docs/hub/PROJECT_OVERVIEW.md
- Current state: docs/hub/CURRENT_STATE.md
- Module ownership: docs/hub/MODULE_OWNERSHIP.md
- Executive report: docs/hub/EXECUTIVE_REPORT.md
- Pipeline: docs/hub/PIPELINE.md
- New chat starter: docs/hub/NEW_CHAT_STARTER.md
- Rules and handoff: docs/hub/RULES_AND_HANDOFF.md

Operational shortcuts:
- Snapshot gates: make snapshot-check
- Snapshot regeneration: make snapshot (commit required)
- Handoff bundle: make handoff
"""


def make_commands_block(repo_root: Path) -> Optional[str]:
    makefile = repo_root / "Makefile"
    if not makefile.exists():
        return None

    # Keep this minimal and stable. Use explicit "Not configured" entries for sections
    # that are intentionally not wired in this repository.
    return """Setup:
- bash tools/install_git_hooks.sh

Run:
- make check

Tests:
- Not configured

Lint:
- Not configured

Type check:
- Not configured

Format:
- Not configured

Snapshot gates:
- make validate
- make snapshot-check

Snapshot regeneration:
- make snapshot

Handoff:
- make handoff
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--snapshot", default="docs/hub/CONTEXT_SNAPSHOT.md", help="Snapshot path")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    snapshot_path = Path(args.snapshot).resolve()

    if not snapshot_path.exists():
        raise SystemExit(f"ERROR: Snapshot file not found: {snapshot_path}")

    text = snapshot_path.read_text(encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rc, rev = run_git(repo_root, ["rev-parse", "--short", "HEAD"])
    if rc != 0:
        rev = "unknown"

    rc, status = run_git(repo_root, ["status", "--porcelain"])
    if rc != 0:
        status = "unknown"
    else:
        status = "clean" if status.strip() == "" else "dirty"

    text = replace_metadata(text, generated_at=now, rev=rev, status=status)

    hub_nav_body = make_hub_navigation_block(repo_root)
    text = update_auto_block(text, "hub_navigation", hub_nav_body)

    commands_body = make_commands_block(repo_root)
    if commands_body is not None:
        text = update_auto_block(text, "commands", commands_body)

    snapshot_path.write_text(text, encoding="utf-8")
    print("OK: Snapshot updated")


if __name__ == "__main__":
    main()
