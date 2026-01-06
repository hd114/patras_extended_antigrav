#!/usr/bin/env python3
"""
Update the repo_inventory AUTO block in docs/hub/CONTEXT_SNAPSHOT.md from docs/hub/modules.toml.

This is intentionally deterministic and file-based, not heuristic.

All output messages are in English.
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


SNAPSHOT_PATH = Path("docs/hub/CONTEXT_SNAPSHOT.md")
MODULES_TOML_PATH = Path("docs/hub/modules.toml")
WORK_ITEMS_DIR = Path("docs/work_items")

BEGIN_MARKER = "<!-- AUTO:BEGIN repo_inventory -->"
END_MARKER = "<!-- AUTO:END repo_inventory -->"


def die(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        die(f"File not found: {path}")


def read_toml(path: Path) -> Dict[str, Any]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        die(f"File not found: {path}")
    try:
        return tomllib.loads(data.decode("utf-8"))
    except Exception as exc:
        die(f"Failed to parse TOML at {path}: {exc}")


def find_block(snapshot: str) -> tuple[int, int]:
    begin = snapshot.find(BEGIN_MARKER)
    end = snapshot.find(END_MARKER)
    if begin == -1 or end == -1 or end <= begin:
        die("repo_inventory AUTO block markers not found or invalid in CONTEXT_SNAPSHOT.md")
    begin_end = begin + len(BEGIN_MARKER)
    return begin_end, end


def render_repo_inventory(cfg: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("Entry points:")
    entry_points = cfg.get("entry_points", {}).get("items", [])
    if not entry_points:
        lines.append("- Path: TBD")
        lines.append("- How to run: TBD")
    else:
        for item in entry_points:
            path = item.get("path", "TBD")
            notes = item.get("notes", "")
            if notes:
                lines.append(f"- Path: {path} ({notes})")
            else:
                lines.append(f"- Path: {path}")

    lines.append("")
    lines.append("How to run:")
    how_to_run = cfg.get("how_to_run", {}).get("items", [])
    if not how_to_run:
        lines.append("- Command: TBD")
    else:
        for item in how_to_run:
            cmd = item.get("command", "TBD")
            notes = item.get("notes", "")
            if notes:
                lines.append(f"- Command: {cmd} ({notes})")
            else:
                lines.append(f"- Command: {cmd}")

    lines.append("")
    lines.append("Key modules:")
    lines.append("")
    lines.append("Module name | Path | Responsibility")
    lines.append("---|---|---")
    for m in cfg.get("modules", []):
        name = str(m.get("name", "TBD"))
        path = str(m.get("path", "TBD"))
        resp = str(m.get("responsibility", "TBD"))
        lines.append(f"{name} | {path} | {resp}")

    lines.append("")
    lines.append("Key configuration files:")
    lines.append("")
    lines.append("Path | Purpose")
    lines.append("---|---")
    for c in cfg.get("config_files", []):
        path = str(c.get("path", "TBD"))
        purpose = str(c.get("purpose", "TBD"))
        lines.append(f"{path} | {purpose}")

    lines.append("")
    lines.append("Dependencies:")
    deps = cfg.get("dependencies", {})
    lines.append(f"- Language and runtime: {deps.get('language_and_runtime', 'TBD')}")
    lines.append(f"- Package manager: {deps.get('package_manager', 'TBD')}")
    core = deps.get("core_libraries", [])
    if isinstance(core, list) and core:
        lines.append(f"- Core libraries: {', '.join(str(x) for x in core)}")
    else:
        lines.append("- Core libraries: TBD")

    lines.append("")
    lines.append("Build artifacts excluded:")
    excluded = cfg.get("build_artifacts_excluded", {}).get("items", [])
    if isinstance(excluded, list) and excluded:
        lines.append(f"- {', '.join(str(x) for x in excluded)}")
    else:
        lines.append("- venv, conda envs, node_modules, caches, large data, models")

    return "\n".join(lines) + "\n"


def update_snapshot(snapshot: str, new_block_body: str) -> str:
    begin_end, end = find_block(snapshot)
    before = snapshot[:begin_end]
    after = snapshot[end:]
    if not before.endswith("\n"):
        before += "\n"
    return before + new_block_body + after


def main() -> None:
    if not SNAPSHOT_PATH.exists():
        die(f"Missing snapshot file: {SNAPSHOT_PATH}")
    if not MODULES_TOML_PATH.exists():
        die(f"Missing modules file: {MODULES_TOML_PATH}")

    cfg = read_toml(MODULES_TOML_PATH)
    new_body = render_repo_inventory(cfg)

    snapshot = read_text(SNAPSHOT_PATH)
    updated = update_snapshot(snapshot, new_body)
    SNAPSHOT_PATH.write_text(updated, encoding="utf-8")

    print("OK: repo_inventory block updated")


if __name__ == "__main__":
    main()
