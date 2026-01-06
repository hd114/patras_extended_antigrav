#!/usr/bin/env python3
"""
Update the task_graph AUTO block in docs/hub/CONTEXT_SNAPSHOT.md by scanning docs/work_items/*.md.

Source:
- Each WI may define: 'Depends on: WI-0002, WI-0003' (or '-' / empty)

Rules:
- Nodes are WI IDs with titles derived from the first H1 heading.
- Edges: dependency -> WI
- Only include edges for dependencies that exist as WI files.
- Deterministic ordering by WI number.

All output messages are in English.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


SNAPSHOT_PATH = Path("docs/hub/CONTEXT_SNAPSHOT.md")
WORK_ITEMS_DIR = Path("docs/work_items")

BEGIN_RE = re.compile(r"^\s*<!--\s*AUTO:BEGIN\s+task_graph\s*-->\s*$")
END_RE = re.compile(r"^\s*<!--\s*AUTO:END\s+task_graph\s*-->\s*$")

H1_RE = re.compile(r"^\s*#\s+(WI-(\d+))\s+(.+?)\s*$")
DEPENDS_RE = re.compile(r"^\s*Depends\s+on:\s*(.+?)\s*$", re.IGNORECASE)

WI_ID_RE = re.compile(r"WI-\d{4}")


@dataclass(frozen=True)
class WorkItem:
    wi_id: str
    wi_num: int
    title: str
    depends_on: Tuple[str, ...]


def die(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        die(f"File not found: {path}")


def parse_depends(value: str) -> Tuple[str, ...]:
    v = value.strip()
    if not v or v == "-":
        return tuple()

    parts = [p.strip() for p in v.split(",")]
    deps: List[str] = []
    for p in parts:
        if not p:
            continue
        m = WI_ID_RE.search(p)
        if m:
            deps.append(m.group(0))
    return tuple(deps)


def parse_work_item(path: Path) -> Optional[WorkItem]:
    lines = read_text(path).splitlines()

    wi_id: Optional[str] = None
    wi_num: Optional[int] = None
    title: Optional[str] = None
    depends_on: Tuple[str, ...] = tuple()

    for line in lines:
        if wi_id is None:
            m = H1_RE.match(line)
            if m:
                wi_id = m.group(1)
                wi_num = int(m.group(2))
                title = m.group(3).strip()
                continue

        m = DEPENDS_RE.match(line)
        if m:
            depends_on = parse_depends(m.group(1))
            break

    if wi_id is None or wi_num is None or title is None:
        return None

    return WorkItem(wi_id=wi_id, wi_num=wi_num, title=title, depends_on=depends_on)


def render_mermaid(items: List[WorkItem], existing_ids: set[str]) -> str:
    items_sorted = sorted(items, key=lambda it: it.wi_num)
    lines: List[str] = []
    lines.append("```mermaid")
    lines.append("flowchart TD")

    for it in items_sorted:
        safe_title = it.title.replace("[", "(").replace("]", ")")
        lines.append(f'  {it.wi_id}["{it.wi_id} {safe_title}"]')

    for it in items_sorted:
        for dep in it.depends_on:
            if dep in existing_ids:
                lines.append(f"  {dep} --> {it.wi_id}")

    lines.append("```")
    return "\n".join(lines) + "\n"


def update_snapshot(snapshot_text: str, new_body: str) -> str:
    lines = snapshot_text.splitlines(keepends=True)

    begin_idx: Optional[int] = None
    end_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if begin_idx is None and BEGIN_RE.match(line.strip("\n")):
            begin_idx = i
            continue
        if begin_idx is not None and END_RE.match(line.strip("\n")):
            end_idx = i
            break

    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        die("task_graph AUTO block markers not found or invalid in CONTEXT_SNAPSHOT.md")

    out: List[str] = []
    out.extend(lines[: begin_idx + 1])
    if not out[-1].endswith("\n"):
        out[-1] += "\n"

    out.append(new_body)
    if not new_body.endswith("\n"):
        out.append("\n")

    out.extend(lines[end_idx:])

    return "".join(out)


def main() -> None:
    if not WORK_ITEMS_DIR.exists():
        die(f"Missing directory: {WORK_ITEMS_DIR}")
    if not SNAPSHOT_PATH.exists():
        die(f"Missing snapshot file: {SNAPSHOT_PATH}")

    md_files = sorted(WORK_ITEMS_DIR.glob("WI-*.md"))
    parsed: List[WorkItem] = []
    for p in md_files:
        it = parse_work_item(p)
        if it is not None:
            parsed.append(it)

    existing_ids = {it.wi_id for it in parsed}
    mermaid = render_mermaid(parsed, existing_ids)

    snapshot_text = read_text(SNAPSHOT_PATH)
    updated = update_snapshot(snapshot_text, mermaid)
    SNAPSHOT_PATH.write_text(updated, encoding="utf-8")

    print("OK: task_graph block updated")


if __name__ == "__main__":
    main()
