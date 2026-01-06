#!/usr/bin/env python3
"""
Update the work_items AUTO block in docs/hub/CONTEXT_SNAPSHOT.md by scanning docs/work_items/*.md.

Rules:
- Active statuses: Planned, In progress, Blocked
- Done status: Done
- Include up to 5 active items and up to 3 recently completed items
- Ordering: active first (In progress, Blocked, Planned), then by WI number ascending
            done next, by WI number descending

All output messages are in English.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


SNAPSHOT_PATH = Path("docs/hub/CONTEXT_SNAPSHOT.md")
WORK_ITEMS_DIR = Path("docs/work_items")

BEGIN_RE = re.compile(r"^\s*<!--\s*AUTO:BEGIN\s+work_items\s*-->\s*$")
END_RE = re.compile(r"^\s*<!--\s*AUTO:END\s+work_items\s*-->\s*$")

H1_RE = re.compile(r"^\s*#\s+(WI-(\d+))\s+(.+?)\s*$")
STATUS_RE = re.compile(r"^\s*Status:\s*(.+?)\s*$", re.IGNORECASE)


ACTIVE_STATUSES = {"planned", "in progress", "blocked"}
DONE_STATUS = "done"

STATUS_PRIORITY = {
    "in progress": 0,
    "blocked": 1,
    "planned": 2,
}


@dataclass(frozen=True)
class WorkItem:
    wi_id: str
    wi_num: int
    title: str
    status: str
    path: Path


def die(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        die(f"File not found: {path}")


def parse_work_item(path: Path) -> Optional[WorkItem]:
    """Parse WI metadata from a markdown file.

    Expected:
    - First heading line: '# WI-000X Title'
    - A line: 'Status: ...'
    """
    text = read_text(path)
    lines = text.splitlines()

    wi_id: Optional[str] = None
    wi_num: Optional[int] = None
    title: Optional[str] = None
    status: Optional[str] = None

    for line in lines:
        if wi_id is None:
            m = H1_RE.match(line)
            if m:
                wi_id = m.group(1)
                wi_num = int(m.group(2))
                title = m.group(3).strip()
                continue

        if status is None:
            m = STATUS_RE.match(line)
            if m:
                status = m.group(1).strip()
                continue

        if wi_id is not None and status is not None:
            break

    if wi_id is None or wi_num is None or title is None:
        return None

    if status is None:
        status = "Planned"

    return WorkItem(
        wi_id=wi_id,
        wi_num=wi_num,
        title=title,
        status=status,
        path=path,
    )


def normalize_status(status: str) -> str:
    return " ".join(status.strip().lower().split())


def classify(items: Iterable[WorkItem]) -> Tuple[List[WorkItem], List[WorkItem]]:
    active: List[WorkItem] = []
    done: List[WorkItem] = []

    for it in items:
        ns = normalize_status(it.status)
        if ns in ACTIVE_STATUSES:
            active.append(it)
        elif ns == DONE_STATUS:
            done.append(it)
        else:
            # Unknown statuses are treated as active to keep visibility.
            active.append(it)

    def active_key(it: WorkItem) -> Tuple[int, int]:
        ns = normalize_status(it.status)
        prio = STATUS_PRIORITY.get(ns, 9)
        return (prio, it.wi_num)

    active_sorted = sorted(active, key=active_key)
    done_sorted = sorted(done, key=lambda it: it.wi_num, reverse=True)

    return active_sorted, done_sorted


def render_block(active: List[WorkItem], done: List[WorkItem]) -> str:
    lines: List[str] = []
    lines.append("Format: ID | Title | Status | Owner | Links | Notes")

    for it in active[:5]:
        lines.append(
            f"- {it.wi_id} | {it.title} | {it.status} | You | {it.path.as_posix()} | -"
        )

    for it in done[:3]:
        lines.append(
            f"- {it.wi_id} | {it.title} | {it.status} | You | {it.path.as_posix()} | -"
        )

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
        die("work_items AUTO block markers not found or invalid in CONTEXT_SNAPSHOT.md")

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

    md_files = sorted(WORK_ITEMS_DIR.glob("WI-*.md"))
    parsed: List[WorkItem] = []
    for p in md_files:
        it = parse_work_item(p)
        if it is not None:
            parsed.append(it)

    if not SNAPSHOT_PATH.exists():
        die(f"Missing snapshot file: {SNAPSHOT_PATH}")

    active, done = classify(parsed)
    new_body = render_block(active, done)

    snapshot_text = read_text(SNAPSHOT_PATH)
    updated = update_snapshot(snapshot_text, new_body)
    SNAPSHOT_PATH.write_text(updated, encoding="utf-8")

    print("OK: work_items block updated")


if __name__ == "__main__":
    main()
