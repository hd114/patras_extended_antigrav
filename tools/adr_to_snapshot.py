#!/usr/bin/env python3
"""
Update the decisions AUTO block in docs/hub/CONTEXT_SNAPSHOT.md by scanning docs/adr/ADR-*.md.

Rules:
- Parse ADR id and title from the first H1 line: "# ADR-XXXX Title"
- Parse Date and Status from lines like:
  - Date: YYYY-MM-DD
  - Status: Accepted
- Show the most recent N ADRs (by ADR number, descending) to keep it compact.
- Deterministic ordering.

All output messages are in English.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


SNAPSHOT_PATH = Path("docs/hub/CONTEXT_SNAPSHOT.md")
ADR_DIR = Path("docs/adr")

BEGIN_RE = re.compile(r"^\s*<!--\s*AUTO:BEGIN\s+decisions\s*-->\s*$")
END_RE = re.compile(r"^\s*<!--\s*AUTO:END\s+decisions\s*-->\s*$")

H1_RE = re.compile(r"^\s*#\s+(ADR-(\d+))\s+(.+?)\s*$")
DATE_RE = re.compile(r"^\s*Date:\s*(.+?)\s*$", re.IGNORECASE)
STATUS_RE = re.compile(r"^\s*Status:\s*(.+?)\s*$", re.IGNORECASE)

MAX_ITEMS = 5


@dataclass(frozen=True)
class ADR:
    adr_id: str
    adr_num: int
    title: str
    date: str
    status: str
    path: str


def die(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        die(f"File not found: {path}")


def parse_adr(path: Path) -> Optional[ADR]:
    lines = read_text(path).splitlines()

    adr_id: Optional[str] = None
    adr_num: Optional[int] = None
    title: Optional[str] = None
    date: str = "TBD"
    status: str = "TBD"

    for line in lines:
        if adr_id is None:
            m = H1_RE.match(line)
            if m:
                adr_id = m.group(1)
                adr_num = int(m.group(2))
                title = m.group(3).strip()
                continue

        m = DATE_RE.match(line)
        if m:
            date = m.group(1).strip()
            continue

        m = STATUS_RE.match(line)
        if m:
            status = m.group(1).strip()
            continue

    if adr_id is None or adr_num is None or title is None:
        return None

    rel_path = str(path.as_posix())
    return ADR(
        adr_id=adr_id,
        adr_num=adr_num,
        title=title,
        date=date,
        status=status,
        path=rel_path,
    )


def render_block(adrs: List[ADR]) -> str:
    lines: List[str] = []
    lines.append("ADR | Title | Date | Status | Link")
    lines.append("---|---|---|---|---")
    for a in adrs:
        safe_title = a.title.replace("|", "\\|")
        link = f"[{a.path}]({a.path})"
        lines.append(f"{a.adr_id} | {safe_title} | {a.date} | {a.status} | {link}")
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
        die("decisions AUTO block markers not found or invalid in CONTEXT_SNAPSHOT.md")

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
    if not SNAPSHOT_PATH.exists():
        die(f"Missing snapshot file: {SNAPSHOT_PATH}")
    if not ADR_DIR.exists():
        die(f"Missing ADR directory: {ADR_DIR}")

    adr_files = sorted(ADR_DIR.glob("ADR-*.md"))
    parsed: List[ADR] = []
    for p in adr_files:
        adr = parse_adr(p)
        if adr is not None:
            parsed.append(adr)

    parsed_sorted = sorted(parsed, key=lambda a: a.adr_num, reverse=True)[:MAX_ITEMS]

    body = render_block(parsed_sorted) if parsed_sorted else "No ADRs recorded yet.\n"

    snapshot_text = read_text(SNAPSHOT_PATH)
    updated = update_snapshot(snapshot_text, body)
    SNAPSHOT_PATH.write_text(updated, encoding="utf-8")

    print("OK: decisions block updated")


if __name__ == "__main__":
    main()
