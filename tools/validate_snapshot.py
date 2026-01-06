#!/usr/bin/env python3
"""
Validate docs/hub/CONTEXT_SNAPSHOT.md formatting.

Checks:
- AUTO:BEGIN and AUTO:END markers are paired and properly nested.
- AUTO markers do not appear inside fenced code blocks.
- Mermaid fences in task_graph and module_graph blocks are closed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple


BEGIN_RE = re.compile(r"^\s*<!--\s*AUTO:BEGIN\s+([A-Za-z0-9_\-]+)\s*-->\s*$")
END_RE = re.compile(r"^\s*<!--\s*AUTO:END\s+([A-Za-z0-9_\-]+)\s*-->\s*$")
FENCE_RE = re.compile(r"^\s*```")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def load_lines(path: Path) -> List[str]:
    if not path.exists():
        fail(f"Missing file: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def validate_markers_not_in_fences(lines: List[str]) -> None:
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if in_fence and (BEGIN_RE.match(line) or END_RE.match(line)):
            fail(f"AUTO marker inside fenced code block at line {i}")


def validate_marker_pairs(lines: List[str]) -> List[Tuple[str, int, int]]:
    stack: List[Tuple[str, int]] = []
    blocks: List[Tuple[str, int, int]] = []
    for i, line in enumerate(lines, start=1):
        m = BEGIN_RE.match(line)
        if m:
            name = m.group(1)
            stack.append((name, i))
            continue
        m = END_RE.match(line)
        if m:
            name = m.group(1)
            if not stack:
                fail(f"AUTO:END without matching BEGIN for '{name}' at line {i}")
            top_name, top_line = stack.pop()
            if top_name != name:
                fail(
                    f"Mismatched AUTO markers at line {i}. "
                    f"Expected END for '{top_name}' (BEGIN at line {top_line}), got '{name}'."
                )
            blocks.append((name, top_line, i))
    if stack:
        name, line_no = stack[-1]
        fail(f"AUTO:BEGIN for '{name}' at line {line_no} has no matching END")
    return blocks


def extract_block(lines: List[str], start: int, end: int) -> str:
    # start/end are line numbers containing the markers
    body = lines[start:end - 1]
    return "\n".join(body)


def validate_mermaid_blocks(lines: List[str], blocks: List[Tuple[str, int, int]]) -> None:
    by_name = {name: (s, e) for name, s, e in blocks}
    for name in ("task_graph", "module_graph"):
        if name not in by_name:
            continue
        s, e = by_name[name]
        body = extract_block(lines, s, e)
        if "```mermaid" not in body:
            fail(f"Block '{name}' must contain a ```mermaid fenced code block")
        mermaid_idx = body.find("```mermaid")
        after = body[mermaid_idx + len("```mermaid") :]
        if "```" not in after:
            fail(f"Block '{name}' has an opening ```mermaid fence but no closing ``` fence")


def main() -> None:
    path = Path("docs/hub/CONTEXT_SNAPSHOT.md")
    lines = load_lines(path)

    validate_markers_not_in_fences(lines)
    blocks = validate_marker_pairs(lines)
    validate_mermaid_blocks(lines, blocks)

    print("OK: CONTEXT_SNAPSHOT.md validated successfully")


if __name__ == "__main__":
    main()
