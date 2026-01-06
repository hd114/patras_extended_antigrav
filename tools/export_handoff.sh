#!/usr/bin/env bash
set -euo pipefail

# Export a minimal handoff bundle for starting a new chat.
# All prints are in English.

OUT_DIR="${1:-.}"
BUNDLE_NAME="${2:-handoff_bundle.zip}"

# Resolve OUT_DIR to an absolute path so it remains valid even if we change directories.
OUT_DIR_ABS="$(python - "$OUT_DIR" << 'PY'
import os
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "."
print(os.path.abspath(p))
PY
)"

echo "Running snapshot and validation..."
make snapshot
make validate

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

echo "Collecting handoff files..."
mkdir -p "${TMP_DIR}/docs/hub" "${TMP_DIR}/docs/work_items" "${TMP_DIR}/docs/adr" "${TMP_DIR}/tools"

# Hub
cp -f docs/hub/CONTEXT_SNAPSHOT.md "${TMP_DIR}/docs/hub/"
cp -f docs/hub/RULES_AND_HANDOFF.md "${TMP_DIR}/docs/hub/" || true
cp -f docs/hub/PIPELINE.md "${TMP_DIR}/docs/hub/" || true
cp -f docs/hub/modules.toml "${TMP_DIR}/docs/hub/" || true
cp -f docs/hub/NEW_CHAT_STARTER.md "${TMP_DIR}/docs/hub/" || true

# Work items
cp -f docs/work_items/README.md "${TMP_DIR}/docs/work_items/" || true
cp -f docs/work_items/WI-*.md "${TMP_DIR}/docs/work_items/" || true
cp -f docs/work_items/_TEMPLATE.md "${TMP_DIR}/docs/work_items/" || true

# ADRs
cp -f docs/adr/README.md "${TMP_DIR}/docs/adr/" || true
cp -f docs/adr/ADR-*.md "${TMP_DIR}/docs/adr/" || true
cp -f docs/adr/_TEMPLATE.md "${TMP_DIR}/docs/adr/" || true

# Tooling (optional but useful for understanding the contract)
cp -f tools/validate_snapshot.py "${TMP_DIR}/tools/" || true
cp -f tools/snapshot.py "${TMP_DIR}/tools/" || true
cp -f tools/work_items_to_snapshot.py "${TMP_DIR}/tools/" || true
cp -f tools/task_graph_to_snapshot.py "${TMP_DIR}/tools/" || true
cp -f tools/repo_inventory_to_snapshot.py "${TMP_DIR}/tools/" || true
cp -f tools/adr_to_snapshot.py "${TMP_DIR}/tools/" || true

echo "Creating zip bundle..."
mkdir -p "${OUT_DIR_ABS}"
( cd "${TMP_DIR}" && zip -r "${OUT_DIR_ABS}/${BUNDLE_NAME}" . >/dev/null )

echo "OK: created ${OUT_DIR_ABS}/${BUNDLE_NAME}"
echo "Upload this zip when starting a new chat."
