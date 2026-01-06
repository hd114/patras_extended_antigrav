#!/usr/bin/env bash
set -euo pipefail

# Export a deterministic handoff bundle zip from the repository.
# This script includes only git-tracked files to avoid leaking local artifacts.
# All output is in English.

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT_ZIP="${REPO_ROOT}/handoff_bundle.zip"

echo "Exporting handoff bundle..."
echo "Repository root: ${REPO_ROOT}"
echo "Output: ${OUT_ZIP}"

if ! command -v zip >/dev/null 2>&1; then
  echo "ERROR: 'zip' is not available in PATH."
  exit 1
fi

echo "Running snapshot gates..."
make -C "${REPO_ROOT}" snapshot-check >/dev/null
echo "OK: snapshot gates passed"

TMP_DIR="$(mktemp -d)"
FILE_LIST="${TMP_DIR}/handoff_files.txt"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

# Collect a curated set of tracked files. Missing optional files are ignored by git ls-files.
git -C "${REPO_ROOT}" ls-files -- \
  docs \
  tools \
  .github \
  Makefile \
  README.md \
  CONTRIBUTING.md \
  SECURITY.md \
  CHANGELOG.md \
  RULES_AND_HANDOFF.md \
  .gitignore \
| LC_ALL=C sort > "${FILE_LIST}"

# Never include the output zip itself, even if tracked by mistake.
grep -vE '^handoff_bundle\.zip$' "${FILE_LIST}" > "${TMP_DIR}/handoff_files.filtered.txt"
mv "${TMP_DIR}/handoff_files.filtered.txt" "${FILE_LIST}"

FILE_COUNT="$(wc -l < "${FILE_LIST}" | tr -d ' ')"
echo "Files to include: ${FILE_COUNT}"

rm -f "${OUT_ZIP}"

(
  cd "${REPO_ROOT}"
  zip -q -X "${OUT_ZIP}" -@ < "${FILE_LIST}"
)

echo "OK: handoff bundle created: ${OUT_ZIP}"
