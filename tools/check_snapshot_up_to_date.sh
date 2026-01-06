#!/usr/bin/env bash
set -euo pipefail

# Check whether the snapshot and auto-generated docs are up to date.
# This script must not mutate the working tree.
# All output is in English.

echo "Checking snapshot drift (non-mutating)..."

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

TMP_DIR="$(mktemp -d)"
TMP_REPO="${TMP_DIR}/worktree"

cleanup() {
  if [[ -d "${TMP_REPO}" ]]; then
    git -C "${REPO_ROOT}" worktree remove --force "${TMP_REPO}" >/dev/null 2>&1 || true
  fi
  git -C "${REPO_ROOT}" worktree prune >/dev/null 2>&1 || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

# Create a detached worktree so git metadata is available without copying .git.
git -C "${REPO_ROOT}" worktree add --detach "${TMP_REPO}" HEAD >/dev/null

# Regenerate snapshot and auto docs in the temporary worktree.
pushd "${TMP_REPO}" >/dev/null
make snapshot >/dev/null
popd >/dev/null

normalize_snapshot() {
  # Normalize volatile metadata so drift checks focus on deterministic content.
  sed -E \
    -e 's/^Generated at:.*$/Generated at: <ignored>/' \
    -e 's/^Repository revision:.*$/Repository revision: <ignored>/' \
    -e 's/^Working tree status:.*$/Working tree status: <ignored>/' \
    "$1"
}

SNAP_A="${TMP_DIR}/snapshot_local.norm.md"
SNAP_B="${TMP_DIR}/snapshot_temp.norm.md"

normalize_snapshot "${REPO_ROOT}/docs/hub/CONTEXT_SNAPSHOT.md" > "${SNAP_A}"
normalize_snapshot "${TMP_REPO}/docs/hub/CONTEXT_SNAPSHOT.md" > "${SNAP_B}"

SNAPSHOT_DIFF_FILE="${TMP_DIR}/diff_snapshot.txt"
AUTO_DIFF_FILE="${TMP_DIR}/diff_auto.txt"

SNAPSHOT_DIFF=0
AUTO_DIFF=0

if ! diff -u "${SNAP_A}" "${SNAP_B}" > "${SNAPSHOT_DIFF_FILE}" 2>&1; then
  SNAPSHOT_DIFF=1
fi

# Compare only tracked files under docs/auto to avoid noise from untracked files
# such as .ipynb_checkpoints.
: > "${AUTO_DIFF_FILE}"
if git -C "${REPO_ROOT}" ls-files --error-unmatch docs/auto >/dev/null 2>&1; then
  mapfile -t AUTO_TRACKED < <(git -C "${REPO_ROOT}" ls-files -- docs/auto | sort)

  for rel_path in "${AUTO_TRACKED[@]}"; do
    local_path="${REPO_ROOT}/${rel_path}"
    temp_path="${TMP_REPO}/${rel_path}"

    if [[ ! -f "${local_path}" || ! -f "${temp_path}" ]]; then
      AUTO_DIFF=1
      {
        echo "Missing tracked file in one workspace: ${rel_path}"
        echo "Local exists: $([[ -f "${local_path}" ]] && echo yes || echo no)"
        echo "Temp exists:  $([[ -f "${temp_path}" ]] && echo yes || echo no)"
        echo ""
      } >> "${AUTO_DIFF_FILE}"
      continue
    fi

    if ! diff -u -- "${local_path}" "${temp_path}" >> "${AUTO_DIFF_FILE}" 2>&1; then
      AUTO_DIFF=1
      echo "" >> "${AUTO_DIFF_FILE}"
    fi
  done
fi

if [[ "${SNAPSHOT_DIFF}" -eq 0 && "${AUTO_DIFF}" -eq 0 ]]; then
  echo "OK: snapshot and tracked auto docs are up to date"
  exit 0
fi

echo "ERROR: snapshot drift detected. Run 'make snapshot' and commit the changes before pushing."

if [[ "${SNAPSHOT_DIFF}" -ne 0 ]]; then
  echo "Changed: docs/hub/CONTEXT_SNAPSHOT.md (excluding volatile metadata)"
fi

if [[ "${AUTO_DIFF}" -ne 0 ]]; then
  echo "Changed: tracked docs/auto/*"
fi

echo ""
echo "Diff preview (first 160 lines):"
if [[ -s "${SNAPSHOT_DIFF_FILE}" ]]; then
  sed -n '1,160p' "${SNAPSHOT_DIFF_FILE}"
fi
if [[ -s "${AUTO_DIFF_FILE}" ]]; then
  sed -n '1,160p' "${AUTO_DIFF_FILE}"
fi

exit 1
