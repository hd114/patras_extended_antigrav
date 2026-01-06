#!/usr/bin/env bash
set -euo pipefail

# Install local git hooks for this repository.
# All output is in English.

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="${REPO_ROOT}/.git/hooks"

echo "Installing git hooks into ${HOOKS_DIR}"

install -m 0755 "${REPO_ROOT}/tools/git-hooks/pre-push" "${HOOKS_DIR}/pre-push"

echo "OK: hooks installed"
echo "Note: hooks are local to this clone and are not versioned by git."
