#!/usr/bin/env bash
set -euo pipefail

# Run local quality gates before pushing.
# All output is in English.

echo "Running pre-push checks..."

echo "1) Validate snapshot formatting"
make validate

echo "2) Check snapshot drift (non-mutating)"
bash tools/check_snapshot_up_to_date.sh

echo "OK: pre-push checks passed"
