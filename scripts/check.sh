#!/usr/bin/env bash
# Run every check locally: lint, unit tests, conformance, benchmark tasks,
# and (with --web) the browser and VS Code extension tests.
#
#     scripts/check.sh           # Python checks, about 3 minutes
#     scripts/check.sh --web     # also the browser tests (needs Node and Playwright)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "== lint";        "$PY" -m ruff check .
echo "== tests";       "$PY" -m pytest -q
echo "== conformance"; "$PY" tests/conformance/run.py --compare | tail -1
echo "== tasks";       "$PY" datasets/matlab-tasks/tools/run_tasks.py | tail -1

if [ "${1:-}" = "--web" ]; then
  echo "== browser app"; PYTHON="$PY" node scripts/test_web.mjs --no-screenshot | tail -1
  echo "== VS Code extension"; (cd editors/vscode && node --test 2>&1 | grep -E "^# (pass|fail)")
fi
echo "All checks finished."
