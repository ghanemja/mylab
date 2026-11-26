#!/usr/bin/env bash
# Run the checks CI runs: lint, unit tests, conformance, benchmark tasks; with --full also the
# MATLAB-behaviour checks and the speed check, and with --web the browser and VS Code tests.
#
#     scripts/check.sh           # Python checks, about 5 minutes
#     scripts/check.sh --full    # also the probe ratchet and the speed check (about 10 more minutes)
#     scripts/check.sh --web     # also the browser tests (needs Node and Playwright)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "== lint";        "$PY" -m ruff check .
echo "== tests";       "$PY" -m pytest -q
echo "== conformance"; "$PY" tests/conformance/run.py --compare | tail -1
echo "== tasks";       "$PY" datasets/matlab-tasks/tools/run_tasks.py | tail -1

if [ "${1:-}" = "--full" ]; then
  echo "== MATLAB-behaviour checks"
  "$PY" tests/probes/check_expected.py --set original --jobs 4 | tail -1
  "$PY" tests/probes/check_expected.py --set heldout --jobs 4 | tail -1
  echo "== speed";       "$PY" benchmarks/speed_check.py --limit 0.25 | tail -1
fi

if [ "${1:-}" = "--web" ]; then
  echo "== browser app"; PYTHON="$PY" node scripts/test_web.mjs --no-screenshot | tail -1
  echo "== VS Code extension"; (cd editors/vscode && node --test 2>&1 | grep -E "^# (pass|fail)")
fi
echo "All checks finished."
