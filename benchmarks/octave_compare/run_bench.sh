#!/bin/bash
# Speed benchmarks on a mylab checkout:  run_bench.sh /path/to/checkout [case-regex]
# Prints one line per case: BENCH <name> <best-of-3 seconds> <value>. Baseline times for mylab and Octave are in BASELINE_TIMES.txt.
SRC="$(cd "${1:-.}" && pwd)"; RE="${2:-.}"
cd "$(dirname "$0")"
for f in b*.m; do
  n="${f%.m}"
  echo "$n" | grep -qE "$RE" || continue
  PYTHONPATH="$SRC" timeout 600 python -m mylab -e "CASE='$n'; one_case" 2>&1 | grep -a -E "^BENCH|Error" | head -2
done
