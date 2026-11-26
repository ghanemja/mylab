#!/bin/bash
# The speed cases on GNU Octave, one process per case:  run_octave.sh [octave-program] [seconds-cap]
# Prints one line per case: BENCH <name> <best-of-3 seconds> <value>, or TIMEOUT when the cap was hit.
OCT="${1:-octave}"; CAP="${2:-360}"
cd "$(dirname "$0")"
export DISPLAY="${DISPLAY:-:99}"
for f in b*.m; do
  n="${f%.m}"
  out=$(timeout "$CAP" "$OCT" --no-init-file --quiet --eval "warning('off','all'); CASE='$n'; one_case" 2>/dev/null | grep -a "^BENCH")
  [ -z "$out" ] && out=$(printf 'BENCH\t%s\tTIMEOUT\tNaN\tover %s s' "$n" "$CAP")
  echo "$out"
done
