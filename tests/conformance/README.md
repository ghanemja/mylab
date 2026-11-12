# Conformance suite

This folder checks that mylab computes the same **values** as an independent
implementation of the MATLAB language. We cannot run MATLAB in CI, so the
reference is GNU Octave, corrected by hand where Octave is known to differ
from MATLAB.

## How it works

1. `cases/<area>/<name>.m` are small original scripts. Each one leaves its
   results in workspace variables.
2. `support/cf_run_case.m` runs one case and writes every variable to JSON
   with `support/cf_dump_value.m`: class, size and contents (numbers with 17
   significant digits, text as character codes, cells and structs
   recursively). The same two files run in Octave and in mylab.
3. `run.py --update-golden` runs all cases in Octave and stores the results
   in `golden.json`.
4. `overrides.json` holds the MATLAB-correct value for each variable where
   Octave differs from MATLAB. Each entry has a MATLAB expression built from
   literals and a one-line reason.
5. `tests/test_conformance.py` runs every case on mylab in-process and
   compares it with golden + overrides. It needs no Octave.

The comparison is exact for class, size, text, logicals and integers. Floating
point values may differ by 1e-10 (relative to `max(1, |value|)`), and single
values by 5e-6. A case can ask for a looser tolerance with a comment line such
as `% tol: 1e-6`. Besides its variables, each case contributes one more check:
whether it ran to the end or stopped with an error.

## Running

```sh
python -m pytest -q tests/test_conformance.py        # mylab against the stored expectations
python tests/conformance/run.py --compare            # the same as a categorized report
python tests/conformance/run.py --compare --filter strings --verbose
python tests/conformance/run.py --stats              # number of cases and checks per area
python tests/conformance/run.py --update-golden      # regenerate golden.json (needs octave-cli)
python tests/conformance/run.py --engine octave --out /tmp/octave.json   # raw results of one engine
```

## Known failures

`known_failures.json` lists the differences that mylab has today, as
case → variable → reason (`__error__` stands for the error state of the case).
The pytest reports those cases as `xfail`. It fails when

- a variable that is not listed differs (a regression), or
- a listed variable now matches. Delete the entry then.

A reason that starts with `FLAKY` marks a result that changes from run to run.
Such an entry may pass or fail.

`run.py --compare --write-known` rewrites the list from the current
differences. It keeps existing reasons and gives new entries a `TODO` reason.
Review each new entry before you commit it.

## Adding a case

1. Write `cases/<area>/<name>.m`. Use only syntax that is valid in MATLAB: no
   `!=`, `printf`, `endif`, `x(1)(2)` chains or double-quoted text. Avoid
   random numbers, the clock, the display and file paths.
2. Variables whose names start with `tmp` are not compared, and neither are
   caught error objects. Use `tmp` names for values that may legitimately
   differ, such as solver step counts.
3. To test that something raises an error, catch it and store a flag or the
   message: `t1 = 0; try, x(5); catch, t1 = 1; end`. Compare `err.message`
   and `err.identifier` only for errors that the case raises with `error`.
4. Helper functions live in `support/funcs/` (a script cannot define
   functions in a way that works in both engines).
5. Run `run.py --update-golden`, then `run.py --compare`, and review **every**
   difference by hand:
   - mylab is wrong: fix mylab, or add the variable to `known_failures.json`;
   - Octave differs from MATLAB: add the MATLAB value to `overrides.json`
     with a reason;
   - the behaviour is implementation-defined (order of eigenvectors, wording
     of built-in error messages): remove or loosen the check.
6. If a line fails in Octave only because Octave lacks the feature, write it
   as `try, v = ...; catch, end` and give `v` an override. Cases under
   `cases/matlab_only/` never run in Octave: all their expectations are
   overrides.

## Limits of the method

- **Octave is not MATLAB.** Where both Octave and mylab agree but MATLAB
  differs, the suite sees nothing unless someone wrote an override.
- **Overrides are human judgment.** They come from documented MATLAB
  behaviour, not from a MATLAB run. An override can be wrong. If you can run
  MATLAB, check them first.
- Values are compared, not printed output. The suite says nothing about
  `disp` or command-window formatting.
- A complex result whose imaginary part is all zero counts as equal to the
  real value, because the products differ on when they drop a zero imaginary
  part. Use `isreal` in a case to test this explicitly.
- Octave is used strictly as a black box, as [CLEANROOM.md](../../CLEANROOM.md)
  requires. No case is copied from Octave's tests or from MathWorks
  documentation.

## Current numbers

| | |
|---|---|
| Cases | 257 in 24 areas |
| Checks | 5,377 (5,120 variables + 257 error states) |
| Overrides (MATLAB differs from Octave, or Octave lacks the feature) | 271, of which 154 are in `matlab_only/` |
| Passing on mylab | 5,377 (100%) |
| Known failures | none |

Golden values were recorded with GNU Octave 8.4.0.
