# MATLAB validation kit (on hold)

**Do not run this kit on MATLAB.** Nobody has ever run it. mylab has never
been run on MATLAB, on MATLAB Online or under a MathWorks CI licence, and
the project holds no MathWorks licence.

MathWorks' terms forbid using MATLAB to test or benchmark it in order to
make software with the same or similar functions. That includes MATLAB
Online's free tier and the free licence for GitHub Actions, and mylab is
that kind of software. The details and quotes are in
[docs/MATLAB-TERMS-REVIEW.md](../docs/MATLAB-TERMS-REVIEW.md). The kit
stays on hold unless MathWorks gives written permission.

An earlier version of this file (added 26 November 2026) gave step-by-step
instructions for running the kit on MATLAB Online. They were written before
the licence terms were checked and were never followed. They were removed
once the review was done.

## What is here

The files are kept so the work isn't lost if permission is ever given.

- `build_kit.py` packs mylab's behaviour checks (`tests/probes/`) and value
  expressions into `matlab/checks.json` and `matlab/exprs.json`.
- `matlab/` holds the MATLAB-side runner and those JSON files.
- `compare_with_matlab.py` compares a results file with mylab.

## How mylab is checked instead

- 705 + 469 behaviour checks, written from MATLAB's public documentation.
- 5,377 values compared with GNU Octave, run as a black box.
- 25 real repositories from the MATLAB Corpus that run unchanged.

See [docs/COMPATIBILITY.md](../docs/COMPATIBILITY.md). If you find a
difference, please
[report it](https://github.com/ghanemja/mylab/issues/new?template=report-a-difference.yml)
with a short example and a link to the MathWorks reference page that
documents what should happen.
