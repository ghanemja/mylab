# Checking mylab against real MATLAB

mylab's behaviour checks (`tests/probes/`) were written from MATLAB's documented behaviour and
compared with GNU Octave, but never run in MATLAB itself. This kit runs them in MATLAB, so that
every disagreement can be found and fixed, and every check whose expectation is wrong about
MATLAB can be corrected.

What runs in MATLAB:

- the 705 original checks (`tests/probes/checks/`),
- the 469 held-out checks (`tests/probes/heldout/`),
- the 438 value-comparison expressions (`tests/probes/values/exprs.txt`), each turned into a
  one-line text form of its class, size and value with 17 significant digits.

The kit only reads its own files and writes one results file. Each check runs in its own
temporary folder, deleted afterwards.

## Step by step

1. **Build the kit** (or use the copy already in `validation/matlab/`):

   ```bash
   python validation/build_kit.py
   ```

   This writes `validation/matlab/checks.json` and `exprs.json`, and the zip
   `dist/mylab-matlab-validation.zip`.

2. **Open MATLAB Online** at https://matlab.mathworks.com and sign in. A free MathWorks account
   gives 20 hours a month, with a limit of 15 minutes of continuous computing. A desktop MATLAB
   (R2021a or newer) works the same way.

3. **Upload the kit.** In the Files panel, choose Upload and pick
   `dist/mylab-matlab-validation.zip`. Right-click the uploaded zip and choose Extract. Then
   double-click the new `mylab-matlab-validation` folder so that it is the current folder.

4. **Run it.** In the Command Window type:

   ```matlab
   run_mylab_validation
   ```

   It prints its progress every 20 seconds and saves as it goes. It stops by itself after
   12 minutes so that MATLAB Online never cuts it off; run the same command again and it
   continues where it stopped. Repeat until it prints `Finished: ...`. The whole run takes
   about 15 to 40 minutes of MATLAB time.

   Options: `run_mylab_validation('MaxMinutes', 30)` on a desktop MATLAB without the limit;
   `run_mylab_validation('Only', '^tables/')` to run part of the checks.

5. **Download the results.** Right-click `mylab_validation_results.json` in the Files panel and
   choose Download.

6. **Compare** on a computer with a mylab checkout (`pip install -e ".[dev]"` done):

   ```bash
   python validation/compare_with_matlab.py ~/Downloads/mylab_validation_results.json
   ```

   This runs the same checks and expressions on mylab (several minutes) and writes
   `validation/report.md` and `validation/report.json`.

## Reading the report

- **MATLAB passes, mylab fails:** mylab is wrong. Each entry shows mylab's error and the check.
- **Checks MATLAB itself fails:** the check expects something MATLAB does not do. Correct the
  check and record the reason in `tests/probes/heldout/CORRECTIONS.md`; never change a check
  just to make an engine pass. If mylab passes such a check, mylab matches the wrong expectation
  and must be fixed as well.
- **Not available in this MATLAB:** the function needs a toolbox or license that the MATLAB used
  does not have. These say nothing either way.
- **Expressions with a different value:** class, size or value differs (numbers are compared to a
  relative tolerance of 1e-10). Error messages are not compared, only whether both raised one.
- The one check of Octave-only syntax is skipped, because MATLAB rejects that syntax by design.
