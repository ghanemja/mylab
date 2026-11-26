mylab validation kit for MATLAB
================================

What it does: runs 705 + 469 small checks of MATLAB behaviour and evaluates 438
expressions in MATLAB, and writes one file, mylab_validation_results.json. Nothing else is
read or written; each check runs in a temporary folder that is deleted afterwards.

1. Upload all files in this folder (or the zip, then right-click it and choose Extract).
2. In the Command Window, go to the folder and run:

       run_mylab_validation

   The run saves its progress. It stops by itself after 12 minutes (below the 15-minute
   compute limit of MATLAB Online basic); run the same command again to continue until it
   prints "Finished".
3. Download mylab_validation_results.json and give it to the mylab compare script:

       python validation/compare_with_matlab.py mylab_validation_results.json
