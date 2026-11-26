"""Build the MATLAB validation kit: validation/matlab/{checks.json, exprs.json} and a zip to upload.

    python validation/build_kit.py            # writes the JSON files and dist/mylab-matlab-validation.zip

The kit runs mylab's MATLAB-behaviour checks (tests/probes/checks and tests/probes/heldout) and the
value-comparison expressions (tests/probes/values/exprs.txt) in real MATLAB. See validation/README.md.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, "validation", "matlab")
sys.path.insert(0, os.path.join(ROOT, "tests", "probes"))
from run_probes import parse  # noqa: E402

SETS = {"original": os.path.join(ROOT, "tests", "probes", "checks"),
        "heldout": os.path.join(ROOT, "tests", "probes", "heldout")}
FILES = ["run_mylab_validation.m", "mlv_eval.m", "mlv_repr.m", "nthout.m", "checks.json", "exprs.json",
         "README.txt"]


def load_checks() -> list[dict]:
    out = []
    for set_name, folder in SETS.items():
        for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
            for c in parse(path):
                out.append({"set": set_name, "id": c["id"], "cat": c["cat"], "code": c["code"],
                            "files": [{"path": k, "text": v} for k, v in sorted(c["files"].items())]})
    return out


def load_exprs() -> list[str]:
    with open(os.path.join(ROOT, "tests", "probes", "values", "exprs.txt"), encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


README = """mylab validation kit for MATLAB
================================

What it does: runs {n_orig} + {n_held} small checks of MATLAB behaviour and evaluates {n_expr}
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
"""


def main() -> None:
    checks = load_checks()
    exprs = load_exprs()
    with open(os.path.join(KIT, "checks.json"), "w", encoding="utf-8") as fh:
        json.dump(checks, fh, ensure_ascii=False)
    with open(os.path.join(KIT, "exprs.json"), "w", encoding="utf-8") as fh:
        json.dump(exprs, fh, ensure_ascii=False)
    n_orig = sum(c["set"] == "original" for c in checks)
    n_held = sum(c["set"] == "heldout" for c in checks)
    with open(os.path.join(KIT, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(README.format(n_orig=n_orig, n_held=n_held, n_expr=len(exprs)))
    os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
    zpath = os.path.join(ROOT, "dist", "mylab-matlab-validation.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(os.path.join(KIT, name), os.path.join("mylab-matlab-validation", name))
    print(f"{n_orig} original checks, {n_held} held-out checks, {len(exprs)} expressions")
    print(f"kit: {KIT}\nzip: {zpath}")


if __name__ == "__main__":
    main()
