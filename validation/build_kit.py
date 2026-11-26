"""Build the MATLAB validation kit (on hold): validation/matlab/{checks.json, exprs.json} and a zip.

    python validation/build_kit.py            # writes the JSON files and dist/mylab-matlab-validation.zip

The kit runs mylab's MATLAB-behaviour checks (tests/probes/checks and tests/probes/heldout) and the
value-comparison expressions (tests/probes/values/exprs.txt) in real MATLAB. It is on hold and must
not be run: see validation/README.md and docs/MATLAB-TERMS-REVIEW.md.
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


README = """mylab validation kit for MATLAB (ON HOLD)
==========================================

Do not run this kit. MathWorks' licence terms forbid using MATLAB, MATLAB Online or
the free CI licence to test software with the same or similar functions, such as
mylab. Nobody has run it, and it stays on hold unless MathWorks gives written
permission. See validation/README.md and docs/MATLAB-TERMS-REVIEW.md in the mylab
repository.

Contents: run_mylab_validation.m and its helpers, and checks.json and exprs.json with
{n_orig} + {n_held} behaviour checks and {n_expr} expressions.
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
