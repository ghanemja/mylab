"""Fail when a MATLAB-behaviour check that mylab passed before now fails (a ratchet).

    python tests/probes/check_expected.py --set original          # checks in tests/probes/checks
    python tests/probes/check_expected.py --set heldout           # checks in tests/probes/heldout
    python tests/probes/check_expected.py --set original --update # record the current passes

The list of checks mylab is expected to pass is tests/probes/baselines/mylab_expected.json. A check
that newly passes is reported; run with --update to add it, so it is protected from then on.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXPECTED = os.path.join(HERE, "baselines", "mylab_expected.json")
FOLDERS = {"original": os.path.join(HERE, "checks"), "heldout": os.path.join(HERE, "heldout")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(FOLDERS), required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "results.json")
        subprocess.run([sys.executable, os.path.join(HERE, "run_probes.py"), "--src", ROOT, "--probes",
                        FOLDERS[a.set], "--jobs", str(a.jobs), "--quiet", "--out", out], check=True)
        with open(out, encoding="utf-8") as fh:
            results = {r["id"]: r for r in json.load(fh)["results"]}
    passing = sorted(i for i, r in results.items() if r["ok"])
    expected_all = {}
    if os.path.isfile(EXPECTED):
        with open(EXPECTED, encoding="utf-8") as fh:
            expected_all = json.load(fh)
    if a.update:
        expected_all[a.set] = passing
        with open(EXPECTED, "w", encoding="utf-8") as fh:
            json.dump(expected_all, fh, indent=0, sort_keys=True)
        print(f"{a.set}: {len(passing)} of {len(results)} checks recorded as expected to pass.")
        return
    expected = set(expected_all.get(a.set, []))
    lost = sorted(i for i in expected if i in results and not results[i]["ok"])
    gained = sorted(i for i in passing if i not in expected)
    print(f"{a.set}: {len(passing)} of {len(results)} checks pass; {len(expected)} expected.")
    if gained:
        print(f"{len(gained)} check(s) pass now that were not expected to; record them with --update:")
        for i in gained:
            print("   ", i)
    if lost:
        print(f"{len(lost)} check(s) that passed before fail now:")
        for i in lost:
            print(f"    {i} :: {results[i]['msg'][:200]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
