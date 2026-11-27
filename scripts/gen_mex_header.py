"""Regenerate mylab/mex/include/mylab_mex_api.h from the table in mylab/mex/spec.py.

    python scripts/gen_mex_header.py            # rewrite the header
    python scripts/gen_mex_header.py --check    # exit 1 if the committed header is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mylab.mex.header import API_HEADER, render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="do not write; fail if the header is out of date")
    args = ap.parse_args()
    text = render()
    if args.check:
        if API_HEADER.read_text(encoding="utf-8") != text:
            print(f"{API_HEADER} is out of date: run python scripts/gen_mex_header.py")
            return 1
        return 0
    API_HEADER.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {API_HEADER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
