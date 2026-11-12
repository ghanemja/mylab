"""Write the function reference that editor extensions bundle.

    python scripts/gen_editor_data.py            # rewrite editors/vscode/data/functions.json
    python scripts/gen_editor_data.py --check    # exit 1 if the committed file is out of date

The file maps each documented function to its calling forms and description,
taken from mylab's own help entries (mylab/help_data.py):

    {"zeros": {"usage": ["A = zeros(n)", ...], "description": "Array filled with 0. ..."}, ...}

Run this after adding or changing help entries. CI runs it with --check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mylab.help_data import HELP  # noqa: E402

OUT = ROOT / "editors" / "vscode" / "data" / "functions.json"


def build() -> dict:
    data = {}
    for name in sorted(HELP):
        usage, description = HELP[name]
        usages = list(usage) if isinstance(usage, (list, tuple)) else [usage]
        data[name] = {"usage": [u.strip() for u in usages], "description": description.strip()}
    return data


def render() -> str:
    # One function per line: small diffs when a single entry changes.
    data = build()
    lines = [f"{json.dumps(name)}: {json.dumps(entry, ensure_ascii=False)}" for name, entry in data.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="do not write; fail if the file is out of date")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    text = render()
    if args.check:
        current = args.out.read_bytes().decode("utf-8") if args.out.exists() else ""
        if current.replace("\r\n", "\n") != text:
            sys.stderr.write(f"{args.out} is out of date. Run: python scripts/gen_editor_data.py\n")
            return 1
        print(f"{args.out.name} is up to date")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(build())} functions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
