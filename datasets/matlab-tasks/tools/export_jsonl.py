"""Export the benchmark as JSON Lines (one task per line) for evaluation pipelines and dataset hubs.

    python tools/export_jsonl.py                 # -> dist/matlab_tasks.jsonl (prompts, no tests or solutions)
    python tools/export_jsonl.py --full          # include reference solutions and tests

Fields: id, title, category, difficulty, function, signature, tags, prompt,
and with --full also solution (dict file -> code), test, data_files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include solutions and tests")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "matlab_tasks.jsonl")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for meta_path in sorted((ROOT / "tasks").rglob("task.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        row = {k: meta[k] for k in ("id", "title", "category", "difficulty", "function", "signature", "tags")}
        row["verified"] = meta.get("verified", {})
        row["prompt"] = (d / "prompt.md").read_text()
        if args.full:
            row["solution"] = {f.name: f.read_text() for f in sorted((d / "solution").glob("*.m"))}
            row["test"] = (d / "test.m").read_text()
            data = d / "data"
            row["data_files"] = {f.name: f.read_text() for f in sorted(data.iterdir())} if data.is_dir() else {}
        rows.append(row)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tasks to {args.out}")


if __name__ == "__main__":
    main()
