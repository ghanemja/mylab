"""Report how well ``.slx`` files import and whether the result simulates.

Usage::

    python -m mylab.blocks.import_report <folder or file> [--json out.json] [--time-limit 20]

For every file the report says whether it imported fully, imported with
unsupported blocks (named and counted) or failed, and whether the imported
model then ran to its stop time. A model that needs workspace variables or
blocks mylab does not have is listed as such, not as a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

from .engine import SimError, compile_model, Sim
from .model import ModelError
from .slx import import_slx


class _TimeUp(Exception):
    pass


def check_file(path: str, time_limit: float = 20.0) -> dict:
    """Import one file and try to simulate it. Returns a plain dict."""
    row: dict = {"file": path, "import": "failed", "blocks": 0, "unsupported": {}, "simulate": "not tried", "detail": ""}
    try:
        model = import_slx(path)
    except ModelError as exc:
        row["detail"] = str(exc)
        return row
    except Exception as exc:  # a reader bug must not stop the survey
        row["detail"] = f"{type(exc).__name__}: {exc}"
        return row
    rep = model.import_report  # type: ignore[attr-defined]
    row["blocks"] = rep["blocks"]
    row["unsupported"] = rep["unsupported"]
    row["import"] = "partial" if rep["unsupported"] else "full"
    if rep["unsupported"]:
        row["simulate"] = "no: unsupported blocks"
        return row
    if rep["blocks"] == 0:
        row["simulate"] = "no: empty model"
        return row
    start = time.time()

    def tick(sim, t):
        if time.time() - start > time_limit:
            raise _TimeUp()

    try:
        comp = compile_model(model, strict=False)
        errors = comp.errors()
        if errors:
            msgs = " | ".join(e["message"] for e in errors[:3])
            missing = [e for e in errors if e.get("kind") == "param" and
                       any(w in e["message"] for w in ("Unrecognized", "Undefined", "not defined", "cannot be evaluated"))]
            if len(missing) == len(errors):
                row["simulate"] = "no: needs workspace variables or toolbox functions"
            elif any(e.get("kind") == "algebraic-loop" for e in errors):
                row["simulate"] = "no: algebraic loop"
            else:
                row["simulate"] = "no: model error"
            row["detail"] = msgs[:400]
            return row
        sim = Sim(comp, None, tick)
        sim.run()
        res = sim.result()
        row["simulate"] = "yes"
        row["detail"] = f"{res.steps} steps to t = {res.t[-1]:g}"
    except _TimeUp:
        row["simulate"] = "no: time limit"
        row["detail"] = f"stopped after {time_limit:g} s of computing"
    except SimError as exc:
        text = exc.message
        needs = any(w in text for w in ("Unrecognized", "Undefined", "not defined"))
        row["simulate"] = "no: needs workspace variables or toolbox functions" if needs else "no: simulation error"
        row["detail"] = text[:400]
    except Exception as exc:
        row["simulate"] = "no: internal error"
        row["detail"] = f"{type(exc).__name__}: {exc}"[:400]
    return row


def survey(root: str, time_limit: float = 20.0) -> list[dict]:
    files = []
    if os.path.isfile(root):
        files = [root]
    else:
        for folder, _dirs, names in os.walk(root):
            files += [os.path.join(folder, n) for n in names if n.lower().endswith(".slx")]
    rows = []
    for f in sorted(files):
        row = check_file(f, time_limit)
        row["file"] = os.path.relpath(f, root) if os.path.isdir(root) else os.path.basename(f)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    imp = Counter(r["import"] for r in rows)
    simc = Counter(r["simulate"] for r in rows)
    missing: Counter = Counter()
    files_with: Counter = Counter()
    for r in rows:
        labels = set()
        for k, v in r["unsupported"].items():
            label = "Library link (toolbox or user library)" if k.startswith("Library link") else k
            missing[label] += v
            labels.add(label)
        files_with.update(labels)
    return {"files": len(rows), "import": dict(imp), "simulate": dict(simc),
            "unsupported_blocks": dict(missing.most_common()), "unsupported_files": dict(files_with.most_common())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--json", help="also write the full result to this file")
    ap.add_argument("--time-limit", type=float, default=20.0, help="seconds of computing allowed per model")
    args = ap.parse_args(argv)
    rows = survey(args.path, args.time_limit)
    for r in rows:
        miss = ", ".join(f"{k} ({v})" for k, v in sorted(r["unsupported"].items(), key=lambda kv: -kv[1])[:6])
        more = len(r["unsupported"]) - 6
        print(f"{r['file']}\n    import: {r['import']}  blocks: {r['blocks']}  simulate: {r['simulate']}")
        if miss:
            print(f"    unsupported: {miss}{f' and {more} more kinds' if more > 0 else ''}")
        if r["detail"]:
            print(f"    {r['detail']}")
    s = summarize(rows)
    print(f"\n{s['files']} files. Import: " + ", ".join(f"{k} {v}" for k, v in sorted(s["import"].items())))
    print("Simulate: " + ", ".join(f"{k} {v}" for k, v in sorted(s["simulate"].items())))
    top = list(s["unsupported_files"].items())[:15]
    if top:
        print("Unsupported blocks by number of files: " + ", ".join(f"{k} {v}" for k, v in top))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": s, "files": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
