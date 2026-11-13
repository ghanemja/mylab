"""Report how well ``.slx`` files import and whether the result simulates.

Usage::

    python -m mylab.blocks.import_report <folder or file> [--sources DIR] [--json out.json] [--time-limit 30]

For every file the report says whether it imported fully, imported with
unsupported blocks (named and counted) or failed, and whether the imported
model then ran to its stop time. A model that needs workspace variables or
blocks mylab does not have is listed as such, not as a pass.

Models often need set-up that is not in the model file: variables made by a
script, and functions in ``.m`` files. With ``--sources DIR`` the tool looks
for those in ``DIR/<first folder of the model's path>/...`` (a checkout of
the same repository): its folders go on the search path, and when a model
names variables that do not exist, scripts that assign those names are run
first. Every script that was run is listed with the result.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager

from .engine import Sim, SimError, compile_model, run_callbacks
from .model import ModelError
from .slx import import_slx

MISSING = re.compile(
    r"(?:Unrecognized function or variable|Undefined function or variable|Undefined function|the S-function) "
    r"'([A-Za-z_]\w*)'(?! failed)"
)


class _TimeUp(BaseException):
    pass


@contextmanager
def time_limit(seconds: float):
    """Stop the enclosed work after ``seconds`` (on the main thread of a POSIX system; elsewhere no limit)."""
    usable = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if not usable or seconds <= 0:
        yield
        return

    def stop(signum, frame):
        raise _TimeUp()

    old = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _m_folders(root: str, limit: int = 400) -> list[str]:
    out = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "+", "@")) and d != "private")
        if any(f.endswith(".m") for f in files):
            out.append(folder)
            if len(out) >= limit:
                break
    return out


def _scripts(folders: list[str]) -> list[tuple[str, str]]:
    """(path, text) of the script files (not functions or classes) in the folders."""
    out = []
    for folder in folders:
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".m"):
                continue
            p = os.path.join(folder, f)
            try:
                if os.path.getsize(p) > 400_000:
                    continue
                with open(p, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            code = re.sub(r"^\s*(%.*)?\n", "", text, flags=re.M).lstrip()
            if re.match(r"(function|classdef)\b", code):
                continue
            out.append((p, text))
    return out


def find_setup(missing: set[str], scripts: list[tuple[str, str]], model_path: str, tried: set[str]) -> str | None:
    """The script most likely to define the missing variables: it assigns the most of them, and sits nearest."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    stem = os.path.splitext(os.path.basename(model_path))[0].lower()
    best, best_score = None, 0.0
    for path, text in scripts:
        if path in tried:
            continue
        hits = sum(
            1 for name in missing if re.search(rf"^\s*{re.escape(name)}\s*(\.[\w.]+|\([^=\n]*\))?\s*=[^=]", text, re.M)
        )
        if not hits:
            continue
        base = os.path.splitext(os.path.basename(path))[0].lower()
        score = 10.0 * hits / len(missing)
        if os.path.dirname(path) == model_dir:
            score += 3
        if re.search(r"init|setup|param|data|startup", base):
            score += 1
        # shared name fragments, e.g. initCS2.m for demoCS2passiveObserverDP.slx
        frag = max((len(m) for m in re.findall(r"[a-z0-9]{3,}", base) if m in stem), default=0)
        score += min(frag, 6) * 0.5
        if score > best_score:
            best, best_score = path, score
    return best


def _missing_names(messages: list[str]) -> set[str]:
    names: set[str] = set()
    for m in messages:
        names.update(MISSING.findall(m))
        if "Dot indexing is not supported" in m:
            # 'quad.nrotors' where quad is a function here: the model expects a workspace struct of that name
            names.update(re.findall(r"(?<![\w.])([A-Za-z_]\w*)\.[A-Za-z_]", m.split("cannot be evaluated")[0]))
    return names


def check_file(path: str, time_limit_s: float = 30.0, sources: str | None = None) -> dict:
    """Import one file and try to simulate it. Returns a plain dict."""
    from ..interpreter import Interpreter

    row: dict = {
        "file": path,
        "import": "failed",
        "blocks": 0,
        "unsupported": {},
        "simulate": "not tried",
        "detail": "",
        "setup": [],
    }
    try:
        with time_limit(max(time_limit_s, 60)):
            model = import_slx(path)
    except ModelError as exc:
        row["detail"] = str(exc)
        return row
    except _TimeUp:
        row["detail"] = "reading the file took too long"
        return row
    except Exception as exc:  # a reader bug must not stop the survey
        row["detail"] = f"{type(exc).__name__}: {exc}"
        return row
    rep = model.import_report  # type: ignore[attr-defined]
    row["blocks"] = rep["blocks"]
    row["unsupported"] = rep["unsupported"]
    row["import"] = "partial" if rep["unsupported"] else "full"
    row["library"] = bool(model.library)
    row["standins"] = sum(1 for n in model.notes if "was replaced by a subsystem" in n)
    if model.library:
        row["simulate"] = "library: not a model to run"
        return row
    if rep["unsupported"]:
        row["simulate"] = "no: unsupported blocks"
        return row
    last: list[str] = []
    if rep["blocks"] == 0:
        row["simulate"] = "no: empty model"
        return row

    interp = Interpreter(out=lambda s: None)
    interp.input_func = lambda prompt="": (_ for _ in ()).throw(RuntimeError("input() is not available"))
    interp.state["dialogs_noninteractive"] = True
    folders = [os.path.dirname(os.path.abspath(path))]
    scripts: list[tuple[str, str]] = []
    if sources:
        folders += [f for f in _m_folders(sources) if f not in folders]
        scripts = _scripts(folders)
    interp.path = list(folders)
    here = os.getcwd()
    tried: set[str] = set()
    start = time.time()

    def tick(sim, t):
        if time.time() - start > time_limit_s:
            raise _TimeUp()

    def needs(messages: list[str]) -> bool:
        """Try to provide missing names by running one more set-up script. True when one was run."""
        missing = _missing_names(messages)
        script = find_setup(missing, scripts, path, tried) if missing else None
        if script is None:
            return False
        tried.add(script)
        try:
            os.chdir(os.path.dirname(script))
            with time_limit(time_limit_s):
                interp.run_file(script)
            row["setup"].append(os.path.relpath(script, sources) if sources else os.path.basename(script))
        except _TimeUp:
            row["setup"].append(os.path.basename(script) + " (stopped: took too long)")
        except Exception as exc:
            row["setup"].append(f"{os.path.basename(script)} (stopped: {str(getattr(exc, 'message', exc))[:80]})")
        finally:
            os.chdir(here)
        return True

    try:
        for _round in range(6):
            start = time.time()
            try:
                with time_limit(time_limit_s + 5):
                    notes = run_callbacks(model, interp)
                    comp = compile_model(model, interp, strict=False)
                    errors = comp.errors()
                    if errors:
                        messages = [e["message"] for e in errors]
                        last[:] = messages
                        if needs(messages + notes):
                            continue
                        missing = _missing_names(messages)
                        if missing:
                            row["simulate"] = "no: needs workspace variables or toolbox functions"
                            row["missing"] = sorted(missing)
                        elif any(e.get("kind") == "algebraic-loop" for e in errors):
                            row["simulate"] = "no: algebraic loop"
                        else:
                            row["simulate"] = "no: model error"
                        row["detail"] = " | ".join(messages[:3])[:400]
                        return row
                    sim = Sim(comp, None, tick)
                    sim.run()
                    res = sim.result()
                stopped = [d["message"] for d in comp.diagnostics if d.get("kind") == "run"]
                row["standins"] += len(stopped)  # a display block that was switched off is not a real block either
                row["simulate"] = "yes (with stand-ins)" if row["standins"] else "yes"
                row["detail"] = f"{res.steps} steps to t = {res.t[-1]:g}" + ("; " + stopped[0] if stopped else "")
                row["result"] = res
                return row
            except _TimeUp:
                row["simulate"] = "no: time limit"
                row["detail"] = f"stopped after {time_limit_s:g} s of computing"
                return row
            except SimError as exc:
                last[:] = [exc.message]
                if needs([exc.message]):
                    continue
                missing = _missing_names([exc.message])
                row["simulate"] = (
                    "no: needs workspace variables or toolbox functions" if missing else "no: simulation error"
                )
                if re.search(
                    r"FileNotFoundError|No such file|Unable to (read|open|find) file|does not exist", exc.message
                ):
                    row["simulate"] = "no: needs data files that are not available"
                if missing:
                    row["missing"] = sorted(missing)
                row["detail"] = exc.message[:400]
                return row
        row["simulate"] = "no: needs workspace variables or toolbox functions"
        row["missing"] = sorted(_missing_names(last))
        row["detail"] = ("set-up scripts were run but names are still missing: " + " | ".join(last[:2]))[:400]
    except Exception as exc:
        row["simulate"] = "no: internal error"
        row["detail"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        os.chdir(here)
    return row


def survey(
    root: str, time_limit: float = 30.0, sources: str | None = None, keep_results: bool = False, only: str = ""
) -> list[dict]:
    files = []
    if os.path.isfile(root):
        files = [root]
    else:
        for folder, _dirs, names in os.walk(root):
            files += [os.path.join(folder, n) for n in names if n.lower().endswith(".slx")]
    rows = []
    for f in sorted(files):
        rel = os.path.relpath(f, root) if os.path.isdir(root) else os.path.basename(f)
        if only and only not in rel.replace(os.sep, "/"):
            continue
        src = None
        if sources:
            first = rel.replace(os.sep, "/").split("/")[0]
            cand = os.path.join(sources, first)
            src = (
                cand
                if os.path.isdir(cand) and "/" in rel.replace(os.sep, "/")
                else (sources if os.path.isdir(sources) else None)
            )
        row = check_file(f, time_limit, src)
        row["file"] = rel
        if not keep_results:
            row.pop("result", None)
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    imp = Counter(r["import"] for r in rows)
    simc = Counter(r["simulate"] for r in rows)
    missing: Counter = Counter()
    files_with: Counter = Counter()
    names: Counter = Counter()
    for r in rows:
        labels = set()
        for k, v in r["unsupported"].items():
            label = "Library link (toolbox or user library)" if k.startswith("Library link") else k
            missing[label] += v
            labels.add(label)
        files_with.update(labels)
        names.update(r.get("missing", []))
    return {
        "files": len(rows),
        "import": dict(imp),
        "simulate": dict(simc),
        "unsupported_blocks": dict(missing.most_common()),
        "unsupported_files": dict(files_with.most_common()),
        "missing_names": dict(names.most_common()),
        "with_setup": sum(1 for r in rows if r["simulate"].startswith("yes") and r.get("setup")),
    }


def _same_value(a: str, b: str) -> bool:
    a, b = a.strip(), b.strip()
    if a.lower() == b.lower():
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        pass
    squeeze = lambda s: re.sub(r"[\s\[\]]", "", s).lower()  # noqa: E731
    return squeeze(a) == squeeze(b)


def defaults_table(root: str) -> list[dict]:
    """What the files say about parameter defaults, next to what the importer assumes.

    A file may list the default parameters of its block types. For every
    parameter that the importer carries over to a mylab block, a row gives
    the value most files list as the default (``listed``, with the number of
    files), the value most often set on a block (``common``), and the value
    the importer uses when a file says nothing (``ours``). ``agree`` is False
    when files list a default and ours differs.
    """
    from .library import find_type
    from .slx import IMPORT_DEFAULTS, PARAM_MAP, TYPE_MAP, _Archive, _params

    listed: dict[tuple[str, str], Counter] = {}
    explicit: dict[tuple[str, str], Counter] = {}
    files = (
        [root]
        if os.path.isfile(root)
        else sorted(os.path.join(d, f) for d, _dirs, fs in os.walk(root) for f in fs if f.lower().endswith(".slx"))
    )
    for path in files:
        try:
            arc = _Archive(path)
            names = [n for n in arc.names() if n.startswith("simulink/") and n.endswith(".xml")]
            for name in names:
                if "blockdiagram" not in name and "/systems/" not in name:
                    continue
                tree = arc.xml(name)
                if tree is None:
                    continue
                skip = set()
                for d in tree.iter("BlockParameterDefaults"):
                    for b in d.findall("Block"):
                        skip.add(id(b))
                        for k, v in _params(b).items():
                            listed.setdefault((b.get("BlockType", ""), k), Counter())[v.strip()] += 1
                for b in tree.iter("Block"):
                    if id(b) in skip:
                        continue
                    for p in b.findall("P"):
                        explicit.setdefault((b.get("BlockType", ""), p.get("Name", "")), Counter())[
                            (p.text or "").strip()
                        ] += 1
        except Exception:  # unreadable file: the import report says so
            continue
    rows = []
    for (btype, k), counts in sorted(listed.items()):
        cls = find_type(TYPE_MAP.get(btype, btype))
        if cls is None:
            continue
        k2 = PARAM_MAP.get(cls.type, {}).get(k, k)
        spec = next((q for q in cls.params if q.name == k2), None)
        if spec is None:
            continue
        value, n = counts.most_common(1)[0]
        ours = IMPORT_DEFAULTS.get(btype, {}).get(k, spec.default)
        common = explicit.get((btype, k), Counter()).most_common(1)
        rows.append(
            {
                "block": btype,
                "parameter": k,
                "listed": value,
                "files": n,
                "of": sum(counts.values()),
                "common": common[0][0] if common else "",
                "ours": str(ours),
                "agree": _same_value(value, str(ours)),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--sources", help="folder with the .m files of the same repositories (set-up scripts, functions)")
    ap.add_argument("--only", default="", help="only files whose path contains this text")
    ap.add_argument("--json", help="also write the full result to this file")
    ap.add_argument("--time-limit", type=float, default=30.0, help="seconds of computing allowed per model")
    ap.add_argument("--defaults", action="store_true", help="print the table of parameter defaults instead")
    args = ap.parse_args(argv)
    if args.defaults:
        for r in defaults_table(args.path):
            mark = "  " if r["agree"] else "!!"
            print(
                f"{mark} {r['block']}.{r['parameter']}: files list {r['listed']!r} ({r['files']} of {r['of']}), "
                f"most often set to {r['common']!r}, mylab assumes {r['ours']!r}"
            )
        return 0
    rows = survey(args.path, args.time_limit, args.sources, only=args.only)
    for r in rows:
        miss = ", ".join(f"{k} ({v})" for k, v in sorted(r["unsupported"].items(), key=lambda kv: -kv[1])[:6])
        more = len(r["unsupported"]) - 6
        print(f"{r['file']}\n    import: {r['import']}  blocks: {r['blocks']}  simulate: {r['simulate']}")
        if miss:
            print(f"    unsupported: {miss}{f' and {more} more kinds' if more > 0 else ''}")
        if r.get("setup"):
            print(f"    set-up scripts run: {', '.join(r['setup'])}")
        if r["detail"]:
            print(f"    {r['detail']}")
    s = summarize(rows)
    print(f"\n{s['files']} files. Import: " + ", ".join(f"{k} {v}" for k, v in sorted(s["import"].items())))
    print("Simulate: " + ", ".join(f"{k} {v}" for k, v in sorted(s["simulate"].items())))
    if s["with_setup"]:
        print(f"Of the models that ran, {s['with_setup']} needed set-up scripts found in --sources.")
    top = list(s["unsupported_files"].items())[:15]
    if top:
        print("Unsupported blocks by number of files: " + ", ".join(f"{k} {v}" for k, v in top))
    if s["missing_names"]:
        print("Names still missing: " + ", ".join(f"{k} ({v})" for k, v in list(s["missing_names"].items())[:25]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": s, "files": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
