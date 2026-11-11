"""Run the MATLAB Tasks benchmark.

Each task folder holds task.json, prompt.md, solution/<function>.m, test.m
and optionally data/ (files the test reads). A task passes when test.m runs
to the end without an error, with the solution on the path.

Examples:
    python tools/run_tasks.py                          # reference solutions on mylab
    python tools/run_tasks.py --engine octave          # ...on GNU Octave
    python tools/run_tasks.py --solutions my_answers/  # grade candidate solutions
    python tools/run_tasks.py --verify --write-status  # check both engines, record status in task.json

Candidate solutions: for task "basics/sum_of_multiples" put the function file at
my_answers/basics/sum_of_multiples/sum_of_multiples.m (or any .m files in that
folder; helper functions are allowed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
REQUIRED = {"id", "title", "category", "difficulty", "function", "signature", "tags"}


def discover(filter_text: str | None = None) -> list[Path]:
    dirs = sorted(p.parent for p in TASKS.rglob("task.json"))
    if filter_text:
        dirs = [d for d in dirs if filter_text in str(d.relative_to(TASKS))]
    return dirs


def load_meta(task_dir: Path) -> dict:
    return json.loads((task_dir / "task.json").read_text())


def validate(task_dir: Path) -> list[str]:
    problems = []
    try:
        meta = load_meta(task_dir)
    except json.JSONDecodeError as e:
        return [f"task.json is not valid JSON: {e}"]
    missing = REQUIRED - meta.keys()
    if missing:
        problems.append(f"task.json missing {sorted(missing)}")
    rel = str(task_dir.relative_to(TASKS)).replace(os.sep, "/")
    if meta.get("id") != rel:
        problems.append(f"id {meta.get('id')!r} does not match folder {rel!r}")
    if meta.get("difficulty") not in ("easy", "medium", "hard"):
        problems.append("difficulty must be easy, medium or hard")
    for f in ("prompt.md", "test.m"):
        if not (task_dir / f).is_file():
            problems.append(f"missing {f}")
    fn = meta.get("function", "")
    if not (task_dir / "solution" / f"{fn}.m").is_file():
        problems.append(f"missing solution/{fn}.m")
    return problems


def engine_command(engine: str, workdir: Path) -> list[str]:
    if engine == "mylab":
        return [sys.executable, "-m", "mylab", "--no-plots", "test.m"]
    if engine == "octave":
        return ["octave-cli", "--norc", "--quiet", "--no-window-system", "--eval",
                "try, run('test.m'); catch err, disp(['Error: ' err.message]); exit(1); end"]
    raise SystemExit(f"unknown engine {engine}")


def run_one(task_dir: Path, engine: str, solutions: Path | None, timeout: float) -> dict:
    meta = load_meta(task_dir)
    result = {"id": meta["id"], "engine": engine, "status": "fail", "seconds": 0.0, "output": ""}
    with tempfile.TemporaryDirectory(prefix="mltask_") as td:
        work = Path(td)
        src = (solutions / meta["id"]) if solutions else (task_dir / "solution")
        if solutions and not src.is_dir():
            single = solutions / f"{meta['id']}.m"
            if single.is_file():
                shutil.copy(single, work / f"{meta['function']}.m")
            else:
                result["status"] = "missing"
                return result
        else:
            for f in src.glob("*.m"):
                shutil.copy(f, work / f.name)
        if (task_dir / "data").is_dir():
            for f in (task_dir / "data").iterdir():
                if f.is_file():
                    shutil.copy(f, work / f.name)
        shutil.copy(task_dir / "test.m", work / "test.m")
        env = dict(os.environ, MPLBACKEND="Agg")
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(engine_command(engine, work), cwd=work, capture_output=True, text=True,
                                  timeout=timeout, env=env)
            out = (proc.stdout + proc.stderr).strip()
            failed = proc.returncode != 0 or "\nError" in "\n" + out
            result["status"] = "fail" if failed else "pass"
            result["output"] = out[-2000:]
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
        result["seconds"] = round(time.perf_counter() - t0, 3)
    return result


def summarize(results: list[dict]) -> str:
    by_cat: dict[str, list[int]] = {}
    for r in results:
        cat = r["id"].split("/")[0]
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][1] += 1
        by_cat[cat][0] += r["status"] == "pass"
    total = sum(v[1] for v in by_cat.values())
    passed = sum(v[0] for v in by_cat.values())
    lines = [f"{'category':16s} passed", "-" * 30]
    lines += [f"{c:16s} {p}/{n}" for c, (p, n) in sorted(by_cat.items())]
    lines += ["-" * 30, f"{'total':16s} {passed}/{total} ({100 * passed / max(total, 1):.1f}%)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mylab", "octave"], default="mylab")
    ap.add_argument("--solutions", type=Path, help="folder of candidate solutions to grade")
    ap.add_argument("--filter", help="only tasks whose id contains this text")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--verify", action="store_true", help="run reference solutions on mylab and octave")
    ap.add_argument("--write-status", action="store_true", help="with --verify: record results in task.json")
    ap.add_argument("--json", type=Path, help="write detailed results to this file")
    ap.add_argument("--show-failures", action="store_true")
    args = ap.parse_args(argv)

    tasks = discover(args.filter)
    bad = {str(t.relative_to(TASKS)): validate(t) for t in tasks}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        for k, v in bad.items():
            print(f"INVALID {k}: {'; '.join(v)}")
        return 2

    engines = ["mylab", "octave"] if args.verify else [args.engine]
    all_results: list[dict] = []
    for engine in engines:
        if engine == "octave" and shutil.which("octave-cli") is None:
            print("octave-cli not found; skipping Octave")
            continue
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda t: run_one(t, engine, args.solutions, args.timeout), tasks))
        all_results += results
        print(f"\n== {engine} ==")
        print(summarize(results))
        if args.show_failures:
            for r in results:
                if r["status"] != "pass":
                    print(f"\n--- {r['id']} [{r['status']}]\n{r['output'][-600:]}")
    if args.write_status and args.verify:
        status: dict[str, dict] = {}
        for r in all_results:
            status.setdefault(r["id"], {})[r["engine"]] = r["status"] == "pass"
        for t in tasks:
            meta = load_meta(t)
            meta["verified"] = status.get(meta["id"], {})
            (t / "task.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    if args.json:
        args.json.write_text(json.dumps(all_results, indent=2))
    return 0 if all(r["status"] == "pass" for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
