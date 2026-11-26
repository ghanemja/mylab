"""Run real code from the corpus: the tests and examples of selected repositories, unchanged.

    python tools/run_corpus.py fetch                 # check out the selected repositories (once)
    python tools/run_corpus.py run                   # run them; fails if a run that passed before fails
    python tools/run_corpus.py run --only PRML/PRMLT --verbose
    python tools/run_corpus.py run --update          # record the current results as expected
    python tools/run_corpus.py survey REPO           # try every test/demo-looking script of a repository

The selection is regression.json next to this folder: for each repository, the scripts to run
(and, for some, the output they must print). expected.json records which runs passed. Nothing
from the repositories is stored in mylab: fetch checks them out at the commit pinned in
manifest.jsonl into ~/.cache/mylab-corpus-runs (or $MYLAB_CORPUS_RUNS).

Each script runs in a fresh mylab process inside its own folder, with the repository's folders on
the path, figures drawn off screen, and a time limit. Only mylab executes the code: nothing in the
repositories is run by Python, MATLAB or a shell.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELECTION = ROOT / "regression.json"
EXPECTED = ROOT / "regression_expected.json"
CACHE = Path(os.environ.get("MYLAB_CORPUS_RUNS", Path.home() / ".cache" / "mylab-corpus-runs"))
SKIP_DIRS = {".git", "mex", "private"}

# The program each run executes in a child process (mylab only; the script is MATLAB code).
CHILD = r"""
import json, os, sys, time, traceback
os.environ.setdefault("MPLBACKEND", "Agg")
spec = json.loads(sys.argv[1])
from mylab import Interpreter, MatlabError
out = []
interp = Interpreter(out=out.append, path=list(spec["path"]))
interp.input_func = lambda prompt="": ""
os.chdir(spec["cwd"])
t0 = time.perf_counter()
status = "ok"
try:
    if spec.get("code"):
        interp.execute(spec["code"])
    else:
        interp.run_file(spec["file"])
except MatlabError as e:
    status = "error: " + str(e)
except RecursionError:
    status = "crash: RecursionError"
except Exception as e:
    status = "crash: " + type(e).__name__ + ": " + str(e) + " | " + traceback.format_exc(limit=-3).replace("\n", " ")
text = "".join(out)
result = {"status": status, "seconds": time.perf_counter() - t0, "output": text[-20000:]}
print("\n@@MYLAB-RESULT@@" + json.dumps(result))
"""


def load_rows() -> dict[str, dict]:
    return {r["repo"]: r for r in map(json.loads, (ROOT / "manifest.jsonl").read_text().splitlines())}


def repo_dir(repo: str) -> Path:
    return CACHE / repo.replace("/", "__")


def git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=900)


def fetch_one(row: dict) -> str:
    dest = repo_dir(row["repo"])
    marker = dest / ".mylab-corpus-commit"
    if marker.exists() and marker.read_text().strip() == row["commit"]:
        return "cached"
    try:
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], check=True)
        dest.mkdir(parents=True)
        git(["init", "-q"], dest)
        git(["remote", "add", "origin", row["clone_url"]], dest)
        # every file up to 20 MB: sources plus the data files examples load
        git(["fetch", "-q", "--depth", "1", "--filter=blob:limit=20m", "origin", row["commit"]], dest)
        git(["checkout", "-q", "FETCH_HEAD"], dest)
        subprocess.run(["rm", "-rf", str(dest / ".git")], check=True)  # the files are all that is needed
        marker.write_text(row["commit"] + "\n")
        return "ok"
    except subprocess.CalledProcessError as e:
        return f"failed: {(e.stderr or '')[:300].strip()}"
    except subprocess.TimeoutExpired:
        return "failed: timeout"


def search_path(root: Path, add: list[str] | None) -> list[str]:
    """The repository's folders (all of them, or the listed ones), as genpath would give them."""
    if add is not None:
        return [str(root / p) for p in add]
    dirs = []
    for d, sub, _files in os.walk(root):
        sub[:] = sorted(s for s in sub if s not in SKIP_DIRS and not s.startswith((".", "+", "@")))
        dirs.append(d)
    return dirs


def run_one(repo: str, run: dict, timeout: float) -> dict:
    root = repo_dir(repo)
    spec = {"path": search_path(root, run.get("path"))}
    if "file" in run:
        file = root / run["file"]
        spec["file"] = str(file)
        spec["cwd"] = str(root / run.get("cwd", file.parent.relative_to(root)))
    else:
        spec["code"] = run["code"]
        spec["cwd"] = str(root / run.get("cwd", "."))
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-c", CHILD, json.dumps(spec)], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL,
                           timeout=run.get("timeout", timeout), env={**os.environ, "MPLBACKEND": "Agg"})
        marker = p.stdout.rfind("@@MYLAB-RESULT@@")
        if marker < 0:
            res = {"status": f"crash: exit {p.returncode}: {p.stderr[-400:]}", "output": p.stdout[-2000:]}
        else:
            res = json.loads(p.stdout[marker + len("@@MYLAB-RESULT@@"):])
    except subprocess.TimeoutExpired:
        res = {"status": f"timeout after {run.get('timeout', timeout):.0f} s", "output": ""}
    res["seconds"] = round(time.perf_counter() - t0, 1)
    if res["status"] == "ok":
        for pattern in run.get("expect_output", []):
            if not re.search(pattern, res["output"], re.M):
                res["status"] = f"wrong output: no match for {pattern!r}"
                break
    return res


def run_name(run: dict) -> str:
    return run.get("name") or run.get("file") or run["code"][:60]


def cmd_fetch(args) -> int:
    rows = load_rows()
    sel = json.loads(SELECTION.read_text())
    repos = [e["repo"] for e in sel["repositories"] if not args.only or e["repo"] in args.only]
    CACHE.mkdir(parents=True, exist_ok=True)
    bad = 0
    with ThreadPoolExecutor(6) as pool:
        for repo, status in zip(repos, pool.map(lambda r: fetch_one(rows[r]), repos)):
            print(f"{repo}: {status}")
            bad += status.startswith("failed")
    print(f"checked out in {CACHE}")
    return 1 if bad else 0


def cmd_run(args) -> int:
    sel = json.loads(SELECTION.read_text())
    expected = json.loads(EXPECTED.read_text()) if EXPECTED.exists() else {}
    jobs = []
    for entry in sel["repositories"]:
        if args.only and entry["repo"] not in args.only:
            continue
        if not (repo_dir(entry["repo"]) / ".mylab-corpus-commit").exists():
            print(f"{entry['repo']}: not fetched (run: python tools/run_corpus.py fetch)")
            return 2
        for run in entry["runs"]:
            jobs.append((entry["repo"], run))
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda j: run_one(j[0], j[1], args.timeout), jobs))
    passed, lost, gained, per_repo = 0, [], [], {}
    for (repo, run), res in zip(jobs, results):
        key = f"{repo} :: {run_name(run)}"
        ok = res["status"] == "ok"
        passed += ok
        per_repo.setdefault(repo, [0, 0])
        per_repo[repo][0] += ok
        per_repo[repo][1] += 1
        if expected.get(key) and not ok:
            lost.append((key, res["status"]))
        if ok and not expected.get(key):
            gained.append(key)
        if args.verbose or (not ok and expected.get(key)):
            detail = "" if ok else f"\n      {res['status'][:400]}"
            print(f"{'PASS' if ok else 'FAIL'}  {key}  ({res['seconds']} s){detail}")
    print(f"\n{passed} of {len(jobs)} runs pass, in {len(per_repo)} repositories; "
          f"{sum(1 for v in per_repo.values() if v[0] == v[1])} repositories pass completely.")
    if args.update:
        for (repo, run), res in zip(jobs, results):
            expected[f"{repo} :: {run_name(run)}"] = res["status"] == "ok"
        EXPECTED.write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n")
        print(f"recorded in {EXPECTED.name}")
        return 0
    if gained:
        print(f"{len(gained)} run(s) pass that were not expected to; record them with --update.")
    if lost:
        print(f"{len(lost)} run(s) that passed before fail now:")
        for key, status in lost:
            print(f"    {key} :: {status[:300]}")
        return 1
    return 0


def cmd_survey(args) -> int:
    rows = load_rows()
    repo = args.repo
    status = fetch_one(rows[repo])
    if status.startswith("failed"):
        print(status)
        return 1
    root = repo_dir(repo)
    pattern = re.compile(r"(?i)(^|/)(tests?|demos?|examples?|tutorials?|run)[^/]*\.m$|(test|demo|example)[^/]*\.m$"
                         r"|(^|/)(tests?|demos?|examples?|tutorials?|scripts?)/")
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*.m") if pattern.search(str(p.relative_to(root))))
    if args.all:
        files = sorted(str(p.relative_to(root)) for p in root.rglob("*.m"))
    runs = []
    for f in files:
        try:
            head = (root / f).read_text(encoding="utf-8", errors="replace").lstrip()
        except OSError:
            continue
        code = "\n".join(line for line in head.splitlines() if line.strip() and not line.lstrip().startswith("%"))
        if code.startswith("function") and not args.functions:
            continue  # a function file: needs arguments (use --functions to try calling it with none)
        runs.append({"file": f})
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda r: run_one(repo, r, args.timeout), runs))
    for run, res in zip(runs, results):
        print(f"{'PASS' if res['status'] == 'ok' else 'FAIL'}  {run['file']}  ({res['seconds']} s)"
              + ("" if res["status"] == "ok" else f"\n      {res['status'][:300]}"))
    print(f"{sum(r['status'] == 'ok' for r in results)} of {len(runs)} scripts ran without an error.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--only", nargs="*")
    r = sub.add_parser("run")
    r.add_argument("--only", nargs="*")
    r.add_argument("--jobs", type=int, default=3)
    r.add_argument("--timeout", type=float, default=180)
    r.add_argument("--update", action="store_true")
    r.add_argument("--verbose", action="store_true")
    s = sub.add_parser("survey")
    s.add_argument("repo")
    s.add_argument("--jobs", type=int, default=3)
    s.add_argument("--timeout", type=float, default=60)
    s.add_argument("--all", action="store_true")
    s.add_argument("--functions", action="store_true")
    a = ap.parse_args()
    return {"fetch": cmd_fetch, "run": cmd_run, "survey": cmd_survey}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
