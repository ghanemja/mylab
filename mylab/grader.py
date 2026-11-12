"""Autograder for teachers: grade MATLAB-language assignments with mylab.

An assignment is a folder (see docs/AUTOGRADER.md)::

    my-assignment/
      assignment.json
      problems/<id>/
        problem.json  prompt.md  starter/  solution/  tests/  data/

Every test is a script of ``assert`` calls. It passes when it runs to the end
without an error, and each test is worth points. Every test runs in its own
``python -m mylab`` process with a time limit, so a crash, an infinite loop or
``exit`` in a student's code only fails that one test.

Command line::

    mylab-grade init NAME
    mylab-grade check ASSIGNMENT
    mylab-grade run ASSIGNMENT --submission DIR
    mylab-grade batch ASSIGNMENT --submissions DIR --csv grades.csv
    mylab-grade export ASSIGNMENT --out assignment.student.json
    mylab-grade gradescope ASSIGNMENT --out autograder.zip

Python::

    from mylab.grader import grade_submission
    result = grade_submission("my-assignment", "submissions/alice")
    result.score, result.max_score, result.to_dict()
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "mylab-assignment-1"
DEFAULT_TIMEOUT = 10.0
DRIVER = "mylab_grader_driver__.m"
MAX_OUTPUT_CHARS = 1500
MAX_OUTPUT_LINES = 30

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TEST_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*\.m$")
_FUNCTION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_META_RE = re.compile(r"^%+\s*(name|points|hidden)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_ERROR_IN_RE = re.compile(r"^Error in (.+) \(line (\d+)\)\s*$")
# Files that are never copied from a submission into the test folder.
_SKIP_SUFFIXES = {".py", ".pyc", ".pyd", ".pyo", ".pth", ".so", ".dll", ".dylib", ".exe", ".zip"}
# A submitted file with one of these names would replace the function the tests rely on.
_PROTECTED = {"assert", "isequal", "isequaln", "run", "error"}
_MAX_COPY_BYTES = 20 * 1024 * 1024


class AssignmentError(Exception):
    """The assignment folder is missing something or is not valid."""

    def __init__(self, problems: list[str] | str):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("\n".join(self.problems))


# ---------------------------------------------------------------------------------------
# The assignment on disk
# ---------------------------------------------------------------------------------------

@dataclass
class Test:
    __test__ = False   # not a pytest test class

    file: str          # file name inside tests/, e.g. "01_basic.m"
    name: str
    points: float
    hidden: bool
    code: str


@dataclass
class Problem:
    id: str
    dir: Path
    title: str
    points: float
    function: str
    timeout: float
    prompt: str
    tests: list[Test]

    @property
    def data_dir(self) -> Path:
        return self.dir / "data"


@dataclass
class Assignment:
    dir: Path
    title: str
    description: str
    problems: list[Problem]

    @property
    def points(self) -> float:
        return _round(sum(p.points for p in self.problems))


def _round(x: float) -> float:
    return round(float(x) + 0.0, 2)


def _num(x: float) -> str:
    """Points as short text: 3, 2.5, 3.33."""
    x = _round(x)
    return str(int(x)) if x == int(x) else f"{x:.2f}".rstrip("0").rstrip(".")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")


def _default_test_name(file: str) -> str:
    stem = re.sub(r"^\d+[_-]*", "", Path(file).stem) or Path(file).stem
    text = stem.replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:]


def parse_test_metadata(code: str) -> dict:
    """Read ``% name:``, ``% points:`` and ``% hidden:`` from the first comment lines."""
    meta: dict = {}
    for line in code.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith("%"):
            break
        m = _META_RE.match(s)
        if not m:
            continue
        key, value = m.group(1).lower(), m.group(2)
        if key == "name" and value:
            meta["name"] = value
        elif key == "points":
            try:
                meta["points"] = float(value)
            except ValueError:
                meta["points_error"] = value
        elif key == "hidden":
            meta["hidden"] = value.lower() in ("true", "yes", "1", "on")
    return meta


def _load_json(path: Path, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"{path}: file is missing")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        errors.append(f"{path}: not valid JSON ({e})")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: must contain a JSON object {{...}}")
        return None
    return data


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _load_problem(pdir: Path, pid: str, errors: list[str]) -> Problem | None:
    n_before = len(errors)
    where = f"problems/{pid}"
    if not pdir.is_dir():
        errors.append(f"{where}: folder is missing (it is listed in assignment.json)")
        return None
    meta = _load_json(pdir / "problem.json", errors)
    if meta is None:
        return None
    function = meta.get("function")
    if not isinstance(function, str) or not _FUNCTION_RE.match(function):
        errors.append(f'{where}/problem.json: "function" must be the name of the function students write')
        function = ""
    title = meta.get("title") if isinstance(meta.get("title"), str) and meta.get("title") else pid
    timeout = meta.get("timeout", DEFAULT_TIMEOUT)
    if not _is_number(timeout) or timeout <= 0:
        errors.append(f'{where}/problem.json: "timeout" must be a positive number of seconds')
        timeout = DEFAULT_TIMEOUT
    points = meta.get("points")
    if points is not None and (not _is_number(points) or points < 0):
        errors.append(f'{where}/problem.json: "points" must be a number, 0 or more')
        points = None
    prompt = ""
    if (pdir / "prompt.md").is_file():
        prompt = _read_text(pdir / "prompt.md")
    else:
        errors.append(f"{where}/prompt.md: file is missing (the problem statement)")

    tests_dir = pdir / "tests"
    files = sorted(f.name for f in tests_dir.glob("*.m")) if tests_dir.is_dir() else []
    if not files:
        errors.append(f"{where}/tests: needs at least one test script (for example 01_basic.m)")
    raw = []
    for name in files:
        if not _TEST_FILE_RE.match(name):
            errors.append(f"{where}/tests/{name}: use only letters, digits, '_' and '-' in test file names")
            continue
        if Path(name).stem == function:
            errors.append(f"{where}/tests/{name}: a test cannot have the same name as the function it tests")
            continue
        code = _read_text(tests_dir / name)
        m = parse_test_metadata(code)
        if "points_error" in m:
            errors.append(f"{where}/tests/{name}: '% points: {m['points_error']}' is not a number")
        if m.get("points", 0) < 0:
            errors.append(f"{where}/tests/{name}: points cannot be negative")
        raw.append((name, code, m))

    explicit = [m["points"] for _, _, m in raw if "points" in m]
    n_auto = len(raw) - len(explicit)
    if points is None:
        if n_auto or not raw:
            errors.append(f'{where}/problem.json: "points" is missing')
            points = 0.0
        else:
            points = sum(explicit)
    share = last_share = 0.0
    if n_auto:
        rest = points - sum(explicit)
        if rest < -1e-9:
            errors.append(f"{where}: the tests' '% points:' add up to {_num(sum(explicit))}, "
                          f"more than the problem's {_num(points)} points")
            rest = 0.0
        share = _round(rest / n_auto)
        last_share = _round(rest - share * (n_auto - 1))
    elif raw and abs(sum(explicit) - points) > 1e-6:
        errors.append(f"{where}: the tests' '% points:' add up to {_num(sum(explicit))}, "
                      f"but problem.json says {_num(points)} points")
    tests = []
    seen_auto = 0
    for name, code, m in raw:
        if "points" in m:
            pts = _round(m["points"])
        else:
            seen_auto += 1
            pts = last_share if seen_auto == n_auto else share
        tests.append(Test(file=name, name=m.get("name") or _default_test_name(name), points=pts,
                          hidden=bool(m.get("hidden", False)), code=code))
    if len(errors) > n_before:
        return None
    return Problem(id=pid, dir=pdir, title=title, points=_round(points), function=function,
                   timeout=float(timeout), prompt=prompt, tests=tests)


def load_assignment(assignment_dir: str | os.PathLike) -> Assignment:
    """Read and validate an assignment folder. Raises AssignmentError listing every problem found."""
    root = Path(assignment_dir)
    if not root.is_dir():
        raise AssignmentError(f"{root}: not a folder")
    errors: list[str] = []
    meta = _load_json(root / "assignment.json", errors)
    if meta is None:
        raise AssignmentError(errors)
    if meta.get("format") != FORMAT:
        errors.append(f'assignment.json: "format" must be "{FORMAT}"')
    title = meta.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append('assignment.json: "title" is missing')
        title = root.name
    description = meta.get("description", "")
    if not isinstance(description, str):
        errors.append('assignment.json: "description" must be text')
        description = ""
    ids = meta.get("problems")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        errors.append('assignment.json: "problems" must be a list of problem folder names, for example ["p1_area"]')
        ids = []
    problems = []
    for pid in ids:
        if not _ID_RE.match(pid):
            errors.append(f"assignment.json: problem id {pid!r} may only use letters, digits, '_', '-' and '.'")
            continue
        if ids.count(pid) > 1:
            if pid not in [p.id for p in problems]:
                errors.append(f"assignment.json: problem {pid!r} is listed more than once")
        p = _load_problem(root / "problems" / pid, pid, errors)
        if p is not None and pid not in [q.id for q in problems]:
            problems.append(p)
    if errors:
        raise AssignmentError(errors)
    return Assignment(dir=root, title=title.strip(), description=description, problems=problems)


# ---------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------

@dataclass
class TestResult:
    __test__ = False   # not a pytest test class

    name: str
    file: str
    hidden: bool
    max_points: float
    points: float = 0.0
    status: str = "failed"            # passed | failed | timeout | missing
    message: str = ""                 # why it failed (empty when passed)
    error_file: str | None = None     # file the error happened in
    error_line: int | None = None
    source_line: str | None = None    # the text of that line, when it may be shown
    output: str = ""                  # what the code printed (shortened)

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict:
        return {
            "name": self.name, "file": self.file, "hidden": self.hidden,
            "points": self.points, "max_points": self.max_points, "status": self.status,
            "message": self.message, "error_file": self.error_file, "error_line": self.error_line,
            "source_line": self.source_line, "output": self.output,
        }


@dataclass
class ProblemResult:
    id: str
    title: str
    function: str
    tests: list[TestResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return _round(sum(t.points for t in self.tests))

    @property
    def max_score(self) -> float:
        return _round(sum(t.max_points for t in self.tests))

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "function": self.function, "score": self.score,
                "max_score": self.max_score, "notes": list(self.notes),
                "tests": [t.to_dict() for t in self.tests]}


@dataclass
class Result:
    assignment: str
    submission: str
    problems: list[ProblemResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return _round(sum(p.score for p in self.problems))

    @property
    def max_score(self) -> float:
        return _round(sum(p.max_score for p in self.problems))

    @property
    def percent(self) -> float:
        return round(100.0 * self.score / self.max_score, 1) if self.max_score else 0.0

    def to_dict(self) -> dict:
        return {"assignment": self.assignment, "submission": self.submission, "score": self.score,
                "max_score": self.max_score, "percent": self.percent,
                "problems": [p.to_dict() for p in self.problems]}

    def to_gradescope(self) -> dict:
        return gradescope_results(self)

    def report(self, markdown: bool = False) -> str:
        return format_markdown(self) if markdown else format_report(self)


# ---------------------------------------------------------------------------------------
# Running one test in its own process
# ---------------------------------------------------------------------------------------

def _subprocess_env() -> dict:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Make sure the child finds this same copy of mylab, installed or not.
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + old if old else "")
    return env


def _run_process(cmd: list[str], cwd: Path, timeout: float, log_dir: Path) -> tuple[int | None, str, str]:
    """Run cmd; returns (return code or None on timeout, stdout, stderr). Output goes to
    files (outside the folder the code runs in) so that code printing in an endless loop
    cannot fill the grader's memory."""
    out_path, err_path = log_dir / "stdout.txt", log_dir / "stderr.txt"
    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:  # pragma: no cover - Windows
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
        proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe,
                                env=_subprocess_env(), **kwargs)
        try:
            rc: int | None = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = None
            _kill(proc)
    return rc, _read_limited(out_path), _read_limited(err_path)


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def _read_limited(path: Path, limit: int = 200_000) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > limit:
                fh.seek(size - limit)
            data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def _shorten_output(text: str) -> str:
    text = text.strip("\n")
    if not text.strip():
        return ""
    lines = text.split("\n")
    cut = len(lines) > MAX_OUTPUT_LINES
    lines = lines[:MAX_OUTPUT_LINES]
    text = "\n".join(lines)
    if len(text) > MAX_OUTPUT_CHARS:
        text, cut = text[:MAX_OUTPUT_CHARS], True
    return text + ("\n... (output shortened)" if cut else "")


def _parse_error(stderr: str) -> tuple[str, str | None, int | None]:
    """Pick the mylab error message, file and line out of what the process wrote."""
    lines = stderr.strip("\n").split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.startswith("Error: "):
            start = i
    if start is None:
        tail = [ln for ln in lines if ln.strip()]
        if tail and any(ln.startswith("Traceback") for ln in tail):
            return f"mylab stopped with an internal error: {tail[-1].strip()}", None, None
        return (tail[-1].strip() if tail else "The test stopped with an error."), None, None
    body = lines[start:]
    file, line_no = None, None
    m = _ERROR_IN_RE.match(body[-1]) if len(body) > 1 else None
    if m:
        file, line_no = m.group(1), int(m.group(2))
        body = body[:-1]
    message = "\n".join(body)[len("Error: "):].strip()
    return message, file, line_no


_SYNTAX_CHECK = r"""
import sys
from mylab.errors import MatlabError
from mylab.parser import parse
for path in sys.argv[1:]:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            parse(fh.read())
    except MatlabError as e:
        print(path)
        print(e.message.replace("\n", " "))
    except Exception:
        pass
"""


def _syntax_errors(files: dict[str, Path], timeout: float) -> dict[str, str]:
    """Which .m files do not parse? Runs in a separate process; nothing is executed."""
    paths = [str(p.resolve()) for n, p in sorted(files.items()) if n.lower().endswith(".m")]
    if not paths:
        return {}
    with tempfile.TemporaryDirectory(prefix="mylab_grade_") as td:
        cwd = Path(td) / "empty"
        cwd.mkdir()
        rc, out, _ = _run_process([sys.executable, "-c", _SYNTAX_CHECK, *paths], cwd, max(timeout, 20), Path(td))
    if rc is None:
        return {}
    found: dict[str, str] = {}
    lines = out.split("\n")
    for i in range(0, len(lines) - 1, 2):
        found[Path(lines[i]).name] = lines[i + 1]
    return found


def _source_line(text: str, line_no: int | None) -> str | None:
    if not line_no:
        return None
    lines = text.split("\n")
    if 1 <= line_no <= len(lines):
        s = lines[line_no - 1].strip()
        return s[:200] if s else None
    return None


class _ProblemRun:
    """Runs the tests of one problem against one set of files."""

    def __init__(self, problem: Problem, files: dict[str, Path]):
        self.problem = problem
        self.files = files
        self._syntax: dict[str, str] | None = None

    def syntax_errors(self) -> dict[str, str]:
        if self._syntax is None:
            self._syntax = _syntax_errors(self.files, self.problem.timeout)
        return self._syntax

    def run(self, test: Test) -> TestResult:
        p = self.problem
        res = TestResult(name=test.name, file=test.file, hidden=test.hidden, max_points=test.points)
        token = "MYLAB-GRADER-DONE-" + secrets.token_hex(12)
        with tempfile.TemporaryDirectory(prefix="mylab_grade_") as td:
            work = Path(td) / "work"
            work.mkdir()
            for name, src in sorted(self.files.items()):
                shutil.copyfile(src, work / name)
            if p.data_dir.is_dir():
                for f in sorted(p.data_dir.iterdir()):
                    if f.is_file():
                        shutil.copyfile(f, work / f.name)
            (work / test.file).write_text(test.code, encoding="utf-8")
            (work / DRIVER).write_text(f"run('{test.file}');\nfprintf('\\n%s\\n', '{token}');\n", encoding="utf-8")
            rc, out, err = _run_process([sys.executable, "-m", "mylab", "--no-plots", DRIVER], work, p.timeout,
                                        Path(td))
        finished = token in out
        res.output = _shorten_output("\n".join(ln for ln in out.split("\n") if token not in ln))
        if rc is None:
            res.status = "timeout"
            res.message = (f"Timed out: the code was stopped after {_num(p.timeout)} seconds. "
                           "It may contain a loop that never ends.")
            return res
        if rc == 0 and finished:
            res.status, res.points = "passed", test.points
            return res
        if rc == 0:
            res.message = "The code stopped before the test finished. Do not call exit or quit in your functions."
            return res
        if rc < 0:
            res.message = f"mylab stopped unexpectedly (signal {-rc}). The code may use too much memory."
            return res
        message, file, line_no = _parse_error(err)
        if file is None and message.startswith("Parse error"):
            bad = self.syntax_errors()
            name = f"{p.function}.m" if f"{p.function}.m" in bad else next(iter(sorted(bad)), None)
            if name:
                message, file = bad[name], name
                m = re.search(r"line (\d+)", message)
                line_no = int(m.group(1)) if m else None
            message = "Syntax error. " + message
        res.message = message or "The test stopped with an error."
        if file == DRIVER:
            file, line_no = None, None
        res.error_file, res.error_line = file, line_no
        if file == test.file:
            if test.hidden:
                res.error_line = None      # do not give away where in a hidden test
            else:
                res.source_line = _source_line(test.code, line_no)
        elif file in self.files:
            res.source_line = _source_line(_read_text(self.files[file]), line_no)
        return res


def _missing_results(problem: Problem, message: str) -> list[TestResult]:
    return [TestResult(name=t.name, file=t.file, hidden=t.hidden, max_points=t.points, status="missing",
                       message=message) for t in problem.tests]


def _usable_files(directory: Path) -> tuple[dict[str, Path], list[str]]:
    """Files of one folder that may be copied next to a test, plus notes about skipped ones."""
    files: dict[str, Path] = {}
    notes: list[str] = []
    if not directory.is_dir():
        return files, notes
    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.is_symlink() or f.name.startswith("."):
            continue
        if f.suffix.lower() in _SKIP_SUFFIXES or f.name == DRIVER:
            continue
        if f.suffix.lower() == ".m" and f.stem in _PROTECTED:
            notes.append(f"{f.name} was ignored: a submission may not replace the built-in function '{f.stem}'.")
            continue
        try:
            if f.stat().st_size > _MAX_COPY_BYTES:
                notes.append(f"{f.name} was ignored: the file is too large.")
                continue
        except OSError:
            continue
        files[f.name] = f
    return files, notes


def _map(fn, items: list, jobs: int) -> list:
    """Like map(), optionally on several threads (each test is its own process anyway)."""
    if jobs <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, items))


def grade_problem(problem: Problem, source_dir: str | os.PathLike, jobs: int = 1) -> ProblemResult:
    """Grade one problem using the .m files in ``source_dir`` exactly as they are."""
    files, notes = _usable_files(Path(source_dir))
    pr, runner = _prepare(problem, files, notes)
    if runner is not None:
        pr.tests = _map(runner.run, problem.tests, jobs)
    return pr


def _prepare(problem: Problem, files: dict[str, Path], notes: list[str]) -> tuple[ProblemResult, _ProblemRun | None]:
    """The (still empty) result for a problem and the runner for its tests; the runner is
    None when the function file is missing (the tests are then already marked as such)."""
    pr = ProblemResult(id=problem.id, title=problem.title, function=problem.function, notes=list(notes))
    wanted = f"{problem.function}.m"
    if wanted not in files:
        other = [n for n in files if n.lower() == wanted.lower()]
        if other:
            files = dict(files)
            files[wanted] = files.pop(other[0])
            pr.notes.append(f"Found {other[0]}; used it as {wanted}. File names must match the function name exactly.")
    if wanted not in files:
        pr.tests = _missing_results(problem, f"File {wanted} was not found in the submission.")
        return pr, None
    names = {t.file for t in problem.tests}
    files = {k: v for k, v in files.items() if k not in names}
    return pr, _ProblemRun(problem, files)


# ---------------------------------------------------------------------------------------
# Finding a student's files (flat folder, per-problem folders, wrapped in extra folders...)
# ---------------------------------------------------------------------------------------

_JUNK_DIRS = {"__MACOSX", "__pycache__", "node_modules"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _walk_dirs(root: Path, max_depth: int = 6) -> list[tuple[int, Path]]:
    found = [(0, root)]
    i = 0
    while i < len(found):
        depth, d = found[i]
        i += 1
        if depth >= max_depth:
            continue
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir() and not c.is_symlink())
        except OSError:
            continue
        for c in children:
            if c.name.startswith(".") or c.name in _JUNK_DIRS:
                continue
            found.append((depth + 1, c))
    return found


def _matches_problem(directory: Path, root: Path, problem: Problem) -> bool:
    keys = {_norm(problem.id), _norm(problem.function)}
    head = _norm(re.split(r"[_.-]", problem.id)[0])
    if len(head) >= 2:
        keys.add(head)
    try:
        parts = directory.relative_to(root).parts
    except ValueError:
        parts = ()
    return any(_norm(part) in keys for part in parts)


def find_problem_dir(submission: Path, problem: Problem) -> Path | None:
    """The folder of a submission that holds the files for a problem, or None.

    Looks for ``<function>.m`` anywhere in the submission. If several folders have it,
    one named after the problem wins, then the one closest to the top."""
    wanted = f"{problem.function}.m".lower()
    candidates = []
    for depth, d in _walk_dirs(submission):
        try:
            names = [f.name.lower() for f in d.iterdir() if f.is_file()]
        except OSError:
            continue
        if wanted in names:
            candidates.append((not _matches_problem(d, submission, problem), depth, str(d), d))
    if not candidates:
        return None
    return sorted(candidates)[0][3]


class _Submission:
    """A submission folder, or a .zip that is unpacked to a temporary folder."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._tmp: tempfile.TemporaryDirectory | None = None
        self.root = self.path
        self.name = self.path.stem if self.path.suffix.lower() == ".zip" else self.path.name

    def __enter__(self) -> _Submission:
        if self.path.is_file() and self.path.suffix.lower() == ".zip":
            self._tmp = tempfile.TemporaryDirectory(prefix="mylab_sub_")
            self.root = Path(self._tmp.name)
            _safe_extract(self.path, self.root)
        elif not self.path.is_dir():
            raise AssignmentError(f"{self.path}: submission not found (expected a folder or a .zip file)")
        return self

    def __exit__(self, *exc) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


def _safe_extract(zip_path: Path, dest: Path, max_bytes: int = 200 * 1024 * 1024, max_files: int = 5000) -> None:
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        return                      # an unreadable zip is graded as an empty submission
    with zf:
        total = 0
        for n, info in enumerate(zf.infolist()):
            if n >= max_files:
                break
            name = info.filename.replace("\\", "/")
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if info.is_dir() or not parts or ".." in parts or name.startswith("/") or ":" in parts[0]:
                continue
            total += info.file_size
            if total > max_bytes:
                break
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError):
                continue


def grade_submission(assignment_dir: str | os.PathLike | Assignment, submission_dir: str | os.PathLike,
                     name: str | None = None, jobs: int = 1) -> Result:
    """Grade one submission (a folder of .m files, or a .zip of one) against an assignment.

    ``jobs`` is how many tests may run at the same time; the result does not depend on it."""
    assignment = assignment_dir if isinstance(assignment_dir, Assignment) else load_assignment(assignment_dir)
    with _Submission(submission_dir) as sub:
        result = Result(assignment=assignment.title, submission=name or sub.name)
        todo: list[tuple[ProblemResult, _ProblemRun, Test]] = []
        for problem in assignment.problems:
            d = find_problem_dir(sub.root, problem)
            files, notes = _usable_files(d) if d is not None else ({}, [])
            pr, runner = _prepare(problem, files, notes)
            result.problems.append(pr)
            if runner is not None:
                todo += [(pr, runner, t) for t in problem.tests]
        for (pr, _, _), tr in zip(todo, _map(lambda item: item[1].run(item[2]), todo, jobs)):
            pr.tests.append(tr)
    return result


# ---------------------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------------------

_TAGS = {"passed": "PASS", "failed": "FAIL", "timeout": "TIME", "missing": "MISS"}


def _failure_lines(t: TestResult) -> list[str]:
    lines = []
    first, *rest = (t.message or "The test failed.").split("\n")
    lines.append(("" if t.status in ("timeout", "missing") else "Error: ") + first)
    lines += rest
    if t.error_file and t.error_line:
        where = f"in {t.error_file}, line {t.error_line}"
        lines.append(where + (f":  {t.source_line}" if t.source_line else ""))
    elif t.error_file:
        where = "in a hidden test" if t.hidden and t.error_file == t.file else f"in {t.error_file}"
        lines.append(where)
    return lines


def format_report(result: Result) -> str:
    """A plain-text report for one submission."""
    width = 72
    head = f"{result.assignment} - {result.submission}"
    out = [head, "=" * max(len(head), width)]
    for i, p in enumerate(result.problems, 1):
        left = f"Problem {i}: {p.title} ({p.function}.m)"
        right = f"{_num(p.score)} / {_num(p.max_score)}"
        out += ["", f"{left:<{width - len(right) - 1}} {right}"]
        for note in p.notes:
            out.append(f"  Note: {note}")
        for t in p.tests:
            left = f"  [{_TAGS[t.status]}] {t.name}" + (" (hidden test)" if t.hidden else "")
            right = f"{_num(t.points)} / {_num(t.max_points)}"
            out.append(f"{left:<{width - len(right) - 1}} {right}")
            if not t.passed:
                out += ["         " + ln for ln in _failure_lines(t)]
                if t.output and t.status != "missing":
                    out.append("         Printed before it stopped:")
                    out += ["           | " + ln for ln in t.output.split("\n")]
    out += ["", "-" * width]
    left, right = "Total", f"{_num(result.score)} / {_num(result.max_score)}  ({result.percent:g}%)"
    out.append(f"{left:<{width - len(right) - 1}} {right}")
    return "\n".join(ln.rstrip() for ln in out) + "\n"


def format_markdown(result: Result) -> str:
    """A Markdown report for one submission (to hand back to a student)."""
    out = [f"# {result.assignment}", "", f"**Submission:** {result.submission}  ",
           f"**Score:** {_num(result.score)} / {_num(result.max_score)} ({result.percent:g}%)", ""]
    for i, p in enumerate(result.problems, 1):
        out += [f"## Problem {i}: {p.title} - {_num(p.score)} / {_num(p.max_score)}", "",
                f"Function file: `{p.function}.m`", ""]
        for note in p.notes:
            out += [f"> Note: {note}", ""]
        for t in p.tests:
            mark = "x" if t.passed else " "
            status = {"passed": "passed", "failed": "failed", "timeout": "timed out", "missing": "file missing"}
            hidden = " (hidden test)" if t.hidden else ""
            out.append(f"- [{mark}] **{t.name}**{hidden}: {status[t.status]}, "
                       f"{_num(t.points)} / {_num(t.max_points)}")
            if not t.passed:
                block = _failure_lines(t)
                if t.output and t.status != "missing":
                    block += ["", "Printed before it stopped:"] + t.output.split("\n")
                out += ["", "  ```text"] + ["  " + ln for ln in block] + ["  ```", ""]
        out.append("")
    return "\n".join(ln.rstrip() for ln in out).rstrip("\n") + "\n"


def gradescope_results(result: Result, hidden_visibility: str = "after_published") -> dict:
    """The result in Gradescope's results.json format."""
    tests = []
    for p in result.problems:
        for t in p.tests:
            if t.passed:
                output = "Passed."
            else:
                lines = _failure_lines(t)
                if t.output and t.status != "missing":
                    lines += ["", "Printed before it stopped:"] + t.output.split("\n")
                output = "\n".join(lines)
            tests.append({
                "name": f"{p.title}: {t.name}",
                "score": t.points,
                "max_score": t.max_points,
                "output": output,
                "visibility": hidden_visibility if t.hidden else "visible",
                "status": "passed" if t.passed else "failed",
            })
    return {
        "score": result.score,
        "output": f"{result.assignment}: {_num(result.score)} / {_num(result.max_score)} points.",
        "tests": tests,
    }


# ---------------------------------------------------------------------------------------
# check, export, gradescope, batch, init
# ---------------------------------------------------------------------------------------

@dataclass
class CheckReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


def _is_text_file(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def check_assignment(assignment_dir: str | os.PathLike, jobs: int = 1) -> CheckReport:
    """Validate an assignment: the structure, the reference solutions (must earn every
    point) and the starter files (should not pass any test)."""
    rep = CheckReport(ok=True)
    try:
        a = load_assignment(assignment_dir)
    except AssignmentError as e:
        rep.ok = False
        rep.errors = list(e.problems)
        return rep
    rep.lines.append(f"{a.title}: {len(a.problems)} problem(s), {_num(a.points)} points")
    for p in a.problems:
        where = f"problems/{p.id}"
        n_hidden = sum(t.hidden for t in p.tests)
        rep.lines.append(f"  {p.id}: {_num(p.points)} points, {len(p.tests)} test(s), {n_hidden} hidden")
        if n_hidden == len(p.tests):
            rep.warnings.append(f"{where}: every test is hidden, so students cannot check their work in the app")
        if p.data_dir.is_dir():
            for f in sorted(p.data_dir.iterdir()):
                if f.is_file() and not _is_text_file(f):
                    rep.warnings.append(f"{where}/data/{f.name}: not a text file, so it is left out of the "
                                        "student bundle for the web app")
        if not (p.dir / "solution" / f"{p.function}.m").is_file():
            rep.errors.append(f"{where}/solution/{p.function}.m: the reference solution is missing")
        else:
            sol = grade_problem(p, p.dir / "solution", jobs)
            for t in sol.tests:
                if t.passed:
                    rep.lines.append(f"    solution passes  {t.file}  ({_num(t.max_points)} points)")
                else:
                    detail = "; ".join(_failure_lines(t))
                    rep.errors.append(f"{where}: the reference solution fails test {t.file} ({t.name}): {detail}")
        starter = p.dir / "starter"
        if not (starter / f"{p.function}.m").is_file():
            rep.warnings.append(f"{where}/starter/{p.function}.m: no starter file for students")
        else:
            st = grade_problem(p, starter, jobs)
            passing = [t for t in st.tests if t.passed]
            for t in passing:
                rep.warnings.append(f"{where}: the starter file already passes test {t.file} ({t.name}), "
                                    f"worth {_num(t.max_points)} points; add a check that an empty answer fails")
    rep.ok = not rep.errors
    return rep


def build_bundle(assignment_dir: str | os.PathLike | Assignment) -> dict:
    """The student bundle: prompts, starter files, data and visible tests. Hidden tests and
    reference solutions are never included."""
    a = assignment_dir if isinstance(assignment_dir, Assignment) else load_assignment(assignment_dir)
    problems = []
    for p in a.problems:
        starter: dict[str, str] = {}
        if (p.dir / "starter").is_dir():
            for f in sorted((p.dir / "starter").iterdir()):
                if f.is_file() and not f.name.startswith(".") and _is_text_file(f):
                    starter[f.name] = _read_text(f)
        data: dict[str, str] = {}
        if p.data_dir.is_dir():
            for f in sorted(p.data_dir.iterdir()):
                if f.is_file() and not f.name.startswith(".") and _is_text_file(f):
                    data[f.name] = _read_text(f)
        problems.append({
            "id": p.id,
            "title": p.title,
            "points": _json_num(p.points),
            "function": p.function,
            "prompt": p.prompt,
            "starter": starter,
            "data": data,
            "tests": [{"name": t.name, "points": _json_num(t.points), "code": t.code}
                      for t in p.tests if not t.hidden],
        })
    return {"format": FORMAT, "title": a.title, "description": a.description, "problems": problems}


def _json_num(x: float):
    x = _round(x)
    return int(x) if x == int(x) else x


SETUP_SH = """#!/usr/bin/env bash
# Gradescope runs this once when the autograder image is built.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-pip python3-venv git

python3 -m venv /autograder/venv
/autograder/venv/bin/pip install --upgrade pip

# mylab is published on PyPI as "mylab-lang".
/autograder/venv/bin/pip install mylab-lang || true
# Fallback: install from the GitHub repository if PyPI did not provide the grader.
if ! /autograder/venv/bin/python -c "import mylab.grader" 2>/dev/null; then
    /autograder/venv/bin/pip install "git+https://github.com/ghanemja/mylab"
fi
/autograder/venv/bin/python -c "import mylab.grader"
"""

RUN_AUTOGRADER = """#!/usr/bin/env bash
# Gradescope runs this for every submission.
RESULTS=/autograder/results/results.json
mkdir -p /autograder/results
cd /autograder/source
export MPLBACKEND=Agg
/autograder/venv/bin/python -m mylab.grader run assignment \\
    --submission /autograder/submission --gradescope "$RESULTS" > /autograder/results/report.txt 2>&1
status=$?
cat /autograder/results/report.txt
if [ $status -ne 0 ] || [ ! -s "$RESULTS" ]; then
    cat > "$RESULTS" <<'JSON'
{"score": 0, "output": "The autograder could not run. Please tell your teacher.", "tests": []}
JSON
fi
"""


def build_gradescope_zip(assignment_dir: str | os.PathLike, out: str | os.PathLike) -> list[str]:
    """Write a Gradescope autograder zip. Returns the names of the files in it."""
    a = load_assignment(assignment_dir)
    entries: list[tuple[str, bytes, int]] = [
        ("setup.sh", SETUP_SH.encode(), 0o755),
        ("run_autograder", RUN_AUTOGRADER.encode(), 0o755),
        ("assignment/assignment.json", (a.dir / "assignment.json").read_bytes(), 0o644),
    ]
    for p in a.problems:
        base = f"assignment/problems/{p.id}"
        for rel in ("problem.json", "prompt.md"):
            entries.append((f"{base}/{rel}", (p.dir / rel).read_bytes(), 0o644))
        for sub in ("tests", "data"):
            d = p.dir / sub
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.is_file() and not f.name.startswith("."):
                        entries.append((f"{base}/{sub}/{f.name}", f.read_bytes(), 0o644))
    out = Path(out)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return [e[0] for e in entries]


def find_submissions(submissions_dir: str | os.PathLike) -> list[tuple[str, Path]]:
    """(student name, path) for every folder or .zip directly inside ``submissions_dir``."""
    root = Path(submissions_dir)
    if not root.is_dir():
        raise AssignmentError(f"{root}: not a folder")
    found: dict[str, Path] = {}
    for entry in sorted(root.iterdir(), key=lambda p: (p.name.lower(), p.name)):
        if entry.name.startswith(".") or entry.name in _JUNK_DIRS:
            continue
        if entry.is_dir():
            found[entry.name] = entry           # a folder wins over a zip of the same name
        elif entry.is_file() and entry.suffix.lower() == ".zip":
            found.setdefault(entry.stem, entry)
    return sorted(found.items(), key=lambda kv: (kv[0].lower(), kv[0]))


def grade_batch(assignment_dir: str | os.PathLike, submissions_dir: str | os.PathLike,
                jobs: int = 1, progress=None) -> list[Result]:
    """Grade every submission in a folder. Results are sorted by student name."""
    assignment = load_assignment(assignment_dir)
    subs = find_submissions(submissions_dir)

    def one(item: tuple[str, Path]) -> Result:
        r = grade_submission(assignment, item[1], name=item[0])
        if progress:
            progress(r)
        return r

    return _map(one, subs, jobs)


def write_csv(results: list[Result], assignment: Assignment, path: str | os.PathLike) -> None:
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["student", *[p.id for p in assignment.problems], "total", "max", "percent"])
        for r in results:
            w.writerow([r.submission, *[_num(p.score) for p in r.problems], _num(r.score),
                        _num(r.max_score), f"{r.percent:g}"])


_INIT_FILES = {
    "assignment.json": json.dumps({
        "format": FORMAT,
        "title": "{title}",
        "description": "Write a short introduction for your students here. You can use **Markdown**.",
        "problems": ["p1_add_two"],
    }, indent=2) + "\n",
    "problems/p1_add_two/problem.json": json.dumps({
        "title": "Add two", "points": 10, "function": "add_two", "timeout": 10,
    }, indent=2) + "\n",
    "problems/p1_add_two/prompt.md": (
        "# Add two\n\n"
        "Write a function `y = add_two(x)` that adds 2 to its input.\n\n"
        "`x` can be a number, a vector or a matrix; `y` has the same size as `x`.\n\n"
        "```matlab\n>> add_two(5)\nans =\n     7\n```\n"
    ),
    "problems/p1_add_two/starter/add_two.m": (
        "function y = add_two(x)\n% ADD_TWO  Add 2 to every element of x.\n\n"
        "y = [];   % replace this line with your code\n\nend\n"
    ),
    "problems/p1_add_two/solution/add_two.m": (
        "function y = add_two(x)\n% ADD_TWO  Add 2 to every element of x.\n\ny = x + 2;\n\nend\n"
    ),
    "problems/p1_add_two/tests/01_numbers.m": (
        "% name: Single numbers\n% points: 6\n"
        "assert(isequal(add_two(5), 7), 'add_two(5) should be 7');\n"
        "assert(isequal(add_two(-2), 0), 'add_two(-2) should be 0');\n"
        "assert(abs(add_two(0.5) - 2.5) < 1e-12, 'add_two(0.5) should be 2.5');\n"
    ),
    "problems/p1_add_two/tests/02_arrays.m": (
        "% name: Vectors and matrices\n% points: 4\n% hidden: true\n"
        "assert(isequal(add_two([1 2 3]), [3 4 5]));\n"
        "assert(isequal(add_two([1 2; 3 4]), [3 4; 5 6]));\n"
        "assert(isequal(size(add_two(zeros(3, 1))), [3 1]));\n"
    ),
}


def init_assignment(target: str | os.PathLike) -> Path:
    """Create a new assignment folder with one example problem."""
    root = Path(target)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise AssignmentError(f"{root} already exists; choose a new name")
    title = root.name.replace("_", " ").replace("-", " ").strip().title() or "My assignment"
    for rel, text in _INIT_FILES.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text.replace("{title}", title) if rel == "assignment.json" else text, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------

def _write_json(path: str | os.PathLike, data) -> None:
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cmd_init(args) -> int:
    root = init_assignment(args.name)
    print(f"Created {root}/ with one example problem (problems/p1_add_two).")
    print("Next steps:")
    print(f"  1. Edit the files in {root}/problems/ (see docs/AUTOGRADER.md).")
    print(f"  2. mylab-grade check {root}")
    print(f"  3. mylab-grade export {root} --out {root.name}.student.json")
    return 0


def _cmd_check(args) -> int:
    rep = check_assignment(args.assignment, jobs=args.jobs)
    for line in rep.lines:
        print(line)
    for w in rep.warnings:
        print(f"WARNING: {w}")
    for e in rep.errors:
        print(f"ERROR: {e}")
    if rep.ok:
        extra = f" ({len(rep.warnings)} warning(s))" if rep.warnings else ""
        print(f"OK: the assignment is ready{extra}.")
        return 0
    print(f"NOT READY: {len(rep.errors)} problem(s) to fix.")
    return 1


def _cmd_run(args) -> int:
    result = grade_submission(args.assignment, args.submission, jobs=args.jobs)
    sys.stdout.write(format_report(result))
    if args.json:
        _write_json(args.json, result.to_dict())
    if args.gradescope:
        _write_json(args.gradescope, gradescope_results(result))
    return 0


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@+-]+", "_", name).strip("._") or "student"


def _cmd_batch(args) -> int:
    assignment = load_assignment(args.assignment)
    subs = find_submissions(args.submissions)
    if not subs:
        raise AssignmentError(f"{args.submissions}: no submissions found (expected one folder or .zip per student)")
    done = [0]

    def progress(r: Result) -> None:
        done[0] += 1
        sys.stderr.write(f"  graded {done[0]}/{len(subs)}: {r.submission}\n")
        sys.stderr.flush()

    results = grade_batch(args.assignment, args.submissions, jobs=args.jobs, progress=progress)
    ids = [p.id for p in assignment.problems]
    name_w = max([len("student")] + [len(r.submission) for r in results])
    col_w = [max(len(i), 6) for i in ids]
    print(f"{assignment.title}: {len(results)} submission(s), {_num(assignment.points)} points")
    print()
    print("  ".join([f"{'student':<{name_w}}"] + [f"{i:>{w}}" for i, w in zip(ids, col_w)] + [f"{'total':>7}"]))
    print("  ".join(["-" * name_w] + ["-" * w for w in col_w] + ["-" * 7]))
    for r in results:
        cells = [f"{_num(p.score):>{w}}" for p, w in zip(r.problems, col_w)]
        print("  ".join([f"{r.submission:<{name_w}}"] + cells + [f"{_num(r.score):>7}"]))
    if results:
        mean = sum(r.score for r in results) / len(results)
        print()
        print(f"Average: {_num(mean)} / {_num(assignment.points)}")
    if args.csv:
        write_csv(results, assignment, args.csv)
        print(f"Grades written to {args.csv}")
    if args.reports:
        out = Path(args.reports)
        out.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        for r in results:
            base = stem = _safe_name(r.submission)
            n = 2
            while stem.lower() in used:
                stem = f"{base}_{n}"
                n += 1
            used.add(stem.lower())
            (out / f"{stem}.md").write_text(format_markdown(r), encoding="utf-8")
            (out / f"{stem}.txt").write_text(format_report(r), encoding="utf-8")
        print(f"Reports written to {out}/ (one .md and one .txt per student)")
    return 0


def _cmd_export(args) -> int:
    a = load_assignment(args.assignment)
    bundle = build_bundle(a)
    _write_json(args.out, bundle)
    n_visible = sum(len(p["tests"]) for p in bundle["problems"])
    n_hidden = sum(t.hidden for p in a.problems for t in p.tests)
    print(f"Wrote {args.out}: {len(a.problems)} problem(s), {n_visible} visible test(s).")
    print(f"Not included: {n_hidden} hidden test(s) and all reference solutions.")
    for p in a.problems:
        if all(t.hidden for t in p.tests):
            print(f"WARNING: {p.id} has no visible tests; students cannot check it in the app.")
        if p.data_dir.is_dir():
            for f in sorted(p.data_dir.iterdir()):
                if f.is_file() and not f.name.startswith(".") and not _is_text_file(f):
                    print(f"WARNING: {p.id}/data/{f.name} is not a text file and was left out.")
    return 0


def _cmd_gradescope(args) -> int:
    names = build_gradescope_zip(args.assignment, args.out)
    print(f"Wrote {args.out} ({len(names)} files). Upload it to Gradescope under 'Configure Autograder'.")
    print("Reference solutions and starter files are not included in the zip.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mylab-grade",
        description="Autograder for MATLAB-language assignments, powered by mylab. See docs/AUTOGRADER.md.",
    )
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p = sub.add_parser("init", help="create a new assignment folder with one example problem")
    p.add_argument("name", metavar="NAME", help="folder to create")
    p.set_defaults(func=_cmd_init)

    p = sub.add_parser("check", help="validate an assignment and run its reference solutions")
    p.add_argument("assignment", metavar="ASSIGNMENT", help="assignment folder")
    p.add_argument("--jobs", type=int, default=1, metavar="N", help="run N tests at the same time (default 1)")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("run", help="grade one submission and print a report")
    p.add_argument("assignment", metavar="ASSIGNMENT", help="assignment folder")
    p.add_argument("--submission", required=True, metavar="DIR", help="folder (or .zip) with the student's .m files")
    p.add_argument("--json", metavar="FILE", help="also write the full result as JSON")
    p.add_argument("--gradescope", metavar="FILE", help="also write Gradescope's results.json")
    p.add_argument("--jobs", type=int, default=1, metavar="N", help="run N tests at the same time (default 1)")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("batch", help="grade a folder with one submission per student")
    p.add_argument("assignment", metavar="ASSIGNMENT", help="assignment folder")
    p.add_argument("--submissions", required=True, metavar="DIR", help="folder with one folder or .zip per student")
    p.add_argument("--csv", metavar="FILE", help="write the grades to a CSV file")
    p.add_argument("--reports", metavar="DIR", help="write a report per student into this folder")
    p.add_argument("--jobs", type=int, default=1, metavar="N", help="grade N students at the same time (default 1)")
    p.set_defaults(func=_cmd_batch)

    p = sub.add_parser("export", help="build the student bundle for the mylab web app")
    p.add_argument("assignment", metavar="ASSIGNMENT", help="assignment folder")
    p.add_argument("--out", required=True, metavar="FILE", help="for example assignment.student.json")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("gradescope", help="build a Gradescope autograder zip")
    p.add_argument("assignment", metavar="ASSIGNMENT", help="assignment folder")
    p.add_argument("--out", required=True, metavar="FILE", help="for example autograder.zip")
    p.set_defaults(func=_cmd_gradescope)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except AssignmentError as e:
        for line in e.problems:
            sys.stderr.write(f"ERROR: {line}\n")
        return 2
    except OSError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
