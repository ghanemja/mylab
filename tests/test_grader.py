"""Tests for the autograder (mylab/grader.py, the mylab-grade command)."""

import csv
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from mylab import grader

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "assignment-intro"
JOBS = 4      # tests run in separate processes; several at a time keeps this file quick


# ---------------------------------------------------------------------------------------
# Helpers: build small assignments and submissions in a temp folder
# ---------------------------------------------------------------------------------------

def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


DOUBLE_SOLUTION = "function y = double_it(x)\ny = 2 * x;\nend\n"
DOUBLE_STARTER = "function y = double_it(x)\ny = [];\nend\n"
SIGN_SOLUTION = ("function s = sign_word(x)\nif x > 0\n    s = 'positive';\nelseif x < 0\n    s = 'negative';\n"
                 "else\n    s = 'zero';\nend\nend\n")


def make_assignment(root: Path, timeout: float = 20) -> Path:
    """Two problems: double_it (3 tests, one hidden, 10 points) and sign_word (2 tests, 5 points)."""
    write(root / "assignment.json", json.dumps({
        "format": "mylab-assignment-1", "title": "Tiny", "description": "A *tiny* assignment.",
        "problems": ["p1_double", "p2_sign"]}))
    p1 = root / "problems" / "p1_double"
    write(p1 / "problem.json", json.dumps({"title": "Double it", "points": 10, "function": "double_it",
                                           "timeout": timeout}))
    write(p1 / "prompt.md", "Write `y = double_it(x)`.\n")
    write(p1 / "starter" / "double_it.m", DOUBLE_STARTER)
    write(p1 / "solution" / "double_it.m", DOUBLE_SOLUTION)
    write(p1 / "tests" / "01_numbers.m",
          "% name: Single numbers\n% points: 4\nassert(isequal(double_it(2), 4), 'double_it(2) should be 4');\n")
    write(p1 / "tests" / "02_vectors.m", "assert(isequal(double_it([1 2 3]), [2 4 6]));\n")
    write(p1 / "tests" / "03_data.m",
          "% name: Numbers from a file\n% hidden: true\nv = readmatrix('numbers.csv');\n"
          "assert(isequal(double_it(v), [2 4 6 8]));\n")
    write(p1 / "data" / "numbers.csv", "1,2,3,4\n")
    p2 = root / "problems" / "p2_sign"
    write(p2 / "problem.json", json.dumps({"title": "Sign word", "points": 5, "function": "sign_word"}))
    write(p2 / "prompt.md", "Write `s = sign_word(x)`.\n")
    write(p2 / "starter" / "sign_word.m", "function s = sign_word(x)\ns = '';\nend\n")
    write(p2 / "solution" / "sign_word.m", SIGN_SOLUTION)
    write(p2 / "tests" / "01_positive.m", "assert(strcmp(sign_word(3), 'positive'));\n")
    write(p2 / "tests" / "02_others.m",
          "assert(strcmp(sign_word(-3), 'negative'));\nassert(strcmp(sign_word(0), 'zero'));\n")
    return root


def make_one_problem(root: Path, tests: dict[str, str], timeout: float = 20, points=10) -> Path:
    write(root / "assignment.json", json.dumps({
        "format": "mylab-assignment-1", "title": "One", "description": "", "problems": ["p1"]}))
    p = root / "problems" / "p1"
    write(p / "problem.json", json.dumps({"title": "Double it", "points": points, "function": "double_it",
                                          "timeout": timeout}))
    write(p / "prompt.md", "Write `y = double_it(x)`.\n")
    write(p / "starter" / "double_it.m", DOUBLE_STARTER)
    write(p / "solution" / "double_it.m", DOUBLE_SOLUTION)
    for name, code in tests.items():
        write(p / "tests" / name, code)
    return root


def solution_submission(assignment: Path, dest: Path) -> Path:
    """A flat submission folder holding every reference solution."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in assignment.glob("problems/*/solution/*.m"):
        shutil.copy(f, dest / f.name)
    return dest


@pytest.fixture
def tiny(tmp_path):
    return make_assignment(tmp_path / "tiny")


# ---------------------------------------------------------------------------------------
# Loading and validating
# ---------------------------------------------------------------------------------------

def test_load_assignment_and_point_split(tiny):
    a = grader.load_assignment(tiny)
    assert a.title == "Tiny" and [p.id for p in a.problems] == ["p1_double", "p2_sign"]
    p1, p2 = a.problems
    assert [t.points for t in p1.tests] == [4, 3, 3]          # 4 explicit, the rest shared equally
    assert [t.name for t in p1.tests] == ["Single numbers", "Vectors", "Numbers from a file"]
    assert [t.hidden for t in p1.tests] == [False, False, True]
    assert [t.points for t in p2.tests] == [2.5, 2.5]
    assert p2.timeout == grader.DEFAULT_TIMEOUT
    assert a.points == 15


def test_equal_shares_add_up_exactly(tmp_path):
    root = make_one_problem(tmp_path / "a", {f"0{i}_t.m": "assert(true);\n" for i in range(1, 4)})
    pts = [t.points for t in grader.load_assignment(root).problems[0].tests]
    assert pts == [3.33, 3.33, 3.34] and abs(sum(pts) - 10) < 1e-9


def test_metadata_parsing():
    meta = grader.parse_test_metadata("%% name:  Handles empty input \n% points: 2.5\n% Hidden: TRUE\nx = 1;\n"
                                      "% points: 99\n")
    assert meta == {"name": "Handles empty input", "points": 2.5, "hidden": True}
    assert grader.parse_test_metadata("x = 1;\n% hidden: true\n") == {}


def test_structure_errors_are_all_reported(tmp_path):
    root = make_assignment(tmp_path / "bad")
    (root / "problems" / "p1_double" / "prompt.md").unlink()
    write(root / "problems" / "p2_sign" / "problem.json", "{not json")
    with pytest.raises(grader.AssignmentError) as e:
        grader.load_assignment(root)
    text = str(e.value)
    assert "prompt.md" in text and "not valid JSON" in text


@pytest.mark.parametrize("change, expected", [
    (lambda r: write(r / "assignment.json", json.dumps({"format": "other", "title": "x", "problems": ["p1"]})),
     '"format" must be'),
    (lambda r: write(r / "assignment.json", json.dumps({"format": "mylab-assignment-1", "title": "x",
                                                        "problems": ["p1", "nope"]})), "folder is missing"),
    (lambda r: write(r / "problems/p1/tests/01_a.m", "% points: 30\nassert(true);\n"), "more than the problem"),
    (lambda r: write(r / "problems/p1/tests/01_a.m", "% points: lots\nassert(true);\n"), "not a number"),
    (lambda r: write(r / "problems/p1/problem.json", json.dumps({"title": "t", "points": 10})), '"function"'),
    (lambda r: write(r / "problems/p1/problem.json", json.dumps({"title": "t", "function": "double_it"})),
     '"points" is missing'),
    (lambda r: shutil.rmtree(r / "problems/p1/tests"), "at least one test"),
])
def test_structure_error_messages(tmp_path, change, expected):
    root = make_one_problem(tmp_path / "a", {"01_a.m": "assert(true);\n", "02_b.m": "assert(true);\n"})
    change(root)
    with pytest.raises(grader.AssignmentError) as e:
        grader.load_assignment(root)
    assert expected in str(e.value)


def test_cli_structure_error_exit_status(tmp_path, capsys):
    assert grader.main(["check", str(tmp_path / "missing")]) == 1
    assert "ERROR" in capsys.readouterr().out
    assert grader.main(["run", str(tmp_path / "missing"), "--submission", str(tmp_path)]) == 2
    assert "ERROR" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        grader.main(["nonsense"])
    assert e.value.code != 0


# ---------------------------------------------------------------------------------------
# init and check
# ---------------------------------------------------------------------------------------

def test_init_creates_a_valid_assignment(tmp_path, capsys):
    target = tmp_path / "my-first_assignment"
    assert grader.main(["init", str(target)]) == 0
    assert "Created" in capsys.readouterr().out
    a = grader.load_assignment(target)
    assert a.title == "My First Assignment" and len(a.problems) == 1
    assert grader.main(["check", str(target), "--jobs", "4"]) == 0
    assert "OK" in capsys.readouterr().out
    # refuses to overwrite
    assert grader.main(["init", str(target)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_check_example_assignment(capsys):
    assert grader.main(["check", str(EXAMPLE), "--jobs", "4"]) == 0
    out = capsys.readouterr().out
    assert "4 problem(s), 100 points" in out
    assert "OK: the assignment is ready." in out
    assert "WARNING" not in out and "ERROR" not in out


def test_check_reports_wrong_solution_and_passing_starter(tmp_path, capsys):
    root = make_assignment(tmp_path / "t")
    write(root / "problems/p1_double/solution/double_it.m", "function y = double_it(x)\ny = x + 2;\nend\n")
    write(root / "problems/p2_sign/starter/sign_word.m", "function s = sign_word(x)\ns = 'positive';\nend\n")
    rep = grader.check_assignment(root, jobs=JOBS)
    assert not rep.ok
    assert any("reference solution fails test 02_vectors.m" in e for e in rep.errors)
    assert not any("01_numbers.m" in e for e in rep.errors)       # 2 + 2 == 2 * 2
    assert any("starter file already passes test 01_positive.m" in w for w in rep.warnings)
    assert grader.main(["check", str(root), "--jobs", "4"]) == 1
    assert "NOT READY" in capsys.readouterr().out


def test_check_missing_solution(tmp_path):
    root = make_one_problem(tmp_path / "a", {"01_a.m": "assert(double_it(1) == 2);\n"})
    shutil.rmtree(root / "problems/p1/solution")
    rep = grader.check_assignment(root, jobs=JOBS)
    assert not rep.ok and "reference solution is missing" in rep.errors[0]


# ---------------------------------------------------------------------------------------
# Grading one submission
# ---------------------------------------------------------------------------------------

def test_example_solutions_score_full_marks(tmp_path):
    sub = solution_submission(EXAMPLE, tmp_path / "perfect")
    result = grader.grade_submission(EXAMPLE, sub, jobs=JOBS)
    assert (result.score, result.max_score, result.percent) == (100, 100, 100.0)
    assert all(t.passed for p in result.problems for t in p.tests)
    assert result.submission == "perfect"


def test_example_starters_score_zero(tmp_path):
    sub = tmp_path / "starter"
    sub.mkdir()
    for f in EXAMPLE.glob("problems/*/starter/*.m"):
        shutil.copy(f, sub / f.name)
    result = grader.grade_submission(EXAMPLE, sub, jobs=JOBS)
    assert result.score == 0 and result.max_score == 100


def test_partial_credit_and_error_details(tiny, tmp_path):
    sub = tmp_path / "student"
    # Right for single numbers, wrong for vectors.
    write(sub / "double_it.m", "function y = double_it(x)\ndisp('working...')\ny = 2 * x(1);\nend\n")
    write(sub / "sign_word.m", "function s = sign_word(x)\ns = 'positive';\nt = undefined_thing + 1;\nend\n")
    result = grader.grade_submission(tiny, sub, jobs=JOBS)
    p1, p2 = result.problems
    assert [t.status for t in p1.tests] == ["passed", "failed", "failed"]
    assert (p1.score, p1.max_score) == (4, 10)
    failed = p1.tests[1]
    assert failed.message == "Assertion failed." and failed.error_file == "02_vectors.m"
    assert failed.error_line == 1 and "double_it([1 2 3])" in failed.source_line
    assert failed.output == "working..."
    # a hidden test never reveals its source line
    assert p1.tests[2].hidden and p1.tests[2].source_line is None and p1.tests[2].error_line is None
    assert p2.score == 0
    t = p2.tests[0]
    assert "undefined_thing" in t.message
    assert (t.error_file, t.error_line, t.source_line) == ("sign_word.m", 3, "t = undefined_thing + 1;")
    assert result.score == 4 and result.max_score == 15

    d = result.to_dict()
    assert d["score"] == 4 and d["problems"][0]["tests"][0]["status"] == "passed"
    json.dumps(d)

    text = grader.format_report(result)
    assert "[PASS] Single numbers" in text and "[FAIL] Vectors" in text
    assert "in sign_word.m, line 3:  t = undefined_thing + 1;" in text
    assert "(hidden test)" in text and "in a hidden test" in text
    assert "readmatrix" not in text                 # hidden test code is not shown
    assert "| working..." in text
    assert text.rstrip().endswith("4 / 15  (26.7%)")
    assert grader.format_report(grader.grade_submission(tiny, sub, jobs=JOBS)) == text      # deterministic
    md = grader.format_markdown(result)
    assert md.startswith("# Tiny") and "- [x] **Single numbers**" in md and "- [ ] **Vectors**" in md


def test_assert_message_is_reported(tiny, tmp_path):
    sub = tmp_path / "s"
    write(sub / "double_it.m", "function y = double_it(x)\ny = 0;\nend\n")
    write(sub / "sign_word.m", SIGN_SOLUTION)
    result = grader.grade_submission(tiny, sub, jobs=JOBS)
    assert result.problems[0].tests[0].message == "double_it(2) should be 4"
    assert result.problems[1].score == 5


def test_missing_file(tiny, tmp_path):
    sub = tmp_path / "s"
    write(sub / "double_it.m", DOUBLE_SOLUTION)
    result = grader.grade_submission(tiny, sub, jobs=JOBS)
    assert result.problems[0].score == 10
    assert [t.status for t in result.problems[1].tests] == ["missing", "missing"]
    assert "sign_word.m was not found" in result.problems[1].tests[0].message
    assert "[MISS]" in grader.format_report(result)


def test_syntax_error_names_the_file(tmp_path):
    root = make_one_problem(tmp_path / "a", {"01_a.m": "assert(double_it(1) == 2);\n"})
    sub = tmp_path / "s"
    write(sub / "double_it.m", "function y = double_it(x)\ny = [2 * x;\nend\n")
    t = grader.grade_submission(root, sub, jobs=JOBS).problems[0].tests[0]
    assert t.status == "failed" and t.message.startswith("Syntax error.")
    assert t.error_file == "double_it.m" and t.error_line


def test_timeout_only_fails_those_tests(tmp_path):
    # p1_double has a 2-second limit; p2_sign keeps the default limit.
    root = make_assignment(tmp_path / "a", timeout=2)
    sub = tmp_path / "s"
    write(sub / "double_it.m",
          "function y = double_it(x)\nwhile true\n    fprintf('still going\\n');\nend\ny = 2 * x;\nend\n")
    write(sub / "sign_word.m", SIGN_SOLUTION)
    result = grader.grade_submission(root, sub, jobs=JOBS)
    p1, p2 = result.problems
    assert [t.status for t in p1.tests] == ["timeout", "timeout", "timeout"] and p1.score == 0
    assert "Timed out" in p1.tests[0].message and "2 seconds" in p1.tests[0].message
    assert len(p1.tests[0].output) < 3000           # endless printing is cut short
    assert "still going" in p1.tests[0].output
    assert p2.score == 5                            # the other problem is graded normally
    report = grader.format_report(result)
    assert "[TIME] Single numbers" in report and "5 / 15" in report
    assert grader.gradescope_results(result)["tests"][0]["output"].startswith("Timed out")


def test_exit_crash_and_output_do_not_break_grading(tmp_path):
    root = make_one_problem(tmp_path / "a", {
        "01_exit.m": "assert(double_it(1) == 2);\n",
        "02_error.m": "assert(double_it(2) == 4);\n",
        "03_figure.m": "assert(double_it(3) == 6);\n",
        "04_input.m": "assert(double_it(4) == 8);\n",
    }, points=8)
    sub = tmp_path / "s"
    write(sub / "double_it.m", (
        "function y = double_it(x)\ny = 2 * x;\n"
        "if x == 1\n    exit\nend\n"
        "if x == 2\n    error('Custom:id', 'I give up on %d', x);\nend\n"
        "if x == 3\n    figure; plot(1:10); title('hello'); disp(magic(3)); close all; clc\nend\n"
        "if x == 4\n    n = input('Type a number: ');\nend\n"
        "end\n"))
    tests = grader.grade_submission(root, sub, jobs=JOBS).problems[0].tests
    assert [t.status for t in tests] == ["failed", "failed", "passed", "failed"]
    assert "exit" in tests[0].message
    assert tests[1].message == "I give up on 2" and tests[1].error_file == "double_it.m"
    assert tests[3].error_file == "double_it.m" and tests[3].error_line == 13


def test_submission_cannot_replace_assert_or_the_test(tmp_path):
    root = make_one_problem(tmp_path / "a", {"01_a.m": "assert(double_it(1) == 2);\n"})
    sub = tmp_path / "s"
    write(sub / "double_it.m", "function y = double_it(x)\ny = 0;\nend\n")
    write(sub / "assert.m", "function assert(varargin)\nend\n")
    write(sub / "01_a.m", "disp('nothing to see');\n")
    write(sub / "numpy.py", "raise SystemExit(0)\n")
    pr = grader.grade_submission(root, sub, jobs=JOBS).problems[0]
    assert pr.score == 0 and pr.tests[0].message == "Assertion failed."
    assert any("assert.m was ignored" in n for n in pr.notes)


def test_helper_files_are_available(tmp_path):
    root = make_one_problem(tmp_path / "a", {"01_a.m": "assert(double_it(4) == 8);\n"})
    sub = tmp_path / "s"
    write(sub / "double_it.m", "function y = double_it(x)\ny = my_helper(x) + my_helper(x);\nend\n")
    write(sub / "my_helper.m", "function y = my_helper(x)\ny = x;\nend\n")
    assert grader.grade_submission(root, sub, jobs=JOBS).score == 10


# ---------------------------------------------------------------------------------------
# Forgiving layouts and zip files
# ---------------------------------------------------------------------------------------

def test_per_problem_subfolders(tiny, tmp_path):
    sub = tmp_path / "s"
    write(sub / "p1_double" / "double_it.m", DOUBLE_SOLUTION)
    write(sub / "p2_sign" / "sign_word.m", SIGN_SOLUTION)
    assert grader.grade_submission(tiny, sub, jobs=JOBS).score == 15


def test_nested_wrapper_folders_and_wrong_case(tiny, tmp_path):
    sub = tmp_path / "s"
    write(sub / "Homework 1" / "src" / "Double_It.m", DOUBLE_SOLUTION)
    write(sub / "Homework 1" / "Problem 2" / "sign_word.m", SIGN_SOLUTION)
    write(sub / "__MACOSX" / "sign_word.m", "garbage(")
    write(sub / ".git" / "double_it.m", "garbage(")
    result = grader.grade_submission(tiny, sub, jobs=JOBS)
    assert result.score == 15
    assert any("Double_It.m" in n for n in result.problems[0].notes)


def test_problem_folder_wins_over_stray_copy(tiny, tmp_path):
    sub = tmp_path / "s"
    write(sub / "drafts" / "double_it.m", "function y = double_it(x)\ny = 0;\nend\n")
    write(sub / "work" / "p1" / "double_it.m", DOUBLE_SOLUTION)        # "p1" matches p1_double
    write(sub / "sign_word.m", SIGN_SOLUTION)
    assert grader.grade_submission(tiny, sub, jobs=JOBS).score == 15


def test_zip_submission(tiny, tmp_path):
    z = tmp_path / "carol.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("carol_hw/double_it.m", DOUBLE_SOLUTION)
        zf.writestr("carol_hw/sign_word.m", SIGN_SOLUTION)
        zf.writestr("../escape.m", "x = 1;")
        zf.writestr("/abs.m", "x = 1;")
    result = grader.grade_submission(tiny, z, jobs=JOBS)
    assert result.submission == "carol" and result.score == 15
    assert not (tmp_path / "escape.m").exists()


def test_broken_zip_is_an_empty_submission(tiny, tmp_path):
    z = write(tmp_path / "dave.zip", "this is not a zip file")
    result = grader.grade_submission(tiny, z, jobs=JOBS)
    assert result.score == 0 and result.problems[0].tests[0].status == "missing"


def test_submission_not_found(tiny, tmp_path):
    with pytest.raises(grader.AssignmentError):
        grader.grade_submission(tiny, tmp_path / "nobody")


# ---------------------------------------------------------------------------------------
# run / batch command line
# ---------------------------------------------------------------------------------------

def test_cli_run_writes_json_and_gradescope(tiny, tmp_path, capsys):
    sub = tmp_path / "erin"
    write(sub / "double_it.m", DOUBLE_SOLUTION)
    write(sub / "sign_word.m", "function s = sign_word(x)\ns = 'positive';\nend\n")
    out_json, out_gs = tmp_path / "out" / "r.json", tmp_path / "results.json"
    rc = grader.main(["run", str(tiny), "--submission", str(sub), "--json", str(out_json),
                      "--gradescope", str(out_gs)])
    assert rc == 0                                   # graded, whatever the score
    printed = capsys.readouterr().out
    assert printed.startswith("Tiny - erin\n") and "12.5 / 15" in printed
    data = json.loads(out_json.read_text())
    assert data["score"] == 12.5 and data["max_score"] == 15 and data["submission"] == "erin"
    gs = json.loads(out_gs.read_text())
    assert gs["score"] == 12.5
    assert len(gs["tests"]) == 5
    for t in gs["tests"]:
        assert {"name", "score", "max_score", "output", "visibility"} <= set(t)
    assert [t["visibility"] for t in gs["tests"]] == ["visible", "visible", "after_published", "visible", "visible"]
    assert gs["tests"][0]["name"] == "Double it: Single numbers"
    assert sum(t["score"] for t in gs["tests"]) == 12.5
    assert sum(t["max_score"] for t in gs["tests"]) == 15
    assert "Assertion failed" in gs["tests"][4]["output"]


def test_cli_batch_csv_and_reports(tiny, tmp_path, capsys):
    subs = tmp_path / "subs"
    write(subs / "bob" / "double_it.m", DOUBLE_SOLUTION)
    write(subs / "bob" / "sign_word.m", SIGN_SOLUTION)
    write(subs / "Alice Smith" / "hw" / "double_it.m", "function y = double_it(x)\ny = x + x;\nend\n")
    with zipfile.ZipFile(subs / "carol.zip", "w") as zf:
        zf.writestr("sign_word.m", SIGN_SOLUTION)
    write(subs / "notes.txt", "not a submission")
    (subs / "empty").mkdir()
    csv_path, reports = tmp_path / "grades.csv", tmp_path / "reports"
    rc = grader.main(["batch", str(tiny), "--submissions", str(subs), "--csv", str(csv_path),
                      "--reports", str(reports), "--jobs", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "4 submission(s)" in out and "Average: 7.5 / 15" in out
    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    assert rows[0] == ["student", "p1_double", "p2_sign", "total", "max", "percent"]
    assert rows[1:] == [
        ["Alice Smith", "10", "0", "10", "15", "66.7"],
        ["bob", "10", "5", "15", "15", "100"],
        ["carol", "0", "5", "5", "15", "33.3"],
        ["empty", "0", "0", "0", "15", "0"],
    ]
    assert sorted(f.name for f in reports.iterdir()) == [
        "Alice_Smith.md", "Alice_Smith.txt", "bob.md", "bob.txt", "carol.md", "carol.txt", "empty.md", "empty.txt"]
    assert "**Score:** 5 / 15" in (reports / "carol.md").read_text()
    assert "double_it.m was not found" in (reports / "carol.txt").read_text()


def test_batch_without_submissions_is_an_error(tiny, tmp_path, capsys):
    (tmp_path / "none").mkdir()
    assert grader.main(["batch", str(tiny), "--submissions", str(tmp_path / "none")]) == 2
    assert "no submissions found" in capsys.readouterr().err


# ---------------------------------------------------------------------------------------
# export and gradescope
# ---------------------------------------------------------------------------------------

def test_export_bundle_format_and_hidden_tests(tiny, tmp_path, capsys):
    out = tmp_path / "tiny.student.json"
    assert grader.main(["export", str(tiny), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "4 visible test(s)" in printed and "1 hidden test(s)" in printed
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert list(bundle) == ["format", "title", "description", "problems"]
    assert bundle["format"] == "mylab-assignment-1" and bundle["title"] == "Tiny"
    assert bundle["description"] == "A *tiny* assignment."
    p1, p2 = bundle["problems"]
    assert list(p1) == ["id", "title", "points", "function", "prompt", "starter", "data", "tests"]
    assert (p1["id"], p1["title"], p1["points"], p1["function"]) == ("p1_double", "Double it", 10, "double_it")
    assert p1["prompt"] == "Write `y = double_it(x)`.\n"
    assert p1["starter"] == {"double_it.m": DOUBLE_STARTER}
    assert p1["data"] == {"numbers.csv": "1,2,3,4\n"}
    assert [(t["name"], t["points"]) for t in p1["tests"]] == [("Single numbers", 4), ("Vectors", 3)]
    assert all(list(t) == ["name", "points", "code"] for t in p1["tests"])
    assert "double_it(2)" in p1["tests"][0]["code"]
    assert p2["data"] == {} and [t["points"] for t in p2["tests"]] == [2.5, 2.5]
    text = out.read_text(encoding="utf-8")
    assert "Numbers from a file" not in text and "readmatrix" not in text     # the hidden test
    assert "y = 2 * x" not in text and "elseif" not in text                   # the solutions


def test_export_example_assignment(tmp_path):
    bundle = grader.build_bundle(EXAMPLE)
    assert len(bundle["problems"]) == 4
    assert sum(p["points"] for p in bundle["problems"]) == 100
    for p in bundle["problems"]:
        assert len(p["tests"]) == 2 and f"{p['function']}.m" in p["starter"]
        assert "hidden: true" not in json.dumps(p)
    assert "rain_week.csv" in bundle["problems"][3]["data"]
    # binary data files cannot travel in the JSON bundle
    root = make_one_problem(tmp_path / "a", {"01_a.m": "assert(double_it(1) == 2);\n"})
    (root / "problems/p1/data").mkdir()
    (root / "problems/p1/data/blob.bin").write_bytes(b"\x00\x01\xff")
    assert grader.build_bundle(root)["problems"][0]["data"] == {}
    assert any("blob.bin" in w for w in grader.check_assignment(root, jobs=JOBS).warnings)


def test_gradescope_zip(tiny, tmp_path, capsys):
    out = tmp_path / "autograder.zip"
    assert grader.main(["gradescope", str(tiny), "--out", str(out)]) == 0
    assert "Wrote" in capsys.readouterr().out
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        setup = zf.read("setup.sh").decode()
        run = zf.read("run_autograder").decode()
        assert zf.getinfo("run_autograder").external_attr >> 16 & 0o111
        zf.extractall(tmp_path / "unzipped")
    assert {"setup.sh", "run_autograder", "assignment/assignment.json",
            "assignment/problems/p1_double/tests/03_data.m",
            "assignment/problems/p1_double/data/numbers.csv"} <= set(names)
    assert not any("/solution/" in n or "/starter/" in n for n in names)
    assert "pip install mylab-lang" in setup and "git+https://github.com/ghanemja/mylab" in setup
    assert run.startswith("#!/usr/bin/env bash")
    assert "/autograder/submission" in run and "/autograder/results/results.json" in run
    # the copy inside the zip is a complete assignment that can grade a submission
    sub = solution_submission(tiny, tmp_path / "sub")
    assert grader.grade_submission(tmp_path / "unzipped" / "assignment", sub).score == 15
    # building it twice gives the same file
    out2 = tmp_path / "again.zip"
    grader.build_gradescope_zip(tiny, out2)
    assert out.read_bytes() == out2.read_bytes()
