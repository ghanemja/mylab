"""Tracer, interactive debugger, lint and the CLI flags built on them."""

import json
import subprocess
import sys

from mylab import Interpreter
from mylab.api import Session
from mylab.debugger import Debugger, trace
from mylab.lint import check_source


def test_trace_records_lines_and_variable_changes():
    t = trace(Session(), "x = 1;\nfor k = 1:3\n  x = x * 2;\nend\nv = zeros(1, 2); v(2) = 5;\n")
    lines = [s["line"] for s in t.steps]
    assert lines[:2] == [1, 2] and lines.count(3) == 3 and t.steps[-1].get("end")
    assert t.error is None and not t.truncated
    last = t.workspace_at(len(t.steps) - 1)
    assert last["x"]["preview"] == "8" and last["k"]["preview"] == "3" and last["v"]["preview"] == "[0 5]"
    before_loop = t.workspace_at(1)
    assert before_loop["x"]["preview"] == "1" and "k" not in before_loop


def test_trace_functions_errors_and_limit():
    code = "function r = sq(a)\n  r = a^2;\nend\ny = sq(3);\nz = undefined_thing;\n"
    t = trace(Session(), code)
    inner = [s for s in t.steps if s["function"] == "sq"]
    assert inner and inner[0]["depth"] == 2 and inner[0]["set"]["a"]["preview"] == "3"
    assert "undefined_thing" in t.error and t.error_line == 5
    t2 = trace(Session(), "k = 0; while k < 1000, k = k + 1; end", max_steps=50)
    assert t2.truncated and len(t2.steps) == 50
    assert json.dumps(t.to_dict())


def test_interactive_debugger_commands():
    out: list[str] = []
    interp = Interpreter(out=out.append)
    commands = iter(["b 4", "c", "x", "x = 100;", "n", "w", "c"])
    dbg = Debugger(interp, read=lambda prompt: next(commands), write=out.append)
    code = "x = 1;\ny = 2;\nz = 3;\nw = x + y;\nq = w * 2;\n"
    dbg.set_source(None, code)
    dbg.attach(stop_at_start=True)
    interp.execute(code)
    text = "".join(out)
    assert "base:1  x = 1;" in text and "base:4  w = x + y;" in text and "x = 1\n" in text
    assert interp.get_var("w").item() == 102 and interp.get_var("q").item() == 204


def test_debugger_quit_stops_program():
    out: list[str] = []
    interp = Interpreter(out=out.append)
    dbg = Debugger(interp, read=lambda prompt: "q", write=out.append)
    dbg.attach(stop_at_start=True)
    try:
        interp.execute("a = 1;\nb = 2;\n")
    except Exception as e:  # noqa: BLE001
        assert "stopped" in str(e)
    assert not interp.base.has("a")


def test_lint():
    assert check_source("x = [1 2\n")[0]["severity"] == "error"
    probs = check_source("a = 1;\nb = nosuchfunction_xyz(a);\nc = sin(a);\n")
    assert len(probs) == 1 and probs[0]["line"] == 2 and "nosuchfunction_xyz" in probs[0]["message"]
    assert check_source("function r = f(x)\n r = g(x);\nend\nfunction r = g(x)\n r = x;\nend\n") == []
    assert check_source("load data.mat\ny = mystery(1);\n") == []


def test_cli_check_and_help(tmp_path):
    f = tmp_path / "bad.m"
    f.write_text("x = 1 +* 2;\n")
    r = subprocess.run([sys.executable, "-m", "mylab", "--check", str(f)], capture_output=True, text=True)
    data = json.loads(r.stdout)
    assert data["ok"] is False and data["problems"][0]["line"] == 1
    r = subprocess.run([sys.executable, "-m", "mylab", "--help-for", "linspace"], capture_output=True, text=True)
    assert r.returncode == 0 and "linspace(a, b" in r.stdout
