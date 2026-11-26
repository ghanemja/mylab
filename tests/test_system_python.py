"""Calling Python from MATLAB code: py.*, conversions both ways, pyrun, pyrunfile, pyenv."""

import sys

import numpy as np
import pytest

from mylab import Interpreter
from mylab.errors import MatlabError


def test_calls_and_automatic_conversion(m):
    m.run("a = py.math.sqrt(16); b = py.len(py.list({1, 2, 3})); c = py.bool(1); d = py.complex(1, 2);"
          "e = py.math.pi; f = py.operator.add(2, 3); g = py.int(7);")
    assert m.num("a") == 4 and m.num("a").dtype == np.float64
    assert m.num("b") == 3 and m.num("b").dtype == np.int64 and m.num("g").dtype == np.int64
    assert m.num("c").dtype == bool and m.num("d") == 1 + 2j
    assert abs(m.num("e").item() - np.pi) < 1e-15 and m.num("f") == 5
    assert m.str("class(py.list({1}))") == "py.list" and m.str("class(py.str('x'))") == "py.str"


def test_explicit_conversions(m):
    m.run("L = py.list({1, 2, 3}); d = double(L); c = cell(L); s = py.str('abc'); t = py.tuple({1, 'a'});"
          "D = py.dict(pyargs('a', 1, 'b', 'two')); S = struct(D); n = py.list({'x', 'yz'});")
    assert m.num("d").tolist() == [[1, 2, 3]] and m.num("iscell(c) && c{3} == 3")
    assert m.str("char(s)") == "abc" and m.num("string(s) == \"abc\"") and m.num("numel(t) == 2")
    assert m.num("S.a") == 1 and m.str("char(S.b)") == "two"
    assert m.num("isequal(string(n), [\"x\" \"yz\"])") and m.str("strjoin(cellstr(n), '+')") == "x+yz"
    assert m.num("isa(L, 'py.list') && length(L) == 3 && ~isempty(L)")
    m.run("i = int32(py.list({1.7, -2.2})); z = logical(py.list({0, 3}));")
    assert m.num("i").dtype == np.int32 and m.num("i").tolist() == [[2, -2]] and m.num("z").tolist() == [[False, True]]
    with pytest.raises(MatlabError, match="str cannot be converted"):
        m.run("double(py.str('12'))")
    with pytest.raises(MatlabError, match="dict"):
        m.run("struct(py.list({1}))")


def test_arguments_from_matlab(m):
    m.run("s = py.sum([1 2 3]); t = py.type(5); u = py.type(int8(5)); v = py.type('txt'); w = py.type(true);"
          "x = py.type({1, 2}); y = py.type(struct('a', 1)); z = py.type([1 2 3]); q = py.type([1 2; 3 4]);")
    assert m.num("s") == 6
    names = [m.str(f"char(py.getattr({k}, '__name__'))") for k in "tuvwxyzq"]
    assert names == ["float", "int", "str", "bool", "tuple", "dict", "array", "ndarray"]
    m.run("r = py.sorted([3 1 2], pyargs('reverse', true)); j = py.json.dumps(struct('k', {{1, 'v'}}));"
          "sq = py.list(py.map(@(v) v^2, [1 2 3]));")
    assert m.num("double(r)").tolist() == [[3, 2, 1]] and m.str("char(j)") == '{"k": [1.0, "v"]}'
    assert m.num("double(sq)").tolist() == [[1, 4, 9]]
    with pytest.raises(MatlabError, match="cannot be passed to Python"):
        m.run("py.len(datetime(2024, 1, 1))")


def test_objects_methods_indexing_and_display(m):
    m.run("L = py.list({1, 2}); L.append(5); a = L{3}; L{1} = 9; b = double(L); d = py.dict(struct('x', 5));"
          "c = d{'x'}; d{'y'} = 'new'; k = cell(py.list(d.keys())); u = py.str('abc').upper();"
          "n = py.str('a,b').split(','); f = @py.math.floor; g = py.getattr(py.math, 'floor');")
    assert m.num("a") == 5 and m.num("b").tolist() == [[9, 2, 5]] and m.num("c") == 5 and m.num("numel(k)") == 2
    assert m.str("char(u)") == "ABC" and m.num("numel(n)") == 2 and m.num("f(2.7) == 2 && g(3.9) == 3")
    out = m.run("L")
    assert "Python list" in out and "[9.0, 2.0, 5.0]" in out
    assert m.num("isequal(py.list({1, 2}), py.list({1, 2})) && ~isequal(py.list({1}), py.list({2}))")
    with pytest.raises(MatlabError, match="out of range"):
        m.run("L{7}")


def test_numpy_arrays_keep_matlab_orientation(m):
    m.run("n = py.numpy.array([1 2 3; 4 5 6]); a = double(n); t = double(n.T); s = double(n.sum());"
          "sh = cell(n.shape); v = double(py.numpy.arange(4)); i = int64(py.numpy.array(int32([1 2 3])));")
    assert m.num("a").tolist() == [[1, 2, 3], [4, 5, 6]] and m.num("t").tolist() == [[1, 4], [2, 5], [3, 6]]
    assert m.num("s") == 21 and m.num("sh{1} == 2 && sh{2} == 3") and m.num("v").tolist() == [[0, 1, 2, 3]]
    assert m.num("i").dtype == np.int64 and m.str("class(n)") == "py.numpy.ndarray"


def test_errors_and_output(m):
    with pytest.raises(MatlabError, match="Python Error: ValueError") as e:
        m.run("py.math.sqrt(-1)")
    assert e.value.identifier == "MATLAB:Python:PyException"
    with pytest.raises(MatlabError, match="no built-in or module named 'nosuchmodule'"):
        m.run("py.nosuchmodule.f(1)")
    with pytest.raises(MatlabError, match="'nothing' was not found"):
        m.run("py.math.nothing(1)")
    out = m.run("py.print('from python', 3); x = evalc('py.print(''captured'')');")
    assert "from python 3" in out and m.str("x").strip() == "captured"


def test_pyrun_keeps_variables_and_returns_outputs(m):
    m.run("r = pyrun(\"y = a + b\", \"y\", a=2, b=3); pyrun(\"z = [i * i for i in range(4)]\");"
          "z = pyrun(\"\", \"z\"); [p, q] = pyrun(\"p = 1.5; q = 'txt'\", [\"p\" \"q\"]);"
          "t = pyrun([\"import math\", \"t = math.factorial(5)\"], \"t\");")
    assert m.num("r") == 5 and m.num("double(z)").tolist() == [[0, 1, 4, 9]]
    assert m.num("p") == 1.5 and m.str("char(q)") == "txt" and m.num("t") == 120
    with pytest.raises(MatlabError, match="did not define a variable named 'nope'"):
        m.run("pyrun(\"k = 1\", \"nope\")")
    with pytest.raises(MatlabError, match="Python Error: ZeroDivisionError"):
        m.run("pyrun(\"1 / 0\")")
    with pytest.raises(MatlabError, match="SyntaxError"):
        m.run("pyrun(\"def\")")


def test_pyrunfile_and_pyenv(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scr.py").write_text("import sys\nres = len(sys.argv) + k\nwords = sys.argv[1:]\n")
    m.run("r = pyrunfile(\"scr.py one two\", \"res\", k=10); w = pyrunfile(\"scr.py a\", \"words\", k=0); e = pyenv;")
    assert m.num("r") == 13 and m.num("numel(w)") == 1
    assert m.str("char(e.Version)") == f"{sys.version_info[0]}.{sys.version_info[1]}"
    assert m.num("e.Status == \"Loaded\" && e.ExecutionMode == \"InProcess\"")
    assert sys.argv[0] != "scr.py"                      # the script's arguments do not leak out
    with pytest.raises(MatlabError, match="was not found"):
        m.run("pyrunfile('missing.py')")
    with pytest.raises(MatlabError, match="cannot be changed"):
        m.run("pyenv('Version', '3.9')")


def test_python_can_be_switched_off(monkeypatch):
    out: list = []
    interp = Interpreter(out=out.append)
    assert interp.python_enabled
    interp.python_enabled = False
    for code in ("py.math.sqrt(4)", "pyrun('x = 1')", "pyenv", "pyrunfile('a.py')"):
        with pytest.raises(MatlabError, match="turned off") as e:
            interp.execute(code)
        assert e.value.identifier == "MATLAB:Python:Disabled"
    interp.execute("y = sqrt(4);")                       # everything else still works
    assert interp.get_var("y")[0, 0] == 2
    monkeypatch.setenv("MYLAB_PYTHON", "0")
    assert not Interpreter(out=out.append).python_enabled


def test_python_object_made_before_switching_off_cannot_be_used(m):
    m.run("L = py.list({1, 2});")
    m.interp.python_enabled = False
    with pytest.raises(MatlabError, match="turned off"):
        m.run("L.append(3)")
    with pytest.raises(MatlabError, match="turned off"):
        m.run("d = double(L);")
