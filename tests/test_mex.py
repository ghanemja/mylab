"""MEX files: building C/C++ sources with mex and calling them (tests compile the C files in fixtures/mex)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mylab import Interpreter
from mylab.errors import MatlabError
from mylab.mex import MEX_SUFFIX, MEXEXT
from mylab.mex.build import find_c_compiler, find_cxx_compiler

FIXTURES = Path(__file__).parent / "fixtures" / "mex"
ROOT = Path(__file__).parent.parent

pytestmark = pytest.mark.skipif(find_c_compiler() is None or sys.platform == "emscripten",
                                reason="no C compiler")

HELPERS = {
    "fixture_fails.m": "function y = fixture_fails(x)\nerror('fixture:fails:neg', 'negative input %g', x);\nend\n",
    "fixture_identifier.m": "function id = fixture_identifier(e)\nid = e.identifier;\nend\n",
    "times_hundred.m": "function y = times_hundred(x)\ny = x * 100;\nend\n",
}


class Session:
    def __init__(self, folder: Path | None = None):
        self.output: list[str] = []
        self.interp = Interpreter(out=self.output.append)
        if folder is not None:
            self.interp.execute(f"addpath('{folder}')")

    def run(self, code: str) -> str:
        self.output.clear()
        self.interp.execute(code)
        return "".join(self.output)

    def val(self, name: str):
        return self.interp.get_var(name)

    def num(self, expr: str):
        return np.asarray(self.interp.eval_expression(expr))

    def str(self, expr: str) -> str:
        from mylab.values import to_str
        return to_str(self.interp.eval_expression(expr))

    @property
    def runtime(self):
        return self.interp.state.get("mex_runtime")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """All fixture sources, compiled once into one folder."""
    folder = tmp_path_factory.mktemp("mexbuilt")
    for src in FIXTURES.iterdir():
        shutil.copy(src, folder / src.name)
    for name, text in HELPERS.items():
        (folder / name).write_text(text)
    s = Session()
    s.run(f"cd('{folder}')")
    try:
        for src in sorted(FIXTURES.glob("*.c")):
            flags = "'-R2018a', " if src.stem == "mx_complex2018" else ""
            s.run(f"mex({flags}'-silent', '{src.name}')")
        if find_cxx_compiler() is not None:
            s.run("mex -silent mx_cpp.cpp")
    finally:
        s.run(f"cd('{ROOT}')")
    return folder


@pytest.fixture
def m(built) -> Session:
    return Session(built)


def test_numeric_classes_round_trip(m):
    m.run("[a, c] = mx_scale([1 2; 3 4]); [b, cb] = mx_scale(single([1.5 -2]), 3); d = mx_scale(int16([-3 7]));"
          "e = mx_scale(uint64(5)); f = mx_scale([true false true]); g = mx_scale(ones(2, 3, 2), 4);")
    assert m.val("a").tolist() == [[2, 4], [6, 8]] and m.str("c") == "double"
    assert m.val("b").dtype == np.float32 and m.val("b").tolist() == [[4.5, -6]] and m.str("cb") == "single"
    assert m.val("d").dtype == np.int16 and m.val("d").tolist() == [[-6, 14]]
    assert m.val("e").dtype == np.uint64 and m.val("e").item() == 10
    assert m.val("f").dtype == bool and m.val("f").tolist() == [[False, True, False]]
    g = m.val("g")
    assert g.shape == (2, 3, 2) and np.all(g == 4)
    for cls in ("int8", "uint8", "uint16", "int32", "uint32", "int64"):
        m.run(f"[v, k] = mx_scale({cls}([1 2 3]));")
        assert m.str("class(v)") == cls and m.str("k") == cls and m.val("v").tolist() == [[2, 4, 6]]


def test_sizes_classes_and_predicates(m):
    m.run("s = mx_info(zeros(2, 3, 4));")
    assert m.num("s.m") == 2 and m.num("s.n") == 12 and m.num("s.numel") == 24 and m.num("s.ndims") == 3
    assert m.num("s.dims").tolist() == [[2, 3, 4]] and m.str("s.flags") == "ND"
    m.run("s = mx_info(int8(1)); t = mx_info({}); u = mx_info(struct('a', 1)); v = mx_info('ab'); "
          "w = mx_info(@sin); x = mx_info(sparse(eye(2))); y = mx_info(2i); z = mx_info(true);")
    assert m.str("s.class") == "int8" and m.str("s.flags") == "N1"
    assert m.str("t.class") == "cell" and m.str("t.flags") == "cE"
    assert m.str("u.class") == "struct" and m.str("u.flags") == "s1"
    assert m.str("v.class") == "char" and m.str("v.flags") == "T" and m.num("v.n") == 2
    assert m.str("w.class") == "function_handle" and m.str("w.flags") == "1F"
    assert m.str("x.flags") == "NSD"
    assert m.str("y.flags") == "NC1D"
    assert m.str("z.class") == "logical" and m.str("z.flags") == "L1"
    assert m.val("s").fields == ["m", "n", "numel", "ndims", "class", "dims", "flags"]


def test_complex_separate_and_interleaved(m):
    m.run("[w, r, i] = mx_complex([1+2i, 3-4i]); w2 = mx_complex([1 2]);")
    assert m.val("w").tolist() == [[2 - 4j, 6 + 8j]]
    assert m.val("r").tolist() == [[1, 3]] and m.val("i").tolist() == [[2, -4]]
    assert m.val("w2").dtype == np.complex128 and m.val("w2").tolist() == [[2, 4]]
    m.run("q = mx_complex2018([1+2i 3i]); p = mx_complex2018(single(1+1i));")
    assert m.val("q").tolist() == [[-3 + 4j, -9]]
    assert m.val("p").dtype == np.complex64 and m.val("p").item() == 2j
    with pytest.raises(MatlabError) as e:
        m.run("mx_complex2018(5)")
    assert e.value.identifier == "fixture:complex:real"


def test_text(m):
    m.run("[u, rows, t, n, g] = mx_text('hello World');")
    assert m.str("u") == "HELLO WORLD"
    assert m.val("rows").shape == (3, 6) and m.str("rows(2,:)") == "be    "
    assert m.str("t") == "hel" and m.num("n") == 1
    assert m.str("g") == "αz"
    m.run("[u, rows, t, n] = mx_text('ab');")
    assert m.str("t") == "ab" and m.num("n") == 0
    with pytest.raises(MatlabError) as e:
        m.run("mx_text(5)")
    assert e.value.identifier == "fixture:text:notChar"


def test_cells_and_structs(m):
    m.run("[r, s] = mx_cells({1, 'two', [3 4]}, struct('a', 1, 'bb', 2));")
    assert m.str("class(r)") == "cell" and m.num("size(r)").tolist() == [[1, 3]]
    assert m.num("r{1}").tolist() == [[3, 4]] and m.str("r{2}") == "two" and m.num("r{3}") == 1
    assert m.num("size(s)").tolist() == [[1, 3]] and m.val("s").fields == ["index", "value", "names"]
    assert m.num("[s.index]").tolist() == [[1, 2, 3]] and m.str("s(2).value") == "two"
    assert m.run("disp(s(3).names)").split() == ["{'a'}", "{'bb'}"]
    m.run("r = mx_cells(cell(2, 2));")
    assert m.num("size(r)").tolist() == [[2, 2]] and m.num("isempty(r{1})") == 1


def test_sparse(m):
    m.run("[B, I] = mx_sparse(sparse([1 0; 0 3])); C = mx_sparse(sparse([0 2i; 1 0]));")
    assert m.num("issparse(B)") and m.num("full(B)").tolist() == [[2, 0], [0, 6]]
    assert m.num("issparse(I)") and m.num("full(I)").tolist() == np.eye(3).tolist()
    assert m.num("full(C)").tolist() == [[0, 2j], [2, 0]]  # the fixture doubles only the real parts (Pr)
    with pytest.raises(MatlabError) as e:
        m.run("mx_sparse([1 2])")
    assert e.value.identifier == "fixture:sparse:full"


def test_errors_are_mylab_errors(m):
    m.run("try, mx_errors(1); catch e1, end; try, mx_errors(2); catch e2, end; "
          "try, mx_errors(4); catch e4, end; try, mx_errors(5); catch e5, end")
    assert m.str("e1.identifier") == "fixture:errors:bad" and m.str("e1.message") == "Bad value 42 (forty-two)."
    assert m.str("class(e1)") == "MException"
    assert m.str("e2.identifier") == "" and m.str("e2.message") == "Plain failure."
    assert m.str("e4.identifier") == "fixture:errors:inner"
    assert m.str("e4.message") == "raised inside mexCallMATLAB"
    assert m.str("e5.identifier") == "mylab:mex:notCell"
    with pytest.raises(MatlabError, match="Bad value 42"):
        m.run("mx_errors(1)")
    # still usable after errors, and nothing leaked
    assert m.num("mx_errors(0)") == 0
    assert not m.runtime.arrays and not m.runtime.mem


def test_warnings_and_traps(m):
    out = m.run("r = mx_errors(3);")
    assert "Warning: Careful with 3." in out and m.num("r") == 7
    m.run("[msg, id] = lastwarn;")
    assert m.str("id") == "fixture:errors:warn"
    m.run("warning('off', 'fixture:errors:warn');")
    assert "Warning" not in m.run("r = mx_errors(3);")
    m.run("warning('error', 'fixture:errors:warn');")
    with pytest.raises(MatlabError, match="Careful with 3"):
        m.run("mx_errors(3)")
    assert m.num("mx_errors(6)") == 1   # mexSetTrapFlag: mexCallMATLAB returns 1 instead of raising
    assert m.str("mx_errors(7)") == "fixture:fails:neg"   # mexCallMATLABWithTrap returns the MException


def test_printf_output_is_captured(m):
    assert m.run("x = mx_nargs(1, 2);") == "nlhs=1 nrhs=2\n"
    m.run("t = evalc('mx_nargs(5);');")
    assert m.str("t") == "nlhs=0 nrhs=1\n"


def test_call_back_into_mylab(m):
    out = m.run("[y, z] = mx_callback('cumsum', [3 1 2]);")
    assert out == "called cumsum from mx_callback\n"
    assert m.val("y").tolist() == [[3, 4, 6]] and m.num("z") == 3
    m.run("y2 = mx_callback('times_hundred', 2);")
    assert m.num("y2") == 200
    # mexEvalString runs in the caller's workspace, mexPutVariable writes to it and to the base workspace
    assert m.num("mex_seen_by_caller") == 41 and m.num("mex_put") == 42 and m.str("mex_put_base") == "base!"
    m.run("function r = wrap()\n  r = mx_callback('abs', -5);\n  r = [r, mex_put];\nend\nv = wrap();")
    assert m.val("v").tolist() == [[5, 42]]
    with pytest.raises(MatlabError) as e:
        m.run("mx_callback('no_such_function_abc', 1)")
    assert e.value.identifier == "MATLAB:UndefinedFunction"


def test_nargout(m):
    m.run("[a, b, c] = mx_nargs();")
    assert [m.num(v).item() for v in "abc"] == [10, 20, 30]
    out = m.run("mx_nargs")
    assert "nlhs=0 nrhs=0" in out and "ans =" in out and m.num("ans") == 10
    m.run("q = mx_nargs(1, 2, 3);")
    assert m.num("q") == 13
    with pytest.raises(MatlabError) as e:
        m.run("q = mx_nargs('skip');")
    assert e.value.identifier == "MATLAB:unassignedOutputs"
    m.run("mx_nargs('skip');")  # nothing requested, nothing returned: fine
    assert m.num("nargin('mx_nargs')") == -1 and m.num("nargout('mx_nargs')") == -1


def test_handles_feval_and_loops(m):
    m.run("f = @mx_scale; a = f(3); b = feval('mx_scale', 4); c = cellfun(@mx_scale, {1, 2}); "
          "d = arrayfun(@(x) mx_scale(x, 3), 1:3);")
    assert m.num("a") == 6 and m.num("b") == 8 and m.val("c").tolist() == [[2, 4]]
    assert m.val("d").tolist() == [[3, 6, 9]]
    m.run("function s = loopit(n)\n s = 0;\n for k = 1:n\n  s = s + mx_scale(k);\n end\nend\n"
          "t = loopit(300);")
    assert m.num("t") == 300 * 301


def test_memory_and_temporary_arrays(m):
    m.run("y = mx_memory(6);")
    assert m.val("y").tolist() == [[1, 2, 3, 4, 5, 6]]
    rt = m.runtime
    assert not rt.arrays and not rt.mem   # everything made during the call was released


def test_persistent_state_atexit_and_locks(m):
    m.run("a = mx_counter; b = mx_counter;")
    assert m.val("a").tolist() == [[1, 10, 100]] and m.val("b").tolist() == [[2, 20, 200]]
    rt = m.runtime
    assert len(rt.arrays) == 1 and len(rt.mem) == 1    # the persistent array and memory
    m.run("mx_counter lock")
    assert m.num("mx_counter('locked')")
    assert m.run("clear mx_counter") == ""            # locked: stays loaded
    m.run("c = mx_counter;")
    assert m.val("c").tolist() == [[3, 30, 300]]
    m.run("mx_counter unlock")
    assert m.run("clear mex") == "mx_counter: goodbye after 3 calls\n"
    assert not rt.arrays and not rt.mem and not rt.libs
    m.run("d = mx_counter;")
    assert m.val("d").tolist() == [[1, 10, 100]]
    assert m.run("clear all") == "mx_counter: goodbye after 1 calls\n"
    m.run("e = mx_counter;")
    assert m.run("clear functions") == "mx_counter: goodbye after 1 calls\n"


def test_exist_which_what_help_mexext(m, built):
    assert m.num("exist('mx_scale')") == 3 and m.num("exist('mx_scale', 'file')") == 3
    assert m.num(f"exist('mx_scale{MEX_SUFFIX}')") == 3
    assert m.str("which('mx_scale')") == str(built / f"mx_scale{MEX_SUFFIX}")
    assert m.run("which mx_scale").strip() == str(built / f"mx_scale{MEX_SUFFIX}")
    m.run(f"w = what('{built}');")
    assert f"mx_scale{MEX_SUFFIX}" in m.run("disp(w.mex)")
    assert "MEX file" in m.run("help mx_scale")
    assert m.str("mexext") == MEXEXT
    assert m.num("numel(mexext('all'))") == 4


@pytest.mark.skipif(find_cxx_compiler() is None, reason="no C++ compiler")
def test_cpp_mex_file(m):
    assert m.num("mx_cpp(0, [1 2 3.5])") == 6.5
    m.run("try, mx_cpp(1); catch e, end")
    assert m.str("e.identifier") == "mylab:mex:cppException" and m.str("e.message") == "thrown from C++"
    out = m.run("try, mx_cpp(2); catch e, end")
    assert out == "destructor ran\n"   # the C++ unwinding ran the destructor
    assert m.str("e.identifier") == "fixture:cpp:err"


VERSIONED = """#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    (void)nlhs; (void)nrhs; (void)prhs;
    plhs[0] = mxCreateDoubleScalar(VERSION);
}
"""


def test_rebuild_and_clear_mex(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "versioned.c").write_text(VERSIONED)
    m = Session()
    m.run("mex -silent -DVERSION=1 versioned.c")
    assert m.num("versioned") == 1
    m.run("mex -silent -DVERSION=2 versioned.c")      # mex replaces the loaded copy
    assert m.num("versioned") == 2
    # a file replaced behind mylab's back keeps running until clear mex
    (tmp_path / "other").mkdir()
    m.run("mex -silent -DVERSION=3 -outdir other versioned.c")
    shutil.copy(tmp_path / "other" / f"versioned{MEX_SUFFIX}", tmp_path / f"versioned{MEX_SUFFIX}")
    m.interp.lookup_changed()
    assert m.num("versioned") == 2
    m.run("clear mex")
    assert m.num("versioned") == 3
    m.run("mex -silent -DVERSION=4 versioned.c -output renamed")
    assert m.num("renamed") == 4 and m.num("versioned") == 3


def test_mex_file_wins_over_m_file_in_same_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shadowed.c").write_text(VERSIONED)
    (tmp_path / "shadowed.m").write_text("function y = shadowed()\ny = -1;\nend\n")
    m = Session()
    assert m.num("shadowed") == -1
    m.run("mex -silent -DVERSION=7 shadowed.c")
    assert m.num("shadowed") == 7
    assert m.str("which('shadowed')").endswith(MEX_SUFFIX)
    # a .m file in a folder earlier on the search order (the current folder) still wins over a MEX file later
    (tmp_path / "later").mkdir()
    shutil.move(str(tmp_path / f"shadowed{MEX_SUFFIX}"), tmp_path / "later" / f"shadowed{MEX_SUFFIX}")
    m.run(f"addpath('{tmp_path / 'later'}')")
    m.interp.lookup_changed()
    assert m.num("shadowed") == -1


def test_build_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = Session()
    with pytest.raises(MatlabError) as e:
        m.run("mex missing.c")
    assert e.value.identifier == "mylab:mex:fileNotFound"
    (tmp_path / "broken.c").write_text('#include "mex.h"\nvoid mexFunction(int a) { this is not C }\n')
    with pytest.raises(MatlabError) as e:
        m.run("mex broken.c")
    assert e.value.identifier == "mylab:mex:buildFailed" and "broken.c" in e.value.message
    with pytest.raises(MatlabError, match="unknown or unsupported option"):
        m.run("mex -frobnicate broken.c")
    with pytest.raises(MatlabError, match="not a C or C"):
        m.run("mex notes.txt")
    assert "Usage: mex" in m.run("mex -help")
    (tmp_path / "ok.c").write_text(VERSIONED)
    out = m.run("mex -n -DVERSION=1 ok.c")
    assert "-DVERSION=1" in out and not (tmp_path / f"ok{MEX_SUFFIX}").exists()


def test_foreign_mex_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cc = find_c_compiler()
    (tmp_path / "foreign.c").write_text("int mexFunction(void) { return 0; }\n")
    out = tmp_path / f"foreign{MEX_SUFFIX}"
    proc = subprocess.run([cc, "-shared", "-fPIC", "foreign.c", "-o", str(out)], capture_output=True)
    if proc.returncode != 0:
        pytest.skip("cannot build a plain shared library")
    m = Session()
    assert m.num("exist('foreign')") == 3
    with pytest.raises(MatlabError) as e:
        m.run("foreign()")
    assert e.value.identifier == "mylab:mex:notMylabMex" and "rebuild" in e.value.message


def test_browser_gives_a_clear_error(m, monkeypatch):
    import mylab.mex.runtime as rt
    monkeypatch.setattr(rt, "IN_BROWSER", True)
    with pytest.raises(MatlabError) as e:
        m.run("mx_scale(1)")
    assert e.value.identifier == "mylab:mex:browser" and "browser" in e.value.message


def test_header_matches_table():
    from mylab.mex.header import API_HEADER, render
    assert API_HEADER.read_text(encoding="utf-8") == render(), "run python scripts/gen_mex_header.py"


def test_example_builds_and_runs(tmp_path, monkeypatch):
    example = ROOT / "examples" / "mex"
    for f in example.iterdir():
        shutil.copy(f, tmp_path / f.name)
    monkeypatch.chdir(tmp_path)
    out: list[str] = []
    interp = Interpreter(out=out.append)
    interp.run_file(str(tmp_path / "mex_demo.m"))
    text = "".join(out)
    assert "Error" not in text and "column means" in text
