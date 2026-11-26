"""MAT-files: save, load, whos -file, matfile, and exchange with SciPy, Octave and MATLAB-written files."""

import os
import shutil
import subprocess

import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture
def w(m, tmp_path, monkeypatch):
    """A session whose current folder is an empty temporary folder."""
    monkeypatch.chdir(tmp_path)
    return m


def test_standard_types_round_trip(w):
    w.run("d = 1:3; i8 = int8([1 -2; 3 4]); u16 = uint16(65535); sg = single(1.5); z = 3 + 4i; L = [true false];"
          "ch = ['ab'; 'cd']; e = []; n3 = reshape(1:24, 2, 3, 4); i64 = int64(9007199254740993);"
          "c = {1, 'two', {3, [4 5]}}; s.a = 'q'; s.b.c = [1 2 3]; sa = struct('q', {1, 'x', [1 2 3]});"
          "ec = {}; es = struct('f', {}); sp = sparse([1 0 2; 0 0 3]); spl = sparse([true false true]);"
          "spc = sparse([1+2i 0 3]); uni = char([945 946 8364]); neg0 = -0;"
          "save('all.mat'); clear all; load('all.mat');")
    assert w.num("isequal(d, 1:3) && isa(d, 'double')")
    assert w.num("isa(i8, 'int8') && isequal(i8, int8([1 -2; 3 4])) && isa(u16, 'uint16') && u16 == 65535")
    assert w.num("isa(sg, 'single') && sg == 1.5 && z == 3 + 4i && islogical(L) && isequal(L, [true false])")
    assert w.num("ischar(ch) && isequal(ch, ['ab'; 'cd']) && isempty(e) && isa(e, 'double')")
    assert w.num("isequal(size(n3), [2 3 4]) && n3(2, 3, 4) == 24 && i64 == int64(9007199254740993)")
    assert w.num("iscell(c) && strcmp(c{2}, 'two') && isequal(c{3}{2}, [4 5]) && strcmp(s.a, 'q')")
    assert w.num("isequal(s.b.c, [1 2 3]) && isequal(size(sa), [1 3]) && strcmp(sa(2).q, 'x')")
    assert w.num("iscell(ec) && isempty(ec) && isstruct(es) && isempty(es) && isfield(es, 'f')")
    assert w.num("issparse(sp) && isequal(full(sp), [1 0 2; 0 0 3]) && issparse(spl) && islogical(spl)")
    assert w.num("issparse(spc) && spc(1) == 1 + 2i && isequal(double(uni), [945 946 8364])")
    assert np.signbit(w.num("neg0")).all()


def test_mylab_types_round_trip(w):
    w.run("a = 2; f = @(x) a * x.^2 + 1; g = @sin; st = [\"a\" \"bb\"; \"ccc\" missing]; s1 = \"one\";"
          "t = table([1; 2], {'p'; 'q'}, 'VariableNames', {'n', 'c'}, 'RowNames', {'r1', 'r2'});"
          "d = datetime(2024, 1, 2, 3, 4, 5); du = hours(3) + minutes(5); cat = categorical({'r', 'g', 'r'});"
          "mp = containers.Map({'k1', 'k2'}, {1, 'v'}); cd = calmonths(2) + caldays(3);"
          "save('obj.mat'); clear all; S = load('obj.mat');")
    assert w.num("S.f(3)") == 19 and w.num("S.g(0)") == 0
    assert w.num("isstring(S.st) && isequal(size(S.st), [2 2]) && S.st(2, 1) == \"ccc\" && ismissing(S.st(2, 2))")
    assert w.num("isstring(S.s1) && S.s1 == \"one\"")
    assert w.num("istable(S.t) && isequal(S.t.n, [1; 2]) && isequal(S.t.c, {'p'; 'q'})")
    assert w.num("isequal(S.t.Properties.RowNames, {'r1'; 'r2'})")
    assert w.num("isdatetime(S.d) && S.d == datetime(2024, 1, 2, 3, 4, 5) && isduration(S.du)")
    assert w.num("S.du == hours(3) + minutes(5) && iscategorical(S.cat) && isequal(cellstr(S.cat), {'r', 'g', 'r'})")
    assert w.num("S.mp.Count == 2 && S.mp('k1') == 1 && strcmp(S.mp('k2'), 'v')")
    assert w.num("iscalendarduration(S.cd) && isequal(S.cd, calmonths(2) + caldays(3))")


CLASSDEFS = {
    "Pt.m": "classdef Pt\n  properties\n    x = 0\n    y = 0\n  end\n  properties (Transient)\n    cache = 7\n  end\n"
            "  methods\n    function o = Pt(x, y)\n      if nargin > 0, o.x = x; o.y = y; end\n    end\n  end\nend\n",
    "Node.m": "classdef Node < handle\n  properties\n    value\n    next\n  end\nend\n",
    "Temp.m": "classdef Temp\n  properties\n    celsius = 0\n  end\n  methods\n"
              "    function s = saveobj(o)\n      s.kelvin = o.celsius + 273.15;\n    end\n  end\n"
              "  methods (Static)\n    function o = loadobj(s)\n      o = Temp; o.celsius = s.kelvin - 273.15;\n"
              "    end\n  end\nend\n",
    "Color.m": "classdef Color\n  enumeration\n    Red, Green\n  end\nend\n",
}


def test_classdef_objects(w, tmp_path):
    for name, body in CLASSDEFS.items():
        (tmp_path / name).write_text(body)
    w.run("p = Pt(3, 4); p.cache = 99; n = Node; n.value = 1; n.next = n; pair = {n, n}; arr = [Pt(1, 2), Pt(5, 6)];"
          "tk = Temp; tk.celsius = 25; col = Color.Green; save('cls.mat'); clear all; load('cls.mat');")
    assert w.num("isa(p, 'Pt') && p.x == 3 && p.y == 4")
    assert w.num("p.cache") == 7                        # transient properties are not stored
    assert w.num("isa(n, 'Node') && n.value == 1 && n.next == n")     # a handle that refers to itself
    assert w.num("pair{1} == pair{2}")                  # one handle object stays one object
    assert w.num("numel(arr) == 2 && arr(2).y == 6")
    assert abs(w.num("tk.celsius").item() - 25) < 1e-12   # through saveobj and loadobj
    assert w.num("col == Color.Green")
    data = (tmp_path / "cls.mat").read_bytes()
    assert b"kelvin" in __import__("zlib").decompress(_first_compressed(data, b"tk"))


def _first_compressed(data: bytes, name: bytes) -> bytes:
    import struct
    import zlib
    pos = 128
    while pos < len(data):
        kind, size = struct.unpack_from("<II", data, pos)
        assert kind == 15
        raw = data[pos + 8:pos + 8 + size]
        if name in zlib.decompress(raw)[:64]:
            return raw
        pos += 8 + size
    raise AssertionError("variable not found")


def test_missing_class_gives_struct_and_warning(w, tmp_path):
    (tmp_path / "Pt.m").write_text(CLASSDEFS["Pt.m"])
    w.run("p = Pt(3, 4); save('one.mat', 'p'); clear all")
    os.remove(tmp_path / "Pt.m")
    w.interp._file_cache.clear()
    w.interp.classes.clear()
    out = w.run("S = load('one.mat');")
    assert "Pt" in out and "Warning" in out
    assert w.num("isstruct(S.p) && S.p.x == 3")


def test_unsavable_value_is_an_error_naming_the_variable(w):
    w.run("ok = 1; h = matfile('nothing.mat');")
    with pytest.raises(MatlabError, match="'h'"):
        w.run("save('bad.mat')")
    with pytest.raises(MatlabError, match="'zz' not found"):
        w.run("save('bad.mat', 'ok', 'zz')")
    with pytest.raises(MatlabError, match="unknown option"):
        w.run("save('bad.mat', 'ok', '-bogus')")
    with pytest.raises(MatlabError, match="'c'"):
        w.run("c = {1}; save('bad.txt', 'c', '-ascii')")


def test_selection_append_struct_regexp(w):
    w.run("a1 = 1; a2 = 2; b = 3; save('s.mat', 'a*'); S = load('s.mat');")
    assert w.str("strjoin(fieldnames(S)', ',')") == "a1,a2"
    w.run("save('s.mat', 'b', '-append'); a1 = 10; save('s.mat', 'a1', '-append'); S = load('s.mat');")
    assert w.num("S.a1 == 10 && S.a2 == 2 && S.b == 3 && numel(fieldnames(S)) == 3")
    w.run("save('r.mat', '-regexp', '^a\\d$'); S = load('r.mat');")
    assert w.str("strjoin(fieldnames(S)', ',')") == "a1,a2"
    w.run("q.x = 1; q.y = 'two'; q.z = 3; save('q.mat', '-struct', 'q'); S = load('q.mat');")
    assert w.num("S.x == 1 && strcmp(S.y, 'two') && S.z == 3")
    w.run("save('q2.mat', '-struct', 'q', 'x', 'z'); S = load('q2.mat');")
    assert w.str("strjoin(fieldnames(S)', ',')") == "x,z"
    w.run("S = load('s.mat', 'a*'); T = load('s.mat', '-regexp', 'b');")
    assert w.num("numel(fieldnames(S)) == 2 && T.b == 3")
    out = w.run("U = load('s.mat', 'nothere');")
    assert "not found" in out and w.num("isempty(fieldnames(U))")
    w.run("clear all; load s.mat a2")            # command form
    assert w.num("a2 == 2 && ~exist('b', 'var')")
    w.run("x = 5; save; clear x; load;")          # default file name
    assert w.num("x") == 5 and os.path.isfile("matlab.mat")
    w.run("save noext x; clear x; load noext;")   # the extension is added
    assert os.path.isfile("noext.mat") and w.num("x") == 5


def test_global_variables(w):
    w.run("global G; G = 42; local = 1; save('g.mat'); clear all; clear global; load('g.mat');")
    w.run("s = whos('G');")
    assert w.num("G") == 42 and w.num("s.global")
    assert w.interp.global_vars["G"][0, 0] == 42


def test_whos_and_who_on_files(w):
    w.run("x = magic(4); name = 'abc'; c = {1, 2}; z = 1 + 2i; sp = sparse(eye(3)); save('f.mat');")
    w.run("s = whos('-file', 'f.mat'); n = who('-file', 'f.mat');")
    assert w.str("strjoin(n', ',')") == "c,name,sp,x,z"
    assert w.num("s(4).bytes") == 128 and w.str("s(4).class") == "double"
    assert w.num("isequal(s(4).size, [4 4])")
    assert w.num("s(2).bytes") == 6 and w.str("s(2).class") == "char"       # two bytes per character
    assert w.num("s(5).complex && s(5).bytes == 16 && s(3).sparse")
    out = w.run("whos('-file', 'f.mat', 'x')")
    assert "4x4" in out and "128" in out and "name" not in out.split("\n", 2)[2]
    w.run("t = whos('-file', 'f.mat', 'n*');")
    assert w.num("numel(t)") == 1


def test_whos_of_the_workspace(w):
    w.run("x = 5; y = 'abc'; q = single([1 2 3]); s = whos; one = whos('x');")
    assert w.num("numel(s)") == 3 and w.str("s(1).name") == "q"
    assert w.str("one.class") == "double" and w.num("one.bytes") == 8
    assert w.num("s(1).bytes") == 12 and w.num("isequal(s(1).size, [1 3])")
    assert w.num("isfield(s, 'global') && isfield(s, 'sparse') && isfield(s, 'complex') && ~one.global")
    out = w.run("whos")
    assert "Name" in out and "Bytes" in out and "single" in out


def test_matfile_object(w):
    w.run("x = magic(4); t = 'txt'; save('h.mat', 'x', 't'); m = matfile('h.mat');")
    assert w.num("isequal(m.x(2, :), [5 11 10 8]) && strcmp(m.t, 'txt')")
    assert w.num("isequal(size(m, 'x'), [4 4]) && isequal(who(m), {'t'; 'x'})")
    assert w.num("~m.Properties.Writable")
    with pytest.raises(MatlabError, match="read-only"):
        w.run("m.y = 1;")
    w.run("m = matfile('h.mat', 'Writable', true); m.y = [1 2 3]; m.x(1, 1) = 100; S = load('h.mat');")
    assert w.num("isequal(S.y, [1 2 3]) && S.x(1, 1) == 100 && S.x(2, 2) == 11 && strcmp(S.t, 'txt')")
    w.run("n = matfile('new.mat'); n.a = 5; n.b = 'q'; T = load('new.mat');")
    assert w.num("T.a == 5 && strcmp(T.b, 'q')")
    with pytest.raises(MatlabError, match="does not exist"):
        w.run("m.nothing")


def test_ascii_files(w, tmp_path):
    w.run("x = [1.5 2.5; -3 4e10]; save('a.txt', 'x', '-ascii'); y = load('a.txt');")
    assert w.num("isequal(y, x)")
    lines = (tmp_path / "a.txt").read_text().splitlines()
    assert lines[0] == "   1.5000000e+00   2.5000000e+00"       # eight digits, 16 characters per number
    w.run("p = pi; save('d.txt', 'p', '-ascii', '-double'); q = load('d.txt'); save('t.txt', 'x', '-ascii', '-tabs');")
    assert w.num("q == pi") and "\t" in (tmp_path / "t.txt").read_text()
    (tmp_path / "num.dat").write_text("% comment\n1, 2, 3\n4 5 6 % trailing\n\n")
    w.run("load num.dat; A = load('num.dat', '-ascii');")
    assert w.num("isequal(num, [1 2 3; 4 5 6]) && isequal(A, num)")
    (tmp_path / "ragged.txt").write_text("1 2 3\n4 5\n")
    with pytest.raises(MatlabError, match="same number of columns"):
        w.run("load('ragged.txt')")
    (tmp_path / "words.txt").write_text("hello world\n")
    with pytest.raises(MatlabError, match="not numeric"):
        w.run("load('words.txt')")
    with pytest.raises(MatlabError, match="Unable to find"):
        w.run("load('nofile.mat')")


def test_formats_v6_v4_and_damaged_files(w, tmp_path):
    w.run("x = magic(3); s = 'txt'; sp = sparse([0 2; 3 0]); save('v6.mat', 'x', 's', '-v6');"
          "save('v4.mat', 'x', 's', 'sp', '-v4'); A = load('v6.mat'); B = load('v4.mat');")
    assert w.num("isequal(A.x, magic(3)) && strcmp(A.s, 'txt') && isequal(B.x, magic(3)) && strcmp(B.s, 'txt')")
    assert w.num("issparse(B.sp) && isequal(full(B.sp), [0 2; 3 0])")
    assert (tmp_path / "v6.mat").read_bytes()[128:132] == b"\x0e\x00\x00\x00"     # not compressed
    with pytest.raises(MatlabError, match="version 4"):
        w.run("c = {1}; save('bad4.mat', 'c', '-v4')")
    data = (tmp_path / "v6.mat").read_bytes()
    (tmp_path / "cut.mat").write_bytes(data[:140])
    with pytest.raises(MatlabError, match="damaged"):
        w.run("load('cut.mat')")
    (tmp_path / "text.mat").write_text("just some text\n")
    with pytest.raises(MatlabError, match="not a binary MAT-file"):
        w.run("load('text.mat')")


def test_scipy_reads_what_mylab_writes(w, tmp_path):
    sio = pytest.importorskip("scipy.io")
    w.run("d = [1.5 2; 3 4]; i = int16([-5 6]); L = [true; false]; c = 'hello'; z = 1 - 2i; sp = sparse([0 7; 0 0]);"
          "ce = {1, 'a'}; st.f = [1 2 3]; st.g = 'x'; whole = [1 2 300]; save('py.mat');")
    r = sio.loadmat(tmp_path / "py.mat")
    assert r["d"].tolist() == [[1.5, 2], [3, 4]] and r["i"].dtype == np.int16 and r["i"].tolist() == [[-5, 6]]
    assert r["L"].dtype == np.uint8 or r["L"].dtype == bool
    assert r["c"][0] == "hello" and r["z"][0, 0] == 1 - 2j
    assert r["sp"].toarray().tolist() == [[0, 7], [0, 0]]
    assert r["ce"][0, 1][0] == "a" and r["st"]["f"][0, 0].tolist() == [[1, 2, 3]]
    assert r["whole"].tolist() == [[1, 2, 300]]      # stored as small integers, read back as numbers


def test_mylab_reads_what_scipy_writes(w, tmp_path):
    sio = pytest.importorskip("scipy.io")
    import scipy.sparse as sp
    sio.savemat(tmp_path / "sp.mat", {
        "a": np.array([[1.5, 2.5]]), "i": np.array([[1, -2]], dtype=np.int32), "t": "text",
        "c": np.array([[1 + 2j]]), "s": sp.csc_matrix(np.array([[0.0, 5.0], [6.0, 0.0]])),
        "st": {"p": np.array([[1.0, 2.0]]), "q": "w"}, "b": np.array([[True, False]])})
    w.run("S = load('sp.mat');")
    assert w.num("isequal(S.a, [1.5 2.5]) && isa(S.i, 'int32') && isequal(S.i, int32([1 -2])) && strcmp(S.t, 'text')")
    assert w.num("S.c == 1 + 2i && issparse(S.s) && isequal(full(S.s), [0 5; 6 0])")
    assert w.num("isequal(S.st.p, [1 2]) && strcmp(S.st.q, 'w') && islogical(S.b)")


def _scipy_data(name: str) -> str:
    sio = pytest.importorskip("scipy.io")
    path = os.path.join(os.path.dirname(sio.matlab.__file__), "tests", "data", name)
    if not os.path.isfile(path):
        pytest.skip("SciPy's MATLAB-written sample files are not installed")
    return path


@pytest.mark.parametrize("name", ["testdouble_7.4_GLNX86.mat", "testdouble_6.1_SOL2.mat",
                                  "testdouble_4.2c_SOL2.mat"])
def test_files_written_by_matlab(w, name):
    # the same vector, written by three MATLAB versions (the second on a big-endian machine)
    w.interp.set_var("f", _scipy_data(name))
    w.run("S = load(f);")
    assert np.allclose(w.num("S.testdouble"), np.arange(9) * np.pi / 4, atol=1e-12)


def test_matlab_function_handles_objects_and_strings(w):
    w.interp.set_var("f", _scipy_data("some_functions.mat"))
    w.run("S = load(f);")
    assert w.num("S.sqr(3)") == 9
    # the anonymous function's captured a, b and c are stored with it, and also as variables of the file
    assert abs(w.num("S.parabola(2) - (S.a * 4 + S.b * 2 + S.c)").item()) < 1e-12
    assert w.str("func2str(S.nCf)") == "nCountFun"
    w.interp.set_var("f", _scipy_data("testmatlabstring_7_WIN64.mat"))
    w.run("S = load(f);")
    assert w.num("isstring(S.matstring1) && strlength(S.matstring1) > 0")
    w.interp.set_var("f", _scipy_data("logical_sparse.mat"))
    w.run("S = load(f); v = S.sp_log_5_4;")
    assert w.num("issparse(v) && islogical(v) && isequal(size(v), [5 4]) && nnz(v) == 5")
    w.interp.set_var("f", _scipy_data("testobject_7.4_GLNX86.mat"))
    out = w.run("S = load(f);")
    assert "inline" in out and w.num("isstruct(S.testobject)")       # an object of a class that is not here


OCTAVE = shutil.which("octave")


def _octave(code: str, cwd) -> str:
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":99"))
    r = subprocess.run([OCTAVE, "--no-gui", "--no-init-file", "--quiet", "--eval", code], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.mark.skipif(OCTAVE is None, reason="GNU Octave is not installed")
def test_exchange_with_octave(w, tmp_path):
    w.run("a = 2.5; M = magic(4); i8 = int8([1 -2; 3 4]); L = [true false]; c = {1, 'two', [1 2 3]};"
          "s.x = 1; s.name = 'hi'; sa = struct('q', {1, 2, 3}); z = 1 + 2i; ch = ['ab'; 'cd'];"
          "n3 = reshape(1:24, 2, 3, 4); sp = sparse([1 0 2]); e = []; sg = single(0.5); u = uint8(200);"
          "save('from_mylab.mat'); x = [1 2; 3 4]; save('from_mylab.txt', 'x', '-ascii');")
    out = _octave(
        "S = load('from_mylab.mat');"
        "assert(S.a == 2.5 && isequal(S.M, magic(4)) && isa(S.i8, 'int8') && isequal(S.i8, int8([1 -2; 3 4])));"
        "assert(islogical(S.L) && isequal(S.L, [true false]) && isequal(S.c, {1, 'two', [1 2 3]}));"
        "assert(S.s.x == 1 && strcmp(S.s.name, 'hi') && isequal(size(S.sa), [1 3]) && S.sa(3).q == 3);"
        "assert(S.z == 1 + 2i && isequal(S.ch, ['ab'; 'cd']) && isequal(S.n3, reshape(1:24, 2, 3, 4)));"
        "assert(issparse(S.sp) && isequal(full(S.sp), [1 0 2]) && isempty(S.e) && isa(S.sg, 'single'));"
        "assert(isa(S.u, 'uint8') && S.u == 200); y = load('from_mylab.txt'); assert(isequal(y, [1 2; 3 4]));"
        "a = 7; M = magic(3); i16 = int16([5 -6]); L = [false true]; c = {2, 'abc'}; s.p = 'q'; s.v = [4 5];"
        "sa = struct('k', {'a', 'b'}); z = 2 - 3i; ch = ['xy'; 'zw']; n3 = reshape(1:8, 2, 2, 2);"
        "sp = sparse([0 4; 5 0]); global G; G = 11; small = 'ab';"
        "save('-v7', 'oct7.mat', 'a', 'M', 'i16', 'L', 'c', 's', 'sa', 'z', 'ch', 'n3', 'sp', 'G', 'small');"
        "save('-v6', 'oct6.mat', 'a', 'M', 'i16', 'L', 'c', 's', 'sa', 'z', 'ch', 'n3', 'sp', 'G', 'small');"
        "save('-v4', 'oct4.mat', 'a', 'M', 'ch'); disp('octave done')", tmp_path)
    assert "octave done" in out
    for name in ("oct7.mat", "oct6.mat"):
        w.run(f"S = load('{name}');")
        assert w.num("S.a == 7 && isequal(S.M, magic(3)) && isa(S.i16, 'int16') && isequal(S.i16, int16([5 -6]))")
        assert w.num("islogical(S.L) && isequal(S.L, [false true]) && isequal(S.c, {2, 'abc'})")
        assert w.num("strcmp(S.s.p, 'q') && isequal(S.s.v, [4 5]) && strcmp(S.sa(2).k, 'b') && S.z == 2 - 3i")
        assert w.num("isequal(S.ch, ['xy'; 'zw']) && isequal(S.n3, reshape(1:8, 2, 2, 2))")
        assert w.num("issparse(S.sp) && isequal(full(S.sp), [0 4; 5 0]) && S.G == 11 && strcmp(S.small, 'ab')")
    w.run("S = load('oct4.mat');")
    assert w.num("S.a == 7 && isequal(S.M, magic(3)) && isequal(S.ch, ['xy'; 'zw'])")


def test_version_73(w, tmp_path):
    h5py = pytest.importorskip("h5py")
    w.run("a = 2; M = [1 2 3; 4 5 6]; st = struct('q', {1, 'x'}); c = {1, 'two'; [1 2], {3}}; ch = 'text';"
          "L = [true false]; z = 1 + 2i; sp = sparse([1 0 2; 0 0 3]); e = []; s1.m = magic(3); i8 = int8(-5);"
          "str = [\"a\" \"bb\"]; t = table([1; 2]); big = reshape(1:20000, 100, 200);"
          "save('v73.mat', '-v7.3'); S = load('v73.mat');")
    assert w.num("S.a == 2 && isequal(S.M, [1 2 3; 4 5 6]) && strcmp(S.st(2).q, 'x') && strcmp(S.c{1, 2}, 'two')")
    assert w.num("isequal(S.c{2, 1}, [1 2]) && S.c{2, 2}{1} == 3 && strcmp(S.ch, 'text') && islogical(S.L)")
    assert w.num("S.z == 1 + 2i && issparse(S.sp) && isequal(full(S.sp), [1 0 2; 0 0 3]) && isempty(S.e)")
    assert w.num("isequal(S.s1.m, magic(3)) && isa(S.i8, 'int8') && S.str(2) == \"bb\" && istable(S.t)")
    assert w.num("isequal(S.big, reshape(1:20000, 100, 200))")
    with h5py.File(tmp_path / "v73.mat", "r") as f:
        assert f["M"].shape == (3, 2) and f["M"].attrs["MATLAB_class"] == b"double"   # dimensions reversed
        assert f["ch"].dtype == np.uint16 and f["ch"].attrs["MATLAB_class"] == b"char"
        assert f.userblock_size == 512
    assert (tmp_path / "v73.mat").read_bytes()[:19] == b"MATLAB 7.3 MAT-file"
    w.run("q = 9; save('v73.mat', 'q', '-append'); T = load('v73.mat', 'q', 'a'); wf = whos('-file', 'v73.mat');")
    assert w.num("T.q == 9 && T.a == 2 && numel(wf) == 15")
    w.run("m = matfile('v73.mat'); r = m.M(2, :);")
    assert w.num("isequal(r, [4 5 6])")
    w.interp.set_var("f", _scipy_data("testhdf5_7.4_GLNX86.mat"))       # written by MATLAB
    w.run("S = load(f);")
    assert np.allclose(w.num("S.testdouble"), np.arange(9) * np.pi / 4, atol=1e-12)


def test_version_73_without_h5py_says_what_is_missing(w, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "h5py":
            raise ImportError("no h5py")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(MatlabError, match="h5py"):
        w.run("x = 1; save('f.mat', 'x', '-v7.3')")
