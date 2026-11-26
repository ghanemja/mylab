"""matfile on version 7.3 files: indexing reads, and index assignment writes, only the selected part.

The values are compared with ordinary indexing of the same arrays, and the stored datasets are checked
with plain h5py. Reading the whole of a variable is made to fail where it must not happen.
"""

import h5py  # a dependency of mylab: these tests are never skipped
import numpy as np
import pytest

from mylab.builtins import fileformats
from mylab.errors import MatlabError


@pytest.fixture
def w(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m.run("A = reshape(1:60, 5, 4, 3); M = magic(6); z = complex(M, -2 * M); c = 'partial read'; v = (1:10)';"
          "r = 11:20; L = M > 10; i16 = int16(M - 20); s1 = single(M) / 4; one = 7;"
          "save('p.mat', 'A', 'M', 'z', 'c', 'v', 'r', 'L', 'i16', 's1', 'one', '-v7.3');"
          "m = matfile('p.mat', 'Writable', true);")
    return m


@pytest.fixture
def no_whole_reads(monkeypatch):
    """Make reading a whole variable through the matfile object an error."""
    def refuse(self, interp, name):
        raise AssertionError(f"the whole of '{name}' was read")
    monkeypatch.setattr(fileformats.MatFileObject, "value", refuse)


READS = [
    "M(2, :)", "M(:, end)", "M(1:2:end, [2 5 3])", "M(end:-1:1, 1)", "M(end, end)", "M(4)", "M([3 9; 20 1])",
    "M(logical([1 0 1 0 0 1]), 2)", "M(:, :)", "M(2, 3, 1)", "M(1:3, 2:4)'",
    "A(2:3, :, end)", "A(:, 2, :)", "A(7)", "A(1, 7)", "A(5, 4, 3)", "A(1:2:5, 1:3:4, [1 3])", "A(:, :, 2)",
    "z(2, 3:4)", "z(5)", "c(2:4)", "c(end)", "v(3:5)", "v([2 9])", "r(3:5)", "r(end)", "L(2, :)", "i16(3, 3)",
    "i16(1:2, :)", "s1(:, 2)", "one(1)", "one(1, 1)",
]


@pytest.mark.parametrize("expr", READS)
def test_indexing_reads_only_the_part(w, no_whole_reads, expr):
    w.run(f"got = m.{expr}; want = {expr};")
    assert w.num("isequal(got, want) && strcmp(class(got), class(want)) && isreal(got) == isreal(want)")
    assert w.num(f"isequal(size(got), size({expr}))")


def test_size_and_whos_read_no_data(w, no_whole_reads):
    w.run("sz = size(m, 'A'); n3 = size(m, 'A', 3); [r, c] = size(m, 'M'); s = whos(m); txt = evalc('whos(m)');"
          "d = evalc('disp(m)');")
    assert w.num("isequal(sz, [5 4 3]) && n3 == 3 && r == 6 && c == 6")
    w.run("names = {s.name}; k = find(strcmp(names, 'z')); zk = s(k); ka = find(strcmp(names, 'A'));")
    assert w.num("isequal(zk.size, [6 6]) && zk.complex && zk.bytes == 576 && strcmp(zk.class, 'double')")
    assert w.num("s(ka).bytes == 480 && isequal(s(ka).size, [5 4 3])")
    assert "5x4x3" in w.str("txt") and "complex" in w.str("txt") and "[6x6 logical]" in w.str("d")


def test_whos_sizes_match_the_workspace(w):
    w.run("q.a = 1; q.b = 'xy'; cc = {1, 'ab'; [1 2 3], {}}; sp = sparse([1 0; 0 2]); e = zeros(0, 3);"
          "save('p.mat', 'q', 'cc', 'sp', 'e', '-append', '-v7.3'); s = whos(m); clear m;"
          "S = load('p.mat'); t = whos('-file', 'p.mat');")
    got = {r["name"]: r for r in _rows(w.val("s"))}
    want = {r["name"]: r for r in _rows(w.val("t"))}
    assert set(got) == set(want)
    for name in got:
        assert got[name] == want[name], name


def _rows(s):
    out = []
    for d in s.data.flat:
        out.append({"name": "".join(d["name"].reshape(-1)), "size": tuple(np.asarray(d["size"]).ravel()),
                    "bytes": float(np.asarray(d["bytes"]).flat[0]), "class": "".join(d["class"].reshape(-1)),
                    "complex": bool(np.asarray(d["complex"]).flat[0]), "sparse": bool(np.asarray(d["sparse"]).flat[0])})
    return out


WRITES = [
    "M(3, :) = 100", "M(2:3, [1 6]) = [1 2; 3 4]", "M(1:2:end, 2) = -1", "M(end, end) = 0.5", "M(7) = 70",
    "M([1 36]) = [-5 -6]", "A(2, 2:3, 3) = [-1 -2]", "A(:, 1, 2) = (1:5)'", "A(17) = 1.25", "c(1) = 'P'",
    "c(end-3:end) = 'READ'", "v(4) = 44", "r(2:3) = [0 0]", "z(1, 1) = 5i", "z(2, :) = 1:6", "i16(1, 1) = 30000.7",
    "i16(2, :) = -1", "L(1, 1:2) = [false true]", "L(3, 3) = 1", "s1(2, 2) = 1/3", "one(1) = 8",
]


@pytest.mark.parametrize("stmt", WRITES)
def test_assignment_writes_only_the_part(w, no_whole_reads, stmt, tmp_path):
    w.run(f"m.{stmt}; {stmt};")
    name = stmt.split("(")[0]
    with h5py.File(tmp_path / "p.mat", "r") as f:
        cls = f[name].attrs["MATLAB_class"].decode()
    w.run(f"clear m; S = load('p.mat'); ok = isequal(S.{name}, {name}) && strcmp(class(S.{name}), class({name}));")
    assert w.num("ok") and cls == w.str(f"class({name})")


def test_partial_write_cross_checked_with_h5py(w, tmp_path):
    w.run("m.M(2, 3:5) = [-7 -8 -9]; m.A(4, 2, 3) = -100;")
    with h5py.File(tmp_path / "p.mat", "r") as f:
        stored = f["M"][()]                  # stored transposed: stored[col, row]
        assert stored.shape == (6, 6) and list(stored[2:5, 1]) == [-7, -8, -9]
        assert f["A"].shape == (3, 4, 5) and f["A"][2, 1, 3] == -100
    assert w.num("m.M(2, 4) == -8 && m.A(4, 2, 3) == -100")


def test_growth_and_class_changes_rewrite_the_variable(w):
    w.run("m.M(8, 1) = 1; M(8, 1) = 1; m.v(12) = 3; v(12) = 3; m.i16(1, 1) = 2.5; i16(1, 1) = 2.5;"
          "m.r(1) = 1+2i; r(1) = 1+2i; m.M(1, :) = []; M(1, :) = []; m.c(3) = 65; c(3) = 65;"
          "m.nv(2, 2) = 4; nv(2, 2) = 4; S = load('p.mat');")
    for name in ("M", "v", "i16", "r", "c", "nv"):
        assert w.num(f"isequal(S.{name}, {name}) && strcmp(class(S.{name}), class({name}))"), name


def test_index_errors_match_ordinary_indexing(w, no_whole_reads):
    with pytest.raises(MatlabError, match="position 1 exceeds array bounds"):
        w.run("m.M(7, 1)")
    with pytest.raises(MatlabError, match="position 3 exceeds array bounds"):
        w.run("m.M(1, 2, 3)")
    with pytest.raises(MatlabError, match="positive integers"):
        w.run("m.M(0, 1)")
    with pytest.raises(MatlabError, match="exceeds the number of array elements"):
        w.run("m.v(11)")


def test_read_only_and_side_effects(w, tmp_path):
    out = w.run("f = @() fprintf('#') + 1; k = m.M(f(), 1); m.M(f(), 2) = -3; ro = matfile('p.mat');")
    assert out.count("#") == 2                                        # each index evaluated once
    assert w.num("k == M(2, 1) && m.M(2, 2) == -3")
    with pytest.raises(MatlabError, match="read-only"):
        w.run("ro.M(1, 1) = 0;")
    with h5py.File(tmp_path / "p.mat", "r") as f:
        assert f["M"][0, 0] == 35                                     # magic(6)(1, 1) is unchanged


def test_cells_structs_and_version_7_files_still_work(w):
    w.run("cc = {1, 'two'; [3 4], {5}}; q.a = 3; save('p.mat', 'cc', 'q', '-append');"
          "c2 = m.cc(2, :); a = m.q.a; X = magic(4); save('v7.mat', 'X'); o = matfile('v7.mat', 'Writable', true);"
          "x2 = o.X(2:3, end); o.X(1, :) = 0; X(1, :) = 0; T = load('v7.mat'); sz = size(o, 'X');")
    assert w.num("isequal(c2, cc(2, :)) && a == 3 && isequal(x2, [8; 12]) && isequal(T.X, X) && isequal(sz, [4 4])")
