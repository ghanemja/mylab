"""HDF5: version 7.3 MAT-files (save, load, matfile) in MATLAB's layout, and the h5* functions.

h5py is installed with mylab, so nothing here is skipped for want of it. The checks against hdf5storage and
mat73 (independent readers and writers of MATLAB's format) run only where those packages are installed.
"""

import h5py  # a dependency of mylab: these tests are never skipped
import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture
def w(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return m


# Every MATLAB class that a version 7.3 MAT-file holds in the standard way, plus mylab's own (string arrays).
ALL_CLASSES = (
    "d = [1.5 -2; 3 4e300]; dn = [NaN Inf -Inf -0]; sg = single([1.5 2.25]); i8 = int8([-128 127]);"
    "u8 = uint8([0 255]); i16 = int16(-32768); u16 = uint16(65535); i32 = int32([-2147483648; 2147483647]);"
    "u32 = uint32(4294967295); i64 = int64(-9223372036854775807); u64 = uint64(18446744073709551615);"
    "L = [true false; false true]; L1 = true; ch = ['ab'; 'cd']; uni = char([945 8364 65 20013 233]);"
    "z = [1+2i; 3-4i]; zs = single(1-2i);"
    "sp = sparse([1 0 2; 0 0 3]); spc = sparse([1+2i 0 3]); spl = sparse([true false true]); spe = sparse(3, 4);"
    "e = zeros(0, 3); e00 = []; esg = single(zeros(2, 0)); ei8 = int8(zeros(3, 0, 2)); eu64 = uint64(zeros(0, 1));"
    "el = true(0, 1); ech = ''; ech2 = char(zeros(0, 5)); ec = cell(2, 0); es = struct('f', {});"
    "es2 = repmat(struct('a', 1, 'b', 2), 0, 3);"
    "s.a = 1; s.b.c = 'x'; s.b.d = {1, 'q'}; sa = struct('q', {1, 'two', [1 2 3]}, 'r', {{}, [], 'z'});"
    "sa2 = repmat(struct('k', 5), 2, 3); s0 = struct(); s0arr = repmat(struct(), 2, 3);"
    "c = {1, 'two'; {3, {4, 'five'}}, []}; cn = {int8(1), single(2); sparse(1), {}}; c1 = {'only'};"
    "str = [\"a\" \"bc\"; \"\" \"déf\"]; str1 = \"hello\";"
    "n3 = reshape(1:24, 2, 3, 4); n4 = reshape(1:24, 2, 3, 2, 2) / 7; n3i = int16(reshape(1:12, 2, 3, 2));"
    "n3L = reshape(mod(1:8, 3) == 0, 2, 2, 2); n3z = complex(n3, -n3); n3c = reshape({1, 'a', [], {2}}, 1, 2, 2);"
    "n3ch = reshape('abcdefgh', 2, 2, 2);"
)


def test_v73_round_trip_of_every_class(w):
    w.run(ALL_CLASSES + "names = who; save('all.mat', '-v7.3'); S = load('all.mat'); bad = {};"
          "for k = 1:numel(names), n = names{k}; a = eval(n); b = S.(n);"
          "  same = isequaln(a, b) && strcmp(class(a), class(b)) && isequal(size(a), size(b)) && "
          "    issparse(a) == issparse(b);"
          "  if same && isnumeric(a), same = isreal(a) == isreal(b); end;"
          "  if same && isstruct(a), same = isequal(fieldnames(a), fieldnames(b)); end;"
          "  if ~same, bad{end + 1} = n; end;"
          "end; nbad = numel(bad); badlist = strjoin(bad, ', ');")
    assert w.num("numel(names)") == 49
    assert w.num("nbad") == 0, "changed by save -v7.3 / load: " + w.str("badlist")
    # a few values checked one by one as well
    assert w.num("isequal(S.uni, char([945 8364 65 20013 233])) && S.u64 == intmax('uint64')")
    assert w.num("isequal(size(S.ei8), [3 0 2]) && isa(S.ei8, 'int8') && isequal(size(S.ech2), [0 5])")
    assert w.num("isequal(size(S.es2), [0 3]) && isequal(fieldnames(S.es2), {'a'; 'b'}) && isstruct(S.s0)")
    assert w.num("isempty(fieldnames(S.s0)) && isequal(size(S.s0arr), [2 3]) && isequal(size(S.sa2), [2 3])")
    assert w.num("strcmp(S.c{2, 1}{2}{2}, 'five') && iscell(S.cn{2, 2}) && issparse(S.cn{2, 1})")
    assert w.num("S.str(2, 2) == \"déf\" && S.n3z(2, 3, 4) == 24 - 24i && S.n3ch(2, 2, 2) == 'h'")
    w.run("S.s0.x = 1; S.s0arr(2, 3).y = 'ok';")       # the structs without fields are ordinary structs
    assert w.str("S.s0arr(2, 3).y") == "ok"


def test_v73_append_replaces_and_keeps_references(w, tmp_path):
    w.run("c = {1, 'a'}; s.x = {2}; save('ap.mat', 'c', 's', '-v7.3'); c = {'new', {3}}; d = 4;"
          "save('ap.mat', 'c', 'd', '-append'); S = load('ap.mat');")
    assert w.num("isequal(S.c, {'new', {3}}) && isequal(S.s.x, {2}) && S.d == 4")
    with pytest.raises(MatlabError, match="another format"):
        w.run("save('ap.mat', 'd', '-append', '-v7')")
    w.run("save('old.mat', 'd');")
    with pytest.raises(MatlabError, match="older MAT-file"):
        w.run("save('old.mat', 'd', '-append', '-v7.3')")


def test_v73_values_mylab_stores_its_own_way(w):
    w.run("t = table([1; 2], {'a'; 'b'}, 'VariableNames', {'n', 'k'}); m = containers.Map({'x'}, {5});"
          "f = @(x) x.^2 + 1; dt = datetime(2024, 3, 1); save('own.mat', 't', 'm', 'f', 'dt', '-v7.3');"
          "S = load('own.mat');")
    assert w.num("istable(S.t) && isequal(S.t.n, [1; 2]) && strcmp(S.t.k{2}, 'b') && S.m('x') == 5")
    assert w.num("S.f(2) == 5 && isdatetime(S.dt) && year(S.dt) == 2024")


# -- the layout MATLAB uses ---------------------------------------------------------------------------

def _cls(obj) -> str:
    return obj.attrs["MATLAB_class"].decode("ascii")


def test_v73_layout_is_matlabs(w, tmp_path):
    w.run(ALL_CLASSES + "save('lay.mat', '-v7.3');")
    raw = (tmp_path / "lay.mat").read_bytes()
    # the 512-byte user block: 116 bytes of header text, 8 bytes of subsystem offset, version 0x0200, 'IM'
    head = raw[:116].decode("ascii")
    assert head.startswith("MATLAB 7.3 MAT-file, Platform: ") and ", Created on: " in head
    assert head.rstrip(" ").endswith(" HDF5 schema 1.00 .")
    assert raw[116:124] == b"\0" * 8 and raw[124:126] == b"\x00\x02" and raw[126:128] == b"IM"
    assert raw[128:512] == b"\0" * 384 and raw[512:520] == b"\x89HDF\r\n\x1a\n"
    with h5py.File(tmp_path / "lay.mat", "r") as f:
        assert f.userblock_size == 512
        # MATLAB_class: a scalar, fixed-length, null-terminated ASCII string on every variable
        for name in f:
            if name.startswith("#"):
                continue
            aid = f[name].attrs.get_id("MATLAB_class")
            t = aid.get_type()
            assert aid.shape == () and t.get_class() == h5py.h5t.STRING and not t.is_variable_str()
            assert t.get_strpad() == h5py.h5t.STR_NULLTERM and t.get_cset() == h5py.h5t.CSET_ASCII
        # numbers: dimensions reversed, the data in column-major order
        assert f["d"].shape == (2, 2) and f["d"][()].tolist() == [[1.5, 3], [-2, 4e300]] and _cls(f["d"]) == "double"
        assert f["n3"].shape == (4, 3, 2) and (f["n3"][()] == np.arange(1, 25).reshape(4, 3, 2)).all()
        assert f["n4"].shape == (2, 2, 3, 2) and f["i32"].shape == (1, 2) and f["i32"].dtype == np.int32
        for name, cls, dt in (("sg", "single", np.float32), ("i8", "int8", np.int8), ("u64", "uint64", np.uint64),
                              ("i64", "int64", np.int64), ("u16", "uint16", np.uint16)):
            assert _cls(f[name]) == cls and f[name].dtype == dt
        # char: uint16 codes with MATLAB_int_decode 2; logical: uint8 with MATLAB_int_decode 1
        assert f["ch"].dtype == np.uint16 and f["ch"][()].tolist() == [[97, 99], [98, 100]] and _cls(f["ch"]) == "char"
        assert f["ch"].attrs["MATLAB_int_decode"] == 2 and f["ch"].attrs["MATLAB_int_decode"].dtype == np.int32
        assert f["uni"][()].ravel().tolist() == [945, 8364, 65, 20013, 233]
        assert f["L"].dtype == np.uint8 and _cls(f["L"]) == "logical" and f["L"].attrs["MATLAB_int_decode"] == 1
        # complex: a compound of real and imag
        assert f["z"].dtype.names == ("real", "imag") and f["z"].dtype["real"] == np.float64
        assert f["z"][()]["imag"].ravel().tolist() == [2, -4] and f["zs"].dtype["imag"] == np.float32
        # empty arrays: the dimensions (MATLAB's order) as uint64 data, and MATLAB_empty
        for name, dims, cls in (("e", [0, 3], "double"), ("e00", [0, 0], "double"), ("ei8", [3, 0, 2], "int8"),
                                ("ech2", [0, 5], "char"), ("ec", [2, 0], "cell"), ("el", [0, 1], "logical"),
                                ("es2", [0, 3], "struct"), ("eu64", [0, 1], "uint64"), ("esg", [2, 0], "single")):
            ds = f[name]
            assert ds.dtype == np.uint64 and ds[()].tolist() == dims and _cls(ds) == cls, name
            assert ds.attrs["MATLAB_empty"] == 1 and ds.attrs["MATLAB_empty"].dtype == np.uint8
        assert f["ech"].attrs["MATLAB_int_decode"] == 2 and f["el"].attrs["MATLAB_int_decode"] == 1
        # cells: object references into #refs#, dimensions reversed
        c = f["c"]
        assert c.dtype == h5py.ref_dtype and c.shape == (2, 2) and _cls(c) == "cell"
        assert all(f[r].name.startswith("/#refs#/") for r in c[()].ravel())
        assert _cls(f[c[0, 1]]) == "cell" and _cls(f[c[1, 0]]) == "char"      # c{2,1} is a cell, c{1,2} is text
        assert f[c[1, 0]][()].ravel().tolist() == [ord(x) for x in "two"]
        assert f[c[1, 1]].attrs["MATLAB_empty"] == 1                           # c{2,2} = []
        # sparse: a group with data, ir and jc (zero-based, uint64) and MATLAB_sparse = number of rows
        sp = f["sp"]
        assert isinstance(sp, h5py.Group) and sp.attrs["MATLAB_sparse"] == 2 and _cls(sp) == "double"
        assert sp["data"][()].tolist() == [1, 2, 3] and sp["ir"][()].tolist() == [0, 0, 1]
        assert sp["jc"][()].tolist() == [0, 1, 1, 3] and sp["ir"].dtype == np.uint64 and sp["jc"].dtype == np.uint64
        assert _cls(f["spl"]) == "logical" and f["spl/data"].dtype == np.uint8
        assert f["spc/data"].dtype.names == ("real", "imag") and "data" not in f["spe"] and "ir" not in f["spe"]
        assert f["spe/jc"][()].tolist() == [0] * 5 and f["spe"].attrs["MATLAB_sparse"] == 3
        # structs: a group per scalar struct, MATLAB_fields as variable-length strings of single characters
        s = f["s"]
        assert isinstance(s, h5py.Group) and _cls(s) == "struct" and isinstance(s["b"], h5py.Group)
        assert [b"".join(x).decode() for x in s.attrs["MATLAB_fields"]] == ["a", "b"]
        assert h5py.check_vlen_dtype(s.attrs.get_id("MATLAB_fields").dtype) == np.dtype("S1")
        assert isinstance(f["s0"], h5py.Group) and len(f["s0"]) == 0 and _cls(f["s0"]) == "struct"
        # a struct array: a group with one dataset of references per field, in the array's (reversed) shape
        sa = f["sa"]
        assert isinstance(sa, h5py.Group) and sa["q"].dtype == h5py.ref_dtype and sa["q"].shape == (3, 1)
        assert "MATLAB_class" not in sa["q"].attrs and f[sa["r"][2, 0]][()].ravel().tolist() == [ord("z")]
        assert f["sa2/k"].shape == (3, 2)


def test_v73_files_in_matlabs_layout_are_read(w, tmp_path):
    """A file built here the way MATLAB writes one: canonical empties, null-terminated class names, no data
    for an all-zero sparse matrix, field names as single characters."""
    def cls(obj, name, **more):
        tid = h5py.h5t.C_S1.copy()
        tid.set_size(len(name))
        tid.set_strpad(h5py.h5t.STR_NULLTERM)
        aid = h5py.h5a.create(obj.id, b"MATLAB_class", tid, h5py.h5s.create(h5py.h5s.SCALAR))
        aid.write(np.array(name.encode(), dtype=f"S{len(name)}"), mtype=tid)
        for k, v in more.items():
            obj.attrs[k] = v

    def fields(obj, names):
        obj.attrs.create("MATLAB_fields", np.array([np.frombuffer(n.encode(), "S1") for n in names] + [None],
                                                   dtype=object)[:-1], dtype=h5py.vlen_dtype(np.dtype("S1")))

    path = tmp_path / "matlab.mat"
    with h5py.File(path, "w", userblock_size=512) as f:
        refs = f.create_group("#refs#")
        cls(refs.create_dataset("a", data=np.array([0, 0], dtype=np.uint64)), "canonical empty",
            MATLAB_empty=np.uint8(1))
        cls(refs.create_dataset("b", data=np.array([[104], [105]], dtype=np.uint16)), "char",
            MATLAB_int_decode=np.int32(2))
        cls(refs.create_dataset("c", data=np.array([[7.0]])), "double")
        cls(f.create_dataset("x", data=np.arange(6.0).reshape(3, 2)), "double")          # MATLAB's 2x3
        cls(f.create_dataset("t", data=np.array([[72], [105]], dtype=np.uint16)), "char", MATLAB_int_decode=np.int32(2))
        cls(f.create_dataset("b", data=np.array([[1, 0, 1]], dtype=np.uint8).T), "logical",
            MATLAB_int_decode=np.int32(1))
        cls(f.create_dataset("e", data=np.array([0, 4], dtype=np.uint64)), "double", MATLAB_empty=np.uint8(1))
        cls(f.create_dataset("ce", data=np.array([[refs["c"].ref, refs["b"].ref, refs["a"].ref]],
                                                 dtype=h5py.ref_dtype)), "cell")          # MATLAB's 3x1
        g = f.create_group("sp")
        cls(g, "double", MATLAB_sparse=np.uint64(4))
        g.create_dataset("jc", data=np.zeros(3, dtype=np.uint64))
        s = f.create_group("s")
        cls(s, "struct")
        fields(s, ["alpha", "b"])
        cls(s.create_dataset("alpha", data=np.array([[2.5]])), "double")
        cls(s.create_dataset("b", data=np.array([[np.int8(-1)]], dtype=np.int8)), "int8")
        sa = f.create_group("sa")
        cls(sa, "struct")
        fields(sa, ["v"])
        sa.create_dataset("v", data=np.array([[refs["c"].ref], [refs["b"].ref]], dtype=h5py.ref_dtype))
    text = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Thu Jan  1 00:00:00 2026 HDF5 schema 1.00 ."
    with open(path, "r+b") as fh:
        fh.write(text.ljust(116, b" ") + b"\0" * 8 + b"\x00\x02IM")
    w.run("S = load('matlab.mat'); wf = whos('-file', 'matlab.mat');")
    assert w.num("isequal(S.x, [0 2 4; 1 3 5]) && strcmp(S.t, 'Hi') && isequal(S.b, logical([1 0 1]))")
    assert w.num("isequal(size(S.e), [0 4]) && iscell(S.ce) && isequal(size(S.ce), [3 1]) && S.ce{1} == 7")
    assert w.num("strcmp(S.ce{2}, 'hi') && isempty(S.ce{3}) && isa(S.ce{3}, 'double')")
    assert w.num("issparse(S.sp) && isequal(size(S.sp), [4 2]) && nnz(S.sp) == 0")
    assert w.num("S.s.alpha == 2.5 && isa(S.s.b, 'int8') && isequal(fieldnames(S.s), {'alpha'; 'b'})")
    assert w.num("isequal(size(S.sa), [1 2]) && S.sa(1).v == 7 && strcmp(S.sa(2).v, 'hi') && numel(wf) == 8")


# -- independent readers and writers ------------------------------------------------------------------

def test_mat73_package_reads_what_mylab_writes(w, tmp_path):
    mat73 = pytest.importorskip("mat73")
    w.run(ALL_CLASSES + "clear n3c; save('x.mat', '-v7.3');")      # mat73 does not read cells of 3 dimensions
    r = mat73.loadmat(str(tmp_path / "x.mat"))
    assert r["d"].tolist() == [[1.5, -2], [3, 4e300]] and r["sg"].dtype == np.float32
    assert r["i8"].tolist() == [-128, 127] and r["u64"] == np.iinfo(np.uint64).max
    assert r["L"].tolist() == [[True, False], [False, True]] and r["uni"] == "α€A中é"
    assert r["z"].tolist() == [1 + 2j, 3 - 4j] and r["zs"].dtype == np.complex64
    assert r["sp"].toarray().tolist() == [[1, 0, 2], [0, 0, 3]] and r["spe"].shape == (3, 4)
    assert r["s"]["a"] == 1 and r["s"]["b"]["c"] == "x" and r["s"]["b"]["d"][1] == "q"
    assert r["sa"]["q"][1] == "two" and r["c"][0][1] == "two" and r["c"][1][0][1][1] == "five"
    assert r["n3"].shape == (2, 3, 4) and r["n3"][1, 2, 3] == 24 and r["s0"] == {}


def test_hdf5storage_reads_what_mylab_writes(w, tmp_path):
    hdf5storage = pytest.importorskip("hdf5storage")
    w.run(ALL_CLASSES + "save('x.mat', '-v7.3');")

    def get(name):
        return hdf5storage.loadmat(str(tmp_path / "x.mat"), variable_names=[name])[name]

    assert get("d").tolist() == [[1.5, -2], [3, 4e300]] and get("sg").dtype == np.float32
    assert get("i64").dtype == np.int64 and get("u64")[0, 0] == np.iinfo(np.uint64).max
    assert get("L").dtype == bool and get("ch").tolist() == [["ab"], ["cd"]]
    assert get("uni")[0, 0] == "α€A中é" and get("z").tolist() == [[1 + 2j], [3 - 4j]]
    assert get("zs").dtype == np.complex64
    for name, shape, dtype in (("e", (0, 3), np.float64), ("ei8", (3, 0, 2), np.int8), ("el", (0, 1), bool),
                               ("ech2", (0, 5), "<U1"), ("esg", (2, 0), np.float32)):
        v = get(name)
        assert v.shape == shape and v.dtype == np.dtype(dtype), name
    assert get("ec").shape == (2, 0) and get("ec").dtype == object
    s = get("s")
    assert s["a"][0].tolist() == [[1.0]] and s["b"][0]["c"][0, 0].tolist() == ["x"]
    sa = get("sa")
    assert sa.shape == (1, 3) and sa["q"][0, 1].tolist() == [["two"]]
    c = get("c")
    assert c.shape == (2, 2) and c[0, 1].tolist() == [["two"]] and c[1, 0][0, 1][0, 1].tolist() == [["five"]]
    assert get("n3")[1, 2, 3] == 24 and get("n3z")[1, 2, 3] == 24 - 24j
    assert get("sp")["ir"][0].tolist() == [0, 0, 1]         # hdf5storage shows a sparse matrix as its parts


def test_mylab_reads_what_hdf5storage_writes(w, tmp_path):
    hdf5storage = pytest.importorskip("hdf5storage")
    data = {
        "d": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "sg": np.array([[1.5, 2.0]], dtype=np.float32),
        "i16": np.array([[-3, 4]], dtype=np.int16), "u8": np.uint8(200), "L": np.array([[True, False]]),
        "ch": np.str_("hello"), "uni": np.str_("α€A"), "z": np.array([[1 + 2j, 3 - 4j]]),
        "e": np.zeros((0, 3)), "ei": np.zeros((2, 0, 4), dtype=np.int16), "n3": np.arange(24.0).reshape(2, 3, 4),
        "c": np.array([[1.0, "two"]], dtype=object), "st": {"a": np.array([[1.0]]), "b": np.str_("x")},
        "sa": np.array([[(np.array([[1.0]]),), (np.str_("q"),)]], dtype=[("v", object)]),
    }
    for name, v in data.items():                 # one file per variable, so one problem does not hide another
        hdf5storage.savemat(str(tmp_path / f"h_{name}.mat"), {name: v}, format="7.3")
    w.run("S = struct(); for n = {'d', 'sg', 'i16', 'u8', 'L', 'ch', 'uni', 'z', 'e', 'ei', 'n3', 'c', 'st', 'sa'},"
          " T = load(['h_' n{1} '.mat']); S.(n{1}) = T.(n{1}); end")
    assert w.num("isequal(S.d, [1 2 3; 4 5 6]) && isa(S.sg, 'single') && isequal(S.i16, int16([-3 4]))")
    assert w.num("isa(S.u8, 'uint8') && S.u8 == 200 && isequal(S.L, [true false]) && strcmp(S.ch, 'hello')")
    assert w.num("isequal(double(S.uni), [945 8364 65]) && isequal(S.z, [1+2i 3-4i])")
    assert w.num("isequal(size(S.e), [0 3]) && isequal(size(S.ei), [2 0 4]) && isa(S.ei, 'int16')")
    assert w.num("isequal(S.n3, reshape(permute(reshape(0:23, 4, 3, 2), [3 2 1]), 2, 3, 4))")
    assert w.num("isequal(S.c, {1, 'two'}) && S.st.a == 1 && strcmp(S.st.b, 'x')")
    assert w.num("isequal(size(S.sa), [1 2]) && S.sa(1).v == 1 && strcmp(S.sa(2).v, 'q')")


# -- matfile ----------------------------------------------------------------------------------------

def test_matfile_reads_and_writes_parts_of_v73_files(w, tmp_path):
    w.run("M = magic(5); c = {1, 'two'}; s.a = 3; save('m.mat', 'M', 'c', 's', '-v7.3'); m = matfile('m.mat');"
          "r = m.M(2, :); col = m.M(:, end); blk = m.M(2:3, [1 5]); sz = size(m, 'M'); n = size(m, 'M', 1);"
          "c2 = m.c(1, 2); a = m.s.a; w = who(m);")
    assert w.num("isequal(r, [23 5 7 14 16]) && isequal(col, [15; 16; 22; 3; 9]) && isequal(blk, [23 16; 4 22])")
    assert w.num("isequal(sz, [5 5]) && n == 5 && isequal(c2, {'two'}) && a == 3 && isequal(w, {'M'; 'c'; 's'})")
    with pytest.raises(MatlabError, match="read-only"):
        w.run("m.M(1, 1) = 0;")
    w.run("m.Properties.Writable = true; m.M(1, 1) = 100; m.M(6, :) = 1:5; m.N = 'new'; m.c{3} = [7 8];"
          "S = load('m.mat');")
    assert w.num("S.M(1, 1) == 100 && isequal(S.M(6, :), 1:5) && isequal(S.M(2:5, :), M(2:5, :))")
    assert w.num("strcmp(S.N, 'new') && isequal(S.c, {1, 'two', [7 8]}) && S.s.a == 3")
    assert (tmp_path / "m.mat").read_bytes()[512:516] == b"\x89HDF"        # still a version 7.3 file


def test_matfile_creates_version_73_files(w, tmp_path):
    w.run("n = matfile('new.mat'); n.a = 5; n.b = {1, 'q'}; n.a(2, 2) = 9; T = load('new.mat');"
          "wf = whos('-file', 'new.mat');")
    assert w.num("isequal(T.a, [5 0; 0 9]) && isequal(T.b, {1, 'q'}) && numel(wf) == 2")
    raw = (tmp_path / "new.mat").read_bytes()
    assert raw.startswith(b"MATLAB 7.3 MAT-file") and raw[512:520] == b"\x89HDF\r\n\x1a\n"
    w.run("x = 1; save('v7.mat', 'x'); o = matfile('v7.mat', 'Writable', true); o.y = 2; U = load('v7.mat');")
    assert w.num("U.x == 1 && U.y == 2") and (tmp_path / "v7.mat").read_bytes()[:10] == b"MATLAB 5.0"


# -- h5create, h5write, h5read ------------------------------------------------------------------------

def test_h5_start_count_stride(w, tmp_path):
    w.run("A = reshape(1:60, 5, 4, 3); h5create('s.h5', '/A', [5 4 3]); h5write('s.h5', '/A', A);"
          "p = h5read('s.h5', '/A', [2 1 2], [3 2 1]); q = h5read('s.h5', '/A', [1 1 1], [3 2 2], [2 3 2]);"
          "r = h5read('s.h5', '/A', [4 2 1], [Inf 1 Inf]); one = h5read('s.h5', '/A', [5 4 3], [1 1 1]);")
    assert w.num("isequal(p, A(2:4, 1:2, 2)) && isequal(q, A(1:2:5, 1:3:4, 1:2:3)) && isequal(r, A(4:5, 2, :))")
    assert w.num("one == 60")
    w.run("h5create('s.h5', '/B', [4 6], 'FillValue', -1); h5write('s.h5', '/B', [1 2; 3 4], [2 3], [2 2]);"
          "h5write('s.h5', '/B', [7 8 9], [1 1], [1 3], [1 2]); B = h5read('s.h5', '/B');")
    assert w.num("isequal(B, [7 -1 8 -1 9 -1; -1 -1 1 2 -1 -1; -1 -1 3 4 -1 -1; -1 -1 -1 -1 -1 -1])")
    with h5py.File(tmp_path / "s.h5", "r") as f:
        assert f["A"].shape == (3, 4, 5) and f["A"][1, 0, 2] == 23        # A(3, 1, 2): dimensions reversed
    for code, msg in (("h5read('s.h5', '/A', [5 1 1], [2 1 1])", "beyond the size"),
                      ("h5read('s.h5', '/A', [1 1], [1 1])", "one value per dimension"),
                      ("h5read('s.h5', '/A', [0 1 1], [1 1 1])", "positive"),
                      ("h5read('s.h5', '/A', [1 1 1], [1 1 1], [0 1 1])", "positive"),
                      ("h5write('s.h5', '/B', [1 2 3], [4 6], [1 3])", "beyond the size"),
                      ("h5write('s.h5', '/B', [1 2 3], [1 1], [1 2])", "does not match")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)


def test_h5_types_chunks_filters_and_fill_values(w, tmp_path):
    w.run("for t = {'double', 'single', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'},"
          " n = ['/' t{1}]; h5create('t.h5', n, [2 3], 'Datatype', t{1}, 'FillValue', 7);"
          " h5write('t.h5', n, cast([1 2 3; 4 5 6], t{1})); back.(t{1}) = h5read('t.h5', n); end;"
          "h5create('t.h5', '/z', [10 20], 'ChunkSize', [5 4], 'Deflate', 9, 'Shuffle', true, 'Fletcher32', true);"
          "Z = reshape(1:200, 10, 20); h5write('t.h5', '/z', Z); Zb = h5read('t.h5', '/z');"
          "h5create('t.h5', '/f', [3 2], 'Datatype', 'int16', 'FillValue', int16(-5), 'ChunkSize', [3 1]);"
          "F = h5read('t.h5', '/f'); I = h5info('t.h5', '/z'); J = h5info('t.h5', '/f');")
    for t in ("double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"):
        assert w.num(f"isa(back.{t}, '{t}') && isequal(back.{t}, [1 2 3; 4 5 6])"), t
    assert w.num("isequal(Zb, Z) && isequal(F, int16(-5 * ones(3, 2))) && isa(F, 'int16')")
    assert w.num("isequal(I.ChunkSize, [5 4]) && isequal(J.ChunkSize, [3 1]) && J.FillValue == -5")
    assert w.str("strjoin({I.Filters.Name}, ',')") == "shuffle,deflate,fletcher32"
    assert w.num("I.Filters(2).Data == 9 && isa(J.FillValue, 'int16') && strcmp(J.Datatype.Type, 'H5T_STD_I16LE')")
    with h5py.File(tmp_path / "t.h5", "r") as f:
        z = f["z"]
        assert z.chunks == (4, 5) and z.compression == "gzip" and z.compression_opts == 9
        assert z.shuffle and z.fletcher32 and f["f"].fillvalue == -5 and f["uint64"].dtype == np.uint64
    for code, msg in (("h5create('t.h5', '/q', [2 2], 'Deflate', 5)", "need a ChunkSize"),
                      ("h5create('t.h5', '/q', [2 2], 'ChunkSize', [1 1 1])", "one positive value per dimension"),
                      ("h5create('t.h5', '/q', [2 2], 'ChunkSize', [2 2], 'Deflate', 10)", "between 0 and 9"),
                      ("h5create('t.h5', '/q', [2 2], 'Datatype', 'int3')", "unknown Datatype"),
                      ("h5create('t.h5', '/q', [2 2], 'Colour', 1)", "unknown or unsupported option 'Colour'"),
                      ("h5create('t.h5', '/q', [2 2], 'ChunkSize')", "name-value pairs"),
                      ("h5create('t.h5', '/q', [2 -1])", "non-negative"),
                      ("h5writeatt('t.h5', '/z', 'a', 1, 'Bogus', 2)", "unknown or unsupported option 'Bogus'")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)


def test_h5_unlimited_dimensions_grow(w, tmp_path):
    w.run("h5create('g.h5', '/log', [3 Inf], 'ChunkSize', [3 10]); for k = 1:4, h5write('g.h5', '/log', [k; 2*k; 3*k],"
          " [1 k], [3 1]); end; L = h5read('g.h5', '/log'); I = h5info('g.h5', '/log');")
    assert w.num("isequal(L, [1 2 3 4; 2 4 6 8; 3 6 9 12]) && isequal(I.Dataspace.Size, [3 4])")
    assert w.num("I.Dataspace.MaxSize(1) == 3 && isinf(I.Dataspace.MaxSize(2))")


def test_h5_reads_files_from_other_programs(w, tmp_path):
    with h5py.File(tmp_path / "o.h5", "w") as f:
        f["vec"] = np.arange(4.0)                                        # one dimension: a column
        f["mat"] = np.arange(6, dtype=np.int32).reshape(2, 3)            # C order: MATLAB sees the transpose
        f["scalar"] = 2.5
        f["flags"] = np.array([True, False, True])
        f["names"] = np.array(["ab", "cé"], dtype=h5py.string_dtype())
        f["fixed"] = np.array([b"xy", b"zw"])
        f["big"] = np.arange(3, dtype=">i2")                             # big-endian
        rec = np.zeros(2, dtype=[("id", np.int32), ("val", np.float64)])
        rec["id"], rec["val"] = [1, 2], [0.5, 1.5]
        f["rec"] = rec
        f.create_group("grp/sub")["x"] = [1.0]
    w.run("v = h5read('o.h5', '/vec'); M = h5read('o.h5', '/mat'); s = h5read('o.h5', '/scalar');"
          "b = h5read('o.h5', '/flags'); n = h5read('o.h5', '/names'); fx = h5read('o.h5', '/fixed');"
          "be = h5read('o.h5', '/big'); r = h5read('o.h5', '/rec'); x = h5read('o.h5', '/grp/sub/x');")
    assert w.num("isequal(v, (0:3)') && isa(M, 'int32') && isequal(M, int32([0 3; 1 4; 2 5])) && s == 2.5")
    assert w.num("islogical(b) && isequal(b, [true; false; true]) && isequal(n, [\"ab\"; \"cé\"])")
    assert w.num("isequal(fx, [\"xy\"; \"zw\"]) && isa(be, 'int16') && isequal(be, int16([0; 1; 2]))")
    assert w.num("isequal(r.id, int32([1; 2])) && isequal(r.val, [0.5; 1.5]) && x == 1")


# -- attributes, h5info and h5disp ------------------------------------------------------------------------

def test_h5_attributes(w, tmp_path):
    w.run("h5create('a.h5', '/g/d', [2 2]);"
          "h5writeatt('a.h5', '/g/d', 'units', 'm/s'); h5writeatt('a.h5', '/g/d', 'scale', 0.5);"
          "h5writeatt('a.h5', '/g', 'range', [1 2 3]); h5writeatt('a.h5', '/g', 'grid', [1 2 3; 4 5 6]);"
          "h5writeatt('a.h5', '/', 'count', int32(4)); h5writeatt('a.h5', '/', 'tags', [\"x\" \"yz\"]);"
          "h5writeatt('a.h5', '/', 'note', \"text\"); h5writeatt('a.h5', '/g/d', 'units', 'km');"
          "u = h5readatt('a.h5', '/g/d', 'units'); sc = h5readatt('a.h5', '/g/d', 'scale');"
          "rg = h5readatt('a.h5', '/g', 'range'); gr = h5readatt('a.h5', '/g', 'grid');"
          "cn = h5readatt('a.h5', '/', 'count'); tg = h5readatt('a.h5', '/', 'tags');"
          "nt = h5readatt('a.h5', '/', 'note');"
          "I = h5info('a.h5', '/g/d'); A = I.Attributes;")
    assert w.str("u") == "km" and w.num("ischar(u) && sc == 0.5 && isequal(rg(:)', [1 2 3])")
    assert w.num("isequal(gr, [1 2 3; 4 5 6]) && isa(cn, 'int32') && cn == 4 && isequal(tg(:)', [\"x\" \"yz\"])")
    assert w.num("ischar(nt) && strcmp(nt, 'text') && numel(A) == 2")
    assert w.str("strjoin(sort({A.Name}), ',')") == "scale,units"
    with h5py.File(tmp_path / "a.h5", "r") as f:
        assert f["g/d"].attrs["units"] == "km" and f["g"].attrs["grid"].shape == (3, 2)    # dimensions reversed
        assert f.attrs["count"].dtype == np.int32
    for code, msg in (("h5readatt('a.h5', '/g', 'nope')", "was not found"),
                      ("h5readatt('a.h5', '/missing', 'x')", "not found"),
                      ("h5writeatt('a.h5', '/g', 'bad', {1})", "numeric or text"),
                      ("h5writeatt('a.h5', 'g', 'x', 1)", "starting with '/'")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)


def test_h5info_and_h5disp(w, tmp_path):
    w.run("h5create('i.h5', '/data/temp', [4 Inf], 'ChunkSize', [4 8], 'Deflate', 6, 'Datatype', 'single');"
          "h5write('i.h5', '/data/temp', single(ones(4, 3)), [1 1], [4 3]);"
          "h5create('i.h5', '/data/ids', 5, 'Datatype', 'uint16'); h5create('i.h5', '/top', [2 3]);"
          "h5create('i.h5', '/s', [1 2], 'Datatype', 'string');"
          "h5writeatt('i.h5', '/data', 'source', 'sensor'); h5writeatt('i.h5', '/', 'version', 2);"
          "I = h5info('i.h5'); G = I.Groups(1); T = h5info('i.h5', '/data/temp');")
    assert w.str("G.Name") == "/data" and w.num("numel(I.Groups) == 1 && numel(I.Datasets) == 2")
    assert w.str("strjoin(sort({G.Datasets.Name}), ',')") == "ids,temp" and w.str("I.Attributes(1).Name") == "version"
    assert w.str("G.Attributes(1).Value") == "sensor" and w.num("I.Attributes(1).Value == 2")
    assert w.num("isequal(T.Dataspace.Size, [4 3]) && isequal(T.Dataspace.MaxSize(1), 4) && "
                 "isinf(T.Dataspace.MaxSize(2)) && isequal(T.ChunkSize, [4 8]) && T.Datatype.Size == 4")
    assert w.str("T.Datatype.Class") == "H5T_FLOAT" and w.str("T.Datatype.Type") == "H5T_IEEE_F32LE"
    assert w.str("T.Filters(1).Name") == "deflate" and w.num("T.Filters(1).Data == 6")
    assert w.num("isfield(T, 'Filename') && isfield(T, 'FillValue') && isfield(I, 'Filename')")
    assert w.str("I.Datasets(1).Name") in ("s", "top")
    w.run("Sd = h5info('i.h5', '/s');")
    assert w.str("Sd.Datatype.Class") == "H5T_STRING" and w.num("isequal(Sd.Dataspace.Size, [1 2])")
    out = w.run("h5disp('i.h5')")
    for line in ("HDF5 i.h5", "Group '/'", "'version':  2", "Group '/data'", "'source':  'sensor'",
                 "Dataset 'temp'", "Size:  4x3", "MaxSize:  4xInf", "Datatype:   H5T_IEEE_F32LE (single)",
                 "ChunkSize:  4x8", "Filters:  deflate(6)", "Dataset 'ids'", "Size:  5",
                 "Datatype:   H5T_STD_U16LE (uint16)", "Dataset 'top'", "Size:  2x3", "ChunkSize:  []",
                 "Filters:  none"):
        assert line in out, line
    part = w.run("h5disp('i.h5', '/data/ids')")
    assert "Dataset 'ids'" in part and "temp" not in part
    for code, msg in (("h5info('i.h5', '/nothing')", "not found"), ("h5disp('nofile.h5')", "does not exist"),
                      ("h5info('i.h5', 'data')", "starting with '/'")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)


def test_h5_and_v73_files_are_the_same_kind(w, tmp_path):
    """A version 7.3 MAT-file is an HDF5 file: the h5 functions read it, with MATLAB's dimension order."""
    w.run("M = [1 2 3; 4 5 6]; save('v.mat', 'M', '-v7.3'); R = h5read('v.mat', '/M');"
          "c = h5readatt('v.mat', '/M', 'MATLAB_class'); I = h5info('v.mat');")
    assert w.num("isequal(R, M)") and w.str("c") == "double" and w.str("I.Datasets(1).Name") == "M"


# -- without h5py ---------------------------------------------------------------------------------------

def _without_h5py(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "h5py":
            raise ImportError("no h5py")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_in_the_browser_a_missing_h5py_is_explained(w, monkeypatch):
    import sys
    _without_h5py(monkeypatch)
    monkeypatch.setattr(sys, "platform", "emscripten")
    with pytest.raises(MatlabError, match="browser version of mylab could not load") as e:
        w.run("h5read('a.h5', '/d')")
    assert e.value.identifier == "mylab:hdf5:notInstalled" and "pip" not in str(e.value)
    with pytest.raises(MatlabError, match=r"Version 7.3 MAT-files .* need the Python package 'h5py'"):
        w.run("x = 1; save('f.mat', 'x', '-v7.3')")


def test_matfile_without_h5py_creates_a_version_7_file(w, monkeypatch, tmp_path):
    _without_h5py(monkeypatch)
    w.run("n = matfile('plain.mat'); n.a = 5; T = load('plain.mat');")
    assert w.num("T.a == 5") and (tmp_path / "plain.mat").read_bytes()[:10] == b"MATLAB 5.0"
