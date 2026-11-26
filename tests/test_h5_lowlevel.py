"""The low-level HDF5 interface (H5F, H5G, H5D, H5S, H5A, H5T, H5ML), cross-checked with plain h5py.

Dimensions given to and returned by the H5S functions are in the C library's order; MATLAB arrays read
and written have them reversed, so a dataspace [3 5] goes with a 5-by-3 array.
"""

import h5py  # a dependency of mylab: these tests are never skipped
import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture
def w(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return m


CREATE = (
    "fid = H5F.create('ll.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');"
    "gid = H5G.create(fid, 'grp', 'H5P_DEFAULT', 'H5P_DEFAULT', 'H5P_DEFAULT');"
    "sid = H5S.create_simple(2, [3 5], []);"
    "tid = H5T.copy('H5T_NATIVE_DOUBLE');"
    "did = H5D.create(gid, 'data', tid, sid, 'H5P_DEFAULT');"
    "A = reshape(1:15, 5, 3);"
    "H5D.write(did, 'H5ML_DEFAULT', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', A);"
    "asid = H5S.create_simple(1, 2, []);"
    "aid = H5A.create(did, 'pair', 'H5T_NATIVE_INT', asid, 'H5P_DEFAULT');"
    "H5A.write(aid, 'H5ML_DEFAULT', int32([7 8]));"
    "H5A.close(aid); H5S.close(asid); H5D.close(did); H5T.close(tid); H5S.close(sid); H5G.close(gid);"
    "H5F.close(fid);"
)


def test_create_write_and_check_with_h5py(w, tmp_path):
    w.run(CREATE)
    with h5py.File(tmp_path / "ll.h5", "r") as f:
        d = f["grp/data"]
        assert d.shape == (3, 5) and d.dtype == np.float64
        assert d[()].tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        assert d.attrs["pair"].tolist() == [7, 8] and d.attrs["pair"].dtype == np.int32


def test_read_whole_and_hyperslab(w):
    w.run(CREATE)
    w.run("fid = H5F.open('ll.h5'); did = H5D.open(fid, '/grp/data'); B = H5D.read(did);"
          "B2 = H5D.read(did, 'H5T_NATIVE_FLOAT', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT');"
          "space = H5D.get_space(did); [nd, dims, maxdims] = H5S.get_simple_extent_dims(space);"
          "H5S.select_hyperslab(space, 'H5S_SELECT_SET', [1 1], [], [2 3], []);"
          "part = H5D.read(did, 'H5ML_DEFAULT', 'H5S_ALL', space, 'H5P_DEFAULT');"
          "H5S.select_hyperslab(space, 'H5S_SELECT_SET', [0 0], [2 2], [2 3], [1 1]);"
          "strided = H5D.read(did, 'H5ML_DEFAULT', 'H5S_ALL', space, 'H5P_DEFAULT');"
          "aid = H5A.open(did, 'pair'); av = H5A.read(aid); H5A.close(aid);"
          "H5S.close(space); H5D.close(did); H5F.close(fid);")
    assert w.num("isequal(B, A) && isa(B2, 'single') && isequal(B2, single(A))")
    assert w.num("nd == 2 && isequal(dims, [3 5]) && isequal(maxdims, [3 5])")
    assert w.num("isequal(part, A(2:4, 2:3)) && isequal(strided, A(1:2:5, 1:2:3))")
    assert w.num("isequal(av, int32([7; 8]))")


def test_write_a_hyperslab_through_a_memory_space(w, tmp_path):
    w.run(CREATE)
    w.run("fid = H5F.open('ll.h5', 'H5F_ACC_RDWR', 'H5P_DEFAULT'); did = H5D.open(fid, '/grp/data');"
          "fs = H5D.get_space(did); H5S.select_hyperslab(fs, 'H5S_SELECT_SET', [2 0], [1 2], [1 3], [1 1]);"
          "ms = H5S.create_simple(2, [1 3], []);"
          "H5D.write(did, 'H5T_NATIVE_DOUBLE', ms, fs, 'H5P_DEFAULT', [-1 -2 -3]);"
          "H5S.close(ms); H5S.close(fs); H5D.close(did); H5F.close(fid);")
    with h5py.File(tmp_path / "ll.h5", "r") as f:
        assert f["grp/data"][2].tolist() == [-1, 12, -2, 14, -3] and f["grp/data"][1].tolist() == [6, 7, 8, 9, 10]


def test_files_written_by_h5py(w, tmp_path):
    with h5py.File(tmp_path / "py.h5", "w") as f:
        f["i16"] = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        f["vec"] = np.array([1.5, 2.5, 3.5])
        f["txt"] = np.array([b"abc", b"xyz"], dtype="S3")
        f["vs"] = np.array(["one", "three"], dtype=h5py.string_dtype())
        f.attrs["version"] = np.float64(2.5)
        f.attrs["label"] = np.bytes_(b"root")
        f.create_dataset("grow", shape=(2, 3), maxshape=(None, 3), dtype="u1")
    w.run("fid = H5F.open('py.h5', 'H5F_ACC_RDONLY', 'H5P_DEFAULT');"
          "d = H5D.open(fid, 'i16'); X = H5D.read(d); t = H5D.get_type(d); H5T.close(t); H5D.close(d);"
          "d = H5D.open(fid, 'vec'); v = H5D.read(d); H5D.close(d);"
          "d = H5D.open(fid, 'txt'); T = H5D.read(d); H5D.close(d);"
          "d = H5D.open(fid, 'vs'); V = H5D.read(d); H5D.close(d);"
          "d = H5D.open(fid, 'grow'); s = H5D.get_space(d); [n, dims, mx] = H5S.get_simple_extent_dims(s);"
          "H5S.close(s); H5D.close(d);"
          "a = H5A.open(fid, 'version'); ver = H5A.read(a); H5A.close(a);"
          "a = H5A.open(fid, 'label'); lab = H5A.read(a); H5A.close(a); H5F.close(fid);")
    x = w.num("X")
    assert x.shape == (4, 3, 2) and x.dtype == np.int16
    assert (x == np.arange(24, dtype=np.int16).reshape(2, 3, 4).transpose()).all()
    assert w.num("isequal(v, [1.5; 2.5; 3.5]) && isequal(T, ['ax'; 'by'; 'cz']) && isequal(V, {'one'; 'three'})")
    assert w.num("isequal(dims, [2 3]) && isequal(mx, [-1 3]) && n == 2 && ver == 2.5 && strcmp(lab, 'root')")


def test_scalar_and_text_attributes_and_groups(w, tmp_path):
    w.run("fid = H5F.create('t.h5', 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');"
          "g = H5G.create(fid, 'a', 'H5P_DEFAULT', 'H5P_DEFAULT', 'H5P_DEFAULT'); H5G.close(g);"
          "g = H5G.open(fid, '/a'); g2 = H5G.create(g, 'b', 'H5P_DEFAULT', 'H5P_DEFAULT', 'H5P_DEFAULT');"
          "s = H5S.create('H5S_SCALAR'); tt = H5T.copy('H5T_C_S1'); H5T.set_size(tt, 5);"
          "at = H5A.create(g2, 'name', tt, s, 'H5P_DEFAULT'); H5A.write(at, 'H5ML_DEFAULT', 'hello'); H5A.close(at);"
          "an = H5A.create(fid, 'count', 'H5T_STD_U16LE', s, 'H5P_DEFAULT'); H5A.write(an, 'H5ML_DEFAULT', 42);"
          "H5A.close(an); H5T.close(tt); H5S.close(s); H5G.close(g2); H5G.close(g); H5F.close(fid);"
          "back = h5readatt('t.h5', '/a/b', 'name');")
    with h5py.File(tmp_path / "t.h5", "r") as f:
        assert f["a/b"].attrs["name"] == b"hello" and int(f.attrs["count"]) == 42
        assert f.attrs.get_id("count").dtype == np.dtype("<u2")
    assert w.str("back") == "hello"


def test_constants_and_identifiers(w):
    w.run(CREATE)
    w.run("r = H5ML.get_constant_value('H5F_ACC_RDWR'); fid = H5F.open('ll.h5', r, 'H5P_DEFAULT');"
          "c = class(fid); tf = isa(fid, 'H5ML.id'); txt = evalc('disp(fid)'); H5F.close(fid);"
          "u = H5ML.get_constant_value('H5S_UNLIMITED');")
    assert w.str("c") == "H5ML.id" and w.num("tf") and "file identifier" in w.str("txt")
    assert w.num("u == -1")


def test_unlimited_dataspace_and_errors(w):
    w.run(CREATE)
    w.run("s = H5S.create_simple(2, [2 3], [-1 3]); [n, d, mx] = H5S.get_simple_extent_dims(s); H5S.close(s);")
    assert w.num("isequal(d, [2 3]) && isequal(mx, [-1 3])")
    for code, msg in (("H5F.open('missing.h5')", "does not exist"),
                      ("fid = H5F.open('ll.h5'); H5D.open(fid, '/nope')", "unable to open the dataset"),
                      ("fid = H5F.open('ll.h5'); H5F.close(fid); H5D.open(fid, '/grp/data')", "closed"),
                      ("fid = H5F.open('ll.h5', 'H5F_ACC_TRUNC')", "not allowed"),
                      ("fid = H5F.open('ll.h5'); H5S.close(fid)", "dataspace identifier"),
                      ("H5T.copy('H5T_NOT_A_TYPE')", "not a predefined"),
                      ("fid = H5F.open('ll.h5'); H5G.open(fid, 'grp', 'something')", "H5P_DEFAULT"),
                      ("H5S.create_simple(2, [1 2 3], [])", "rank is 2"),
                      ("fid = H5F.open('ll.h5'); d = H5D.open(fid, '/grp/data');"
                       "H5D.write(d, 'H5ML_DEFAULT', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT', 1:3)", "15"),
                      ("H5ML.get_constant_value('NOPE')", "unknown constant"),
                      ("H5F.create('ll.h5', 'H5F_ACC_EXCL', 'H5P_DEFAULT', 'H5P_DEFAULT')", "unable to create")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)
