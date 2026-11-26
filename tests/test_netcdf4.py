"""NetCDF-4 files (HDF5 storage): nccreate, ncwrite, ncread, ncinfo, ncdisp, ncreadatt, ncwriteatt.

The storage layout is checked with plain h5py against the published NetCDF-4 conventions (dimension
scales, _Netcdf4Dimid, DIMENSION_LIST, _NCProperties, _nc3_strict). Where the netCDF4 package is installed,
files are also exchanged with it; it is never required.
"""

import h5py  # a dependency of mylab: these tests are never skipped
import numpy as np
import pytest

from mylab.errors import MatlabError

DIM_TAG = b"This is a netCDF dimension but not a netCDF variable."


@pytest.fixture
def w(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return m


def _build(w):
    w.run("nccreate('a.nc', 'temp', 'Dimensions', {'x', 3, 't', Inf}, 'Format', 'netcdf4', 'FillValue', -999,"
          "  'DeflateLevel', 4, 'Shuffle', true, 'ChunkSize', [3 2]);"
          "ncwrite('a.nc', 'temp', [1 2 3; 4 5 6]');"
          "nccreate('a.nc', 'x', 'Dimensions', {'x'}, 'Datatype', 'int32'); ncwrite('a.nc', 'x', int32([10 20 30]));"
          "nccreate('a.nc', '/grp/sub/gv', 'Dimensions', {'x', 3, 'y', 2}, 'Datatype', 'int16');"
          "ncwrite('a.nc', '/grp/sub/gv', int16([1 2; 3 4; 5 6]));"
          "nccreate('a.nc', 'names', 'Dimensions', {'x'}, 'Datatype', 'string');"
          "ncwrite('a.nc', 'names', [\"a\" \"bb\" \"ccc\"]);"
          "nccreate('a.nc', 'word', 'Dimensions', {'n', 5}, 'Datatype', 'char'); ncwrite('a.nc', 'word', 'hello');"
          "nccreate('a.nc', 'scal', 'Datatype', 'single'); ncwrite('a.nc', 'scal', 2.5);"
          "nccreate('a.nc', 'big', 'Dimensions', {'x'}, 'Datatype', 'uint64');"
          "ncwriteatt('a.nc', '/', 'title', 'test file'); ncwriteatt('a.nc', 'temp', 'units', 'K');"
          "ncwriteatt('a.nc', '/grp', 'levels', int8([1 2 3]));"
          "ncwriteatt('a.nc', '/grp/sub/gv', 'tags', [\"p\" \"qq\"]);")


def test_round_trip_of_types_groups_and_attributes(w):
    _build(w)
    w.run("T = ncread('a.nc', 'temp'); X = ncread('a.nc', 'x'); G = ncread('a.nc', '/grp/sub/gv');"
          "N = ncread('a.nc', 'names'); C = ncread('a.nc', 'word'); s = ncread('a.nc', 'scal');"
          "B = ncread('a.nc', 'big'); u = ncreadatt('a.nc', 'temp', 'units'); t = ncreadatt('a.nc', '/', 'title');"
          "lv = ncreadatt('a.nc', '/grp', 'levels'); tg = ncreadatt('a.nc', 'grp/sub/gv', 'tags');")
    assert w.num("isequal(T, [1 4; 2 5; 3 6]) && isa(X, 'int32') && isequal(X, int32([10; 20; 30]))")
    assert w.num("isequal(G, int16([1 2; 3 4; 5 6])) && isequal(N, [\"a\"; \"bb\"; \"ccc\"])")
    assert w.num("isequal(C, ('hello').')")
    assert w.num("isa(s, 'single') && s == 2.5 && isa(B, 'uint64') && all(B == intmax('uint64') - 1)")
    assert w.num("strcmp(u, 'K') && strcmp(t, 'test file') && isequal(lv, int8([1 2 3]))")
    assert w.num("isequal(tg, [\"p\" \"qq\"])")


def test_start_count_stride_and_unlimited_growth(w):
    w.run("A = reshape(1:24, 4, 6); nccreate('s.nc', 'A', 'Dimensions', {'r', 4, 'c', 6}, 'Format', 'netcdf4');"
          "ncwrite('s.nc', 'A', A); p = ncread('s.nc', 'A', [2 3], [2 2]);"
          "q = ncread('s.nc', 'A', [1 1], [2 Inf], [3 2]);"
          "e = ncread('s.nc', 'A', [4 6], [1 1]);"
          "nccreate('s.nc', 'R', 'Dimensions', {'r', 4, 'time', Inf, 'run', Inf}, 'Datatype', 'int32');"
          "ncwrite('s.nc', 'R', int32(ones(4, 2, 1)), [1 1 1]);"
          "ncwrite('s.nc', 'R', int32(2 * ones(4, 1, 2)), [1 4 2]);"
          "R = ncread('s.nc', 'R'); ncwrite('s.nc', 'A', [-1 -2 -3], [1 1], [1 2]); A2 = ncread('s.nc', 'A');"
          "info = ncinfo('s.nc', 'R');")
    assert w.num("isequal(p, A(2:3, 3:4)) && isequal(q, A(1:3:4, 1:2:6)) && e == 24")
    r = w.num("R")
    assert r.shape == (4, 4, 3) and r.dtype == np.int32
    assert (r[:, :2, 0] == 1).all() and (r[:, 3, 1:] == 2).all() and (r[:, 2, :] == -2147483647).all()
    assert w.num("isequal(A2(1, [1 3 5]), [-1 -2 -3]) && isequal(A2(2:end, :), A(2:end, :))")
    assert w.num("isequal(info.Size, [4 4 3]) && all([info.Dimensions(2:3).Unlimited]) && "
                 "~info.Dimensions(1).Unlimited")


def test_storage_follows_the_netcdf4_conventions(w, tmp_path):
    _build(w)
    with h5py.File(tmp_path / "a.nc", "r") as f:
        props = f.attrs["_NCProperties"]
        assert props.startswith(b"version=2,") and b"hdf5=" in props and "_nc3_strict" not in f.attrs
        t = f["t"]                                                   # a dimension without a variable
        assert t.attrs["CLASS"] == b"DIMENSION_SCALE" and t.attrs["NAME"].startswith(DIM_TAG)
        assert t.maxshape == (None,) and t.attrs["_Netcdf4Dimid"].dtype == np.int32
        x = f["x"]                                                   # a coordinate variable is its own scale
        assert x.attrs["CLASS"] == b"DIMENSION_SCALE" and x.attrs["NAME"] == b"x" and x[()].tolist() == [10, 20, 30]
        temp = f["temp"]
        assert temp.shape == (2, 3) and temp.maxshape == (None, 3) and temp.chunks == (2, 3)
        assert temp.compression == "gzip" and temp.compression_opts == 4 and temp.shuffle
        assert temp.fillvalue == -999 and temp.attrs["_FillValue"].tolist() == [-999.0]
        assert [temp.dims[k][0].name for k in range(2)] == ["/t", "/x"]
        ids = {n: int(f[n].attrs["_Netcdf4Dimid"]) for n in ("x", "t", "grp/sub/y", "n")}
        assert sorted(ids.values()) == [0, 1, 2, 3]                  # unique in the whole file
        assert temp.attrs["_Netcdf4Coordinates"].tolist() == [ids["t"], ids["x"]]
        gv = f["grp/sub/gv"]
        assert gv.shape == (2, 3) and [gv.dims[k][0].name for k in range(2)] == ["/grp/sub/y", "/x"]
        assert gv[()].tolist() == [[1, 3, 5], [2, 4, 6]]
        assert h5py.check_string_dtype(f["names"].dtype).length is None
        assert f["word"].dtype == np.dtype("S1") and b"".join(f["word"][()].tolist()) == b"hello"
        assert f["scal"].shape == () and f["big"].fillvalue == 18446744073709551614
        assert f.attrs["title"] == b"test file"


def test_classic_model_files(w, tmp_path):
    w.run("nccreate('c.nc', 'v', 'Dimensions', {'r', 2, 'rec', Inf}, 'Format', 'netcdf4_classic', 'Datatype', 'int16');"
          "ncwrite('c.nc', 'v', int16([1 2; 3 4])); v = ncread('c.nc', 'v'); i = ncinfo('c.nc');")
    assert w.num("isequal(v, int16([1 2; 3 4])) && strcmp(i.Format, 'netcdf4_classic') && isempty(i.Groups)")
    with h5py.File(tmp_path / "c.nc", "r") as f:
        assert int(f.attrs["_nc3_strict"]) == 1
    for code, msg in (("nccreate('c.nc', '/g/v2', 'Dimensions', {'r'})", "cannot have groups"),
                      ("nccreate('c.nc', 'u', 'Dimensions', {'r'}, 'Datatype', 'uint8')", "netcdf4_classic"),
                      ("nccreate('c.nc', 'w', 'Dimensions', {'k', Inf, 'r'})", "last dimension"),
                      ("ncwriteatt('c.nc', '/', 'big', int64(3))", "netcdf4_classic"),
                      ("nccreate('c.nc', 'z', 'Format', 'netcdf4')", "cannot be changed")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)


def test_ncinfo_and_ncdisp(w):
    _build(w)
    w.run("i = ncinfo('a.nc'); vi = ncinfo('a.nc', 'temp'); gi = ncinfo('a.nc', '/grp');"
          "names = {i.Variables.Name}; dims = {i.Dimensions.Name};")
    assert w.num("strcmp(i.Format, 'netcdf4') && strcmp(i.Name, '/') && isequal(names, "
                 "{'temp', 'x', 'names', 'word', 'scal', 'big'}) && isequal(dims, {'x', 't', 'n'})")
    assert w.num("isequal(vi.Size, [3 2]) && strcmp(vi.Datatype, 'double') && vi.FillValue == -999 && "
                 "vi.DeflateLevel == 4 && vi.Shuffle && isequal(vi.ChunkSize, [3 2]) && "
                 "isequal({vi.Dimensions.Name}, {'x', 't'}) && [vi.Dimensions.Unlimited] * [1; 2] == 2")
    assert w.num("strcmp(i.Groups(1).Name, 'grp') && strcmp(i.Groups(1).Groups(1).Name, 'sub') && "
                 "strcmp(i.Groups(1).Groups(1).Variables(1).Name, 'gv') && strcmp(gi.Name, 'grp') && "
                 "strcmp(gi.Attributes(1).Name, 'levels')")
    out = w.run("ncdisp('a.nc')")
    for piece in ("Format:", "netcdf4", "t = 2  (UNLIMITED)", "temp", "Size:       3x2", "Dimensions: x,t",
                  "units      = 'K'", "Groups:", "/grp/sub", "gv", "Datatype:   int16", "title = 'test file'"):
        assert piece in out, piece
    one = w.run("ncdisp('a.nc', '/grp/sub/gv')")
    assert "gv" in one and "temp" not in one


def test_errors(w):
    _build(w)
    for code, msg in (("ncread('a.nc', 'nothere')", "no variable named"),
                      ("ncread('a.nc', '/nogroup/v')", "no group"),
                      ("nccreate('a.nc', 'temp')", "already exists"),
                      ("nccreate('a.nc', 'q', 'Dimensions', {'x', 4})", "another length"),
                      ("nccreate('a.nc', 'q', 'Dimensions', {'zz'})", "does not exist yet"),
                      ("nccreate('a.nc', 'q', 'Datatype', 'cell')", "unknown Datatype"),
                      ("nccreate('new.nc', 'q', 'Format', 'hdf5')", "not supported"),
                      ("ncwrite('a.nc', 'x', int32([1 2 3 4]))", "does not fit"),
                      ("ncwrite('a.nc', 'names', [1 2 3])", "holds strings"),
                      ("ncwrite('a.nc', 'temp', 'abc')", "must not be text"),
                      ("ncread('a.nc', 'temp', [4 1], [1 1])", "beyond the length"),
                      ("ncread('a.nc', 'temp', [1 1], [2 3])", "reach beyond"),
                      ("ncwriteatt('a.nc', 'temp', 'DIMENSION_LIST', 1)", "reserved"),
                      ("nccreate('a.nc', 'q', 'Dimensions', {'x'}, 'Format', 'netcdf4', 'DeflateLevel', 12)",
                       "between 0 and 9")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)


def test_fill_values_scale_and_offset(w, tmp_path):
    w.run("nccreate('f.nc', 'p', 'Dimensions', {'x', 4}, 'Datatype', 'int16', 'FillValue', -1, 'Format', 'netcdf4');"
          "ncwriteatt('f.nc', 'p', 'scale_factor', 0.5); ncwriteatt('f.nc', 'p', 'add_offset', 100);"
          "ncwrite('f.nc', 'p', [100.5 NaN 102], [1]); p = ncread('f.nc', 'p');"
          "nccreate('f.nc', 'd', 'Dimensions', {'x'}); d = ncread('f.nc', 'd');"
          "nccreate('f.nc', 'nf', 'Dimensions', {'x'}, 'FillValue', 'disable'); ni = ncinfo('f.nc', 'nf');")
    assert w.num("isequal(p(1), 100.5) && isnan(p(2)) && p(3) == 102 && isnan(p(4)) && all(isnan(d))")
    assert w.num("strcmp(ni.FillValue, 'disable')")
    with h5py.File(tmp_path / "f.nc", "r") as f:
        assert f["p"][()].tolist() == [1, -1, 4, -1] and "_FillValue" not in f["nf"].attrs


def test_coordinate_variable_after_its_dimension(w, tmp_path):
    w.run("nccreate('k.nc', 'v', 'Dimensions', {'lat', 3}, 'Format', 'netcdf4'); ncwrite('k.nc', 'v', [7 8 9]);"
          "nccreate('k.nc', 'lat', 'Dimensions', {'lat'}); ncwrite('k.nc', 'lat', [-10 0 10]);"
          "v = ncread('k.nc', 'v'); lat = ncread('k.nc', 'lat'); i = ncinfo('k.nc');")
    assert w.num("isequal(v, [7; 8; 9]) && isequal(lat, [-10; 0; 10]) && numel(i.Dimensions) == 1 && "
                 "numel(i.Variables) == 2")
    with h5py.File(tmp_path / "k.nc", "r") as f:
        assert f["lat"].attrs["NAME"] == b"lat" and f["v"].dims[0][0].name == "/lat"


def test_reads_files_laid_out_by_other_writers(w, tmp_path):
    with h5py.File(tmp_path / "o.nc", "w") as f:          # the layout the NetCDF library writes
        f.attrs["_NCProperties"] = np.bytes_(b"version=2,netcdf=4.9.2,hdf5=1.14.3")
        f.attrs["history"] = np.bytes_(b"made by h5py")
        tdim = f.create_dataset("time", shape=(0,), maxshape=(None,), dtype=">f4", chunks=(1,))
        tdim.make_scale((DIM_TAG + b"         0").decode())
        tdim.attrs["_Netcdf4Dimid"] = np.int32(0)
        lon = f.create_dataset("lon", data=np.array([0.0, 90.0, 180.0]))
        lon.make_scale("lon")
        lon.attrs["_Netcdf4Dimid"] = np.int32(1)
        v = f.create_dataset("v", data=np.arange(6, dtype=np.float32).reshape(2, 3), maxshape=(None, 3), chunks=(1, 3))
        v.dims[0].attach_scale(tdim)
        v.dims[1].attach_scale(lon)
        v.attrs["_FillValue"] = np.array([np.float32(-1)])
        plain = f.create_dataset("plain", data=np.arange(4, dtype=np.uint16))  # no dimension scales
        plain.attrs["long_name"] = np.bytes_(b"no dims")
        g = f.create_group("g")
        g.create_dataset("s", data=np.array(["one", "two"], dtype=h5py.string_dtype()))
    w.run("v = ncread('o.nc', 'v'); i = ncinfo('o.nc'); p = ncread('o.nc', 'plain'); s = ncread('o.nc', '/g/s');"
          "h = ncreadatt('o.nc', '/', 'history'); vi = ncinfo('o.nc', 'v');")
    assert w.num("isa(v, 'single') && isequal(v, single([0 3; 1 4; 2 5])) && isequal(p, uint16(0:3)')")
    assert w.num("isequal(s, [\"one\"; \"two\"]) && strcmp(h, 'made by h5py') && isequal(vi.Size, [3 2])")
    assert w.num("isequal({vi.Dimensions.Name}, {'lon', 'time'}) && vi.Dimensions(2).Unlimited && "
                 "vi.Dimensions(2).Length == 2 && strcmp(i.Format, 'netcdf4')")


def test_classic_files_are_unchanged(w, tmp_path):
    w.run("nccreate('cl.nc', 'v', 'Dimensions', {'x', 3}); ncwrite('cl.nc', 'v', [1 2 3]); v = ncread('cl.nc', 'v');")
    assert (tmp_path / "cl.nc").read_bytes()[:4] == b"CDF\x01" and w.num("isequal(v, [1; 2; 3])")
    with pytest.raises(MatlabError, match="NetCDF-4 files only"):
        w.run("nccreate('cl.nc', 'w', 'Dimensions', {'x'}, 'DeflateLevel', 2)")


# -- exchange with the netCDF4 package (optional) ----------------------------------------------------

def test_files_written_here_open_in_netcdf4(w, tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    _build(w)
    w.run("nccreate('c.nc', 'v', 'Dimensions', {'r', 2, 'rec', Inf}, 'Format', 'netcdf4_classic');"
          "ncwrite('c.nc', 'v', [1 2; 3 4]);")
    with netCDF4.Dataset(tmp_path / "a.nc") as d:
        assert d.data_model == "NETCDF4" and list(d.dimensions) == ["x", "t", "n"]
        assert d.dimensions["t"].isunlimited() and len(d.dimensions["t"]) == 2
        assert d["temp"].dimensions == ("t", "x") and d["temp"][:].tolist() == [[1, 2, 3], [4, 5, 6]]
        assert d["temp"].units == "K" and d.title == "test file" and d["x"][:].tolist() == [10, 20, 30]
        assert d["names"][:].tolist() == ["a", "bb", "ccc"] and float(d["scal"][...]) == 2.5
        gv = d["grp"]["sub"]["gv"]
        assert gv.dimensions == ("y", "x") and gv[:].tolist() == [[1, 3, 5], [2, 4, 6]]
        assert list(gv.tags) == ["p", "qq"]
    with netCDF4.Dataset(tmp_path / "c.nc") as d:
        assert d.data_model == "NETCDF4_CLASSIC" and d["v"][:].tolist() == [[1, 3], [2, 4]]
    with netCDF4.Dataset(tmp_path / "a.nc", "a") as d:  # the NetCDF library can add to the file
        d.createDimension("z", 2)
        d.createVariable("added", "f4", ("z", "x"))[:] = np.arange(6).reshape(2, 3)
        d["temp"][2, :] = [7, 8, 9]
    w.run("A = ncread('a.nc', 'added'); T = ncread('a.nc', 'temp');")
    assert w.num("isequal(A, single([0 3; 1 4; 2 5])) && isequal(T, [1 4 7; 2 5 8; 3 6 9])")


def test_files_written_by_netcdf4_open_here(w, tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(tmp_path / "n.nc", "w", format="NETCDF4") as d:
        d.setncattr("title", "from netCDF4")
        d.createDimension("x", 3)
        d.createDimension("t", None)
        v = d.createVariable("temp", "f8", ("t", "x"), fill_value=-999.0, zlib=True, complevel=2)
        v.units = "degC"
        v[0:2, :] = np.arange(6).reshape(2, 3)
        d.createVariable("x", "i4", ("x",))[:] = [5, 6, 7]
        d.createVariable("u", "u1", ("x",))
        g = d.createGroup("grp")
        g.createDimension("y", 2)
        g.createVariable("gv", "i2", ("y", "x"))[:] = [[1, 2, 3], [4, 5, 6]]
        g.createVariable("s", str, ("y",))[:] = np.array(["p", "qr"], dtype=object)
    w.run("T = ncread('n.nc', 'temp'); x = ncread('n.nc', 'x'); u = ncread('n.nc', 'u'); g = ncread('n.nc', '/grp/gv');"
          "s = ncread('n.nc', '/grp/s'); i = ncinfo('n.nc'); un = ncreadatt('n.nc', 'temp', 'units');"
          "part = ncread('n.nc', 'temp', [2 1], [2 2], [1 1]);")
    assert w.num("isequal(T, [0 3; 1 4; 2 5]) && isequal(x, int32([5; 6; 7])) && isequal(u, uint8([255; 255; 255]))")
    assert w.num("isequal(g, int16([1 4; 2 5; 3 6])) && isequal(s, [\"p\"; \"qr\"]) && strcmp(un, 'degC')")
    assert w.num("isequal(part, [1 4; 2 5]) && strcmp(i.Format, 'netcdf4') && strcmp(i.Groups(1).Name, 'grp')")
    assert w.num("isequal({i.Dimensions.Name}, {'x', 't'}) && i.Dimensions(2).Unlimited")
    w.run("ncwrite('n.nc', 'temp', [9 9 9]', [1 3]); ncwriteatt('n.nc', '/grp', 'note', 'added here');")
    with netCDF4.Dataset(tmp_path / "n.nc") as d:
        assert d["temp"].shape == (3, 3) and d["temp"][2, :].tolist() == [9, 9, 9] and d["grp"].note == "added here"
