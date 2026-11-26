"""Data formats: XML documents, struct files, NetCDF, HDF5, video (AVI) and audio (FLAC)."""

import io
import shutil
import struct
import subprocess

import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture
def w(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return m


# -- XML ----------------------------------------------------------------------------------------------

def test_xmlread_dom_methods(w, tmp_path):
    (tmp_path / "x.xml").write_text('<r><i v="1">t</i><i v="2">u</i><j/></r>')
    w.run("d = xmlread('x.xml'); n = d.getElementsByTagName('i'); e = n.item(0); root = d.getDocumentElement;"
          "c = root.getFirstChild;")
    assert w.num("n.getLength()") == 2 and w.num("n.getLength") == 2 and w.num("getLength(n)") == 2
    assert w.str("char(e.getAttribute('v'))") == "1" and w.str("char(e.getTextContent())") == "t"
    assert w.str("char(e.getNodeName)") == "i" and w.str("char(root.getTagName)") == "r"
    assert w.num("root.getChildNodes.getLength == 3 && root.hasChildNodes && ~root.hasAttributes")
    assert w.num("c.getNodeType == c.ELEMENT_NODE && c.getFirstChild.getNodeType == c.TEXT_NODE")
    assert w.str("char(c.getFirstChild.getData)") == "t" and w.str("char(c.getNextSibling.getAttribute('v'))") == "2"
    assert w.num("isempty(n.item(5)) && isempty(root.getLastChild.getFirstChild)")
    assert w.str("char(c.getParentNode.getNodeName)") == "r" and w.num("c.hasAttribute('v') && ~c.hasAttribute('w')")
    assert w.str("char(root.getTextContent)") == "tu" and w.str("char(e.getAttribute('none'))") == ""
    w.run("a = e.getAttributes; first = a.item(0);")
    assert w.num("a.getLength") == 1 and w.str("char(first.getName)") == "v" and w.str("char(first.getValue)") == "1"
    assert w.str("class(d)") == "org.w3c.dom.Document" and w.num("isequal(root, d.getDocumentElement)")


def test_xml_documents_can_be_changed_and_written(w, tmp_path):
    (tmp_path / "x.xml").write_text("<r><i>t</i></r>")
    w.run("d = xmlread('x.xml'); root = d.getDocumentElement; k = d.createElement('k'); k.setAttribute('id', 'new');"
          "k.setAttribute('n', 5); k.appendChild(d.createTextNode('hello')); root.appendChild(k);"
          "root.getFirstChild.setTextContent('changed'); xmlwrite('y.xml', d); s = xmlwrite(d);")
    text = (tmp_path / "y.xml").read_text()
    assert text.startswith('<?xml version="1.0" encoding="utf-8"?>')
    assert "<i>changed</i>" in text and '<k id="new" n="5">hello</k>' in text and "\n   <i>" in text
    assert w.str("s") == text
    w.run("d2 = xmlread('y.xml'); kk = d2.getElementsByTagName('k').item(0); root.removeChild(k);"
          "left = root.getElementsByTagName('k').getLength;")
    assert w.str("char(kk.getAttribute('id'))") == "new" and w.num("left") == 0
    (tmp_path / "bad.xml").write_text("<r><i></r>")
    with pytest.raises(MatlabError, match="mismatched tag"):
        w.run("xmlread('bad.xml')")
    with pytest.raises(MatlabError, match="Unable to find"):
        w.run("xmlread('nofile.xml')")
    with pytest.raises(MatlabError, match="does not apply"):
        w.run("root.getFirstChild.getFirstChild.setAttribute('a', 'b')")


def test_struct_files(w, tmp_path):
    w.run("p.name = \"Ann\"; p.age = 30; p.tags = [\"a\" \"b\"]; p.addr.city = 'Rome'; p.addr.zipAttribute = 123;"
          "p.kids = struct('n', {'x', 'y'}); p.ok = true; p.v = [1.5 2.5];"
          "writestruct(p, 'p.xml', 'StructNodeName', 'person'); q = readstruct('p.xml');"
          "writestruct(p, 'p.json'); j = readstruct('p.json');")
    text = (tmp_path / "p.xml").read_text()
    assert "<person>" in text and '<addr zip="123">' in text and text.count("<tags>") == 2
    assert w.num("q.name == \"Ann\" && q.age == 30 && isequal(q.tags, [\"a\" \"b\"]) && q.addr.city == \"Rome\"")
    assert w.num("q.addr.zipAttribute == 123 && numel(q.kids) == 2 && q.kids(2).n == \"y\" && q.ok == true")
    assert w.num("isequal(q.v, [1.5 2.5]) && islogical(q.ok) && isstring(q.name)")
    assert w.num("j.name == \"Ann\" && j.age == 30 && j.addr.city == \"Rome\" && isequal(j.v(:)', [1.5 2.5])")
    (tmp_path / "m.xml").write_text('<lib><book id="1"><t>A</t><y>2001</y></book><book id="2"><t>B</t>'
                                    "<y>1999</y></book><note lang=\"en\">hi</note></lib>")
    w.run("L = readstruct('m.xml'); N = readstruct('m.xml', 'ImportAttributes', false); B = readstruct('m.xml', "
          "'StructNodeName', 'book');")
    assert w.num("numel(L.book) == 2 && L.book(2).idAttribute == 2 && L.book(1).t == \"A\" && L.book(2).y == 1999")
    assert w.num("L.note.langAttribute == \"en\" && L.note.Text == \"hi\" && N.note == \"hi\" && B.t == \"A\"")
    w.run("s.a = 1; s.b = 'x'; writestruct(s, 's.xml'); r = readstruct('s.xml');")
    assert w.num("r.a == 1 && r.b == \"x\"") and "<struct>" in (tmp_path / "s.xml").read_text()
    with pytest.raises(MatlabError, match="extension"):
        w.run("writestruct(s, 's.dat')")
    with pytest.raises(MatlabError, match="unknown or unsupported option"):
        w.run("readstruct('s.xml', 'Bogus', 1)")


# -- NetCDF -------------------------------------------------------------------------------------------

def test_netcdf_round_trip(w, tmp_path):
    w.run("nccreate('a.nc', 'v', 'Dimensions', {'x', 3}); ncwrite('a.nc', 'v', [1 2 3]); v = ncread('a.nc', 'v');"
          "nccreate('a.nc', 'M', 'Dimensions', {'r', 2, 'c', 4}, 'Datatype', 'int16');"
          "ncwrite('a.nc', 'M', int16([1 2 3 4; 5 6 7 8])); M = ncread('a.nc', 'M');"
          "p = ncread('a.nc', 'M', [2 2], [1 2]); q = ncread('a.nc', 'M', [1 1], [Inf Inf], [1 2]);")
    assert w.num("v").tolist() == [[1], [2], [3]] and w.num("M").dtype == np.int16
    assert w.num("M").tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert w.num("p").tolist() == [[6, 7]] and w.num("q").tolist() == [[1, 3], [5, 7]]
    w.run("nccreate('a.nc', 'T', 'Dimensions', {'x', 't', Inf}); ncwrite('a.nc', 'T', [1; 2; 3], [1 1]);"
          "ncwrite('a.nc', 'T', [4; 5; 6], [1 2]); T = ncread('a.nc', 'T'); nccreate('a.nc', 'u');"
          "u0 = ncread('a.nc', 'u'); ncwrite('a.nc', 'u', 7); u = ncread('a.nc', 'u');"
          "nccreate('a.nc', 'name', 'Dimensions', {'n', 5}, 'Datatype', 'char'); ncwrite('a.nc', 'name', 'hello');"
          "name = ncread('a.nc', 'name')'; nccreate('a.nc', 'w', 'Dimensions', {'x'}); w0 = ncread('a.nc', 'w');")
    assert w.num("T").tolist() == [[1, 4], [2, 5], [3, 6]] and w.num("isnan(u0) && u == 7")
    assert w.str("name") == "hello" and w.num("all(isnan(w0)) && numel(w0) == 3")
    w.run("ncwriteatt('a.nc', 'v', 'units', 'm/s'); ncwriteatt('a.nc', '/', 'title', 'demo');"
          "ncwriteatt('a.nc', 'v', 'scale', 2.5); a1 = ncreadatt('a.nc', 'v', 'units');"
          "a2 = ncreadatt('a.nc', '/', 'title'); a3 = ncreadatt('a.nc', 'v', 'scale'); I = ncinfo('a.nc');"
          "V = ncinfo('a.nc', 'T'); names = {I.Variables.Name};")
    assert w.str("a1") == "m/s" and w.str("a2") == "demo" and w.num("a3") == 2.5
    assert w.num("numel(I.Variables) == 6 && isequal(I.Variables(strcmp(names, 'M')).Size, [2 4])")
    assert w.str("I.Format") == "classic" and w.num("isequal(V.Size, [3 2]) && V.Dimensions(2).Unlimited")
    assert w.str("V.Datatype") == "double" and w.str("I.Attributes(1).Name") == "title"
    out = w.run("ncdisp('a.nc')")
    assert "UNLIMITED" in out and "units = 'm/s'" in out and "Dimensions: r,c" in out
    # an independent reader sees the same file: the dimensions are stored in the opposite order
    from scipy.io import netcdf_file
    with netcdf_file(str(tmp_path / "a.nc"), "r", mmap=False) as f:
        assert f.variables["M"].dimensions == ("c", "r")
        assert f.variables["M"][:].tolist() == [[1, 5], [2, 6], [3, 7], [4, 8]]
        assert f.variables["T"][:].tolist() == [[1, 2, 3], [4, 5, 6]] and f.variables["v"].units == b"m/s"
        assert f.variables["u"].getValue() == 7 and f.title == b"demo"


def test_netcdf_scale_fill_and_errors(w, tmp_path):
    w.run("nccreate('s.nc', 'p', 'Dimensions', {'x', 3}, 'Datatype', 'int16', 'FillValue', -999);"
          "ncwriteatt('s.nc', 'p', 'scale_factor', 0.1); ncwriteatt('s.nc', 'p', 'add_offset', 10);"
          "ncwrite('s.nc', 'p', [10.5 NaN 12]); p = ncread('s.nc', 'p');")
    assert np.allclose(w.num("p").ravel()[[0, 2]], [10.5, 12]) and np.isnan(w.num("p")[1, 0])
    from scipy.io import netcdf_file
    with netcdf_file(str(tmp_path / "s.nc"), "r", mmap=False) as f:
        assert f.variables["p"][:].tolist() == [5, -999, 20]         # stored as (value - offset) / scale
    for code, msg in (("ncread('s.nc', 'zz')", "no variable named 'zz'"),
                      ("nccreate('s.nc', 'q', 'Datatype', 'uint64')", "cannot be stored in a classic NetCDF"),
                      ("nccreate('s.nc', 'p')", "already exists"),
                      ("ncwrite('s.nc', 'p', [1 2 3 4])", "does not fit"),
                      ("ncread('s.nc', 'p', 5)", "beyond the length"),
                      ("nccreate('s.nc', 'r', 'Format', 'netcdf4')", "not supported"),
                      ("ncread('nofile.nc', 'p')", "does not exist"),
                      ("nccreate('s.nc', 'r', 'Dimensions', {'t', Inf, 'x'})", "must be the last")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)


# -- HDF5 ---------------------------------------------------------------------------------------------

def test_hdf5(w, tmp_path):
    h5py = pytest.importorskip("h5py")
    w.run("M = [1 2 3; 4 5 6]; h5create('a.h5', '/d', [2 3]); h5write('a.h5', '/d', M); R = h5read('a.h5', '/d');"
          "h5create('a.h5', '/g/i8', [1 4], 'Datatype', 'int8'); h5write('a.h5', '/g/i8', int8([1 2 3 4]));"
          "x = h5read('a.h5', '/g/i8'); h5create('a.h5', '/ext', [Inf 2], 'ChunkSize', [5 2], 'Deflate', 4);"
          "h5write('a.h5', '/ext', [1 2; 3 4], [1 1], [2 2]); h5write('a.h5', '/ext', [5 6], [3 1], [1 2]);"
          "E = h5read('a.h5', '/ext'); part = h5read('a.h5', '/d', [1 2], [2 2]); st = h5read('a.h5', '/d', [1 1], "
          "[1 2], [1 2]); h5create('a.h5', '/v', 5); h5write('a.h5', '/v', 1:5); v = h5read('a.h5', '/v');"
          "h5create('a.h5', '/s', [1 2], 'Datatype', 'string'); h5write('a.h5', '/s', [\"ab\" \"cde\"]);"
          "s = h5read('a.h5', '/s');")
    assert w.num("isequal(R, M) && isa(x, 'int8') && isequal(x, int8([1 2 3 4])) && isequal(E, [1 2; 3 4; 5 6])")
    assert w.num("isequal(part, [2 3; 5 6]) && isequal(st, [1 3]) && isequal(v, (1:5)')")
    assert w.num("isequal(s, [\"ab\" \"cde\"])")
    with h5py.File(tmp_path / "a.h5", "r") as f:
        assert f["d"].shape == (3, 2) and f["d"][()].tolist() == [[1, 4], [2, 5], [3, 6]]   # reversed dimensions
        assert f["ext"].maxshape == (2, None) and f["ext"].compression == "gzip" and f["ext"].chunks == (2, 5)
        assert f["d"].chunks is None and f["g/i8"].dtype == np.int8
    w.run("h5writeatt('a.h5', '/d', 'units', 'm'); h5writeatt('a.h5', '/', 'version', 3);"
          "h5writeatt('a.h5', '/g', 'vec', [1 2 3]); u = h5readatt('a.h5', '/d', 'units');"
          "ver = h5readatt('a.h5', '/', 'version'); vec = h5readatt('a.h5', '/g', 'vec'); I = h5info('a.h5');"
          "D = h5info('a.h5', '/ext'); names = {I.Datasets.Name};")
    assert w.str("u") == "m" and w.num("ver") == 3 and w.num("vec").ravel().tolist() == [1, 2, 3]
    assert w.str("strjoin(sort(names), ',')") == "d,ext,s,v" and w.str("I.Groups(1).Name") == "/g"
    assert w.num("isequal(D.ChunkSize, [5 2]) && isequal(D.Dataspace.Size, [3 2]) && isinf(D.Dataspace.MaxSize(1))")
    assert w.str("D.Filters.Name") == "deflate" and w.str("D.Datatype.Class") == "H5T_FLOAT"
    out = w.run("h5disp('a.h5')")
    assert "Dataset 'ext'" in out and "MaxSize:  Infx2" in out and "'units':  'm'" in out
    for code, msg in (("h5create('a.h5', '/d', [2 2])", "already exists"), ("h5read('a.h5', '/nope')", "not found"),
                      ("h5write('a.h5', '/d', [1 2])", "does not match"),
                      ("h5create('a.h5', '/z', [Inf 2])", "needs a ChunkSize"),
                      ("h5create('a.h5', 'rel', 3)", "starting with '/'"),
                      ("h5read('nofile.h5', '/d')", "does not exist")):
        with pytest.raises(MatlabError, match=msg):
            w.run(code)


def test_hdf5_without_h5py_says_what_is_missing(w, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "h5py":
            raise ImportError("no h5py")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(MatlabError, match="optional Python package 'h5py'") as e:
        w.run("h5create('a.h5', '/d', [2 3])")
    assert e.value.identifier == "mylab:hdf5:notInstalled"


# -- video --------------------------------------------------------------------------------------------

def test_video_written_and_read_without_ffmpeg(w, tmp_path, monkeypatch):
    from mylab.builtins import plotting
    monkeypatch.setattr(plotting, "_find_ffmpeg", lambda: None)     # prove that no external program is used
    w.run("v = VideoWriter('v.avi'); open(v); writeVideo(v, uint8(zeros(32, 32, 3)));"
          "writeVideo(v, uint8(255 * ones(32, 32, 3))); close(v); r = VideoReader('v.avi'); f = readFrame(r);"
          "more = hasFrame(r); g = readFrame(r); done = ~hasFrame(r); A = read(r); B = read(r, 2);"
          "C = read(r, [1 Inf]);")
    assert w.num("f").shape == (32, 32, 3) and w.num("f").dtype == np.uint8 and w.num("f").max() < 5
    assert w.num("more && done && min(g(:)) > 250 && isequal(size(A), [32 32 3 2]) && isequal(B, g) && isequal(A, C)")
    assert w.num("r.NumFrames == 2 && r.FrameRate == 30 && r.Height == 32 && r.Width == 32")
    assert abs(w.num("r.Duration").item() - 2 / 30) < 1e-12
    w.run("r.CurrentTime = 0; again = hasFrame(r);")
    assert w.num("again")
    with pytest.raises(MatlabError, match="No more frames"):
        w.run("readFrame(r); readFrame(r); readFrame(r);")
    data = (tmp_path / "v.avi").read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"AVI " and b"MJPG" in data[:256] and b"idx1" in data
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    # each frame is an ordinary JPEG image
    from PIL import Image
    start = data.index(b"00dc")
    size = struct.unpack_from("<I", data, start + 4)[0]
    assert Image.open(io.BytesIO(data[start + 8:start + 8 + size])).size == (32, 32)


def test_uncompressed_and_grayscale_avi_are_exact(w):
    w.run("[X, Y] = meshgrid(1:48, 1:40); F = uint8(cat(3, 4 * X, 5 * Y, 100 + 0 * X));"
          "v = VideoWriter('u.avi', 'Uncompressed AVI'); v.FrameRate = 12; open(v);"
          "for k = 1:5, writeVideo(v, F + uint8(k)); end; close(v); r = VideoReader('u.avi'); f3 = read(r, 3);"
          "G = uint8(repmat(0:4:159, 30, 1)); g = VideoWriter('g.avi', 'Grayscale AVI'); open(g); writeVideo(g, G);"
          "close(g); rg = VideoReader('g.avi'); fg = readFrame(rg);"
          "q = VideoWriter('q.avi'); q.Quality = 95; open(q); writeVideo(q, F); close(q);"
          "fq = readFrame(VideoReader('q.avi'));")
    assert w.num("r.NumFrames == 5 && r.FrameRate == 12 && r.Height == 40 && r.Width == 48")
    assert w.num("isequal(f3, F + uint8(3))")
    assert w.num("isequal(fg, G)") and w.str("rg.VideoFormat") == "Grayscale"
    err = np.abs(w.num("double(fq)") - w.num("double(F)"))
    assert err.max() < 12 and err.mean() < 2                          # JPEG is close, not exact
    with pytest.raises(MatlabError, match="was not found"):
        w.run("VideoReader('nofile.avi')")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_avi_files_agree_with_ffmpeg(w, tmp_path):
    w.run("v = VideoWriter('m.avi'); v.FrameRate = 10; open(v);"
          "for k = 1:4, writeVideo(v, uint8(60 * k * ones(24, 40, 3))); end; close(v);")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(tmp_path / "m.avi"), "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-"], capture_output=True)
    frames = np.frombuffer(r.stdout, dtype=np.uint8).reshape(-1, 24, 40, 3)
    assert frames.shape[0] == 4 and abs(float(frames[2].mean()) - 180) < 3 and b"10 fps" in r.stderr
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
                    "-frames:v", "7", "-vcodec", "mjpeg", str(tmp_path / "ff.avi")], check=True)
    w.run("r = VideoReader('ff.avi'); f = readFrame(r);")
    assert w.num("r.NumFrames == 7 && r.FrameRate == 10 && r.Height == 48 && r.Width == 64")
    assert w.num("isequal(size(f), [48 64 3])")


# -- audio --------------------------------------------------------------------------------------------

def test_flac_round_trip_is_exact_to_the_sample(w, tmp_path):
    w.run("y = sin(2 * pi * 440 * (0:799)' / 8000); audiowrite('t.flac', y, 8000); [z, fs] = audioread('t.flac');"
          "n = audioread('t.flac', 'native'); p = audioread('t.flac', [11 20]);"
          "st = [y, 0.5 * cos(2 * pi * 100 * (0:799)' / 8000)]; audiowrite('s.flac', st, 44100, 'BitsPerSample', 24);"
          "[z2, fs2] = audioread('s.flac'); i = audioinfo('s.flac'); audiowrite('a.flac', zeros(100, 1), 8000);"
          "[z0, f0] = audioread('a.flac'); audiowrite('i.flac', int16([100; -200; 300]), 8000);"
          "ni = audioread('i.flac', 'native');")
    y = np.sin(2 * np.pi * 440 * np.arange(800) / 8000)
    expected = np.clip(np.round(y * 32768), -32768, 32767)
    assert w.num("fs") == 8000 and np.array_equal(w.num("double(n)").ravel(), expected)      # lossless
    assert np.array_equal(w.num("z").ravel(), expected / 32768)
    assert np.array_equal(w.num("p").ravel(), expected[10:20] / 32768)
    assert w.num("fs2 == 44100 && isequal(size(z2), [800 2])") and np.abs(w.num("z2 - st")).max() <= 1.0 / 2 ** 23
    assert w.str("i.CompressionMethod") == "FLAC" and w.num("i.BitsPerSample == 24 && i.NumChannels == 2")
    assert w.num("i.TotalSamples == 800 && i.SampleRate == 44100 && f0 == 8000 && isequal(z0, zeros(100, 1))")
    assert w.num("isa(ni, 'int16') && isequal(ni, int16([100; -200; 300]))")
    data = (tmp_path / "t.flac").read_bytes()
    assert data[:4] == b"fLaC" and len(data) < 800 * 2          # smaller than the plain samples
    with pytest.raises(MatlabError, match="was not found"):
        w.run("audioread('nofile.flac')")
    with pytest.raises(MatlabError, match="Supported files"):
        w.run("audiowrite('x.m4a', y, 8000)")


def test_flac_codec_against_reference_decoder(tmp_path):
    from mylab.builtins import _flac
    rng = np.random.default_rng(3)
    for bits in (8, 16, 24):
        hi = 1 << (bits - 1)
        data = rng.integers(-hi, hi, (9000, 2))
        data[:3000] //= 37                                             # a quiet part, which compresses
        blob = _flac.encode(data, 44100, bits)
        back, rate, got_bits = _flac.decode(blob)
        assert rate == 44100 and got_bits == bits and np.array_equal(back, data)
        if shutil.which("ffmpeg"):
            path = tmp_path / f"t{bits}.flac"
            path.write_bytes(blob)
            r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "s32le", "-acodec",
                                "pcm_s32le", "-"], capture_output=True)
            ref = np.frombuffer(r.stdout, dtype="<i4").reshape(-1, 2).astype(np.int64) >> (32 - bits)
            assert r.returncode == 0 and np.array_equal(ref, data)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_flac_from_another_encoder(w, tmp_path):
    # ffmpeg's encoder uses linear prediction and stereo decorrelation, which mylab's own does not
    from scipy.io import wavfile
    t = np.arange(6000) / 8000
    stereo = np.stack([np.sin(2 * np.pi * 440 * t) * 12000, np.sin(2 * np.pi * 300 * t) * 9000], axis=1)
    stereo = stereo.astype(np.int16)
    wavfile.write(tmp_path / "src.wav", 8000, stereo)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "src.wav"), "-compression_level", "8",
                    str(tmp_path / "ff.flac")], check=True)
    w.run("[n, fs] = audioread('ff.flac', 'native');")
    assert w.num("fs") == 8000 and np.array_equal(w.num("n"), stereo)


def test_formats_that_need_soundfile(w):
    try:
        import soundfile  # noqa: F401
        have = True
    except (ImportError, OSError):
        have = False
    if have:
        w.run("y = sin(2 * pi * 440 * (0:7999)' / 8000) / 2; audiowrite('x.ogg', y, 8000);"
              "[z, fs] = audioread('x.ogg'); i = audioinfo('x.ogg');")
        assert w.num("fs == 8000 && abs(numel(z) - 8000) < 200") and w.str("i.CompressionMethod") == "Vorbis"
    else:
        with pytest.raises(MatlabError, match="optional Python package 'soundfile'") as e:
            w.run("audiowrite('x.ogg', zeros(10, 1), 8000)")
        assert e.value.identifier == "mylab:audio:soundfileMissing"
