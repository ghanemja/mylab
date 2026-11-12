"""Binary and text file identifiers, paths, the operating system and compatibility functions."""

import gzip
import os
import struct
import tarfile
import zipfile

import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture
def f(m, tmp_path):
    """Session with `d` (a temporary folder) and `p` (a file name in it) defined."""
    m.interp.set_var("d", str(tmp_path))
    m.interp.set_var("p", str(tmp_path / "data.bin"))
    m.path = tmp_path / "data.bin"
    m.dir = tmp_path
    return m


def scalar(m, expr):
    return float(np.asarray(m.num(expr)).ravel()[0])


# -- fwrite / fread ------------------------------------------------------------------------------

@pytest.mark.parametrize("precision,code", [
    ("uint8", "B"), ("int8", "b"), ("uint16", "H"), ("int16", "h"), ("uint32", "I"), ("int32", "i"),
    ("uint64", "Q"), ("int64", "q"), ("single", "f"), ("float32", "f"), ("double", "d"), ("float64", "d"),
    ("short", "h"), ("int", "i"), ("float", "f"), ("uchar", "B"), ("ushort", "H"), ("real*8", "d"),
])
def test_fwrite_bytes_match_struct(f, precision, code):
    values = [0, 1, 2, 100, 127] if code in "bB" else [0, 1, 2, 1000, 31000]
    f.run(f"fid = fopen(p, 'w'); n = fwrite(fid, {values}, '{precision}'); fclose(fid);")
    assert scalar(f, "n") == 5
    assert f.path.read_bytes() == struct.pack("<5" + code, *values)
    f.run(f"fid = fopen(p, 'w', 'b'); fwrite(fid, {values}, '{precision}'); fclose(fid);")
    assert f.path.read_bytes() == struct.pack(">5" + code, *values)
    f.run(f"fid = fopen(p, 'w'); fwrite(fid, {values}, '{precision}', 0, 'ieee-be'); fclose(fid);")
    assert f.path.read_bytes() == struct.pack(">5" + code, *values)
    f.run(f"fid = fopen(p, 'r', 'ieee-be'); [x, c] = fread(fid, Inf, '{precision}'); fclose(fid);")
    assert f.num("x").ravel().tolist() == values and scalar(f, "c") == 5 and f.num("x").dtype == np.float64


def test_fwrite_defaults_rounding_and_saturation(f):
    f.run("fid = fopen(p, 'w'); fwrite(fid, [1.5 2.4 -3 300 255]); fwrite(fid, 'AZ'); "
          "fwrite(fid, [40000 -40000 1.5], 'int16'); fwrite(fid, uint8(200), 'int8'); fclose(fid);")
    assert f.path.read_bytes() == bytes([2, 2, 0, 255, 255]) + b"AZ" + struct.pack("<3h", 32767, -32768, 2) + b"\x7f"
    f.run("fid = fopen(p, 'w'); fwrite(fid, [1 2; 3 4], 'uint8'); fwrite(fid, pi, 'double'); "
          "fwrite(fid, single(2.5), 'float32'); fwrite(fid, true); fclose(fid);")
    assert f.path.read_bytes() == bytes([1, 3, 2, 4]) + struct.pack("<df", np.pi, 2.5) + b"\x01"
    f.run("fid = fopen(p, 'w'); fwrite(fid, intmax('uint64'), 'uint64'); "
          "fwrite(fid, intmin('int64'), 'int64'); fwrite(fid, intmax('uint64'), 'int32'); fclose(fid);")
    assert f.path.read_bytes() == struct.pack("<Qqi", 2 ** 64 - 1, -2 ** 63, 2 ** 31 - 1)
    f.run("fid = fopen(p); a = fread(fid, 1, '*uint64'); b = fread(fid, 1, 'int64=>int64'); fclose(fid);")
    assert int(f.num("a")[0, 0]) == 2 ** 64 - 1 and int(f.num("b")[0, 0]) == -2 ** 63     # no detour via double


def test_fread_sizes(f):
    f.path.write_bytes(bytes(range(1, 11)))
    f.run("fid = fopen(p); a = fread(fid, 3); b = fread(fid, [2 2]); c = fread(fid, [2 Inf]); e = fread(fid, 4); "
          "at_end = feof(fid); fclose(fid);")
    assert f.num("a").tolist() == [[1], [2], [3]]
    assert f.num("b").tolist() == [[4, 6], [5, 7]]
    assert f.num("c").tolist() == [[8, 10], [9, 0]]                 # the last column is zero-filled
    assert f.num("e").shape == (0, 0) and scalar(f, "at_end") == 1
    f.run("fid = fopen(p); [x, n] = fread(fid, [3 4]); fclose(fid);")
    assert f.num("x").shape == (3, 4) and scalar(f, "n") == 10 and f.num("x")[:, 3].tolist() == [10, 0, 0]
    f.run("fid = fopen(p); [x, n] = fread(fid, 100, 'uint16'); fclose(fid);")
    assert f.num("x").ravel().tolist() == list(struct.unpack("<5H", bytes(range(1, 11)))) and scalar(f, "n") == 5
    f.run("fid = fopen(p); x = fread(fid, [1 4])'; y = fread(fid, 0); fclose(fid);")
    assert f.num("x").shape == (4, 1) and f.num("y").size == 0


def test_fread_precision_conversions(f):
    f.path.write_bytes(struct.pack("<4h", -2, 300, 7, -32768) + b"hey" + struct.pack("<2f", 1.5, -2.25))
    f.run("fid = fopen(p); a = fread(fid, 2, 'int16'); b = fread(fid, 1, '*int16'); c = fread(fid, 1, 'int16=>int8'); "
          "s = fread(fid, 3, 'uint8=>char')'; x = fread(fid, 1, 'float32=>double'); y = fread(fid, 1, '*single'); "
          "fclose(fid);")
    assert f.num("a").dtype == np.float64 and f.num("a").ravel().tolist() == [-2, 300]
    assert f.num("b").dtype == np.int16 and scalar(f, "b") == 7
    assert f.num("c").dtype == np.int8 and scalar(f, "c") == -128          # saturates
    assert f.str("s") == "hey" and f.num("s").dtype.kind == "U"
    assert f.num("x").dtype == np.float64 and scalar(f, "x") == 1.5
    assert f.num("y").dtype == np.float32 and scalar(f, "y") == -2.25
    f.path.write_bytes(b"AB\x00\x01\xff")
    f.run("fid = fopen(p); s = fread(fid, 2, '*char')'; t = fread(fid, 2, 'char'); u = fread(fid, 1, '*uint8'); "
          "fclose(fid);")
    assert f.str("s") == "AB" and f.num("t").ravel().tolist() == [0, 1]
    assert f.num("u").dtype == np.uint8 and scalar(f, "u") == 255
    f.run("fid = fopen(p); z = fread(fid, Inf, 'uint8=>logical')'; fclose(fid);")
    assert f.num("z").dtype == bool and f.num("z").tolist() == [[True, True, False, True, True]]
    with pytest.raises(MatlabError, match="precision"):
        f.run("fid = fopen(p); fread(fid, 1, 'uint7');")
    with pytest.raises(MatlabError, match="not supported"):
        f.run("fread(fid, 1, 'ubit3');")
    f.run("fclose('all');")


def test_fread_big_endian_and_per_call_format(f):
    f.path.write_bytes(struct.pack(">2I", 1, 0x01020304) + struct.pack("<d", 2.5))
    f.run("fid = fopen(p, 'r', 'b'); a = fread(fid, 2, 'uint32'); b = fread(fid, 1, 'double', 0, 'l'); "
          "[name, perm, fmt, enc] = fopen(fid); fclose(fid);")
    assert f.num("a").ravel().tolist() == [1, 0x01020304] and scalar(f, "b") == 2.5
    assert f.str("name") == str(f.path) and f.str("perm") == "r" and f.str("fmt") == "ieee-be"
    assert f.str("enc") == "UTF-8"
    f.run("fid = fopen(p, 'r', 'l'); a = fread(fid, 1, 'uint32', 'ieee-be'); fclose(fid);")
    assert scalar(f, "a") == 1
    with pytest.raises(MatlabError, match="machine format"):
        f.run("fopen(p, 'r', 'vaxd')")


def test_skip(f):
    f.path.write_bytes(bytes(range(20)))
    f.run("fid = fopen(p); a = fread(fid, 4, 'uint8', 2)'; pos = ftell(fid); fclose(fid);")
    assert f.num("a").tolist() == [[0, 3, 6, 9]] and scalar(f, "pos") == 12     # the last skip counts too
    f.run("fid = fopen(p); a = fread(fid, Inf, '2*uint8', 3)'; fclose(fid);")
    assert f.num("a").tolist() == [[0, 1, 5, 6, 10, 11, 15, 16]]
    f.run("fid = fopen(p); a = fread(fid, 3, '2*uint16=>uint16', 1)'; fclose(fid);")
    assert f.num("a").tolist() == [list(struct.unpack("<3H", bytes([0, 1, 2, 3, 5, 6])))]
    f.run("fid = fopen(p, 'w'); fwrite(fid, [1 2 3], 'uint8', 2); fclose(fid);")
    assert f.path.read_bytes() == bytes([0, 0, 1, 0, 0, 2, 0, 0, 3])
    f.path.write_bytes(b"\xaa" * 6)
    f.run("fid = fopen(p, 'r+'); fwrite(fid, [1 2], 'uint8', 2); fclose(fid);")
    assert f.path.read_bytes() == bytes([0xAA, 0xAA, 1, 0xAA, 0xAA, 2])       # skipped bytes are kept


def test_seek_tell_rewind(f):
    f.path.write_bytes(bytes(range(10)))
    f.run("""
fid = fopen(p, 'r');
s1 = fseek(fid, 4, 'bof'); a = fread(fid, 1); t1 = ftell(fid);
s2 = fseek(fid, 2, 'cof'); b = fread(fid, 1);
s3 = fseek(fid, -1, 'eof'); c = fread(fid, 1);
s4 = fseek(fid, 0, -1); e = fread(fid, 1);
fseek(fid, -2, 1); g = fread(fid, 1);
fseek(fid, -1, 0); h = fread(fid, 1);
bad1 = fseek(fid, 50, 'bof'); t2 = ftell(fid); msg = ferror(fid);
bad2 = fseek(fid, -1, 'bof');
frewind(fid); t3 = ftell(fid); z = fread(fid, 1);
fseek(fid, 0, 'eof'); t4 = ftell(fid); end1 = feof(fid);
fclose(fid);
""")
    got = [scalar(f, n) for n in ("s1", "a", "t1", "s2", "b", "s3", "c", "s4", "e", "g", "h")]
    assert got == [0, 4, 5, 0, 7, 0, 9, 0, 0, 8, 8]
    assert scalar(f, "bad1") == -1 and scalar(f, "bad2") == -1 and scalar(f, "t2") == 9
    assert "Offset" in f.str("msg")
    assert scalar(f, "t3") == 0 and scalar(f, "z") == 0 and scalar(f, "t4") == 10 and scalar(f, "end1") == 1
    with pytest.raises(MatlabError, match="origin"):
        f.run("fid = fopen(p); fseek(fid, 0, 'middle');")
    f.run("fclose('all')")


def test_read_write_modes(f):
    f.run("fid = fopen(p, 'w+'); fwrite(fid, 1:5); frewind(fid); a = fread(fid)'; fclose(fid);")
    assert f.num("a").tolist() == [[1, 2, 3, 4, 5]]
    f.run("fid = fopen(p, 'a'); fwrite(fid, [6 7]); fclose(fid);")
    assert f.path.read_bytes() == bytes([1, 2, 3, 4, 5, 6, 7])
    f.run("fid = fopen(p, 'r+'); fseek(fid, 2, 'bof'); fwrite(fid, 99); fseek(fid, 0, 'bof'); a = fread(fid, 4)'; "
          "fwrite(fid, 55); fclose(fid);")
    assert f.num("a").tolist() == [[1, 2, 99, 4]] and f.path.read_bytes() == bytes([1, 2, 99, 4, 55, 6, 7])
    f.run("fid = fopen(p, 'a+'); fwrite(fid, 8); fseek(fid, 0, 'bof'); a = fread(fid, 2)'; fclose(fid);")
    assert f.num("a").tolist() == [[1, 2]] and f.path.read_bytes()[-1] == 8
    for perm in ("rb", "wb", "ab", "rt", "wt", "at", "r+b", "rb+", "w+t"):
        f.run(f"fid = fopen(p, '{perm}'); ok = fid >= 3; fclose(fid);")
        assert scalar(f, "ok") == 1, perm
    f.run("fid = fopen(p, 'wb'); fclose(fid);")
    assert f.path.read_bytes() == b""
    with pytest.raises(MatlabError, match="permission"):
        f.run("fopen(p, 'x')")


def test_record_file_round_trip(f):
    """A small header + matrix format, the way data loggers are usually read."""
    f.run("""
A = reshape(1:12, 3, 4) / 4;
fid = fopen(p, 'w', 'ieee-be');
fwrite(fid, 'MYL1', 'char');
fwrite(fid, size(A), 'uint32');
fwrite(fid, A, 'double');
fwrite(fid, int16([-5 5]), 'int16');
fclose(fid);
fid = fopen(p, 'r', 'ieee-be');
magic = fread(fid, [1 4], '*char');
sz = fread(fid, [1 2], 'uint32');
B = fread(fid, sz, 'double');
tail = fread(fid, [1 2], 'int16=>int16');
rest = fread(fid);
fclose(fid);
""")
    raw = f.path.read_bytes()
    A = np.arange(1, 13).reshape(3, 4, order="F") / 4
    assert raw == b"MYL1" + struct.pack(">2I", 3, 4) + A.astype(">f8").tobytes(order="F") + struct.pack(">2h", -5, 5)
    assert f.str("magic") == "MYL1" and np.array_equal(f.num("B"), A)
    assert f.num("tail").tolist() == [[-5, 5]] and f.num("rest").size == 0


# -- text on file identifiers ---------------------------------------------------------------------

def test_text_functions_share_handles_with_binary_ones(f):
    f.run("""
fid = fopen(p, 'w');
fprintf(fid, 'first line\\n');
fprintf(fid, '%d,%d\\n', [1 2; 3 4]);
fwrite(fid, [104 105 10]);
nbytes = fprintf(fid, 'last');
fclose(fid);
fid = fopen(p, 'rb');
l1 = fgetl(fid); pos = ftell(fid);
l2 = fgets(fid);
b = fread(fid, 2, 'uint8=>char')';
l3 = fgetl(fid); l4 = fgetl(fid); e1 = feof(fid);
l5 = fgetl(fid); l6 = fgetl(fid); e2 = feof(fid); l7 = fgetl(fid); l8 = fgets(fid);
frewind(fid); again = fgetl(fid);
fclose(fid);
""")
    assert f.path.read_bytes() == b"first line\n1,3\n2,4\nhi\nlast"
    assert f.str("l1") == "first line" and scalar(f, "pos") == 11 and f.str("l2") == "1,3\n"
    assert f.str("b") == "2," and f.str("l3") == "4" and f.str("l4") == "hi" and scalar(f, "e1") == 0
    assert f.str("l5") == "last" and scalar(f, "l6") == -1 and scalar(f, "e2") == 1
    assert scalar(f, "l7") == -1 and scalar(f, "l8") == -1 and f.str("again") == "first line"
    assert scalar(f, "nbytes") == 4


def test_line_reading_details(f):
    f.path.write_bytes("a\r\nbb\n\ndéjà vu\nend".encode("utf-8"))
    f.run("fid = fopen(p); c = {}; while ~feof(fid), c{end+1} = fgetl(fid); end; fclose(fid);")
    assert [f.str(f"c{{{k}}}") for k in range(1, 6)] == ["a", "bb", "", "déjà vu", "end"]
    f.run("fid = fopen(p); g1 = fgets(fid); g2 = fgets(fid, 1); g3 = fgets(fid); fclose(fid);")
    assert f.str("g1") == "a\r\n" and f.str("g2") == "b" and f.str("g3") == "b\n"
    f.run("fid = fopen(p); n = 0; t = fgetl(fid); while ischar(t), n = n + 1; t = fgetl(fid); end; fclose(fid);")
    assert scalar(f, "n") == 5
    f.path.write_bytes(b"caf\xe9\n")
    f.run("fid = fopen(p, 'r', 'n', 'ISO-8859-1'); s = fgetl(fid); fclose(fid);")
    assert f.str("s") == "café"
    f.run("fid = fopen(p, 'w', 'n', 'ISO-8859-1'); fprintf(fid, '%s', s); fwrite(fid, s, 'char'); fclose(fid);")
    assert f.path.read_bytes() == b"caf\xe9caf\xe9"
    f.run("fid = fopen(p, 'w'); fwrite(fid, s, 'char'); fwrite(fid, s); fclose(fid);")
    assert f.path.read_bytes() == "café".encode("utf-8") + b"caf\xe9"
    with pytest.raises(MatlabError, match="encoding"):
        f.run("fopen(p, 'r', 'n', 'klingon')")


def test_fscanf_textscan_and_textread(f):
    f.path.write_text("x y name\n1 2.5 ab\n3 4.5 cd\n")
    f.run("fid = fopen(p); hdr = fgetl(fid); C = textscan(fid, '%f %f %s'); fclose(fid);")
    assert f.num("C{1}").ravel().tolist() == [1, 3] and f.num("C{2}").ravel().tolist() == [2.5, 4.5]
    assert f.str("C{3}{2}") == "cd"
    f.path.write_text("1 2 3\n4 5 6\n")
    f.run("fid = fopen(p); A = fscanf(fid, '%d', [3 Inf])'; fclose(fid); T = textread(p);")
    assert f.num("A").tolist() == [[1, 2, 3], [4, 5, 6]] and f.num("T").tolist() == [[1, 2, 3], [4, 5, 6]]
    f.run("[a, b, c] = textread(p, '%f %f %f');")
    assert f.num("a").ravel().tolist() == [1, 4] and f.num("c").ravel().tolist() == [3, 6]
    f.path.write_text("name,score\nann,3.5\nbob,4\n")
    f.run("[names, scores] = textread(p, '%s %f', 'delimiter', ',', 'headerlines', 1);")
    assert f.str("names{2}") == "bob" and f.num("scores").ravel().tolist() == [3.5, 4]
    f.run("first = textread(p, '%s', 1, 'delimiter', ',', 'headerlines', 1);")
    assert f.str("first{1}") == "ann" and scalar(f, "numel(first)") == 1
    with pytest.raises(MatlabError):
        f.run("textread('no_such_file_anywhere.txt', '%f')")


def test_standard_streams_and_fopen_forms(f):
    assert f.run("fwrite(1, 'hi there'); fwrite(2, [33 10]);") == "hi there!\n"
    assert f.run("n = fwrite(1, 'abc');") == "abc" and scalar(f, "n") == 3
    assert f.run("fprintf(1, 'a%d\\n', 5); fprintf(2, 'b\\n');") == "a5\nb\n"
    assert f.str("fopen(1)") == "stdout" and f.str("fopen(2)") == "stderr"
    assert f.run("fclose(1); fflush(1);") == ""
    f.run("x = feof(1); e = ferror(1);")
    assert scalar(f, "x") == 0 and f.str("e") == ""
    f.run("[fid, msg] = fopen(fullfile(d, 'missing', 'nope.bin'));")
    assert scalar(f, "fid") == -1 and f.str("msg") != ""
    f.run("a = fopen(p, 'w'); b = fopen(p, 'r'); ids = fopen('all')'; name = fopen(b); fclose('all'); "
          "none = fopen('all'); gone = fopen(a);")
    assert f.num("ids").tolist() == [[3, 4]] and f.str("name") == str(f.path)
    assert f.num("none").size == 0 and f.str("gone") == ""
    f.run("a = fopen(p, 'w'); b = fopen(p, 'r'); st = fclose([a b]);")
    assert scalar(f, "st") == 0
    for bad in ("fread(99)", "fwrite(99, 1)", "ftell(99)", "fseek(99, 0, 'bof')", "frewind(99)", "fgetl(99)",
                "fclose(99)", "feof(99)", "fread('x')"):
        with pytest.raises(MatlabError, match="file identifier"):
            f.run(bad)
    f.run("fflush(1); fflush(3);")


def test_fopen_searches_the_path(f):
    sub = f.dir / "datafolder"
    sub.mkdir()
    (sub / "values.txt").write_text("42\n")
    f.run("addpath(fullfile(d, 'datafolder')); fid = fopen('values.txt'); v = fscanf(fid, '%d'); fclose(fid);")
    assert scalar(f, "v") == 42
    f.run("rmpath(fullfile(d, 'datafolder')); fid = fopen('values.txt');")
    assert scalar(f, "fid") == -1


# -- search path ----------------------------------------------------------------------------------

def test_genpath_and_addpath(f):
    for sub in ("lib", "lib/util", "lib/private", "lib/@cls", "lib/+pkg", "lib/.git", "other"):
        (f.dir / sub).mkdir(parents=True)
    f.run("g = genpath(d);")
    parts = f.str("g").split(os.pathsep)
    assert parts == [str(f.dir), str(f.dir / "lib"), str(f.dir / "lib" / "util"), str(f.dir / "other")]
    assert f.str("genpath(fullfile(d, 'nope'))") == ""
    before = list(f.interp.path)
    f.run("old = addpath(genpath(d));")
    assert f.interp.path[:4] == parts and f.interp.path[4:] == before
    assert f.str("old") == os.pathsep.join(before)
    f.run("rmpath(genpath(d));")
    assert f.interp.path == before
    f.run("addpath(fullfile(d, 'lib'), fullfile(d, 'other'));")
    assert f.interp.path[:2] == [str(f.dir / "lib"), str(f.dir / "other")]
    f.run("addpath(fullfile(d, 'other'), '-end');")
    assert f.interp.path[0] == str(f.dir / "lib") and f.interp.path[-1] == str(f.dir / "other")
    f.run("addpath(d, '-begin'); addpath(fullfile(d, 'lib', 'util'), 1);")
    assert f.interp.path[0] == str(f.dir) and f.interp.path[-1] == str(f.dir / "lib" / "util")
    (f.dir / "lib" / "util" / "helper_fn.m").write_text("function y = helper_fn(x)\ny = x * 2;\nend\n")
    assert scalar(f, "helper_fn(21)") == 42


def test_mfilename_and_which(f):
    (f.dir / "whoami.m").write_text(
        "function [a, b, c] = whoami()\na = mfilename;\nb = mfilename('fullpath');\n"
        "c = mfilename('fullpathext');\nend\n")
    (f.dir / "notes.txt").write_text("x")
    (f.dir / "scr.m").write_text("here = fileparts(mfilename('fullpath'));\nname = mfilename;\n")
    f.run("addpath(d); [a, b, c] = whoami(); run(fullfile(d, 'scr.m'));")
    assert f.str("a") == "whoami" and f.str("b") == str(f.dir / "whoami") and f.str("c") == str(f.dir / "whoami.m")
    assert f.str("here") == str(f.dir) and f.str("name") == "scr"
    assert f.str("mfilename") == ""
    assert f.str("which('whoami')") == str(f.dir / "whoami.m")
    assert f.str("which('whoami.m')") == str(f.dir / "whoami.m")
    assert f.str("which('notes.txt')") == str(f.dir / "notes.txt")
    assert f.str("fileparts(which('whoami'))") == str(f.dir)
    assert f.str("which('sin')").startswith("built-in")
    assert f.str("which('zz_not_a_thing')") == "" and scalar(f, "isempty(which('zz_not_a_thing'))") == 1
    f.run("myvar = 3;")
    assert f.str("which('myvar')") == "variable"
    assert f.run("which sin") == "built-in (mylab/sin)\n"
    assert "not found" in f.run("which zz_not_a_thing")
    assert f.run("which whoami").strip() == str(f.dir / "whoami.m")
    assert scalar(f, "numel(which('sin', '-all'))") == 1
    f.run("w = what(d);")
    assert f.str("w.m{1}") == "scr.m" and f.str("w.path") == str(f.dir)


# -- operating system ---------------------------------------------------------------------------------

@pytest.mark.skipif(os.name != "posix", reason="shell commands differ on Windows")
def test_system(f):
    f.run("[st, out] = system('echo hello');")
    assert scalar(f, "st") == 0 and f.str("out") == "hello\n"
    assert f.run("st = system('echo shown');") == "shown\n" and scalar(f, "st") == 0
    assert f.run("[st, out] = system('echo both', '-echo');") == "both\n" and f.str("out") == "both\n"
    f.run("[st, out] = system('exit 3');")
    assert scalar(f, "st") == 3 and f.str("out") == ""
    f.run("[st, out] = unix('printf abc'); [st2, out2] = dos('printf xyz'); [st3, out3] = system('echo err 1>&2');")
    assert f.str("out") == "abc" and f.str("out2") == "xyz" and f.str("out3") == "err\n"
    f.run("setenv('MYLAB_TEST_VAR', 'value 1'); [~, out] = system('echo $MYLAB_TEST_VAR'); "
          "g = getenv('MYLAB_TEST_VAR');")
    assert f.str("out") == "value 1\n" and f.str("g") == "value 1"
    f.run("setenv('MYLAB_TEST_VAR'); g = getenv('MYLAB_TEST_VAR');")
    assert f.str("g") == ""


def test_copyfile_movefile_ls(f):
    (f.dir / "a.txt").write_text("A")
    (f.dir / "b.txt").write_text("B")
    (f.dir / "sub").mkdir()
    (f.dir / "sub" / "c.dat").write_text("C")
    f.run("copyfile(fullfile(d, 'a.txt'), fullfile(d, 'a2.txt')); [ok, msg] = copyfile(fullfile(d, 'a.txt'), "
          "fullfile(d, 'sub'));")
    assert (f.dir / "a2.txt").read_text() == "A" and (f.dir / "sub" / "a.txt").read_text() == "A"
    assert scalar(f, "ok") == 1 and f.str("msg") == ""
    f.run("copyfile(fullfile(d, '*.txt'), fullfile(d, 'backup')); copyfile(fullfile(d, 'sub'), fullfile(d, 'sub2'));")
    assert sorted(os.listdir(f.dir / "backup")) == ["a.txt", "a2.txt", "b.txt"]
    assert sorted(os.listdir(f.dir / "sub2")) == ["a.txt", "c.dat"]
    f.run("[ok, msg, id] = copyfile(fullfile(d, 'nothing.txt'), d);")
    assert scalar(f, "ok") == 0 and "nothing.txt" in f.str("msg") and f.str("id") != ""
    with pytest.raises(MatlabError, match="nothing.txt"):
        f.run("copyfile(fullfile(d, 'nothing.txt'), d)")
    f.run("movefile(fullfile(d, 'b.txt'), fullfile(d, 'renamed.txt')); st = movefile(fullfile(d, 'a2.txt'), "
          "fullfile(d, 'sub'));")
    assert not (f.dir / "b.txt").exists() and (f.dir / "renamed.txt").read_text() == "B"
    assert (f.dir / "sub" / "a2.txt").exists() and not (f.dir / "a2.txt").exists() and scalar(f, "st") == 1
    f.run("movefile(fullfile(d, 'sub2'), fullfile(d, 'moved'));")
    assert sorted(os.listdir(f.dir / "moved")) == ["a.txt", "c.dat"] and not (f.dir / "sub2").exists()
    with pytest.raises(MatlabError):
        f.run("movefile(fullfile(d, 'ghost'), d)")
    f.run("names = ls(fullfile(d, 'backup')); txt = ls(fullfile(d, '*.txt'));")
    assert ["".join(r).rstrip() for r in f.num("names")] == ["a.txt", "a2.txt", "b.txt"]
    assert f.num("names").shape == (3, 6)
    assert "renamed.txt" in f.run("ls(d)")
    with pytest.raises(MatlabError):
        f.run("ls(fullfile(d, 'ghost'))")


def test_archives(f):
    src = f.dir / "proj"
    (src / "inner").mkdir(parents=True)
    (src / "one.txt").write_text("1" * 500)
    (src / "inner" / "two.m").write_text("x = 2;\n")
    f.run("cd(d); names = zip('out', 'proj'); cd(d);")
    with zipfile.ZipFile(f.dir / "out.zip") as z:
        assert sorted(z.namelist()) == ["proj/inner/two.m", "proj/one.txt"]
    assert scalar(f, "numel(names)") == 2
    f.run("got = unzip(fullfile(d, 'out.zip'), fullfile(d, 'ex'));")
    assert (f.dir / "ex" / "proj" / "inner" / "two.m").read_text() == "x = 2;\n"
    assert scalar(f, "numel(got)") == 2 and f.str("got{1}").startswith(str(f.dir / "ex"))
    f.run("zip(fullfile(d, 'flat.zip'), {'one.txt', fullfile('inner', '*.m')}, fullfile(d, 'proj'));")
    with zipfile.ZipFile(f.dir / "flat.zip") as z:
        assert sorted(z.namelist()) == ["inner/two.m", "one.txt"]
    f.run("gz = gzip(fullfile(d, 'proj', 'one.txt')); gzip(fullfile(d, 'proj', 'one.txt'), fullfile(d, 'gz'));")
    assert gzip.open(src / "one.txt.gz").read() == b"1" * 500 and (f.dir / "gz" / "one.txt.gz").exists()
    assert f.str("gz{1}") == str(src / "one.txt.gz")
    (src / "one.txt").unlink()
    f.run("out = gunzip(fullfile(d, 'proj', 'one.txt.gz')); "
          "gunzip(fullfile(d, 'gz', 'one.txt.gz'), fullfile(d, 'ungz'));")
    assert (src / "one.txt").read_text() == "1" * 500 and (f.dir / "ungz" / "one.txt").exists()
    f.run("tar(fullfile(d, 'p.tar'), 'proj', d); tar(fullfile(d, 'p.tgz'), 'proj', d); "
          "names = untar(fullfile(d, 'p.tgz'), fullfile(d, 'tx'));")
    with tarfile.open(f.dir / "p.tar") as t:
        assert "proj/inner/two.m" in t.getnames()
    assert (f.dir / "tx" / "proj" / "inner" / "two.m").exists() and scalar(f, "numel(names)") == 3
    f.run("untar(fullfile(d, 'p.tar'), fullfile(d, 'tx2'));")
    assert (f.dir / "tx2" / "proj" / "one.txt").exists()
    with pytest.raises(MatlabError, match="no file"):
        f.run("zip(fullfile(d, 'bad.zip'), 'does_not_exist.txt')")
    with pytest.raises(MatlabError):
        f.run("unzip(fullfile(d, 'nope.zip'))")
    (f.dir / "fake.zip").write_text("not a zip")
    with pytest.raises(MatlabError):
        f.run("unzip(fullfile(d, 'fake.zip'))")


def test_unzip_refuses_entries_outside_the_output_folder(f):
    with zipfile.ZipFile(f.dir / "evil.zip", "w") as z:
        z.writestr("../escaped.txt", "x")
    with pytest.raises(MatlabError, match="outside"):
        f.run("unzip(fullfile(d, 'evil.zip'), fullfile(d, 'safe'))")
    assert not (f.dir / "escaped.txt").exists()


# -- the web -------------------------------------------------------------------------------------------

def test_web_functions_with_a_local_server(f):
    import http.server
    import threading
    (f.dir / "page.txt").write_text("hello web")
    (f.dir / "data.json").write_text('{"a": 1}')
    handler_cls = type("Quiet", (http.server.SimpleHTTPRequestHandler,), {
        "log_message": lambda *a, **k: None,
        "__init__": lambda self, *a, **k: http.server.SimpleHTTPRequestHandler.__init__(
            self, *a, directory=str(f.dir), **k)})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        f.interp.set_var("u", f"http://127.0.0.1:{server.server_address[1]}")
        f.run("s = urlread([u '/page.txt']); [s2, ok] = urlread([u '/page.txt']); w = webread([u '/page.txt']);")
        assert f.str("s") == "hello web" and f.str("s2") == "hello web" and scalar(f, "ok") == 1
        assert f.str("w") == "hello web"
        f.run("out = websave(fullfile(d, 'saved.txt'), [u '/page.txt']); [o2, ok2] = urlwrite([u '/page.txt'], "
              "fullfile(d, 'saved2.txt'));")
        assert (f.dir / "saved.txt").read_text() == "hello web" and f.str("out") == str(f.dir / "saved.txt")
        assert (f.dir / "saved2.txt").read_text() == "hello web" and scalar(f, "ok2") == 1
        f.run("q = webread([u '/page.txt'], 'a', 1, 'b', 'x y', weboptions('Timeout', 10));")
        assert f.str("q") == "hello web"
        f.run("[s3, ok3] = urlread([u '/missing.txt']);")
        assert f.str("s3") == "" and scalar(f, "ok3") == 0
        with pytest.raises(MatlabError, match="404"):
            f.run("webread([u '/missing.txt'])")
    finally:
        server.shutdown()
        server.server_close()


def test_web_errors_are_clear(f, monkeypatch):
    with pytest.raises(MatlabError, match="could not reach"):
        f.run("webread('http://127.0.0.1:9/nothing', 'Timeout', 2)")
    f.run("[s, ok] = urlread('http://127.0.0.1:9/nothing'); [n, ok2] = urlwrite('http://127.0.0.1:9/x', p);")
    assert f.str("s") == "" and scalar(f, "ok") == 0 and scalar(f, "ok2") == 0
    with pytest.raises(MatlabError, match="valid URL"):
        f.run("urlread('not a url')")
    from mylab.builtins import fileops
    monkeypatch.setattr(fileops, "IN_BROWSER", True)
    with pytest.raises(MatlabError, match="browser"):
        f.run("websave(p, 'http://example.invalid/x')")
    with pytest.raises(MatlabError, match="browser"):
        f.run("system('echo hi')")


# -- versions and capabilities -----------------------------------------------------------------------

def test_version_and_ver(m):
    v = m.str("version")
    assert v.startswith("9.14.0 (R2023a)") and "mylab" in v
    assert m.str("version('-release')") == "2023a" and "2023" in m.str("version('-date')")
    m.run("[v, dte] = version; parts = sscanf(version, '%d.%d'); r = str2double(version('-release'));")
    assert m.num("parts").ravel().tolist() == [9, 14] and "2023" in m.str("dte")
    assert "mylab version" in m.run("ver")
    m.run("v = ver; names = {v.Name}; mv = ver('matlab'); none = ver('simulink');")
    assert m.str("names{1}") == "MATLAB" and m.str("mv.Version") == "9.14" and m.str("mv.Release") == "(R2023a)"
    assert scalar(m, "isempty(none)") == 1 and scalar(m, "any(strcmp(names, 'Image Processing Toolbox'))") == 1
    with pytest.raises(MatlabError):
        m.run("version('-bogus')")


def test_verlessthan(m):
    assert scalar(m, "verLessThan('matlab', '9.4')") == 0 and scalar(m, "verLessThan('matlab', '8.4.0')") == 0
    assert scalar(m, "verLessThan('matlab', '9.14')") == 0 and scalar(m, "verLessThan('matlab', '9.14.1')") == 1
    assert scalar(m, "verLessThan('matlab', '9.15')") == 1 and scalar(m, "verLessThan('matlab', '10')") == 1
    assert scalar(m, "verLessThan('MATLAB', '7.10')") == 0            # 10 > 9 as numbers, not text
    assert scalar(m, "verLessThan('images', '9.0')") == 0 and scalar(m, "verLessThan('signal', '99')") == 1
    assert scalar(m, "verLessThan('simulink', '1.0')") == 1           # not available: older than anything
    assert scalar(m, "isMATLABReleaseOlderThan('R2021b')") == 0 and scalar(m, "isMATLABReleaseOlderThan('R2023b')") == 1
    with pytest.raises(MatlabError):
        m.run("verLessThan('matlab', 'new')")
    m.run("if verLessThan('matlab', '8.4'), branch = 'old'; else, branch = 'new'; end")
    assert m.str("branch") == "new"


def test_capabilities(m):
    assert scalar(m, "usejava('jvm')") == 0 and scalar(m, "usejava('desktop')") == 0
    assert scalar(m, "license('test', 'image_toolbox')") == 1 and scalar(m, "license('test', 'Simulink')") == 0
    assert scalar(m, "canUseGPU") == 0 and scalar(m, "gpuDeviceCount") == 0
    m.run("x = gpuArray(single([1 2 3])); y = gather(x * 2); c = class(y);")
    assert m.num("y").tolist() == [[2, 4, 6]] and m.str("c") == "single"
    m.run("p = gcp('nocreate'); pool = parpool; pool2 = parpool(4); n = pool.NumWorkers; q = gcp; delete(pool); "
          "delete(gcp('nocreate'));")
    assert scalar(m, "isempty(p)") == 1 and scalar(m, "n") == 1 and scalar(m, "q.NumWorkers") == 1
    assert m.run("parpool; parpool('local', 2); matlabpool open") == ""
    with pytest.raises(MatlabError, match="MEX"):
        m.run("mex myfile.c")
    m.run("[u, s] = memory;")
    assert scalar(m, "isstruct(u) && isfield(u, 'MaxPossibleArrayBytes') && isfield(s, 'PhysicalMemory')") == 1


def test_preferences(m):
    assert scalar(m, "ispref('grp')") == 0 and scalar(m, "ispref('grp', 'a')") == 0
    m.run("setpref('grp', 'a', 5); setpref('grp', {'b', 'c'}, {'text', [1 2 3]}); addpref('grp', 'e', 1);")
    assert scalar(m, "getpref('grp', 'a')") == 5 and m.str("getpref('grp', 'b')") == "text"
    assert scalar(m, "ispref('grp')") == 1 and m.num("ispref('grp', {'a', 'zz'})").tolist() == [[True, False]]
    assert scalar(m, "getpref('grp', 'new', 7)") == 7 and scalar(m, "getpref('grp', 'new')") == 7
    m.run("all = getpref('grp'); vals = getpref('grp', {'a', 'c'});")
    assert m.num("all.c").tolist() == [[1, 2, 3]] and m.num("vals{2}").tolist() == [[1, 2, 3]]
    m.run("rmpref('grp', 'a');")
    assert scalar(m, "ispref('grp', 'a')") == 0
    with pytest.raises(MatlabError, match="no preference"):
        m.run("getpref('grp', 'a')")
    m.run("rmpref('grp');")
    assert scalar(m, "ispref('grp')") == 0
    with pytest.raises(MatlabError):
        m.run("rmpref('grp')")


# -- workspaces ----------------------------------------------------------------------------------------

def test_assignin_and_evalin(m):
    m.run("""
function setter(v)
    assignin('base', 'from_fn', v);
    assignin('caller', 'in_caller', v * 2);
end
function r = reader()
    r = evalin('base', 'base_only') + evalin('caller', 'local_of_caller');
end
function r = outer()
    local_of_caller = 100;
    setter(4);
    r = reader() + in_caller;
end
function bump()
    evalin('caller', 'counter = counter + 1;');
end
""")
    m.run("base_only = 1; r = outer(); counter = 0; bump(); bump();")
    assert scalar(m, "r") == 109 and scalar(m, "from_fn") == 4 and scalar(m, "counter") == 2
    assert scalar(m, "exist('in_caller', 'var')") == 0 and scalar(m, "exist('local_of_caller', 'var')") == 0
    m.run("assignin('base', 'direct', 'abc'); v = evalin('base', 'direct'); "
          "[q, w] = evalin('base', 'size(ones(2, 3))');")
    assert m.str("v") == "abc" and scalar(m, "q") == 2 and scalar(m, "w") == 3
    m.run("fb = evalin('base', 'undefined_thing_zz', '-1');")
    assert scalar(m, "fb") == -1
    assert m.run("evalin('base', 'disp(direct)')") == "abc\n"
    assert scalar(m, "evalin('base', 'exist(''direct'', ''var'')')") == 1
    for bad in ("assignin('nowhere', 'a', 1)", "assignin('base', '1bad', 1)", "evalin('base', 'undefined_thing_zz')",
                "evalin('elsewhere', '1')"):
        with pytest.raises(MatlabError):
            m.run(bad)
    assert len(m.interp.frames) == 1
    assert m.str("inputname(1)") == ""


def test_diary(f):
    f.interp.set_var("log", str(f.dir / "session.log"))
    out = f.run("diary(log); disp('one'); x = 5\ndiary off; disp('hidden'); diary on; fprintf('two\\n'); diary off;")
    assert "one" in out and "hidden" in out and "two" in out          # the screen still shows everything
    text = (f.dir / "session.log").read_text()
    assert "one\n" in text and "x = 5" in text and "two\n" in text and "hidden" not in text
    f.run("diary; disp('toggled on'); diary; disp('toggled off');")
    text = (f.dir / "session.log").read_text()
    assert "toggled on" in text and "toggled off" not in text
    captured = f.run("s = evalc('disp(42)'); disp('after');")
    assert captured == "after\n" and f.str("s").strip() == "42"


def test_diary_default_file_name(m, tmp_path):
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        m.run("diary on; disp('default name'); diary off; disp('not this');")
        assert (tmp_path / "diary").read_text() == "default name\n"
    finally:
        os.chdir(cwd)


# -- compatibility ---------------------------------------------------------------------------------------

def test_old_string_functions(m):
    assert scalar(m, "isstr('abc')") == 1 and scalar(m, "isstr(5)") == 0 and scalar(m, "isstr({'a'})") == 0
    assert m.num("strmatch('ap', {'apple', 'banana', 'ap', 'grape'})").tolist() == [[1], [3]]
    assert m.num("strmatch('ap', {'apple', 'banana', 'ap'}, 'exact')").tolist() == [[3]]
    assert m.num("strmatch('max', strvcat('max', 'minimax', 'maximum'))").tolist() == [[1], [3]]
    assert m.num("strmatch('max', strvcat('max', 'minimax', 'maximum'), 'exact')").tolist() == [[1]]
    assert m.num("strmatch('zz', {'a', 'b'})").size == 0 and scalar(m, "isempty(strmatch('zz', {'a'}))") == 1
    assert m.num("strmatch('', {'a', 'b'})").ravel().tolist() == [1, 2]
    assert m.num("findstr('abcabc', 'bc')").tolist() == [[2, 5]]
    assert m.num("findstr('bc', 'abcabc')").tolist() == [[2, 5]]
    assert m.num("findstr('aaa', 'aa')").tolist() == [[1, 2]] and m.num("findstr('abc', 'x')").size == 0


def test_nargchk(m):
    assert m.str("nargchk(1, 2, 0)") == "Not enough input arguments."
    assert m.str("nargchk(1, 2, 3)") == "Too many input arguments."
    assert scalar(m, "isempty(nargchk(1, 2, 2))") == 1
    m.run("s = nargchk(1, 2, 5, 'struct'); e = nargchk(1, 2, 1, 'struct');")
    assert m.str("s.identifier") == "MATLAB:nargchk:tooManyInputs" and scalar(m, "isempty(e)") == 1
    m.run("function r = needs_two(varargin)\n error(nargchk(2, 2, nargin));\n r = 1;\nend")
    assert scalar(m, "needs_two(1, 2)") == 1
    with pytest.raises(MatlabError, match="Not enough"):
        m.run("needs_two(1)")


def test_shiftdim(m):
    m.run("A = reshape(1:24, 2, 3, 4);")
    A = np.arange(1, 25).reshape(2, 3, 4, order="F").astype(float)
    assert np.array_equal(m.num("shiftdim(A, 1)"), A.transpose(1, 2, 0))
    assert np.array_equal(m.num("shiftdim(A, 2)"), A.transpose(2, 0, 1))
    assert np.array_equal(m.num("shiftdim(A, 3)"), A) and np.array_equal(m.num("shiftdim(A, 0)"), A)
    assert m.num("shiftdim(A, -1)").shape == (1, 2, 3, 4) and m.num("shiftdim(A, -1)")[0, 1, 2, 3] == 24
    m.run("[B, n] = shiftdim(ones(1, 1, 3));")
    assert m.num("B").shape == (3, 1) and scalar(m, "n") == 2
    m.run("[B, n] = shiftdim([1 2 3]);")
    assert m.num("B").tolist() == [[1], [2], [3]] and scalar(m, "n") == 1
    m.run("[B, n] = shiftdim([1; 2]);")
    assert m.num("B").shape == (2, 1) and scalar(m, "n") == 0
    assert m.num("shiftdim([1 2 3], 1)").shape == (3, 1)
    assert np.array_equal(m.num("shiftdim(ones(1, 4, 2))"), np.ones((4, 2)))
    assert m.num("squeeze(shiftdim(mean(A, 3), -1))").shape == (2, 3)


def test_blkdiag(m):
    from scipy.linalg import block_diag
    assert np.array_equal(m.num("blkdiag([1 2; 3 4], 5, [6 7])"), block_diag([[1, 2], [3, 4]], 5, [6, 7]))
    assert np.array_equal(m.num("blkdiag(1, 2, 3)"), np.diag([1, 2, 3]))
    assert m.num("blkdiag(ones(2, 3), zeros(0, 2), 7)").shape == (3, 6)
    assert m.num("blkdiag([1 2])").tolist() == [[1, 2]] and m.num("blkdiag()").size == 0
    assert m.num("blkdiag(1i, 2)").dtype.kind == "c"


def test_getfield_setfield_nested(m):
    m.run("s = struct('a', struct('b', [10 20 30], 'c', {{'x', 'y'}})); s.list(2).v = 7;")
    assert m.num("getfield(s, 'a', 'b')").tolist() == [[10, 20, 30]]
    assert scalar(m, "getfield(s, 'a', 'b', {2})") == 20
    assert scalar(m, "getfield(s, 'list', {2}, 'v')") == 7
    assert scalar(m, "getfield(s, {1}, 'a', 'b', {1, 3})") == 30
    assert m.num("getfield(s, 'a', 'b', {[1 3]})").tolist() == [[10, 30]]
    assert m.num("getfield(s, 'a', 'b', {':'})").ravel().tolist() == [10, 20, 30]
    m.run("t = setfield(s, 'a', 'b', {2}, 99); t = setfield(t, 'new', 'deep', 'er', 1); t = setfield(t, 'top', 'v');")
    assert m.num("t.a.b").tolist() == [[10, 99, 30]] and scalar(m, "t.new.deep.er") == 1 and m.str("t.top") == "v"
    assert m.num("s.a.b").tolist() == [[10, 20, 30]] and scalar(m, "isfield(s, 'new')") == 0   # input unchanged
    m.run("u = setfield(s, 'list', {3}, 'v', 5); w = setfield(struct(), 'k', {2, 2}, 4);")
    assert scalar(m, "numel(u.list)") == 3 and scalar(m, "u.list(3).v") == 5 and scalar(m, "numel(s.list)") == 2
    assert m.num("w.k").tolist() == [[0, 0], [0, 4]]
    m.run("fn = 'dyn'; z = setfield(struct('dyn', 1), fn, 2);")
    assert scalar(m, "getfield(z, fn)") == 2
    with pytest.raises(MatlabError):
        m.run("getfield(s, 'missing')")
    with pytest.raises(MatlabError):
        m.run("getfield(s, 'a', 'nope', 'b')")
    with pytest.raises(MatlabError):
        m.run("getfield(5, 'a', 'b')")


def test_substruct_subsref_subsasgn(m):
    m.run("S = substruct('.', 'a', '()', {2, ':'}); s.a = magic(3); row = subsref(s, S); t = subsasgn(s, S, [0 0 0]);")
    assert m.num("row").tolist() == [[3, 5, 7]] and m.num("t.a")[1].tolist() == [0, 0, 0]
    assert m.num("s.a")[1].tolist() == [3, 5, 7] and m.str("S(2).type") == "()" and m.str("S(1).subs") == "a"
    m.run("c = {1, 'two', [3 4]}; x = subsref(c, substruct('{}', {3})); c2 = subsasgn(c, substruct('{}', {2}), 2);")
    assert m.num("x").tolist() == [[3, 4]] and scalar(m, "c2{2}") == 2 and m.str("c{2}") == "two"
    assert m.num("subsref(magic(4), substruct('()', {[1 2], 1}))").tolist() == [[16], [5]]
    with pytest.raises(MatlabError):
        m.run("substruct('<>', {1})")
    with pytest.raises(MatlabError):
        m.run("subsref(1, 2)")


def test_help_exists_for_new_functions(m):
    from mylab.help import help_text
    for name in ("fread", "fwrite", "fseek", "genpath", "verLessThan", "gpuArray", "inputname", "evalin", "imread",
                 "imresize", "regionprops", "strel"):
        assert "not found" not in help_text(name)
    assert "CPU" in help_text("gpuArray") and "''" in help_text("inputname")


def test_flipdim_swapbytes_strread_gpudevice(m):
    assert m.num("flipdim([1 2 3; 4 5 6], 2)").tolist() == [[3, 2, 1], [6, 5, 4]]
    assert m.num("flipdim([1 2 3; 4 5 6], 1)").tolist() == [[4, 5, 6], [1, 2, 3]]
    assert scalar(m, "swapbytes(uint16(1))") == 256 and m.num("swapbytes(uint16(1))").dtype == np.uint16
    assert scalar(m, "swapbytes(int32(1))") == 2 ** 24 and scalar(m, "swapbytes(uint8(7))") == 7
    assert m.num("swapbytes(swapbytes(uint32([1 2 70000])))").tolist() == [[1, 2, 70000]]
    assert struct.pack(">d", 1.5) == struct.pack("<d", float(m.num("swapbytes(1.5)")[0, 0]))
    m.run("[a, b] = strread('1 2 3 4', '%d %d'); [n, s] = strread('5,abc', '%f %s', 'delimiter', ','); "
          "v = strread('7 8 9');")
    assert m.num("a").ravel().tolist() == [1, 3] and m.num("b").ravel().tolist() == [2, 4]
    assert m.num("a").dtype == np.float64
    assert scalar(m, "n") == 5 and m.str("s{1}") == "abc" and m.num("v").ravel().tolist() == [7, 8, 9]
    m.run("g = gpuDevice; gpuDevice(1); gpuDevice([]);")
    assert scalar(m, "g.DeviceAvailable") == 0


def test_separators(m):
    assert m.str("filesep") == os.sep and m.str("pathsep") == os.pathsep
    assert m.str("['a' filesep 'b']") == "a" + os.sep + "b"
