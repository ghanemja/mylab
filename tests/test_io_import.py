"""Text import: textscan, readtable/readmatrix/readcell, import options, datastores, bit fields, memmapfile."""

import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture
def w(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return m


def cols(m, expr):
    """The cells of a textscan result as Python lists."""
    m.run(f"C__ = {expr};")
    out = []
    for k in range(int(m.num("numel(C__)").item())):
        if m.num(f"iscell(C__{{{k + 1}}})"):
            count = int(m.num(f"numel(C__{{{k + 1}}})").item())
            out.append([m.str(f"C__{{{k + 1}}}{{{i + 1}}}") for i in range(count)])
        else:
            out.append(m.num(f"double(C__{{{k + 1}}})").ravel().tolist())
    return out


def test_textscan_delimiters_and_line_ends(m):
    assert cols(m, "textscan('1,2;3,4', '%f %f', 'Delimiter', ',', 'EndOfLine', ';')") == [[1, 3], [2, 4]]
    assert cols(m, "textscan(sprintf('1 2 3\\n4 5\\n6 7 8'), '%f %f %f')") == [[1, 4, 7], [2, 5, 8], [3, 6]]
    assert cols(m, "textscan('1;2;3', '%f', 'Delimiter', ';')") == [[1, 2, 3]]
    assert cols(m, "textscan('1,,2,,,3', '%f', 'Delimiter', ',', 'MultipleDelimsAsOne', 1)") == [[1, 2, 3]]
    c = cols(m, "textscan(sprintf('1\\t2\\t\\t4'), '%f %f %f %f', 'Delimiter', '\\t')")
    assert c[:2] == [[1], [2]] and np.isnan(c[2][0]) and c[3] == [4]
    assert cols(m, "textscan('a|b||c', '%s', 'Delimiter', '|')") == [["a", "b", "", "c"]]
    assert cols(m, "textscan('1;2,3', '%f', 'Delimiter', {';', ','})") == [[1, 2, 3]]


def test_textscan_short_lines_are_padded_when_delimited(m):
    c = cols(m, "textscan(sprintf('1,2,3\\n4,5\\n6,7,8\\n'), '%f %f %f', 'Delimiter', ',')")
    assert c[0] == [1, 4, 6] and c[1] == [2, 5, 7] and c[2] == pytest.approx([3, np.nan, 8], nan_ok=True)
    c = cols(m, "textscan(sprintf('a,1\\nb\\nc,3\\n'), '%s %f', 'Delimiter', ',', 'EmptyValue', -1)")
    assert c == [["a", "b", "c"], [1, -1, 3]]
    # no line end after the last, short record: nothing is added
    assert cols(m, "textscan('1,2,3', '%f %f %f %f', 'Delimiter', ',')") == [[1], [2], [3], []]
    assert cols(m, "textscan(sprintf('a,b\\n\\nc,d\\n'), '%s %s', 'Delimiter', ',')") == [["a", "c"], ["b", "d"]]


def test_textscan_conversions(m):
    assert cols(m, "textscan('x=1 y=2', 'x=%f y=%f')") == [[1], [2]]
    assert cols(m, "textscan('10:20:30', '%f:%f:%f')") == [[10], [20], [30]]
    assert cols(m, "textscan('abcdefgh', '%3s %2s %s')") == [["abc"], ["de"], ["fgh"]]
    assert cols(m, "textscan('aabXcc', '%[ab] %s')") == [["aab"], ["Xcc"]]
    assert cols(m, "textscan('a 1 b 2', '%*s %f')") == [[1, 2]]
    assert cols(m, "textscan('\"a b\",1', '%q %f', 'Delimiter', ',')") == [["a b"], [1]]
    assert cols(m, "textscan('1.5 2 3', '%d %f')") == [[1, 2], [0.5, 3]]       # %d stops at the decimal point
    assert cols(m, "textscan('NaN Inf -Inf 1e3 .5 1D2', '%f')")[0][1:] == [np.inf, -np.inf, 1000, 0.5, 100]
    m.run("C = textscan('1 2 3', '%f32 %d8 %u16'); D = textscan('7 x', '%n %c'); E = textscan('1+2i 3', '%f');")
    assert m.str("class(C{1})") == "single" and m.str("class(C{2})") == "int8" and m.str("class(C{3})") == "uint16"
    assert m.str("D{2}") == "x" and m.num("E{1}(1) == 1 + 2i")
    m.run("C = textscan(sprintf('2024-01-02,5\\n2024-03-04,6'), '%D %f', 'Delimiter', ',');"
          "K = textscan(sprintf('a,1\\nb,2\\na,3'), '%C %f', 'Delimiter', ',');"
          "T = textscan('00:01:30 01:00:00', '%T'); F = textscan('02.01.2024', '%{dd.MM.yyyy}D');")
    assert m.num("isdatetime(C{1}) && C{1}(2) == datetime(2024, 3, 4)") and m.num("iscategorical(K{1})")
    assert m.num("isequal(categories(K{1}), {'a'; 'b'})") and m.num("T{1}(1) == seconds(90)")
    assert m.num("F{1} == datetime(2024, 1, 2)")


def test_textscan_options(m):
    assert cols(m, "textscan(sprintf('h1 h2\\n1 2\\n3 4'), '%f %f', 'HeaderLines', 1)") == [[1, 3], [2, 4]]
    assert cols(m, "textscan(sprintf('1 2 %% c\\n3 4'), '%f %f', 'CommentStyle', '%')") == [[1, 3], [2, 4]]
    assert cols(m, "textscan('1 /* x */ 2', '%f', 'CommentStyle', {'/*', '*/'})") == [[1, 2]]
    assert cols(m, "textscan('1,NA,3', '%f', 'Delimiter', ',', 'TreatAsEmpty', 'NA')")[0][::2] == [1, 3]
    assert cols(m, "textscan('1_2_3', '%f', 'Whitespace', '_')") == [[1, 2, 3]]
    assert cols(m, "textscan('1 2 3 4 5 6', '%f %f', 2)") == [[1, 3], [2, 4]]
    m.run("C = textscan(sprintf('1 2 x\\n3 4 y'), '%f %f %s', 'CollectOutput', 1); S = textscan('a b', '%s', "
          "'TextType', 'string'); [P, pos] = textscan('1 2 3 4', '%f', 2);")
    assert m.num("isequal(C{1}, [1 2; 3 4]) && isequal(C{2}, {'x'; 'y'})") and m.num("isstring(S{1})")
    assert m.num("pos") in (3, 4)
    # a field that does not fit stops the reading; with ReturnOnError false it is an error
    assert cols(m, "textscan(sprintf('1 2\\n3 abc\\n5 6'), '%f %f')") == [[1, 3], [2]]
    with pytest.raises(MatlabError, match="line 1, field 2"):
        m.run("textscan('1 x', '%f %f', 'ReturnOnError', false)")
    with pytest.raises(MatlabError, match="unknown option"):
        m.run("textscan('1', '%f', 'Bogus', 1)")


def test_textscan_leaves_the_file_where_it_stopped(w, tmp_path):
    (tmp_path / "t.txt").write_text("name value\n1 2\n3 4\n5 6\n")
    w.run("fid = fopen('t.txt'); h = fgetl(fid); A = textscan(fid, '%f %f', 1); p = ftell(fid);"
          "B = textscan(fid, '%f %f'); e = feof(fid); fclose(fid);")
    assert w.num("A{1} == 1 && A{2} == 2") and w.num("isequal([B{:}], [3 4; 5 6])") and w.num("e")
    assert w.num("p") in (14, 15)


CSV = "Title line\n\nmy var,x-y,3rd\n1,2,3\n4,,6\n"


def test_readtable_layout_and_names(w, tmp_path):
    (tmp_path / "h.csv").write_text(CSV)
    w.run("T = readtable('h.csv'); P = readtable('h.csv', 'VariableNamingRule', 'preserve');"
          "N = readtable('h.csv', 'NumHeaderLines', 3, 'ReadVariableNames', false);")
    assert w.str("strjoin(T.Properties.VariableNames, ',')") == "myVar,x_y,x3rd"
    assert w.str("strjoin(P.Properties.VariableNames, ',')") == "my var,x-y,3rd"
    assert w.num("isequaln(T.x_y, [2; NaN]) && isequal(T.myVar, [1; 4]) && height(T) == 2")
    assert w.str("strjoin(N.Properties.VariableNames, ',')") == "Var1,Var2,Var3" and w.num("N.Var3(2)") == 6
    (tmp_path / "w.txt").write_text("x y\n1 2\n3 4\n")
    (tmp_path / "t.tsv").write_text("x\ty\n1\t2\n% c\n3\t4\n")
    w.run("A = readtable('w.txt'); B = readtable('t.tsv', 'FileType', 'text', 'CommentStyle', '%');")
    assert w.num("isequal(A.y, [2; 4]) && isequal(B.x, [1; 3])")


def test_readtable_types(w, tmp_path):
    (tmp_path / "d.csv").write_text("when,dur,v,txt\n2024-01-02,00:01:30,1,a\n2024-01-03,01:00:00,2,\n")
    w.run("T = readtable('d.csv'); S = readtable('d.csv', 'TextType', 'string', 'DatetimeType', 'text', "
          "'DurationType', 'text');")
    assert w.num("isdatetime(T.when) && T.when(2) == datetime(2024, 1, 3) && isduration(T.dur)")
    assert w.num("T.dur(1) == seconds(90) && isequal(T.v, [1; 2]) && iscell(T.txt) && isempty(T.txt{2})")
    assert w.num("isstring(S.when) && S.when(1) == \"2024-01-02\" && isstring(S.dur) && ismissing(S.txt(2))")
    (tmp_path / "q.csv").write_text('id,name,note\nr1,"Smith, J","say ""hi"""\nr2,Lee,\n')
    w.run("Q = readtable('q.csv'); R = readtable('q.csv', 'ReadRowNames', true);")
    assert w.str("Q.name{1}") == "Smith, J" and w.str("Q.note{1}") == 'say "hi"'
    assert w.num("isequal(R.Properties.RowNames, {'r1'; 'r2'}) && width(R) == 2")


def test_readtable_number_formats_and_rules(w, tmp_path):
    (tmp_path / "o.csv").write_text("a;b\n1,5;x\n2,25;y\n")
    (tmp_path / "th.csv").write_text('v\n"1,234.5"\n"2,000"\n')
    (tmp_path / "h.csv").write_text(CSV)
    w.run("T = readtable('o.csv', 'Delimiter', ';', 'DecimalSeparator', ','); K = readtable('th.csv', "
          "'ThousandsSeparator', ','); F = readtable('o.csv', 'Delimiter', ';', 'Format', '%s%*s');")
    assert w.num("isequal(T.a, [1.5; 2.25]) && isequal(K.v, [1234.5; 2000])")
    assert w.num("iscell(F.a) && width(F) == 1")
    w.run("R = readtable('h.csv', 'MissingRule', 'omitrow'); V = readtable('h.csv', 'MissingRule', 'omitvar');"
          "M = readtable('h.csv', 'TreatAsMissing', '6');")
    assert w.num("height(R) == 1 && R.myVar == 1") and w.str("strjoin(V.Properties.VariableNames, ',')") == "myVar,x3rd"
    assert w.num("isnan(M.x3rd(2))")
    with pytest.raises(MatlabError, match="line 5, variable 'x_y'"):
        w.run("readtable('h.csv', 'MissingRule', 'error')")
    with pytest.raises(MatlabError, match="unknown option 'Bogus'"):
        w.run("readtable('h.csv', 'Bogus', 1)")
    with pytest.raises(MatlabError, match="Unable to find or open"):
        w.run("readtable('nofile.csv')")


def test_readmatrix_and_readcell(w, tmp_path):
    (tmp_path / "h.csv").write_text(CSV)
    w.run("M = readmatrix('h.csv'); H = readmatrix('h.csv', 'NumHeaderLines', 4); R = readmatrix('h.csv', 'Range', "
          "'B4:C5'); N = readmatrix('h.csv', 'Range', [4 1 4 2]); S = readmatrix('h.csv', 'OutputType', 'string');"
          "I = readmatrix('h.csv', 'OutputType', 'int32'); C = readcell('h.csv', 'NumHeaderLines', 2);")
    assert w.num("isequaln(M, [1 2 3; 4 NaN 6]) && isequaln(H, [4 NaN 6]) && isequaln(R, [2 3; NaN 6])")
    assert w.num("isequal(N, [1 2]) && isstring(S) && S(1, 1) == \"1\" && isa(I, 'int32')")
    assert w.num("isequal(size(C), [3 3]) && strcmp(C{1, 1}, 'my var') && C{2, 3} == 3 && ismissing(C{3, 2})")
    w.run("writematrix([1 2; 3 4], 'm.txt', 'Delimiter', '\\t'); writecell({1, 'a'; 2, 'b'}, 'wc.csv');"
          "A = readmatrix('m.txt'); B = readcell('wc.csv');")
    assert w.num("isequal(A, [1 2; 3 4]) && isequal(B, {1, 'a'; 2, 'b'})")
    (tmp_path / "dt.csv").write_text("2024-05-06,3\n")
    w.run("D = readcell('dt.csv');")
    assert w.num("isdatetime(D{1}) && D{1} == datetime(2024, 5, 6) && D{2} == 3")


def test_import_options(w, tmp_path):
    (tmp_path / "o.csv").write_text("a,b\n1,x\n2,y\n")
    w.run("opts = detectImportOptions('o.csv'); T = readtable('o.csv', opts);")
    assert w.num("isequal(opts.VariableNames, {'a', 'b'}) && isequal(opts.VariableTypes, {'double', 'char'})")
    assert w.num("isequal(opts.Delimiter, {','}) && isequal(opts.DataLines, [2 Inf]) && opts.VariableNamesLine == 1")
    assert w.num("height(T) == 2 && isequal(T.b, {'x'; 'y'})")
    w.run("o2 = setvartype(opts, 'a', 'string'); A = readtable('o.csv', o2); o3 = setvartype(opts, {'a', 'b'}, "
          "{'int32', 'categorical'}); B = readtable('o.csv', o3); o4 = opts; o4.SelectedVariableNames = 'b';"
          "C = readtable('o.csv', o4); o5 = opts; o5.VariableNames = {'p', 'q'}; D = readtable('o.csv', o5);")
    assert w.num("isstring(A.a) && A.a(2) == \"2\" && isa(B.a, 'int32') && iscategorical(B.b)")
    assert w.num("width(C) == 1 && isequal(C.b, {'x'; 'y'})")
    assert w.str("strjoin(D.Properties.VariableNames, ',')") == "p,q"
    assert w.num("isequal(opts.VariableTypes, {'double', 'char'})")           # the original is unchanged
    (tmp_path / "f.csv").write_text("a,b\n1,x\n,y\nNA,z\n")
    w.run("o = detectImportOptions('f.csv'); o = setvaropts(o, 'a', 'TreatAsMissing', 'NA', 'FillValue', -1);"
          "F = readtable('f.csv', o); v = getvaropts(o, 'a'); P = preview('f.csv', o);"
          "o.MissingRule = 'omitrow'; G = readtable('f.csv', o); M = readmatrix('f.csv', setvartype(o, 'char'));")
    assert w.num("isequal(F.a, [1; -1; -1]) && height(P) == 3 && height(G) == 1")
    assert w.str("v.Name") == "a" and w.str("v.Type") == "double" and w.num("iscell(M) && isequal(size(M), [1 2])")
    out = w.run("opts")
    assert "DelimitedTextImportOptions" in out and "VariableNames: {'a', 'b'}" in out
    with pytest.raises(MatlabError, match="not a variable type"):
        w.run("setvartype(opts, 'a', 'complex')")
    with pytest.raises(MatlabError, match="Unknown variable name"):
        w.run("setvartype(opts, 'zz', 'char')")
    with pytest.raises(MatlabError, match="cannot be combined"):
        w.run("readtable('o.csv', opts, 'Delimiter', ';')")


def test_datastore(w, tmp_path):
    (tmp_path / "p1.csv").write_text("a,b\n1,2\n3,4\n")
    (tmp_path / "p2.csv").write_text("a,b\n5,6\n")
    w.run("ds = tabularTextDatastore('p*.csv'); T = readall(ds); ds.ReadSize = 2; n = 0; s = 0;"
          "while hasdata(ds), t = read(ds); n = n + 1; s = s + sum(t.a); end; reset(ds); again = hasdata(ds);"
          "p = preview(ds); files = numel(ds.Files);")
    assert w.num("height(T) == 3 && isequal(T.b, [2; 4; 6])") and w.num("n") == 2 and w.num("s") == 9
    assert w.num("again && height(p) == 2 && files == 2")
    w.run("ds.SelectedVariableNames = {'b'}; B = readall(ds); d2 = datastore('p1.csv'); [t, info] = read(d2);"
          "part = partition(ds, 2, 2); last = readall(part);")
    assert w.num("width(B) == 1 && isequal(d2.VariableNames, {'a', 'b'}) && height(t) == 2")
    assert w.str("info.Filename").endswith("p1.csv") and w.num("isequal(last.b, 6)")
    with pytest.raises(MatlabError, match="No more data"):
        w.run("read(d2)")
    with pytest.raises(MatlabError, match="Cannot find files"):
        w.run("datastore('nothing*.csv')")


def test_bit_field_precisions(w, tmp_path):
    w.run("fid = fopen('b.bin', 'w'); fwrite(fid, 5, 'ubit3'); fwrite(fid, 2, 'ubit5'); fclose(fid);"
          "fid = fopen('b.bin'); a = fread(fid, 1, 'ubit3'); b = fread(fid, 1, 'ubit5'); fclose(fid);")
    assert w.num("a") == 5 and w.num("b") == 2
    assert (tmp_path / "b.bin").read_bytes() == bytes([5 | (2 << 3)])     # the first field is in the low bits
    w.run("fid = fopen('b.bin', 'w'); fwrite(fid, [1 0 1 1 0 0 1 0 1], 'ubit1'); fclose(fid);"
          "fid = fopen('b.bin'); x = fread(fid, [1 9], 'ubit1'); fclose(fid);")
    assert (tmp_path / "b.bin").read_bytes() == bytes([0b01001101, 1])
    assert w.num("x").tolist() == [[1, 0, 1, 1, 0, 0, 1, 0, 1]]
    w.run("fid = fopen('s.bin', 'w'); fwrite(fid, [-3 2 -1 9], 'bit4'); fclose(fid); fid = fopen('s.bin');"
          "y = fread(fid, 4, 'bit4')'; frewind(fid); z = fread(fid, 2, '*bit4')'; fclose(fid);")
    assert w.num("y").tolist() == [[-3, 2, -1, 7]] and w.num("z").dtype == np.int8      # 9 saturates at 7
    w.run("fid = fopen('be.bin', 'w', 'ieee-be'); fwrite(fid, [5 2], 'ubit4'); fclose(fid);"
          "fid = fopen('be.bin', 'r', 'b'); v = fread(fid, 2, 'ubit4')'; fclose(fid);")
    assert (tmp_path / "be.bin").read_bytes() == bytes([0x52]) and w.num("v").tolist() == [[5, 2]]
    w.run("fid = fopen('m.bin', 'w'); fwrite(fid, 1000, 'ubit12'); fwrite(fid, 9, 'ubit4'); fwrite(fid, 200, 'uint8');"
          "fclose(fid); fid = fopen('m.bin'); a = fread(fid, 1, 'ubit12'); b = fread(fid, 1, 'ubit4');"
          "c = fread(fid, 1, 'uint8'); d = fread(fid, 1, 'ubit2'); fclose(fid);")
    assert w.num("a == 1000 && b == 9 && c == 200 && isempty(d)")
    with pytest.raises(MatlabError, match="between 1 and 64"):
        w.run("fid = fopen('m.bin'); fread(fid, 1, 'ubit0');")
    w.run("fclose('all');")


def test_memmapfile(w, tmp_path):
    (tmp_path / "mm.bin").write_bytes(np.arange(1.0, 5.0).tobytes())
    w.run("m = memmapfile('mm.bin', 'Format', 'double'); a = m.Data'; p = memmapfile('mm.bin', 'Format', 'double', "
          "'Offset', 8, 'Repeat', 2); b = p.Data'; r = memmapfile('mm.bin'); n = numel(r.Data);")
    assert w.num("a").tolist() == [[1, 2, 3, 4]] and w.num("b").tolist() == [[2, 3]] and w.num("n") == 32
    assert w.str("class(r.Data)") == "uint8" and w.num("~r.Writable && r.Offset == 0 && isinf(r.Repeat)")
    with pytest.raises(MatlabError, match="read-only"):
        w.run("m.Data(1) = 9;")
    w.run("q = memmapfile('mm.bin', 'Format', 'double', 'Writable', true); q.Data(2) = 20; clear q")
    assert np.frombuffer((tmp_path / "mm.bin").read_bytes()).tolist() == [1, 20, 3, 4]
    w.run("s = memmapfile('mm.bin', 'Format', {'double', [2 1], 'pair'; 'uint8', [1 16], 'raw'}); d = s.Data;"
          "t = memmapfile('mm.bin', 'Format', {'double', [2 2], 'M'}); M = t.Data.M; t.Writable = true;"
          "t.Data.M(1, 2) = 30;")
    assert w.num("isequal(d.pair, [1; 20]) && numel(d.raw) == 16 && isequal(M, [1 3; 20 4])")
    assert np.frombuffer((tmp_path / "mm.bin").read_bytes()).tolist() == [1, 20, 30, 4]
    with pytest.raises(MatlabError, match="too small"):
        w.run("memmapfile('mm.bin', 'Format', 'double', 'Repeat', 9)")
    with pytest.raises(MatlabError, match="does not exist"):
        w.run("memmapfile('nofile.bin')")
    with pytest.raises(MatlabError, match="unknown option"):
        w.run("memmapfile('mm.bin', 'Colour', 1)")
