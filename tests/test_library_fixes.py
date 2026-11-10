"""Library fixes: grouping, tables, text, interpolation, solvers and text-file I/O."""

import numpy as np
import pytest

from mylab import MatlabError
from mylab.builtins import BUILTINS
from mylab.help_data import HELP


def arr(m, expr):
    return np.asarray(m.num(expr), dtype=float)


# -- 1. accumarray --------------------------------------------------------------------------------

def test_accumarray_functions_size_and_fill(m):
    np.testing.assert_array_equal(arr(m, "accumarray([1;2;1;3], [10;20;30;40], [], @max)"), [[30], [20], [40]])
    np.testing.assert_array_equal(arr(m, "accumarray([1;2;1], [1;2;3])"), [[4], [2]])
    np.testing.assert_array_equal(arr(m, "accumarray([1;3], [5;6], [4 1], @mean, NaN)"),
                                  [[5], [np.nan], [6], [np.nan]])
    np.testing.assert_array_equal(arr(m, "accumarray([1 1; 2 2; 1 1], [1 2 3])"), [[4, 0], [0, 2]])
    np.testing.assert_array_equal(arr(m, "accumarray([1 1; 2 3], [5 6], [2 4])"), [[5, 0, 0, 0], [0, 0, 6, 0]])
    m.run("C = accumarray([1;2;1], [7;8;9], [], @(x) {x});")
    assert m.str("class(C)") == "cell"
    np.testing.assert_array_equal(arr(m, "C{1}"), [[7], [9]])
    np.testing.assert_array_equal(arr(m, "accumarray([2;2], 1)"), [[0], [2]])


# -- 2.-4. strsplit, str2double, regexp names ------------------------------------------------

def test_strsplit_collapses_delimiters(m):
    m.run("a = strsplit('a,,b', ','); b = strsplit('a,,b', ',', 'CollapseDelimiters', false);")
    assert m.num("numel(a)").item() == 2 and m.str("a{2}") == "b"
    assert m.num("numel(b)").item() == 3 and m.str("b{2}") == ""
    m.run('s = strsplit("x  y z");')
    assert m.str("class(s)") == "string" and m.num("size(s)").tolist() == [[1, 3]]
    m.run("[p, d] = strsplit('1;2,3', {';', ','});")
    assert m.str("d{1}") == ";" and m.str("p{3}") == "3"


def test_str2double_string_array(m):
    np.testing.assert_array_equal(arr(m, 'str2double(["1.5", "2", "x"])'), [[1.5, 2, np.nan]])
    np.testing.assert_array_equal(arr(m, "str2double({'3', 'no'})"), [[3, np.nan]])
    assert np.isnan(m.num("str2double(7)").item())


def test_regexp_names_struct_array(m):
    m.run("s = regexp('k1=v1;k2=v2', '(?<k>\\w+)=(?<v>\\w+)', 'names');")
    assert m.num("size(s)").tolist() == [[1, 2]]
    assert m.str("s(2).v") == "v2" and m.str("s(1).k") == "k1"
    m.run("t = regexp('k1=v1;k2=v2', '(?<k>\\w+)=(?<v>\\w+)', 'names', 'once');")
    assert m.num("numel(t)").item() == 1 and m.str("t.v") == "v1"
    m.run("e = regexp('zzz', '(?<k>\\d)', 'names');")
    assert m.num("numel(e)").item() == 0


# -- 5. set operations and sortrows ---------------------------------------------------------------

def test_ismember_rows(m):
    m.run("[tf, loc] = ismember([1 2; 3 4; 5 6], [3 4; 1 2], 'rows');")
    np.testing.assert_array_equal(m.val("tf"), [[True], [True], [False]])
    np.testing.assert_array_equal(m.val("loc"), [[2], [1], [0]])
    np.testing.assert_array_equal(arr(m, "unique([1 2; 1 2; 0 1], 'rows')"), [[0, 1], [1, 2]])


def test_sortrows_cells_and_negative_columns(m):
    m.run("c = sortrows({'pear', 2; 'apple', 3; 'fig', 1});")
    assert [m.str(f"c{{{k}, 1}}") for k in (1, 2, 3)] == ["apple", "fig", "pear"]
    m.run("[c2, i] = sortrows({'pear', 2; 'apple', 3; 'fig', 1}, -2);")
    assert m.str("c2{1, 1}") == "apple"
    np.testing.assert_array_equal(m.val("i"), [[2], [1], [3]])
    np.testing.assert_array_equal(arr(m, "sortrows([3 1; 1 2; 3 0], [1 -2])"), [[1, 2], [3, 1], [3, 0]])
    m.run("d = sortrows({'b'; 'a'; 'c'}, 'descend');")
    assert m.str("d{1}") == "c"


# -- 6. tables ----------------------------------------------------------------------------------

@pytest.fixture
def tg(m):
    m.run("T = table([1;2;3;4], {'a';'b';'a';'c'}, [10;20;30;40], 'VariableNames', {'id','g','val'});")
    return m


def test_table2cell_findgroups_splitapply(tg):
    m = tg
    m.run("C = table2cell(T);")
    assert m.num("size(C)").tolist() == [[4, 3]] and m.str("C{3, 2}") == "a"
    assert m.num("C{4, 3}").item() == 40
    m.run("[G, ids] = findgroups(T.g);")
    np.testing.assert_array_equal(m.val("G"), [[1], [2], [1], [3]])
    assert m.str("ids{3}") == "c"
    np.testing.assert_array_equal(arr(m, "splitapply(@mean, T.val, G)"), [[20], [20], [40]])
    np.testing.assert_array_equal(arr(m, "findgroups([5 3 5])"), [[2, 1, 2]])
    np.testing.assert_array_equal(arr(m, "splitapply(@sum, [1 2 3], [1 2 1])"), [[4, 2]])
    m.run('[G2, s2] = findgroups(["x"; "y"; "x"]);')
    np.testing.assert_array_equal(m.val("G2"), [[1], [2], [1]])
    assert m.str("class(s2)") == "string"
    m.run("[G3, TG] = findgroups(T(:, 'g'));")
    assert m.str("class(TG)") == "table" and m.num("height(TG)").item() == 3
    np.testing.assert_array_equal(arr(m, "findgroups([1; NaN; 1])"), [[1], [np.nan], [1]])
    m.run("Y = splitapply(@(x) {x'}, [1;2;3;4], [1;2;1;2]);")
    np.testing.assert_array_equal(arr(m, "Y{2}"), [[2, 4]])


def test_groupsummary_and_grouped_varfun(tg):
    m = tg
    m.run("S = groupsummary(T, 'g', {'mean', 'max'}, 'val');")
    assert m.str("strjoin(S.Properties.VariableNames, ' ')") == "g GroupCount mean_val max_val"
    np.testing.assert_array_equal(m.num("S.GroupCount"), [[2], [1], [1]])
    np.testing.assert_array_equal(m.num("S.mean_val"), [[20], [20], [40]])
    m.run("S2 = groupsummary(T, 'g');")
    assert m.num("width(S2)").item() == 2
    m.run("S3 = groupsummary(T, 'g', 'numel');")
    np.testing.assert_array_equal(m.num("S3.numel_val"), [[2], [1], [1]])
    m.run("V = varfun(@sum, T, 'GroupingVariables', 'g', 'InputVariables', 'val');")
    np.testing.assert_array_equal(m.num("V.sum_val"), [[40], [20], [40]])
    np.testing.assert_array_equal(arr(m, "varfun(@max, T, 'InputVariables', {'id','val'}, 'OutputFormat', 'uniform')"),
                                  [[4, 40]])


def test_table_brace_assignment_and_row_names(tg):
    m = tg
    m.run("T{2, 'val'} = -1; T{1, 'g'} = 'z';")
    np.testing.assert_array_equal(m.num("T.val'"), [[10, -1, 30, 40]])
    assert m.str("T.g{1}") == "z"
    m.run("T{5, 'val'} = 7;")
    assert m.num("height(T)").item() == 5 and m.num("T.val(5)").item() == 7
    m.run("R = table([1;2], [3;4], 'RowNames', {'r1','r2'}, 'VariableNames', {'x','y'});")
    assert m.num("R{'r2', 'y'}").item() == 4
    m.run("Q = R('r2', :);")
    assert m.str("Q.Properties.RowNames{1}") == "r2"
    out = m.run("R")
    assert "r1" in out and "r2" in out
    m.run("R.Properties.RowNames = {'p'; 'q'};")
    assert m.num("R{'q', 'x'}").item() == 2
    m.run("A = array2table([1 2; 3 4], 'RowNames', {'a', 'b'});")
    assert m.str("A.Properties.RowNames{2}") == "b"
    with pytest.raises(MatlabError):
        m.run("R('nope', :)")


def test_joins(m):
    m.run("A = table([1;2;3], [7;8;9], 'VariableNames', {'k','a'});"
          "B = table([3;1;5], {'x';'y';'z'}, 'VariableNames', {'k','b'});")
    m.run("I = innerjoin(A, B);")
    np.testing.assert_array_equal(m.num("I.k"), [[1], [3]])
    assert m.str("I.b{2}") == "x"
    m.run("J = join(table([3;1], 'VariableNames', {'k'}), B);")
    assert m.str("J.b{1}") == "x" and m.str("J.b{2}") == "y"
    m.run("O = outerjoin(A, B, 'MergeKeys', true);")
    np.testing.assert_array_equal(m.num("O.k"), [[1], [2], [3], [5]])
    np.testing.assert_array_equal(m.num("O.a"), [[7], [8], [9], [np.nan]])
    m.run("O2 = outerjoin(A, B, 'Keys', 'k');")
    assert "k_left" in m.run("disp(strjoin(O2.Properties.VariableNames, ','))")
    m.run("s = join({'a', 'b'}, '-');")
    assert m.str("s{1}") == "a-b"
    with pytest.raises(MatlabError):
        m.run("join(A, B)")


def test_rmmissing_fillmissing_movevars(m):
    np.testing.assert_array_equal(arr(m, "rmmissing([1 NaN 3])"), [[1, 3]])
    m.run("[y, tf] = rmmissing([1 2; NaN 3; 4 5]);")
    np.testing.assert_array_equal(m.val("y"), [[1, 2], [4, 5]])
    np.testing.assert_array_equal(m.val("tf"), [[False], [True], [False]])
    np.testing.assert_array_equal(arr(m, "fillmissing([1 NaN NaN 4], 'linear')"), [[1, 2, 3, 4]])
    np.testing.assert_array_equal(arr(m, "fillmissing([NaN 2 NaN 4 NaN], 'previous')"), [[np.nan, 2, 2, 4, 4]])
    np.testing.assert_array_equal(arr(m, "fillmissing([NaN 2 NaN 4 NaN], 'next')"), [[2, 2, 4, 4, np.nan]])
    np.testing.assert_array_equal(arr(m, "fillmissing([1 NaN 3], 'constant', 0)"), [[1, 0, 3]])
    np.testing.assert_array_equal(arr(m, "fillmissing([1; NaN; 5], 'linear')"), [[1], [3], [5]])
    np.testing.assert_array_equal(arr(m, "fillmissing([NaN 1 2], 'linear')"), [[0, 1, 2]])
    m.run("T = table([1;NaN;3], {'a';'';'c'}, 'VariableNames', {'x','s'});")
    m.run("F = fillmissing(T, 'previous'); R = rmmissing(T);")
    np.testing.assert_array_equal(m.num("F.x"), [[1], [1], [3]])
    assert m.str("F.s{2}") == "a" and m.num("height(R)").item() == 2
    m.run("M = movevars(T, 's', 'Before', 'x');")
    assert m.str("M.Properties.VariableNames{1}") == "s"


# -- 7. new functions ------------------------------------------------------------------------------

def test_gradient(m):
    np.testing.assert_array_equal(arr(m, "gradient([1 4 9 16])"), [[3, 4, 6, 7]])
    np.testing.assert_array_equal(arr(m, "gradient([1 4 9 16], 2)"), [[1.5, 2, 3, 3.5]])
    np.testing.assert_allclose(arr(m, "gradient([0 1 4], [0 1 3])"), [[1, 4 / 3, 1.5]])
    m.run("[gx, gy] = gradient([1 2 4; 2 4 8]);")
    np.testing.assert_array_equal(m.val("gx"), [[1, 1.5, 2], [2, 3, 4]])
    np.testing.assert_array_equal(m.val("gy"), [[1, 2, 4], [1, 2, 4]])


def test_repelem_strtok_conv2(m):
    np.testing.assert_array_equal(arr(m, "repelem([1 2], [2 3])"), [[1, 1, 2, 2, 2]])
    np.testing.assert_array_equal(arr(m, "repelem([1; 2], 2)"), [[1], [1], [2], [2]])
    np.testing.assert_array_equal(arr(m, "repelem([1 2; 3 4], 1, 2)"), [[1, 1, 2, 2], [3, 3, 4, 4]])
    m.run("[t, r] = strtok('  red green');")
    assert m.str("t") == "red" and m.str("r") == " green"
    m.run("[t2, r2] = strtok('a,b,c', ',');")
    assert m.str("t2") == "a" and m.str("r2") == ",b,c"
    A = np.arange(1, 10, dtype=float).reshape(3, 3)
    m.run("A = [1 2 3; 4 5 6; 7 8 9]; K = [1 0; 0 -1];")
    from scipy.signal import convolve2d
    full = convolve2d(A, [[1, 0], [0, -1]])
    np.testing.assert_array_equal(arr(m, "conv2(A, K)"), full)
    np.testing.assert_array_equal(arr(m, "conv2(A, K, 'same')"), full[1:4, 1:4])
    np.testing.assert_array_equal(arr(m, "conv2(A, K, 'valid')"), full[1:3, 1:3])
    np.testing.assert_array_equal(arr(m, "conv2([1 2], [1; 1], A)"), convolve2d(A, [[1, 1], [2, 2]]))


def test_issorted_maxk_mink_pascal(m):
    assert m.num("issorted([1 2 2 5])").item() and not m.num("issorted([2 1])").item()
    assert m.num("issorted([3 2 1], 'descend')").item()
    assert m.num("issorted({'a', 'b', 'c'})").item()
    m.run("[v, i] = maxk([3 1 4 1 5 9 2 6], 3);")
    np.testing.assert_array_equal(m.val("v"), [[9, 6, 5]])
    np.testing.assert_array_equal(m.val("i"), [[6, 8, 5]])
    np.testing.assert_array_equal(arr(m, "mink([3 1 4 NaN 5], 2)"), [[1, 3]])
    np.testing.assert_array_equal(arr(m, "maxk([1 5; 7 2], 1)"), [[7, 5]])
    np.testing.assert_array_equal(arr(m, "pascal(3)"), [[1, 1, 1], [1, 2, 3], [1, 3, 6]])
    np.testing.assert_array_equal(arr(m, "pascal(3, 1) * pascal(3, 1)"), np.eye(3))


def test_pchip_and_interp1_methods(m):
    x = np.array([1, 2, 3, 4, 5.0])
    y = np.array([1, 4, 2, 5, 3.0])
    from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
    m.run("x = 1:5; y = [1 4 2 5 3];")
    expect = PchipInterpolator(x, y)(2.5)
    assert m.num("pchip(x, y, 2.5)").item() == pytest.approx(expect)
    assert m.num("interp1(x, y, 2.5, 'cubic')").item() == pytest.approx(expect)
    assert m.num("interp1(x, y, 2.5, 'pchip')").item() == pytest.approx(expect)
    mk = Akima1DInterpolator(x, y, method="makima")(2.5)
    assert m.num("interp1(x, y, 2.5, 'makima')").item() == pytest.approx(mk)
    assert np.isnan(m.num("interp1(x, y, 6)").item())
    assert np.isfinite(m.num("interp1(x, y, 6, 'spline')").item())
    assert m.num("interp1(x, y, 6, 'linear', 'extrap')").item() == pytest.approx(1)
    assert m.num("interp1(x, y, 6, 'linear', 0)").item() == 0
    np.testing.assert_allclose(arr(m, "interp1([1 2 3], [10 20; 30 40; 50 60], [1.5 2.5])"),
                               [[20, 30], [40, 50]])


def test_spline_pp_forms(m):
    m.run("pp = spline(0:4, (0:4).^3);")
    assert m.num("ppval(pp, 2.5)").item() == pytest.approx(2.5 ** 3)
    assert m.str("pp.form") == "pp" and m.num("pp.pieces").item() == 4
    m.run("[b, c, l, k, d] = unmkpp(mkpp([0 1 2], [1 0; -1 1]));")
    assert m.num("l").item() == 2 and m.num("k").item() == 2 and m.num("d").item() == 1
    assert m.num("ppval(mkpp([0 1 2], [1 0; -1 1]), 1.5)").item() == pytest.approx(0.5)
    # clamped end slopes: y has two extra values
    assert m.num("spline([0 1 2], [0 0 1 4 4], 1)").item() == pytest.approx(1)
    m.run("q = pchip(1:4, [1 3 2 4]);")
    assert m.num("ppval(q, 3)").item() == pytest.approx(2)


def test_interp2_forms(m):
    m.run("Z = magic(3);")
    assert m.num("interp2(Z, 1.5, 1.5)").item() == pytest.approx(np.mean([8, 1, 3, 5]))
    assert m.num("size(interp2(Z))").tolist() == [[5, 5]]
    assert m.num("size(interp2(Z, 2))").tolist() == [[9, 9]]
    m.run("[X, Y] = meshgrid(1:3, 1:3);")
    np.testing.assert_allclose(arr(m, "interp2(X, Y, Z, [1.5 2.5], [1.5; 2.5])"), [[4.25, 4.75], [5.25, 5.75]])
    assert np.isnan(m.num("interp2(X, Y, Z, 4, 1)").item())
    assert m.num("interp2(X, Y, Z, 4, 1, 'linear', 0)").item() == 0
    assert m.num("interp2(1:3, 1:3, Z, 2, 2, 'nearest')").item() == 5


def test_text_files(m, tmp_path):
    d = str(tmp_path).replace("\\", "/")
    m.run(f"""
f = '{d}/a.csv';
fid = fopen(f, 'w'); fprintf(fid, 'x,y,z\\n1,2,3\\n4,,6\\n7,8,abc\\n'); fclose(fid);
M = readmatrix(f);
M2 = readmatrix(f, 'NumHeaderLines', 2);
S = importdata(f);
""")
    np.testing.assert_array_equal(m.val("M"), [[1, 2, 3], [4, np.nan, 6], [7, 8, np.nan]])
    np.testing.assert_array_equal(m.val("M2"), [[4, np.nan, 6], [7, 8, np.nan]])
    assert m.str("S.colheaders{2}") == "y"
    np.testing.assert_array_equal(m.num("S.data(1, :)"), [[1, 2, 3]])
    m.run(f"dlmwrite('{d}/b.txt', [1 2; 3 4], '\\t'); writematrix([1.5 2; 3 4], '{d}/c.txt', 'Delimiter', ';');")
    assert (tmp_path / "b.txt").read_text() == "1\t2\n3\t4\n"
    assert (tmp_path / "c.txt").read_text() == "1.5;2\n3;4\n"
    m.run(f"writematrix([1 2], '{d}/s.txt', 'Delimiter', 'space'); dlmwrite('{d}/t.txt', [5 6], 'delimiter', 'tab');")
    assert (tmp_path / "s.txt").read_text() == "1 2\n"
    assert (tmp_path / "t.txt").read_text() == "5\t6\n"
    m.run(f"writematrix({{'a', 1; 'b,c', 2}}, '{d}/e.csv'); C = readcell('{d}/e.csv');")
    assert (tmp_path / "e.csv").read_text() == 'a,1\n"b,c",2\n'
    assert m.str("C{2, 1}") == "b,c" and m.num("C{2, 2}").item() == 2
    m.run(f"writecell({{'p', 3.5}}, '{d}/g.csv'); L = readlines('{d}/g.csv');")
    assert m.str("L") == "p,3.5"
    m.run(f"writelines([\"one\"; \"two\"], '{d}/h.txt'); L2 = readlines('{d}/h.txt');")
    assert m.num("numel(L2)").item() == 2 and m.str("L2(2)") == "two"
    out = m.run(f"type('{d}/h.txt')")
    assert "one\ntwo" in out
    m.run("c = splitlines(sprintf('a\\nb'));")
    assert m.num("numel(c)").item() == 2


def test_fscanf_textscan(m, tmp_path):
    d = str(tmp_path).replace("\\", "/")
    (tmp_path / "n.txt").write_text("1 2 3\n4 5 6\n")
    (tmp_path / "ts.txt").write_text("name,val\nbob,1.5\nann,2\n")
    m.run(f"fid = fopen('{d}/n.txt'); A = fscanf(fid, '%d', [3 Inf]); fclose(fid);")
    np.testing.assert_array_equal(m.val("A"), [[1, 4], [2, 5], [3, 6]])
    m.run(f"fid = fopen('{d}/ts.txt'); c = textscan(fid, '%s %f', 'Delimiter', ',', 'HeaderLines', 1); fclose(fid);")
    assert m.num("size(c)").tolist() == [[1, 2]]
    assert m.str("c{1}{2}") == "ann"
    np.testing.assert_array_equal(m.num("c{2}"), [[1.5], [2]])
    m.run("c2 = textscan('1 2 3 4', '%d %f');")
    assert m.str("class(c2{1})") == "int32"
    np.testing.assert_array_equal(m.num("c2{2}"), [[2], [4]])
    m.run("c3 = textscan('\"a b\" 1', '%q %f');")
    assert m.str("c3{1}{1}") == "a b"


def test_dir_and_rmdir(m, tmp_path):
    for name in ("a.csv", "b.csv", "c.txt"):
        (tmp_path / name).write_text("1\n")
    (tmp_path / "sub").mkdir()
    d = str(tmp_path).replace("\\", "/")
    m.run(f"L = dir('{d}/*.csv');")
    assert m.num("numel(L)").item() == 2 and m.str("L(2).name") == "b.csv"
    assert m.num("L(1).bytes").item() == 2 and not m.num("L(1).isdir").item()
    assert m.num("L(1).datenum").item() > 7e5
    m.run(f"A = dir('{d}');")
    assert m.num("numel(A)").item() == 6
    m.run(f"rmdir('{d}/sub');")
    assert not (tmp_path / "sub").exists()
    m.run(f"mkdir('{d}/deep/er'); [ok, msg] = rmdir('{d}/deep');")
    assert not m.num("ok").item() and (tmp_path / "deep").exists()
    m.run(f"rmdir('{d}/deep', 's');")
    assert not (tmp_path / "deep").exists()


def test_etime_timeit_missing(m):
    assert m.num("etime([2020 1 1 0 1 1.5], [2020 1 1 0 0 0])").item() == pytest.approx(61.5)
    assert m.num("timeit(@() sum(1:10))").item() > 0
    m.run("x = missing;")
    assert m.num("ismissing(x)").item()


# -- 8. histcounts ---------------------------------------------------------------------------------

def test_histcounts_binning(m):
    m.run("[n, e] = histcounts(1:5);")
    np.testing.assert_array_equal(m.val("n"), [[1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(m.val("e"), [[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]])
    m.run("[n, e] = histcounts([0.1 0.5 1.2 2.7 3.3 3.4], 'BinWidth', 1);")
    np.testing.assert_array_equal(m.val("n"), [[2, 1, 1, 2]])
    np.testing.assert_array_equal(m.val("e"), [[0, 1, 2, 3, 4]])
    np.testing.assert_array_equal(arr(m, "histcounts(1:5, 'Normalization', 'cdf')"), [[0.2, 0.4, 0.6, 0.8, 1]])
    np.testing.assert_allclose(arr(m, "histcounts([1 2 2 3], [0 2 4], 'Normalization', 'probability')"),
                               [[0.25, 0.75]])
    np.testing.assert_allclose(arr(m, "histcounts([1 2 2 3], 'BinEdges', [0 2 4], 'Normalization', 'pdf')"),
                               [[0.125, 0.375]])
    assert m.num("numel(histcounts(rand(1, 50), 'NumBins', 7))").item() == 7
    m.run("rng(1); [n, e] = histcounts(randn(1, 1000));")
    edges = m.val("e").flatten()
    widths = np.diff(edges)
    assert np.allclose(widths, widths[0]) and m.num("sum(n)").item() == 1000
    w = widths[0]
    mant = w / 10 ** np.floor(np.log10(w))
    assert any(np.isclose(mant, c) for c in (1, 2, 2.5, 5))
    assert np.allclose(edges / w, np.round(edges / w))


# -- 9.-10. delimiters and headers are covered in test_text_files ----------------------------------

# -- 11.-12. solvers -----------------------------------------------------------------------------

def test_ode45_accuracy_and_events(m):
    m.run("[t, y] = ode45(@(t, y) -0.3 * y, [0 10], 10);")
    t = m.val("t").flatten()
    y = m.val("y").flatten()
    np.testing.assert_allclose(y, 10 * np.exp(-0.3 * t), rtol=1e-3)
    m.run("[t2, y2] = ode45(@(t, y) -0.3 * y, 0:1:10, 10);")
    assert m.num("y2(6)").item() == pytest.approx(10 * np.exp(-1.5), abs=5e-4)
    m.run("sol = ode45(@(t, y) -0.3 * y, [0 10], 10);")
    assert m.num("deval(sol, 5)").item() == pytest.approx(10 * np.exp(-1.5), abs=5e-4)


def test_ode_events_with_function_file(m, tmp_path):
    (tmp_path / "hitground.m").write_text(
        "function [v, term, dir] = hitground(t, y)\n  v = y(1);\n  term = 1;\n  dir = -1;\nend\n")
    m.run(f"addpath('{str(tmp_path).replace(chr(92), '/')}');")
    m.run("opts = odeset('Events', @hitground);"
          "[t, y, te, ye, ie] = ode45(@(t, y) [y(2); -9.81], [0 10], [10; 0], opts);")
    expect = np.sqrt(20 / 9.81)
    assert m.num("te").item() == pytest.approx(expect, rel=1e-4)
    assert m.num("t(end)").item() == pytest.approx(expect, rel=1e-4)
    assert m.num("ie").item() == 1 and abs(m.num("ye(1)").item()) < 1e-6
    m.run("sol = ode45(@(t, y) [y(2); -9.81], [0 10], [10; 0], opts);")
    assert m.num("sol.xe").item() == pytest.approx(expect, rel=1e-4) and m.num("sol.ie").item() == 1
    np.testing.assert_allclose(arr(m, "deval(sol, 1)"), [[10 - 9.81 / 2], [-9.81]], rtol=1e-6)
    m.run("[yy, yp] = deval(sol, [0.5 1]);")
    np.testing.assert_allclose(m.val("yp")[1], [-9.81, -9.81])
    m.run("[t2, y2] = ode45(@(t, y) [y(2); -9.81], 0:0.5:3, [10; 0], opts);")
    np.testing.assert_allclose(m.val("t2").flatten(), [0, 0.5, 1, expect], rtol=1e-4)
    m.run("[t3, y3, te3] = ode45(@(t, y) -y, [0 1], 1);")
    assert m.num("numel(te3)").item() == 0


def test_optimizer_output_structs(m):
    m.run("[x, fv, ef, out] = fminsearch(@(x) (x(1) - 1)^2 + (x(2) - 2)^2, [0 0]);")
    np.testing.assert_allclose(m.val("x"), [[1, 2]], atol=1e-3)
    assert m.num("out.funcCount").item() > 0 and m.num("out.iterations").item() > 0
    assert "Nelder" in m.str("out.algorithm")
    m.run("[x, fv, ef, out] = fzero(@(x) x^2 - 2, 1);")
    assert m.num("x").item() == pytest.approx(np.sqrt(2)) and m.num("out.funcCount").item() > 0
    m.run("[x, fv, ef, out] = fminbnd(@(x) (x - 2)^2, 0, 5);")
    assert m.num("x").item() == pytest.approx(2, abs=1e-3) and m.num("ef").item() == 1
    m.run("[x, fv, ef, out] = fminunc(@(x) sum((x - 3).^2), [0 0]);")
    np.testing.assert_allclose(m.val("x"), [[3, 3]], atol=1e-4)
    m.run("[x, fv, ef, out] = fmincon(@(x) sum((x - 1).^2), [0 0], [], [], [], [], [0 0], [2 2]);")
    assert m.num("out.funcCount").item() > 0 and m.str("class(out.message)") == "char"


def test_polyfit_mu_and_delta(m):
    m.run("x = 1:10; y = 3 * x.^2 - 2 * x + 1;")
    m.run("[p, S, mu] = polyfit(x, y, 2);")
    np.testing.assert_allclose(m.val("mu").flatten(), [5.5, np.std(np.arange(1, 11), ddof=1)])
    assert m.num("polyval(p, 11, [], mu)").item() == pytest.approx(3 * 121 - 22 + 1)
    assert m.num("polyval(p, 11, S, mu)").item() == pytest.approx(342)
    m.run("[p2, S2] = polyfit(x, y + [0 1 0 -1 0 1 0 -1 0 1], 1);")
    assert m.num("S2.df").item() == 8 and m.num("S2.normr").item() > 0
    assert m.num("size(S2.R)").tolist() == [[2, 2]]
    m.run("[yy, delta] = polyval(p2, [1 2], S2);")
    assert np.all(m.val("delta") > 0)


# -- 13. text conversions ------------------------------------------------------------------------

def test_sprintf_s_with_numbers_and_case_mapping(m):
    assert m.str("sprintf('%s', 65)") == "A"
    assert m.str("sprintf('%s', [72 105])") == "Hi"
    assert m.str("sprintf('%s', 1.5)") == "1.500000e+00"
    assert m.str("sprintf('%d', 7)") == "7"
    assert m.str("upper('straße')") == "STRAßE"
    assert m.num("numel(upper('ß'))").item() == 1
    assert m.str("lower('ÀB')") == "àb"


def test_new_io_builtins_have_help():
    names = [n for n, fn in BUILTINS.items() if fn.__module__.endswith(".io")]
    assert names and not [n for n in names if n not in HELP]
