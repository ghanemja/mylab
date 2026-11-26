"""Fixes found by running real repositories from the corpus (datasets/matlab-corpus/tools/run_corpus.py)."""

import os

import numpy as np
import pytest

from mylab import Interpreter, MatlabError
from mylab.sourcefile import decode_source
from mylab.values import to_str


def num(m: Interpreter, expr: str):
    return np.asarray(m.eval_expression(expr))


# -- source files in other encodings ---------------------------------------------------------------

def test_source_encodings():
    assert decode_source("x = 'é';\r\n".encode("utf-8")) == "x = 'é';\n"
    assert decode_source(b"\xef\xbb\xbfx = 1;\n") == "x = 1;\n"  # byte-order mark
    assert decode_source("% é\nx = 1;\n".encode("cp1252")) == "% é\nx = 1;\n"
    sjis = "% ロボット\nx = 1;\r".encode("shift_jis")  # not valid UTF-8 or Windows-1252
    assert decode_source(sjis).endswith("\nx = 1;\n")


def test_scripts_and_functions_in_a_legacy_encoding(tmp_path):
    (tmp_path / "helper_enc.m").write_bytes("function r = helper_enc()\n% 説明\nr = 41;\nend\n".encode("shift_jis"))
    (tmp_path / "main_enc.m").write_bytes("% ロボット工学\nv = helper_enc() + 1;\n".encode("shift_jis"))
    m = Interpreter(out=lambda s: None, path=[str(tmp_path)])
    m.run_file(str(tmp_path / "main_enc.m"))
    assert num(m, "v").tolist() == [[42]]


# -- the current folder comes before the path -------------------------------------------------------

def test_current_folder_shadows_the_path(tmp_path, monkeypatch):
    for sub, value in (("a", 1), ("b", 2)):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "which_one.m").write_text(f"function r = which_one()\nr = {value};\nend\n")
    m = Interpreter(out=lambda s: None, path=[str(tmp_path / "a"), str(tmp_path / "b")])
    monkeypatch.chdir(tmp_path / "b")
    assert num(m, "which_one()").tolist() == [[2]]
    m2 = Interpreter(out=lambda s: None, path=[str(tmp_path / "a"), str(tmp_path / "b")])
    monkeypatch.chdir(tmp_path)
    assert num(m2, "which_one()").tolist() == [[1]]  # not in the current folder: the path order decides
    assert os.getcwd() == str(tmp_path)


# -- randg -------------------------------------------------------------------------------------------

def test_randg():
    m = Interpreter(out=lambda s: None)
    m.execute("rng(3); x = randg(4, 1, 200000); a = randg(4, 1, 5); rng(3); b = randg(4, 1, 5);")
    x = num(m, "x")
    assert x.shape == (1, 200000)
    assert abs(x.mean() - 4) < 0.03 and abs(x.var() - 4) < 0.1  # gamma(4, 1): mean 4, variance 4
    assert num(m, "b").tolist() == num(m, "x(1:5)").tolist()  # same stream after the same seed
    assert num(m, "randg([0 -1 NaN Inf])")[0, [0, 3]].tolist() == [0, np.inf]
    assert np.isnan(num(m, "randg([0 -1 NaN Inf])")[0, 1:3]).all()
    assert num(m, "size(randg(2, [2 3]))").tolist() == [[2, 3]]
    assert num(m, "size(randg(ones(3, 2)))").tolist() == [[3, 2]]
    assert to_str(m.eval_expression("class(randg(2, 2, 'single'))")) == "single"
    with pytest.raises(MatlabError, match="inconsistent"):
        m.execute("randg([1 2 3], 2, 2)")


# -- scatter3 ------------------------------------------------------------------------------------------

def test_scatter3_options_and_mapped_open_markers():
    import matplotlib.pyplot as plt
    m = Interpreter(out=lambda s: None)
    m.execute("x = (1:10)'; h = scatter3(x, x, 100*x, [], x, 'o');")
    coll = plt.gcf().axes[0].collections[0]
    plt.gcf().canvas.draw()
    assert len(coll.get_facecolors()) == 0 and len(coll.get_edgecolors()) == 10  # open, coloured edges
    m.execute("figure; h2 = scatter3(x, x, x, 20, 'r', 's', 'filled', 'MarkerEdgeColor', 'k');")
    assert to_str(m.eval_expression("h2.Marker")) == "s"
    with pytest.raises(MatlabError, match="'MarkerSize' is not supported"):
        m.execute("scatter3(x, x, x, 'MarkerSize', 3)")
    plt.close("all")


# -- echo, pkg ---------------------------------------------------------------------------------------

def test_echo_shows_script_lines(tmp_path):
    (tmp_path / "echo_demo.m").write_text("echo on\nx = 1;\nif x > 0\n  y = 2;\nend\necho off\nz = 3;\n")
    out = []
    m = Interpreter(out=out.append, path=[str(tmp_path)])
    m.run_file(str(tmp_path / "echo_demo.m"))
    assert "".join(out) == "x = 1;\nif x > 0\n  y = 2;\necho off\n"
    out.clear()
    m.run_file(str(tmp_path / "echo_demo.m"))  # echo off at the end of the last run left it off
    assert "".join(out).startswith("x = 1;")


def test_octave_pkg_load():
    m = Interpreter(out=lambda s: None)
    m.execute("pkg load control; pkg load optim symbolic; n = numel(pkg('list'));")
    assert num(m, "n").item() > 10
    with pytest.raises(MatlabError, match="'nosuchpkg' is not available"):
        m.execute("pkg load nosuchpkg")


# -- graphics: zlabel on 2-D axes, colormap expressions, numeric handles ----------------------------------

def test_zlabel_of_2d_axes():
    m = Interpreter(out=lambda s: None)
    m.execute("figure; plot(1:3); zlabel('depth'); a = get(get(gca, 'ZLabel'), 'String'); ax = gca;"
              "ax.ZLabel.String = 'z2'; b = ax.ZLabel.String;")
    assert to_str(m.eval_expression("a")) == "depth" and to_str(m.eval_expression("b")) == "z2"


def test_colormap_given_as_an_expression():
    m = Interpreter(out=lambda s: None)
    m.execute("figure; imagesc(magic(4)); colormap('gray(16)'); a = size(colormap);"
              "colormap jet(10); b = size(colormap);")
    assert num(m, "a").tolist() == [[16, 3]] and num(m, "b").tolist() == [[10, 3]]
    assert np.allclose(num(m, "colormap"), num(m, "jet(10)"))
    with pytest.raises(MatlabError, match="Unknown colormap"):
        m.execute("colormap('nosuchmap(3)')")


def test_numeric_graphics_handles():
    m = Interpreter(out=lambda s: None)
    m.execute("figure; p = plot(1:3); d = double(p); set(d, 'Color', [1 0 0]); c = get(p, 'Color');"
              "hold on; h = zeros(1, 2); h(1) = plot(1:3); h(2) = plot(3:-1:1); set(h(2), 'LineWidth', 3);"
              "w = get(handle(h(2)), 'LineWidth'); legend(h, 'a', 'b'); s = get(legend, 'String');"
              "ok1 = ishandle(h); delete(h(1)); ok2 = ishandle(h); same = isequal(handle(d), p); f = double(gcf);")
    assert num(m, "d").item() != round(num(m, "d").item())  # never taken for a figure number
    assert num(m, "c").tolist() == [[1, 0, 0]] and num(m, "w").item() == 3
    assert m.eval_expression("class(h)") is not None and to_str(m.eval_expression("class(h)")) == "double"
    assert [to_str(x) for x in m.eval_expression("s").values_f()] == ["a", "b"]
    assert num(m, "ok1").tolist() == [[True, True]] and num(m, "ok2").tolist() == [[False, True]]
    assert num(m, "same").item() and num(m, "f").item() == round(num(m, "f").item())
    with pytest.raises(MatlabError, match="Invalid or deleted"):
        m.execute("set(h(1), 'Color', 'b')")


# -- linspace with infinite end points, dlmread with '' and an offset, colour names in a LineSpec ----------

def test_linspace_end_points_are_exact():
    m = Interpreter(out=lambda s: None)
    assert num(m, "linspace(0, Inf, 5)").tolist() == [[0, np.inf, np.inf, np.inf, np.inf]]
    r = num(m, "linspace(-Inf, Inf, 3)")
    assert r[0, 0] == -np.inf and np.isnan(r[0, 1]) and r[0, 2] == np.inf
    assert num(m, "linspace(Inf, Inf, 3)").tolist() == [[np.inf] * 3]
    assert num(m, "linspace(1, 2, 0)").shape == (1, 0)
    assert num(m, "linspace(0.1, 0.7, 7)")[0, -1] == 0.7 and num(m, "linspace(0.1, 0.7, 7)")[0, 0] == 0.1


def test_dlmread_empty_delimiter_and_header_rows(tmp_path):
    (tmp_path / "f.txt").write_text("Title line here\n1 2 3\n4 5 6\n")
    (tmp_path / "g.csv").write_text("1,2\n3,4\n")
    m = Interpreter(out=lambda s: None)
    m.execute(f"cd('{tmp_path.as_posix()}'); a = dlmread('f.txt', '', 1, 0); b = dlmread('g.csv', ',', 1, 1);")
    assert num(m, "a").tolist() == [[1, 2, 3], [4, 5, 6]] and num(m, "b").tolist() == [[4]]


def test_colour_names_in_line_specs():
    m = Interpreter(out=lambda s: None)
    m.execute("figure; h = plot(1:3, [2 3 1], 'black', 'LineWidth', 2); hold on; h2 = plot(1:3, '--red');")
    assert num(m, "h.Color").tolist() == [[0, 0, 0]] and num(m, "h2.Color").tolist() == [[1, 0, 0]]
    assert to_str(m.eval_expression("h2.LineStyle")) == "--"


# -- fscanf finishes its format pass, exist finds data files on the path, cellfun option prefixes, bucky,
#    struct property lists, fields ------------------------------------------------------------------------

def test_fscanf_with_a_size_reads_the_rest_of_the_format(tmp_path):
    (tmp_path / "t.off").write_text("OFF\n4 3 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n3 1 2 3\n")
    m = Interpreter(out=lambda s: None)
    m.execute(f"cd('{tmp_path.as_posix()}'); f = fopen('t.off'); fgetl(f); fgetl(f);"
              "[v, c] = fscanf(f, '%f ', [3 4]); l = fgetl(f); fclose(f);"
              "[a, c2, e, n1] = sscanf('1 2 3', '%d ', 2); [a, c2, e, n2] = sscanf('1,2,3', '%d,', 2);"
              "[a, c2, e, n3] = sscanf('12abc', '%d', 1);")
    assert to_str(m.eval_expression("l")) == "3 0 1 2" and num(m, "c").item() == 12
    assert [num(m, f"n{k}").item() for k in (1, 2, 3)] == [5, 5, 3]  # as Octave gives


def test_exist_finds_data_files_on_the_path(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "mesh.ply").write_text("ply\n")
    monkeypatch.chdir(tmp_path)
    m = Interpreter(out=lambda s: None, path=[str(tmp_path / "data")])
    assert num(m, "exist('mesh.ply', 'file')").item() == 2 and num(m, "exist('mesh.ply')").item() == 2
    assert num(m, "exist('nothere.ply', 'file')").item() == 0


def test_cellfun_option_names_may_be_shortened():
    m = Interpreter(out=lambda s: None)
    m.execute("a = cellfun(@(x) x * 2, {1 2}, 'uni', 0); b = arrayfun(@(x) x * 2, [1 2], 'Un', false);"
              "c = cellfun(@(x) error('boom'), {1 2}, 'Err', @(e, x) -1);")
    assert to_str(m.eval_expression("class(a)")) == "cell" and to_str(m.eval_expression("class(b)")) == "cell"
    assert num(m, "c").tolist() == [[-1, -1]]


def test_bucky_and_fields():
    m = Interpreter(out=lambda s: None)
    m.execute("[B, V] = bucky; e = full(sum(B(:))) / 2; d = unique(full(sum(B, 2)));"
              "r = max(abs(sqrt(sum(V .^ 2, 2)) - 1)); s.a = 1; s.b = 2; f = fields(s);")
    assert num(m, "size(B)").tolist() == [[60, 60]] and num(m, "e").item() == 90 and num(m, "d").item() == 3
    assert num(m, "r").item() < 1e-12 and num(m, "issparse(B)").item()
    assert [to_str(x) for x in m.eval_expression("f").values_f()] == ["a", "b"]


def test_graphics_functions_take_a_structure_of_properties():
    m = Interpreter(out=lambda s: None)
    m.execute("figure; h = patch([0 1 1], [0 0 1], 'r', struct()); o.FaceAlpha = 0.5; o.EdgeColor = 'b';"
              "h2 = patch([0 1 1], [0 0 1], 'g', o); hold on; h3 = plot(1:3, struct('LineWidth', 4));")
    assert num(m, "h2.FaceAlpha").item() == 0.5 and num(m, "h2.EdgeColor").tolist() == [[0, 0, 1]]
    assert num(m, "h3.LineWidth").item() == 4


# -- the data argument is never an option; voronoin --------------------------------------------------

def test_text_data_named_like_an_option():
    m = Interpreter(out=lambda s: None)
    assert num(m, "sum('all')").item() == 313 and num(m, "any('all')").item() and num(m, "all('all')").item()
    assert num(m, "max('all')").item() == 108 and num(m, "prod('ab')").item() == 97 * 98
    assert num(m, "sum([1 2; 3 4], 'all')").item() == 10  # still the option after the data
    m.execute("v = {'all', 'mostVertices'}; p = inputParser;"
              "addOptional(p, 'components', 'mostVertices', @(x) any(validatestring(x, v)));"
              "parse(p, 'all'); r = p.Results.components;")
    assert to_str(m.eval_expression("r")) == "all"


def test_voronoin():
    m = Interpreter(out=lambda s: None)
    m.execute("x = [0 0; 1 0; 0 1; 1 1; 0.5 0.5]; [V, C] = voronoin(x);")
    V = num(m, "V")
    assert V.shape == (5, 2) and np.isinf(V[0]).all()
    inner = num(m, "sort(C{5})").ravel().astype(int)
    assert inner.tolist() == [2, 3, 4, 5]  # the bounded centre cell: all four finite vertices
    corners = sorted(map(tuple, V[inner - 1].round(9).tolist()))
    assert corners == [(0.0, 0.5), (0.5, 0.0), (0.5, 1.0), (1.0, 0.5)]
    assert all(1 in num(m, f"C{{{k}}}").ravel() for k in range(1, 5))  # the corner cells are unbounded


# -- split collapses delimiters, exact bucky coordinates, limits kept when the axes turn 3-D -------------------

def test_split_collapses_consecutive_delimiters():
    m = Interpreter(out=lambda s: None)
    m.execute("t = split('3  16 1 0'); u = split('a,,b', ','); w = split('a,,b', ',', 'CollapseDelimiters', false);")
    assert [to_str(x) for x in m.eval_expression("t").values_f()] == ["3", "16", "1", "0"]
    assert num(m, "numel(u)").item() == 2 and num(m, "numel(w)").item() == 3
    with pytest.raises(MatlabError, match="'Collapse' is not supported|CollapseDelims"):
        m.execute("split('a b', ' ', 'CollapseDelims', true)")


def test_bucky_faces_are_exactly_planar():
    m = Interpreter(out=lambda s: None)
    m.execute("[B, V] = bucky; f = convhulln(V); n = cross(V(f(:,2),:) - V(f(:,1),:), V(f(:,3),:) - V(f(:,1),:));"
              "n = n ./ sqrt(sum(n .^ 2, 2)); d = n * n'; k = sum(d(:) > 1 - 1e-14);")
    # the 116 hull triangles lie in 32 planes (12 pentagons in 3 triangles, 20 hexagons in 4)
    assert num(m, "size(f, 1)").item() == 116 and num(m, "k").item() == 12 * 9 + 20 * 16


def test_axis_limits_survive_view3():
    m = Interpreter(out=lambda s: None)
    m.execute("figure; axis([-10 10 -10 10 -10 10]); hold on; view(3); x = xlim; z = zlim;")
    assert num(m, "x").tolist() == [[-10, 10]] and num(m, "z").tolist() == [[-10, 10]]


# -- growing [] through colons; optional inputParser arguments given by name ------------------------------------

@pytest.mark.parametrize("code, size", [
    ("W = []; W(:,1,:) = ones(3,4);", [3, 1, 4]),
    ("W = []; W(:,:,2) = ones(3,4);", [3, 4, 2]),
    ("W = []; W(2,:,:) = ones(3,4);", [2, 3, 4]),
    ("W = []; W(:,1,:) = ones(1,4);", [1, 1, 4]),
    ("W = []; W(:,1,:) = ones(4,1);", [4, 1]),
    ("clear W; W(1,1,:) = [5 6];", [1, 1, 2]),
    ("W = []; W(1,:,2) = [1 2 3];", [1, 3, 2]),
    ("W = []; W(2,:) = [1 2 3];", [2, 3]),
])
def test_growing_an_empty_array_through_colons(code, size):  # sizes as Octave gives them
    m = Interpreter(out=lambda s: None)
    m.execute(code)
    assert num(m, "size(W)").ravel().tolist() == size


def test_optional_inputs_given_by_name():
    m = Interpreter(out=lambda s: None)
    m.execute("p = inputParser; addOptional(p, 'K', 10); addOptional(p, 'L', 100); addOptional(p, 'lambda', .001);"
              "parse(p, 'K', 3, 'L', 7); r = [p.Results.K p.Results.L p.Results.lambda]; d = p.UsingDefaults;"
              "q = inputParser; addOptional(q, 'units', 'inches'); parse(q, 'cm'); u = q.Results.units;")
    assert num(m, "r").tolist() == [[3, 7, 0.001]]
    assert [to_str(x) for x in m.eval_expression("d").values_f()] == ["lambda"]
    assert to_str(m.eval_expression("u")) == "cm"  # one text argument that names nothing stays positional
