"""Graphics behaviour that follows MATLAB's documented defaults and properties (Phase 5 gaps).

Default figure size and getframe, colour letters and the default colour order, marker names, pie's patches and
texts, xline/yline ConstantLine objects, polar line data, the two rulers of yyaxis and image CData classes.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from mylab import Interpreter  # noqa: E402
from mylab.errors import MatlabError  # noqa: E402
from mylab.values import to_str  # noqa: E402


@pytest.fixture
def m():
    interp = Interpreter(out=lambda s: None)
    yield interp
    plt.close("all")


def num(m, expr):
    return np.asarray(m.eval_expression(expr), dtype=float)


def s(m, expr):
    return to_str(m.eval_expression(expr))


def test_default_figure_is_560_by_420_and_getframe_matches(m):
    m.execute("f = figure('Visible', 'off'); p = f.Position; plot(1:3); F = getframe(f); G = getframe(gca);")
    assert num(m, "p(3:4)").tolist() == [[560, 420]]
    assert num(m, "size(F.cdata)").tolist() == [[420, 560, 3]]
    g = num(m, "size(G.cdata)").ravel()
    assert g[2] == 3 and 300 < g[0] < 420 and 400 < g[1] < 560  # the axes area only
    assert num(m, "get(groot, 'DefaultFigurePosition')").ravel()[2:].tolist() == [560, 420]


@pytest.mark.parametrize("letter,rgb", [("r", [1, 0, 0]), ("g", [0, 1, 0]), ("b", [0, 0, 1]), ("c", [0, 1, 1]),
                                        ("m", [1, 0, 1]), ("y", [1, 1, 0]), ("k", [0, 0, 0]), ("w", [1, 1, 1])])
def test_colour_letters_are_matlabs(m, letter, rgb):
    m.execute(f"figure('Visible', 'off'); h = plot(1:3, '{letter}--o'); hold on; "
              f"s = stem(1:3, '{letter}'); e = errorbar(1:3, 1:3, [1 1 1], '{letter}'); "
              f"h2 = plot(1:3, 'Color', '{letter}'); t = text(1, 1, 'x', 'Color', '{letter}');")
    for v in ("h", "s", "e", "h2", "t"):
        assert num(m, f"{v}.Color").ravel().tolist() == rgb, v
    ax = plt.gcf().axes[0]
    assert to_rgb(ax.lines[0].get_color()) == tuple(float(c) for c in rgb)  # also what is drawn


def test_colour_name_and_option_order(m):
    m.execute("figure('Visible', 'off'); h = plot(1:3, 'green'); hold on; h2 = plot(1:3, 'r', 'Color', [0 0 1]);")
    assert num(m, "h.Color").ravel().tolist() == [0, 1, 0]
    assert num(m, "h2.Color").ravel().tolist() == [0, 0, 1]  # the Color option comes after the spec


def test_default_colour_order_is_matlabs(m):
    m.execute("figure('Visible', 'off'); c = get(gca, 'ColorOrder'); h1 = plot(1:3); hold on; h2 = plot(1:3); "
              "s = scatter(1:3, 1:3);")
    c = num(m, "c")
    assert c.shape == (7, 3)
    np.testing.assert_allclose(c[0], [0, 0.4470, 0.7410])
    np.testing.assert_allclose(c[6], [0.6350, 0.0780, 0.1840])
    np.testing.assert_allclose(num(m, "h1.Color").ravel(), c[0])
    np.testing.assert_allclose(num(m, "h2.Color").ravel(), c[1])


def test_marker_names_read_back_like_matlab(m):
    m.execute("figure('Visible', 'off'); h = plot(1:3, 'gs-'); a = h.Marker; h.Marker = 'd'; b = h.Marker; "
              "h.Marker = 'pentagram'; c = h.Marker; h.Marker = 'h'; d = h.Marker; h.Marker = 'none'; e = h.Marker; "
              "figure('Visible', 'off'); sc = scatter(1:3, 1:3, 'd'); f = sc.Marker;")
    assert [s(m, v) for v in "abcdef"] == ["square", "diamond", "pentagram", "hexagram", "none", "diamond"]


def test_pie_returns_patches_and_texts(m):
    m.execute("figure('Visible', 'off'); p = pie([1 1 2]);")
    assert num(m, "numel(p)") == 6
    assert s(m, "class(p(1))") == "matlab.graphics.primitive.Patch"
    assert s(m, "class(p(2))") == "matlab.graphics.primitive.Text"
    assert [s(m, f"p({k}).String") for k in (2, 4, 6)] == ["25%", "25%", "50%"]
    m.execute("figure('Visible', 'off'); q = pie([0.2 0.3], {'a', 'b'});")
    assert [s(m, "q(2).String"), s(m, "q(4).String")] == ["a", "b"]
    ax = plt.gcf().axes[0]
    import matplotlib.patches as mp
    wedges = [w for w in ax.patches if isinstance(w, mp.Wedge)]
    spans = sorted(w.theta2 - w.theta1 for w in wedges)
    assert spans == pytest.approx([72.0, 108.0], abs=1e-4)  # a partial pie: the values are the fractions themselves
    m.execute("figure('Visible', 'off'); r = pie([1 2 3], [0 1 0]); t = pie([1 999]);")
    assert s(m, "t(2).String") == "<1%"
    with pytest.raises(MatlabError):
        m.execute("pie([1 -1])")


def test_xline_and_yline_are_constant_lines(m):
    m.execute("figure('Visible', 'off'); plot(1:3); xl = xline(2); yl = yline(1.5, '--r', 'limit');")
    assert s(m, "class(xl)") == "matlab.graphics.chart.decoration.ConstantLine"
    assert s(m, "xl.Type") == "constantline"
    assert num(m, "xl.Value") == 2 and num(m, "yl.Value") == 1.5
    assert s(m, "xl.InterceptAxis") == "x" and s(m, "yl.InterceptAxis") == "y"
    assert num(m, "yl.Color").ravel().tolist() == [1, 0, 0] and s(m, "yl.LineStyle") == "--"
    assert s(m, "yl.Label") == "limit"
    m.execute("xl.Value = 2.5; xl.Label = 'here'; xl.LabelVerticalAlignment = 'bottom';")
    line = [ln for ln in plt.gcf().axes[0].lines if getattr(ln, "_mylab_constant", None)][0]
    assert list(line.get_xdata()) == [2.5, 2.5]
    assert s(m, "xl.Label") == "here" and s(m, "xl.LabelVerticalAlignment") == "bottom"
    with pytest.raises(MatlabError):
        m.execute("xl.LabelVerticalAlignment = 'sideways';")
    with pytest.raises(MatlabError, match="read-only"):
        m.execute("xl.InterceptAxis = 'y';")


def test_polar_lines_have_theta_and_r_data(m):
    m.execute("figure('Visible', 'off'); h = polarplot([0 pi/2], [1 2]); r = h.RData; t = h.ThetaData; "
              "h.RData = [3 4];")
    assert num(m, "r").tolist() == [[1, 2]]
    np.testing.assert_allclose(num(m, "t"), [[0, np.pi / 2]])
    assert num(m, "h.RData").tolist() == [[3, 4]]
    m.execute("figure('Visible', 'off'); g = plot(1:3);")
    with pytest.raises(MatlabError, match="Unrecognized property 'RData'"):
        m.execute("g.RData")


def test_yyaxis_has_two_rulers(m):
    m.execute("figure('Visible', 'off'); yyaxis left; plot(1:3); yyaxis right; plot([10 20 30]); ax = gca; "
              "n = size(ax.YAxis); ax.YAxis(2).Color = [1 0 0]; c = ax.YAxis(2).Color; L = ax.YAxis(1).Limits;")
    assert num(m, "n").tolist() == [[2, 1]]
    assert num(m, "c").ravel().tolist() == [1, 0, 0]
    assert num(m, "L").ravel().tolist() == [1, 3]
    m.execute("figure('Visible', 'off'); plot(1:3); k = numel(gca().YAxis);")
    assert num(m, "k") == 1


@pytest.mark.parametrize("expr,cls", [("image(uint8(magic(4)))", "uint8"), ("imagesc(int16(magic(3)))", "int16"),
                                      ("image(true(3))", "logical"), ("imshow(uint8(magic(4)))", "uint8"),
                                      ("image(magic(3) / 9)", "double"),
                                      ("image(uint8(255 * ones(2, 2, 3)))", "uint8")])
def test_image_cdata_keeps_its_class(m, expr, cls):
    m.execute(f"figure('Visible', 'off'); im = {expr}; c = class(im.CData); v = im.CData;")
    assert s(m, "c") == cls
    assert m.eval_expression("isequal(v, " + expr.split("(", 1)[1].rsplit(")", 1)[0] + ")")


def test_setting_image_cdata(m):
    m.execute("figure('Visible', 'off'); im = image(uint8(255 * ones(2, 2, 3))); im.CData = uint8(zeros(2, 2, 3)); "
              "c = class(im.CData); F = getframe(gca); px = double(F.cdata(round(end / 2), round(end / 2), :));")
    assert s(m, "c") == "uint8"
    assert num(m, "px").ravel().tolist() == [0, 0, 0]  # integer truecolour is 0..255, as when it was created
    m.execute("im.CData = rand(2, 2, 3); d = class(im.CData);")
    assert s(m, "d") == "double"
