"""Plotting: data argument forms, graphics handles, get/set and the extra plot functions."""

import struct

import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _st(m):
    return m.interp.state["plot"]


def _ax(m):
    st = _st(m)
    return st.axes[st.current]


def _png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    return struct.unpack(">II", head[16:24])


# -- plot(y) and other data forms ------------------------------------------------------------

def test_plot_single_vector(m):
    m.run("plot([1 5 3])")
    line = _ax(m).lines[0]
    assert list(line.get_xdata()) == [1, 2, 3]
    assert list(line.get_ydata()) == [1, 5, 3]


def test_plot_single_vector_with_spec(m):
    m.run("y = [2 4 3]; plot(y, 'o-')")
    line = _ax(m).lines[0]
    assert line.get_marker() == "o" and line.get_linestyle() == "-"
    assert list(line.get_xdata()) == [1, 2, 3]


def test_plot_column_and_matrix(m):
    m.run("plot(rand(5,1))")
    assert len(_ax(m).lines[0].get_xdata()) == 5
    m.run("plot([1 2;3 4])")
    lines = _ax(m).lines
    assert len(lines) == 2  # one line per column
    assert list(lines[0].get_ydata()) == [1, 3]
    assert list(lines[1].get_ydata()) == [2, 4]
    assert list(lines[0].get_xdata()) == [1, 2]


def test_semilog_single_argument(m):
    m.run("semilogy([1 10 100])")
    assert _ax(m).get_yscale() == "log"
    m.run("semilogx(1:10)")
    assert _ax(m).get_xscale() == "log"
    assert len(_ax(m).lines[0].get_xdata()) == 10


def test_plot_x_matrix_y_and_triples(m):
    m.run("x = (0:4)'; Y = [x, x.^2, x.^3]; h = plot(x, Y);")
    assert len(_ax(m).lines) == 3
    assert m.num("numel(h)") == 3
    m.run("plot(1:3, [1 2 3], 'r-', 1:3, [3 2 1], 'b--', 1:4, 1:4)")
    lines = _ax(m).lines
    assert len(lines) == 3
    assert lines[1].get_linestyle() == "--"
    # row-oriented matrix data is matched against x by its columns
    m.run("plot(1:4, [1 2 3 4; 5 6 7 8])")
    assert len(_ax(m).lines) == 2


def test_plot_complex(m):
    m.run("plot([1+2i, 3+4i, 5+1i])")
    line = _ax(m).lines[0]
    assert list(line.get_xdata()) == [1, 3, 5]
    assert list(line.get_ydata()) == [2, 4, 1]


def test_linespec_orders(m):
    m.run("plot(1:3, 1:3, 'or--')")
    line = _ax(m).lines[0]
    assert line.get_marker() == "o" and line.get_linestyle() == "--"


def test_plot_length_mismatch_errors(m):
    with pytest.raises(MatlabError):
        m.run("plot(1:3, 1:4)")


# -- handles ---------------------------------------------------------------------------------

def test_plot_returns_line_handle(m):
    m.run("h = plot(1:3);")
    assert m.str("class(h)") == "matlab.graphics.chart.primitive.Line"
    m.run("h.LineWidth = 3; h.Color = 'r'; h.LineStyle = '--'; h.Marker = 'square';")
    line = _ax(m).lines[0]
    assert line.get_linewidth() == 3
    assert line.get_color() == "r"
    assert line.get_linestyle() == "--"
    assert line.get_marker() == "s"
    assert m.num("h.LineWidth") == 3
    np.testing.assert_allclose(m.num("h.Color"), [[1, 0, 0]])
    assert m.str("h.Marker") == "s"
    np.testing.assert_allclose(m.num("h.YData"), [[1, 2, 3]])
    m.run("h.YData = [3 2 1];")
    assert list(line.get_ydata()) == [3, 2, 1]
    m.run("h.DisplayName = 'curve';")
    assert m.str("h.DisplayName") == "curve"


def test_no_output_returns_nothing(m):
    out = m.run("plot(1:3)")
    assert out == ""
    with pytest.raises(MatlabError):
        m.run("x = clf;")


def test_handle_display(m):
    out = m.run("h = plot(1:3)")
    assert "Line with properties" in out and "LineWidth" in out


def test_multiple_lines_return_array(m):
    m.run("h = plot([1 2; 3 4; 5 6]);")
    assert m.num("numel(h)") == 2
    m.run("h(2).Color = [0 1 0];")
    np.testing.assert_allclose(_ax(m).lines[1].get_color(), (0, 1, 0))
    m.run("c = get(h, 'Color');")
    assert m.str("class(c)") == "cell"
    assert m.num("numel(c)") == 2
    m.run("set(h, 'LineWidth', 4)")
    assert all(line.get_linewidth() == 4 for line in _ax(m).lines)
    m.run("h.LineStyle = ':';")
    assert all(line.get_linestyle() == ":" for line in _ax(m).lines)


def test_handle_growth_and_concatenation(m):
    m.run("h(1) = plot(1:3); hold on; h(2) = plot(3:-1:1); legend(h, {'up', 'down'});")
    assert m.num("numel(h)") == 2
    leg = _ax(m).get_legend()
    assert [t.get_text() for t in leg.get_texts()] == ["up", "down"]
    m.run("g = [h(2) h(1)];")
    assert m.num("size(g)").tolist() == [[1, 2]]


def test_figure_and_axes_handles(m):
    m.run("f = figure; ax = gca; fh = gcf;")
    assert m.str("class(f)") == "matlab.ui.Figure"
    assert m.str("class(ax)") == "matlab.graphics.axis.Axes"
    n = m.num("f.Number")
    assert n == m.num("fh.Number") == m.num("get(f, 'Number')") == m.num("get(gcf, 'Number')")
    m.run("f.Name = 'demo';")
    assert m.str("f.Name") == "demo"
    m.run("figure(7); f7 = gcf;")
    assert m.num("f7.Number") == 7
    m.run("figure(f);")
    assert m.num("get(gcf, 'Number')") == n
    m.run("close(f)")
    assert int(n.item()) not in _st(m).figures
    m.run("close(7)")
    assert 7 not in _st(m).figures


def test_figure_options(m):
    m.run("f = figure('Name', 'opts', 'Color', 'w', 'Position', [0 0 400 300], 'NumberTitle', 'off');")
    assert m.str("f.Name") == "opts"
    np.testing.assert_allclose(m.num("f.Position")[0, 2:], [400, 300])
    fig = _st(m).figures[int(m.num("f.Number").item())]
    np.testing.assert_allclose(fig.get_size_inches() * fig.dpi, [400, 300])


def test_axes_properties(m):
    m.run("plot(1:10); ax = gca;")
    m.run("ax.XLim = [0 20]; ax.FontSize = 14; ax.YScale = 'log'; ax.XGrid = 'on'; ax.Box = 'off';")
    ax = _ax(m)
    assert ax.get_xlim() == (0, 20)
    assert ax.get_yscale() == "log"
    assert not ax.spines["top"].get_visible()
    assert m.str("ax.XGrid") == "on"
    assert m.num("ax.FontSize") == 14
    assert m.str("ax.Box") == "off"
    m.run("ax.Title.String = 'Hello'; ax.XLabel.String = 'time';")
    assert ax.get_title() == "Hello"
    assert ax.get_xlabel() == "time"
    m.run("ax.YLabel = 'value';")
    assert ax.get_ylabel() == "value"
    m.run("ax.XDir = 'reverse';")
    assert ax.xaxis_inverted()


def test_subplot_returns_axes(m):
    m.run("ax1 = subplot(2,1,1); plot(1:3); ax2 = subplot(2,1,2); plot(3:-1:1);")
    m.run("plot(ax1, 1:5); title(ax1, 'top');")
    fig = _st(m).figures[_st(m).current]
    top = [a for a in fig.axes if a.get_title() == "top"]
    assert len(top) == 1 and len(top[0].lines[0].get_xdata()) == 5
    m.run("set(ax2, 'YLim', [0 10]);")
    assert m.num("ax2.YLim").tolist() == [[0, 10]]
    # selecting an existing subplot again reuses it
    m.run("subplot(2,1,1);")
    assert len(fig.axes) == 2


def test_bar_scatter_histogram_handles(m):
    m.run("b = bar([1 2 3]); b.FaceColor = 'r';")
    assert m.str("class(b)") == "matlab.graphics.chart.primitive.Bar"
    np.testing.assert_allclose(m.num("b.FaceColor"), [[1, 0, 0]])
    m.run("s = scatter(1:3, 4:6); s.SizeData = 100;")
    assert m.num("s.SizeData") == 100
    np.testing.assert_allclose(m.num("s.XData"), [[1, 2, 3]])
    m.run("hh = histogram([1 2 2 3 3 3 4 4 4 4], 4);")
    np.testing.assert_allclose(m.num("hh.Values"), [[1, 2, 3, 4]])
    assert m.num("hh.NumBins") == 4
    np.testing.assert_allclose(m.num("hh.BinEdges"), [[1, 1.75, 2.5, 3.25, 4]])
    np.testing.assert_allclose(m.num("hh.BinWidth"), 0.75)
    assert m.num("numel(hh.Data)") == 10
    m.run("hh.NumBins = 2;")
    np.testing.assert_allclose(m.num("hh.Values"), [[3, 7]])
    m.run("hh.BinEdges = [0 2 5];")
    np.testing.assert_allclose(m.num("hh.Values"), [[1, 9]])
    m.run("hh.Normalization = 'probability';")
    np.testing.assert_allclose(m.num("sum(hh.Values)"), 1)
    m.run("hh.BinWidth = 1;")
    assert m.num("hh.BinWidth") == 1


def test_histogram_options(m):
    m.run("h = histogram(randn(200,1), 'BinWidth', 0.5, 'Normalization', 'pdf', 'FaceColor', 'g');")
    assert m.num("h.BinWidth") == 0.5
    assert m.str("h.Normalization") == "pdf"
    vals = m.num("h.Values")
    assert np.isclose((vals * 0.5).sum(), 1)
    m.run("h2 = histogram(1:10, 'BinEdges', [0 5 10]);")
    np.testing.assert_allclose(m.num("h2.Values"), [[4, 6]])


def test_hist_counts_unchanged(m):
    m.run("[n, c] = hist([1 2 2 3 3 3], 3);")
    np.testing.assert_allclose(m.num("n"), [[1, 2, 3]])
    assert m.num("numel(c)") == 3


def test_title_and_legend_handles(m):
    m.run("plot(1:3); t = title('Main'); l = legend('a'); xl = xlabel('x');")
    assert m.str("t.String") == "Main"
    m.run("t.FontSize = 20; t.String = 'Changed';")
    ax = _ax(m)
    assert ax.title.get_fontsize() == 20 and ax.get_title() == "Changed"
    assert m.str("class(l)") == "matlab.graphics.illustration.Legend"
    m.run("l.Location = 'southwest';")
    assert m.str("l.Location") == "southwest"
    m.run("xl.Color = 'r';")
    assert ax.xaxis.label.get_color() == "r"


def test_title_with_subtitle(m):
    m.run("plot(1:3); title('Main', 'Second line'); s = subtitle('Other');")
    ax = _ax(m)
    assert ax.get_title() == "Main"
    assert ax._mylab_subtitle.get_text() == "Other"
    assert m.str("s.String") == "Other"


def test_get_set_misc(m):
    m.run("h = plot(1:3);")
    m.run("set(h, 'Color', [0 0 1], 'LineWidth', 2.5, 'Marker', 'o', 'MarkerSize', 9)")
    line = _ax(m).lines[0]
    assert line.get_linewidth() == 2.5 and line.get_markersize() == 9
    assert m.num("get(h, 'linewidth')") == 2.5  # names ignore case
    m.run("v = get(h, {'LineWidth', 'MarkerSize'});")
    assert m.num("numel(v)") == 2
    m.run("s = get(h);")
    assert m.str("class(s)") == "struct"
    assert m.num("s.LineWidth") == 2.5
    with pytest.raises(MatlabError, match="Unrecognized property"):
        m.run("set(h, 'NoSuchProperty', 1)")
    with pytest.raises(MatlabError, match="read-only"):
        m.run("f = gcf; f.Number = 3;")
    m.run("ax = gca; set(ax, 'TickLabelInterpreter', 'latex')")  # accepted, no effect
    assert m.str("get(gca, 'Type')") == "axes"
    assert m.str("get(h, 'Type')") == "line"
    m.run("p = get(h, 'Parent');")
    assert m.str("class(p)") == "matlab.graphics.axis.Axes"


def test_groot_and_figure_numbers(m):
    m.run("figure(3); plot(1:3)")
    np.testing.assert_allclose(m.num("get(0, 'ScreenSize')")[0, 2:], [1920, 1080])
    m.run("r = groot; cf = r.CurrentFigure;")
    assert m.num("cf.Number") == 3
    assert m.num("get(3, 'Number')") == 3
    assert bool(m.num("isgraphics(3)"))
    assert not bool(m.num("isgraphics(99)"))


def test_delete_and_isvalid(m):
    m.run("h = plot(1:3); hold on; g = plot(3:-1:1);")
    m.run("delete(h)")
    assert len(_ax(m).lines) == 1
    assert not bool(m.num("isvalid(h)"))
    assert bool(m.num("isgraphics(g)"))
    m.run("f = gcf; close(f)")
    assert not bool(m.num("isgraphics(f)"))


def test_hold_all_and_ishold(m):
    m.run("plot(1:3); hold('all'); plot(3:-1:1)")
    assert len(_ax(m).lines) == 2
    assert bool(m.num("ishold"))
    m.run("hold('off'); plot(1:2)")
    assert len(_ax(m).lines) == 1


# -- limits with inf ---------------------------------------------------------------------------

def test_inf_limits_keep_auto_side(m):
    m.run("plot(1:10, (1:10).^2);")
    auto_top = _ax(m).get_ylim()[1]
    m.run("ylim([0 inf])")
    lo, hi = _ax(m).get_ylim()
    assert lo == 0 and np.isclose(hi, auto_top) and np.isfinite(hi)
    m.run("xlim([-inf 5])")
    lo, hi = _ax(m).get_xlim()
    assert hi == 5 and np.isfinite(lo) and lo < 1
    m.run("axis([0 inf 0 inf])")
    ax = _ax(m)
    assert ax.get_xlim()[0] == 0 and np.isfinite(ax.get_xlim()[1]) and ax.get_xlim()[1] >= 10
    assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] >= 100
    lims = m.num("axis")
    assert lims.shape == (1, 4) and lims[0, 0] == 0


def test_limit_modes(m):
    m.run("plot(1:10); xlim([2 3]); xlim('auto');")
    assert _ax(m).get_xlim()[1] > 3
    assert m.str("xlim('mode')") == "auto"


# -- new functions -------------------------------------------------------------------------------

def test_line_adds_without_clearing(m):
    m.run("plot(1:3); h = line([0 1], [1 0], 'Color', 'r', 'LineWidth', 2);")
    assert len(_ax(m).lines) == 2
    assert m.num("h.LineWidth") == 2
    m.run("line('XData', [1 2], 'YData', [3 4])")
    assert len(_ax(m).lines) == 3


def test_fplot(m):
    m.run("h = fplot(@(x) sin(x));")
    line = _ax(m).lines[0]
    x = np.asarray(line.get_xdata())
    assert np.isclose(x[0], -5) and np.isclose(x[-1], 5)
    np.testing.assert_allclose(line.get_ydata(), np.sin(x))
    m.run("fplot(@(t) t.^2, [0 2], 'r--')")
    line = _ax(m).lines[0]
    assert np.isclose(line.get_xdata()[-1], 2) and np.isclose(line.get_ydata()[-1], 4)
    assert line.get_linestyle() == "--"
    # a function that only works on scalars is evaluated point by point
    m.run("fplot(@(x) max(x, 0), [-1 1])")
    assert np.nanmin(_ax(m).lines[0].get_ydata()) == 0
    m.run("fplot(@cos, @sin, [0 2*pi])")
    np.testing.assert_allclose(_ax(m).lines[0].get_xdata()[0], 1)


def test_tiledlayout_fixed(m):
    m.run("t = tiledlayout(2, 2); ax1 = nexttile; plot(1:3); nexttile; plot(1:4); nexttile(4); bar(1:2);")
    fig = _st(m).figures[_st(m).current]
    assert len(fig.axes) == 3
    assert m.num("t.GridSize").tolist() == [[2, 2]]
    m.run("title(t, 'All tiles'); xlabel(t, 'x');")
    assert fig._suptitle.get_text() == "All tiles"
    m.run("nexttile(1); plot(5:-1:1);")
    assert len(fig.axes) == 3


def test_tiledlayout_span(m):
    m.run("tiledlayout(2, 2); nexttile([1 2]); plot(1:3); nexttile(3); nexttile(4);")
    fig = _st(m).figures[_st(m).current]
    assert len(fig.axes) == 3
    spec = fig.axes[0].get_subplotspec()
    assert spec.colspan == range(0, 2)


def test_tiledlayout_flow(m):
    m.run("tiledlayout('flow'); for k = 1:5, nexttile; plot(rand(3,1)); end")
    fig = _st(m).figures[_st(m).current]
    assert len(fig.axes) == 5
    geoms = {a.get_subplotspec().get_gridspec().get_geometry() for a in fig.axes}
    assert geoms == {(2, 3)}
    m.run("figure; nexttile; nexttile;")  # nexttile alone starts a flow layout
    fig = _st(m).figures[_st(m).current]
    assert len(fig.axes) == 2


def test_peaks(m):
    Z = m.num("peaks")
    assert Z.shape == (49, 49)
    assert m.num("size(peaks(10))").tolist() == [[10, 10]]
    m.run("[X, Y, Z] = peaks(5);")
    assert m.num("size(X)").tolist() == [[5, 5]]
    x = np.linspace(-3, 3, 5)
    X, Y = np.meshgrid(x, x)
    want = 3 * (1 - X) ** 2 * np.exp(-X ** 2 - (Y + 1) ** 2) \
        - 10 * (X / 5 - X ** 3 - Y ** 5) * np.exp(-X ** 2 - Y ** 2) - np.exp(-(X + 1) ** 2 - Y ** 2) / 3
    np.testing.assert_allclose(m.num("Z"), want)
    np.testing.assert_allclose(m.num("peaks(X, Y)"), want)
    np.testing.assert_allclose(m.num("peaks(1)"), m.num("peaks(3, 3)"))
    m.run("peaks(10)")  # no output: draws a surface
    assert getattr(_ax(m), "name", "") == "3d"


def _edges_drawn(m) -> bool:
    ax = _ax(m)
    ax.figure.canvas.draw()  # 3-D collections compute their colours when drawn
    ec = ax.collections[0].get_edgecolor()
    return len(ec) > 0 and ec[0][3] > 0


def test_shading_and_surf(m):
    m.run("s = surf(peaks(10));")
    assert _edges_drawn(m)  # faceted by default
    m.run("shading('interp')")
    assert not _edges_drawn(m)
    m.run("shading('faceted')")
    assert _edges_drawn(m)
    m.run("shading('flat'); surf(peaks(8));")
    assert not _edges_drawn(m)
    m.run("s = surf(peaks(8)); s.EdgeColor = 'k';")
    assert _edges_drawn(m)


def test_surf_in_subplot_keeps_other_axes(m):
    m.run("subplot(1,2,1); plot(1:3); subplot(1,2,2); surf(peaks(5));")
    fig = _st(m).figures[_st(m).current]
    assert len(fig.axes) == 2
    assert {getattr(a, "name", "") for a in fig.axes} == {"rectilinear", "3d"}


def test_colormap_functions(m):
    for name in ("parula", "jet", "gray", "hot", "cool"):
        c = m.num(f"{name}(5)")
        assert c.shape == (5, 3)
        assert c.min() >= 0 and c.max() <= 1
    assert m.num("size(parula)").tolist() == [[256, 3]]
    np.testing.assert_allclose(m.num("gray(3)"), [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]], atol=0.01)
    np.testing.assert_allclose(m.num("cool(2)"), [[0, 1, 1], [1, 0, 1]], atol=1e-6)
    m.run("imagesc(magic(4)); colormap('jet'); colormap(parula); colormap('hot');")
    assert _ax(m).images[0].get_cmap().name == "hot"
    m.run("colormap(gray(8)); c = colormap;")
    assert m.num("size(c)").tolist() == [[8, 3]]
    with pytest.raises(MatlabError):
        m.run("colormap('nosuchmap')")


def test_polarplot(m):
    m.run("th = linspace(0, 2*pi, 30); h = polarplot(th, 1 + cos(th), 'r');")
    assert _ax(m).name == "polar"
    assert m.str("class(h)") == "matlab.graphics.chart.primitive.Line"
    m.run("plot(1:3)")
    assert _ax(m).name == "rectilinear"


def test_quiver(m):
    m.run("[x, y] = meshgrid(1:3); q = quiver(x, y, ones(3), zeros(3));")
    ax = _ax(m)
    assert len(ax.collections) == 1
    assert m.str("class(q)") == "matlab.graphics.chart.primitive.Quiver"
    m.run("quiver([1 2], [1 1], [1 1], [0 1], 0)")
    m.run("quiver(rand(3), rand(3))")


def test_yyaxis(m):
    m.run("plot(1:3); yyaxis('right'); plot([10 20 30]); ylabel('R'); ylim([0 40]); yyaxis('left'); ylabel('L');")
    fig = _st(m).figures[_st(m).current]
    assert len(fig.axes) == 2
    left, right = fig.axes
    assert left.get_ylabel() == "L" and right.get_ylabel() == "R"
    assert right.get_ylim() == (0, 40)
    assert len(left.lines) == 1 and len(right.lines) == 1
    m.run("legend('a', 'b')")
    leg = right.get_legend() or left.get_legend()
    assert [t.get_text() for t in leg.get_texts()] == ["a", "b"]


def test_legend_commands(m):
    m.run("plot(1:3, 'DisplayName', 'first'); hold on; plot(3:-1:1);")
    m.run("legend('show')")
    leg = _ax(m).get_legend()
    assert [t.get_text() for t in leg.get_texts()] == ["first", "data2"]
    m.run("legend('boxoff')")
    assert not _ax(m).get_legend().get_frame_on()
    m.run("legend('boxon')")
    assert _ax(m).get_legend().get_frame_on()
    m.run("legend('off')")
    assert _ax(m).get_legend() is None
    m.run("l = legend({'x', 'y'}, 'Location', 'northeastoutside', 'Orientation', 'horizontal');")
    assert m.str("l.Location") == "northeastoutside"
    assert m.str("l.Orientation") == "horizontal"


def test_colorbar_handle(m):
    m.run("imagesc(magic(3)); c = colorbar; c.Label.String = 'level';")
    assert m.str("class(c)") == "matlab.graphics.illustration.ColorBar"
    np.testing.assert_allclose(m.num("c.Limits"), [[1, 9]])
    fig = _st(m).figures[_st(m).current]
    assert len(fig.axes) == 2
    m.run("colorbar('off')")
    assert len(fig.axes) == 1
    m.run("plot(1:3)")  # plotting replaces the axes contents; no stray colorbar
    m.run("x = 1;")


def test_axis_modes(m):
    m.run("plot(1:3)")
    for mode in ("equal", "tight", "square", "off", "on", "ij", "xy", "image", "normal", "auto"):
        m.run(f"axis('{mode}')")
    m.run("axis('ij')")
    assert _ax(m).yaxis_inverted()
    m.run("axis('xy')")
    assert not _ax(m).yaxis_inverted()
    m.run("axis('off')")
    assert not _ax(m).axison
    with pytest.raises(MatlabError):
        m.run("axis('sideways')")


def test_grid_box(m):
    m.run("plot(1:3); grid('on')")
    assert any(gl.get_visible() for gl in _ax(m).xaxis.get_gridlines())
    m.run("grid('minor')")
    assert _ax(m)._mylab_minor_grid
    m.run("box('off')")
    assert not _ax(m).spines["top"].get_visible()
    m.run("box('on')")
    assert _ax(m).spines["right"].get_visible()


def test_ticks_and_labels(m):
    m.run("bar([3 5 2]); xticks([1 2 3]); xticklabels({'a', 'b', 'c'});")
    ax = _ax(m)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    m.run("labs = xticklabels;")
    assert m.num("numel(labs)") == 3
    assert m.str("labs{2}") == "b"
    m.run("yticks([0 5]); yticklabels(['low'; 'hi!']);")
    assert [t.get_text() for t in ax.get_yticklabels()] == ["low", "hi!"]
    m.run("xtickangle(45)")
    assert m.num("xtickangle") == 45
    np.testing.assert_allclose(m.num("xticks"), [[1, 2, 3]])
    m.run("ax = gca; ax.XTickLabel = {'p', 'q', 'r'};")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["p", "q", "r"]


def test_text_and_annotation(m):
    m.run("plot(1:3); t = text(2, 2, 'here', 'FontSize', 14);")
    assert _ax(m).texts[0].get_text() == "here"
    assert m.num("t.FontSize") == 14
    m.run("annotation('arrow', [0.1 0.3], [0.1 0.3]); annotation('textbox', [0.5 0.5 0.2 0.1], 'String', 'box');")


def test_other_plot_handles(m):
    m.run("h = stem(1:3); s = stairs(1:4); e = errorbar(1:3, [0.1 0.2 0.1]); a = area(1:3);")
    m.run("p = pie([1 2 3]);")
    m.run("figure; c = contour(peaks(10)); i = imagesc(magic(3)); i.CData = eye(3);")
    m.run("f = fill([0 1 0], [0 0 1], 'r'); f.FaceAlpha = 0.5;")
    m.run("pt = patch([0 1 1], [0 0 1], 'b');")
    m.run("h3 = plot3(1:3, 1:3, 1:3); assert(numel(h3.ZData) == 3)")


# -- saving ------------------------------------------------------------------------------------

def test_print_resolution(m, tmp_path):
    a, b = tmp_path / "lo.png", tmp_path / "hi.png"
    m.run(f"plot(1:3); print('-dpng', '{a}'); print('-r300', '-dpng', '{b}');")
    wa, ha = _png_size(a)
    wb, hb = _png_size(b)
    assert wb == pytest.approx(wa * 2, rel=0.05)
    m.run(f"print('-r50', '{tmp_path / 'small'}')")
    assert (tmp_path / "small.png").exists()


def test_save_with_handles(m, tmp_path):
    m.run("f = figure; plot(1:3);")
    m.run(f"saveas(f, '{tmp_path / 'a.png'}'); saveas(gcf, '{tmp_path / 'b.pdf'}');")
    m.run(f"print(f, '-dsvg', '{tmp_path / 'c'}'); saveas(gcf, '{tmp_path / 'd'}', 'png');")
    m.run(f"exportgraphics(gca, '{tmp_path / 'e.png'}', 'Resolution', 72);")
    m.run(f"n = f.Number; saveas(n, '{tmp_path / 'g.png'}');")
    for name in ("a.png", "b.pdf", "c.svg", "d.png", "e.png", "g.png"):
        assert (tmp_path / name).exists(), name


def test_fig_format_error(m, tmp_path):
    m.run("plot(1:3)")
    with pytest.raises(MatlabError, match=r"\.fig format is not supported"):
        m.run(f"saveas(gcf, '{tmp_path / 'x.fig'}')")
    with pytest.raises(MatlabError, match=r"\.fig format is not supported"):
        m.run(f"saveas(gcf, '{tmp_path / 'x'}', 'fig')")
    assert not (tmp_path / "x.fig").exists()


def test_nested_handle_assignment(m):
    m.run("h = plot(1:3); h.Parent.Title.String = 'Nested'; h.Parent.XLabel.FontSize = 15;")
    assert _ax(m).get_title() == "Nested"
    assert _ax(m).xaxis.label.get_fontsize() == 15
