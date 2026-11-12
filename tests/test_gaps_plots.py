"""quiver3, surface, pcolor, tri plots, xline/yline, alpha, boxplot, gobjects."""

import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture
def p(m):
    yield m
    m.run("close all")


def _axes(m):
    st = m.interp.state["plot"]
    return st.axes[st.current]


def test_quiver3(p):
    p.run("[X, Y] = meshgrid(1:3, 1:3); Z = X + Y; h = quiver3(X, Y, Z, ones(3), ones(3), ones(3), 'r');"
          "u = h.UData;")
    ax = _axes(p)
    assert ax.name == "3d" and len(ax.collections) == 1
    assert p.val("u").shape == (3, 3)
    p.run("figure; quiver3(peaks(5), ones(5), ones(5), ones(5), 0);")
    with pytest.raises(MatlabError, match="quiver3 expects"):
        p.run("quiver3(1, 2)")


def test_surface_keeps_existing_plot(p):
    p.run("[X, Y, Z] = sphere(8); plot3([0 1], [0 1], [0 1]); s = surface(X, Y, Z); "
          "surface(X + 2, Y, Z, 'FaceAlpha', 0.5); held = ishold;")
    ax = _axes(p)
    assert len(ax.lines) == 1 and len(ax.collections) == 2 and not p.num("held")
    assert p.num("s.ZData").shape == (9, 9)
    p.run("figure; surface('XData', X, 'YData', Y, 'ZData', Z);")
    assert len(_axes(p).collections) == 1


def test_pcolor_and_shading(p):
    p.run("h = pcolor(magic(4)); c = h.CData;")
    ax = _axes(p)
    assert len(ax.collections) == 1 and ax.get_xlim() == (1.0, 4.0)
    assert ax.collections[0].get_array().size == 9  # one colour per cell, last row/column not drawn
    p.run("figure; shading interp; [X, Y] = meshgrid(0:0.5:1, 0:0.25:1); pcolor(X, Y, X.*Y);")
    assert _axes(p).get_ylim() == (0.0, 1.0)
    with pytest.raises(MatlabError, match="pcolor"):
        p.run("pcolor([1 2 3])")


def test_tri_plots(p):
    p.run("x = [0 1 0 1 0.5]; y = [0 0 1 1 0.5]; z = x.^2 + y; T = delaunay(x, y);"
          "h = trisurf(T, x, y, z); figure; trimesh(T, x, y, z); figure; t = triplot(T, x, y, 'k');")
    assert len(_axes(p).lines) >= 1
    with pytest.raises(MatlabError, match="three columns"):
        p.run("trisurf([1 2], x, y, z)")


def test_xline_yline(p):
    p.run("plot(1:10); h = xline(3, '--r', 'limit'); yline([2 4], 'Color', [0 0 1], 'LineWidth', 2);"
          "v = h.LineStyle;")
    ax = _axes(p)
    assert len(ax.lines) == 4 and p.str("v") == "--"
    assert ax.lines[1].get_linestyle() == "--" and ax.lines[2].get_linewidth() == 2
    assert any(t.get_text() == "limit" for t in ax.texts)
    with pytest.raises(MatlabError, match="needs a position"):
        p.run("xline()")


def test_alpha_and_stem3(p):
    p.run("surf(peaks(10)); alpha(0.4);")
    assert _axes(p).collections[0].get_alpha() == 0.4
    p.run("figure; h = bar([1 2 3]); alpha(h, 0.5); figure; stem3([1 2; 3 4]);")
    with pytest.raises(MatlabError, match="between 0"):
        p.run("alpha(3)")


def test_boxplot(p):
    p.run("x = [1 2 3 4 100; 2 3 4 5 6]'; boxplot(x, 'Labels', {'a', 'b'});")
    ax = _axes(p)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    p.run("figure; boxplot([1 2 3 10 11 12], {'u', 'u', 'u', 'v', 'v', 'v'});")
    assert [t.get_text() for t in _axes(p).get_xticklabels()] == ["u", "v"]
    with pytest.raises(MatlabError, match="one entry per data value"):
        p.run("boxplot([1 2 3], [1 2])")


def test_gobjects_and_noops(p):
    p.run("h = gobjects(1, 3); for k = 1:3, h(k) = plot(1:k); hold on; end; n = numel(h); "
          "h(2).LineWidth = 3; e = gobjects(0); uistack(h(1), 'top'); movegui('center');")
    assert p.num("n") == 3 and p.num("isempty(e)")
    assert np.isclose(_axes(p).lines[1].get_linewidth(), 3)
