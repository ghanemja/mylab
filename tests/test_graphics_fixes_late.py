"""Graphics fixes checked on the rendered figure: what both sides of a yyaxis chart share (x-axis label,
title, ticks, grid, legend, limits), xline/yline on an axis limit and their AffectAutoLimits property, and
subplot finding its axes again after a colorbar or an outside legend has made it smaller."""

import numpy as np
import pytest
from matplotlib.colors import to_rgb



@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _fig(m):
    st = m.interp.state["plot"]
    return st.figures[st.current]


def _image(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)


def _drawn(fig, *artists) -> np.ndarray:
    """Mask of the pixels that change when ``artists`` are hidden, i.e. where they are actually drawn."""
    shown = _image(fig)
    old = [a.get_visible() for a in artists]
    for a in artists:
        a.set_visible(False)
    hidden = _image(fig)
    for a, v in zip(artists, old):
        a.set_visible(v)
    return np.any(shown != hidden, axis=2)


def _inside_figure(fig, artist) -> bool:
    """The artist's box lies in the figure (its text descent may reach a few pixels past the edge)."""
    fig.canvas.draw()
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    if bb.width <= 0 or bb.height <= 0:
        return False
    w = max(0.0, min(bb.x1, fig.bbox.width) - max(bb.x0, 0.0))
    h = max(0.0, min(bb.y1, fig.bbox.height) - max(bb.y0, 0.0))
    return w * h >= 0.8 * bb.width * bb.height


def _sides(m):
    fig = _fig(m)
    base = next(a for a in fig.axes if getattr(a, "_mylab_yyright", None) is not None)
    return base, base._mylab_yyright


_YY = "figure; yyaxis left; plot(1:4, [3 1 4 1]); yyaxis right; plot(0:5, [20 50 40 70 60 90]);"


# -- 1. yyaxis: what the two sides share ---------------------------------------------------------------

def test_xlabel_and_title_after_yyaxis_right_are_drawn(m):
    m.run(_YY + "xlabel('time (s)'); title('Two scales'); ax = gca;")
    assert m.str("ax.XLabel.String") == "time (s)" and m.str("ax.Title.String") == "Two scales"
    fig = _fig(m)
    base, right = _sides(m)
    for t in (base.xaxis.label, base.title):
        assert t.get_text() and _drawn(fig, t).any() and _inside_figure(fig, t)
    assert right.get_title() == "" and right.get_xlabel() == ""  # one title, not one per side
    m.run("yyaxis left; b = gca;")  # the same label and title seen from the left side
    assert m.str("b.XLabel.String") == "time (s)" and m.str("b.Title.String") == "Two scales"


def test_label_set_through_the_right_side_handle_is_drawn(m):
    m.run(_YY + "ax = gca; ax.XLabel.String = 'distance'; set(ax, 'Title', 'unused');")
    m.run("ax.Title.String = 'Set as a property'; subtitle('second line');")
    fig = _fig(m)
    base, right = _sides(m)
    assert base.get_xlabel() == "distance" and base.get_title() == "Set as a property"
    assert _drawn(fig, base.xaxis.label).any() and _drawn(fig, base.title).any()
    sub = base._mylab_subtitle
    assert sub.get_text() == "second line" and _drawn(fig, sub).any()
    assert getattr(right, "_mylab_subtitle", None) is None
    m.run("ax.TitleHorizontalAlignment = 'left'; ax.Color = [1 1 0.8]; yyaxis left; b = gca;"
          "t = b.TitleHorizontalAlignment; c = b.Color;")
    assert m.str("t") == "left" and m.num("c").ravel().tolist() == [1, 1, 0.8]
    assert to_rgb(base.get_facecolor()) == (1, 1, 0.8)  # the background shows: the right axes has none


def test_xlim_and_x_ticks_after_yyaxis_right(m):
    m.run(_YY + "xlim([0 6]); xticks([0 3 6]); xticklabels({'start', 'middle', 'end'}); xtickangle(30);")
    fig = _fig(m)
    base, right = _sides(m)
    assert tuple(base.get_xlim()) == (0, 6) and tuple(right.get_xlim()) == (0, 6)
    fig.canvas.draw()
    shown = [t.get_text() for t in base.get_xticklabels() if t.get_visible()]
    assert shown == ["start", "middle", "end"] and base.xaxis.get_visible()
    assert all(t.get_rotation() == 30 for t in base.get_xticklabels())
    assert _drawn(fig, *base.get_xticklabels()).any()
    m.run("ax = gca; c = ax.XTickLabel; yyaxis left; d = xticklabels; L = xlim;")
    assert m.str("c{2}") == "middle" and m.str("d{3}") == "end"
    assert m.num("L").tolist() == [[0, 6]]


def test_grid_after_yyaxis_right_follows_the_left_ticks(m):
    m.run(_YY + "grid on; ax = gca; g = ax.XGrid;")
    assert m.str("g") == "on"
    fig = _fig(m)
    base, right = _sides(m)
    fig.canvas.draw()
    xgrid = [gl for gl in base.xaxis.get_gridlines() if gl.get_visible()]
    ygrid = [gl for gl in base.yaxis.get_gridlines() if gl.get_visible()]
    assert xgrid and ygrid and _drawn(fig, *xgrid).any() and _drawn(fig, *ygrid).any()
    assert not any(gl.get_visible() for gl in right.yaxis.get_gridlines())
    m.run("grid off")
    fig.canvas.draw()
    assert not any(gl.get_visible() for gl in base.xaxis.get_gridlines() + base.yaxis.get_gridlines())


def test_legend_after_yyaxis_right_lists_both_sides(m):
    m.run(_YY + "lg = legend('left data', 'right data');")
    fig = _fig(m)
    base, right = _sides(m)
    leg = right.get_legend() or base.get_legend()
    assert [t.get_text() for t in leg.get_texts()] == ["left data", "right data"]
    assert _drawn(fig, leg).any() and _inside_figure(fig, leg)
    m.run("yyaxis left; ax = gca; n = numel(ax.Legend.String);")
    assert m.num("n") == 2


def test_axis_tight_after_yyaxis_right(m):
    m.run(_YY + "axis tight; xl = xlim; yl = ylim;")
    assert m.num("xl").tolist() == [[0, 5]]  # the x data of both sides
    assert m.num("yl").tolist() == [[20, 90]]
    base, right = _sides(m)
    assert tuple(base.get_xlim()) == (0, 5)


def test_hold_cla_and_switching_back_after_yyaxis_right(m):
    m.run(_YY + "xlabel('x'); hold on; plot(0:5, 10 * (1:6)); yyaxis left; h = ishold;")
    base, right = _sides(m)
    assert bool(m.num("h")) and len(right.lines) == 2
    m.run("yyaxis right; cla;")  # clears the active side only
    assert len(right.lines) == 0 and len(base.lines) == 1
    fig = _fig(m)
    assert base.get_xlabel() == "x" and _drawn(fig, base.xaxis.label).any()
    m.run("yyaxis left; xlabel('changed on the left'); ax = gca; s = ax.XLabel.String;")
    assert m.str("s") == "changed on the left" and _drawn(fig, base.xaxis.label).any()


def test_cla_reset_removes_the_right_side(m):
    m.run(_YY + "r = gca; cla reset; ax = gca; n = numel(ax.YAxis); k = numel(ax.Children);")
    assert m.num("n") == 1 and m.num("k") == 0
    assert len(_fig(m).axes) == 1
    m.run("plot(r, 1:3); k = numel(ax.Children);")  # a handle taken on the right still names the axes
    assert m.num("k") == 1
