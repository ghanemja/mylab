"""Graphics fixes checked on the rendered figure: what both sides of a yyaxis chart share (x-axis label,
title, ticks, grid, legend, limits), xline/yline on an axis limit and their AffectAutoLimits property, and
subplot finding its axes again after a colorbar or an outside legend has made it smaller."""

import numpy as np
import pytest
from matplotlib.colors import to_rgb

from mylab.errors import MatlabError


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


def _dark(px) -> np.ndarray:
    return np.all(px < 80, axis=-1)


def test_box_off_after_yyaxis_right_keeps_both_y_axis_lines(m):
    m.run(_YY + "box off; ax = gca; b = ax.Box;")
    assert m.str("b") == "off"
    fig = _fig(m)
    base, right = _sides(m)
    img = _image(fig)
    bb = base.get_window_extent()
    H = fig.bbox.height
    top, bottom = int(H - bb.y1), int(H - bb.y0)
    x0, x1 = int(round(bb.x0)), int(round(bb.x1))
    assert _dark(img[top - 1:top + 2, x0 + 20:x1 - 20]).any(axis=0).mean() < 0.1  # no top edge
    assert _dark(img[top + 10:bottom - 10, x1 - 2:x1 + 2]).any(axis=1).mean() > 0.95  # right y-axis line
    assert _dark(img[top + 10:bottom - 10, x0 - 2:x0 + 2]).any(axis=1).mean() > 0.95  # left y-axis line


# -- 2. xline / yline ----------------------------------------------------------------------------------

def _frame_row(fig, ax) -> int:
    """Pixel row of the top edge of the axes frame."""
    fig.canvas.draw()
    return int(round(fig.bbox.height - ax.get_window_extent().y1 - 0.5))


def test_yline_at_the_limit_keeps_the_limits_and_shows_on_the_frame(m):
    m.run("figure; plot(1:3, 'b'); yl = yline(3, 'r'); L = ylim;")
    assert m.num("L").tolist() == [[1, 3]]  # constant lines do not change automatic limits
    fig = _fig(m)
    ax = fig.axes[0]
    line = next(ln for ln in ax.lines if getattr(ln, "_mylab_constant", None))
    img = _image(fig)
    row = _frame_row(fig, ax)
    bb = ax.get_window_extent()
    mid = img[row - 1:row + 2, int(bb.x0) + 10:int(bb.x1) - 10]
    red = (mid[..., 0] > 200) & (mid[..., 1] < 80) & (mid[..., 2] < 80)
    assert red.any(axis=0).mean() > 0.95  # drawn over the frame along its whole length
    assert _drawn(fig, line).any()


def test_yline_at_the_limit_of_the_left_side_of_a_yyaxis_chart_shows(m):
    # the right side is drawn over the left one: it must not draw the frame again on top of the line
    m.run("figure; yyaxis left; plot(1:3); yline(3, 'r'); yyaxis right; plot([10 20 30]);")
    fig = _fig(m)
    base, right = _sides(m)
    img = _image(fig)
    row = _frame_row(fig, base)
    bb = base.get_window_extent()
    mid = img[row - 1:row + 2, int(bb.x0) + 10:int(bb.x1) - 10]
    red = (mid[..., 0] > 200) & (mid[..., 1] < 80) & (mid[..., 2] < 80)
    assert red.any(axis=0).mean() > 0.95


def test_xline_at_the_limit_shows_on_the_frame(m):
    m.run("figure; plot(1:3, 'b'); xline(1, 'r', 'LineWidth', 2);")
    fig = _fig(m)
    ax = fig.axes[0]
    img = _image(fig)
    bb = ax.get_window_extent()
    col = int(round(bb.x0 - 0.5))
    top, bottom = int(fig.bbox.height - bb.y1) + 10, int(fig.bbox.height - bb.y0) - 10
    strip = img[top:bottom, col - 1:col + 2]
    red = (strip[..., 0] > 200) & (strip[..., 1] < 80) & (strip[..., 2] < 80)
    assert red.any(axis=1).mean() > 0.95


def test_constant_line_outside_the_limits_is_not_drawn(m):
    m.run("figure; plot(1:3, 'b'); yl = yline(5, 'r'); xl = xline(-1, 'r'); L = [xlim ylim];")
    assert m.num("L").tolist() == [[1, 3, 1, 3]]
    fig = _fig(m)
    img = _image(fig)
    assert not ((img[..., 0] > 200) & (img[..., 1] < 80) & (img[..., 2] < 80)).any()
    m.run("ylim([0 6])")  # once the value is inside the limits the line appears
    img = _image(fig)
    assert ((img[..., 0] > 200) & (img[..., 1] < 80) & (img[..., 2] < 80)).any()


def test_affect_auto_limits(m):
    m.run("figure; plot(1:3); yl = yline(5); a = yl.AffectAutoLimits; L1 = ylim;")
    assert m.str("a") == "off" and m.num("L1").tolist() == [[1, 3]]
    m.run("yl.AffectAutoLimits = 'on'; L2 = ylim; b = get(yl, 'AffectAutoLimits');")
    assert m.num("L2").tolist() == [[1, 5]] and m.str("b") == "on"
    m.run("set(yl, 'AffectAutoLimits', 'off'); L3 = ylim;")
    assert m.num("L3").tolist() == [[1, 3]]
    m.run("xl = xline(-2, 'AffectAutoLimits', 'on', 'Color', 'r'); L4 = xlim; c = xl.AffectAutoLimits;")
    assert m.num("L4").tolist() == [[-2, 3]] and m.str("c") == "on"
    assert m.num("xl.Color").ravel().tolist() == [1, 0, 0]
    m.run("hs = yline([0 7], '--', 'AffectAutoLimits', true); L5 = ylim;")
    assert m.num("L5").tolist() == [[0, 7]]
    with pytest.raises(MatlabError):
        m.run("yl.AffectAutoLimits = 'sometimes';")


def test_affect_auto_limits_off_keeps_other_objects_in_the_limits(m):
    m.run("figure; scatter([0 10], [0 10]); hold on; yl = yline(-4, 'AffectAutoLimits', 'on'); L1 = ylim;")
    assert m.num("L1").tolist() == [[-4, 10]]
    m.run("yl.AffectAutoLimits = 'off'; L2 = ylim;")
    assert m.num("L2").tolist() == [[0, 10]]


def test_constant_line_with_limits_on_survives_copyobj(m):
    m.run("figure; plot(1:3); yline(6, 'AffectAutoLimits', 'on'); f2 = figure; copyobj(get(1, 'Children'), f2);"
          "L = ylim(gca);")
    assert m.num("L").tolist() == [[1, 6]]


def test_constant_line_still_has_its_label_and_legend_entry(m):
    m.run("figure; plot(1:3, 'DisplayName', 'data'); yline(2, 'r', 'limit', 'DisplayName', 'max'); lg = legend;")
    fig = _fig(m)
    ax = fig.axes[0]
    assert any(t.get_text() == "limit" for t in ax.texts)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["data", "max"]
    line = next(ln for ln in ax.lines if getattr(ln, "_mylab_constant", None))
    assert to_rgb(line.get_color()) == (1, 0, 0)


# -- 3. subplot finds its axes again -------------------------------------------------------------------

def test_subplot_after_colorbar_returns_the_same_axes(m):
    m.run("figure; a1 = subplot(2, 1, 1); imagesc(magic(3)); colorbar; a2 = subplot(2, 1, 1); same = a1 == a2;"
          "ok = isvalid(a1); n = numel(a1.Children);")
    assert bool(m.num("same")) and bool(m.num("ok")) and m.num("n") == 1
    fig = _fig(m)
    assert len(fig.axes) == 2  # the axes and its colorbar
    assert len(fig.axes[0].images) == 1
    m.run("a3 = subplot(2, 1, 2); plot(1:3); colorbar('westoutside'); a4 = subplot(2, 1, 2); same2 = a3 == a4;")
    assert bool(m.num("same2")) and len(fig.axes) == 4


def test_subplot_after_outside_legend_and_axis_image_returns_the_same_axes(m):
    m.run("figure; a1 = subplot(2, 2, 1); plot(1:3); legend('x', 'Location', 'eastoutside');"
          "b1 = subplot(2, 2, 1); a2 = subplot(2, 2, 2); imagesc(magic(4)); axis image; b2 = subplot(2, 2, 2);"
          "a3 = subplot(2, 2, 3); plot(1:3); axis equal; b3 = subplot(2, 2, 3);"
          "same = [a1 == b1, a2 == b2, a3 == b3];")
    assert m.num("same").tolist() == [[True, True, True]]


def test_subplot_of_another_grid_still_replaces_overlapping_axes(m):
    m.run("figure; a1 = subplot(2, 1, 1); imagesc(magic(3)); colorbar; a2 = subplot(2, 2, 1); ok = isvalid(a1);")
    assert not bool(m.num("ok"))
    fig = _fig(m)
    assert len(fig.axes) == 1 and not hasattr(fig.axes[0], "_colorbar")
    m.run("figure; b1 = subplot(2, 2, [1 2]); imagesc(magic(3)); colorbar; b2 = subplot(2, 2, [1 2]);"
          "b3 = subplot(2, 2, 3); same = b1 == b2; ok = isvalid(b1);")
    assert bool(m.num("same")) and bool(m.num("ok"))  # the bottom-left cell does not overlap the top row
    m.run("figure; c1 = subplot(2, 1, 1); imagesc(magic(3)); colorbar; c2 = subplot('Position', [0.1 0.1 0.3 0.3]);"
          "ok = isvalid(c1); p = c2.Position; same = c1 == c2;")
    assert bool(m.num("ok")) and not bool(m.num("same"))
    np.testing.assert_allclose(m.num("p"), [[0.1, 0.1, 0.3, 0.3]])
