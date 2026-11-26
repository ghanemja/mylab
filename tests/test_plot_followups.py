"""Plotting behaviour that real code relies on: EPS preview requests and box plots of matrix columns."""

import numpy as np


def _axes(m):
    st = m.interp.state["plot"]
    return st.axes[st.current]


def test_eps_with_tiff_preview_option(m, tmp_path):
    f = (tmp_path / "fig.eps").as_posix()
    m.run(f"figure('Visible', 'off'); plot(1:3); print('-depsc', '-tiff', '-r100', '{f}'); close all")
    data = (tmp_path / "fig.eps").read_bytes()
    assert data.startswith(b"%!PS")


def test_boxplot_of_a_matrix_with_one_group_per_column(m):
    m.run("figure('Visible', 'off'); x = [1 10 100; 2 20 200; 3 30 300; 4 40 400];"
          "boxplot(x, {'low', 'mid', 'high'});")
    ax = _axes(m)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["low", "mid", "high"]
    medians = sorted(float(np.mean(ln.get_ydata())) for ln in ax.lines if len(ln.get_ydata()) == 2
                     and ln.get_ydata()[0] == ln.get_ydata()[1] and ln.get_ydata()[0] in (2.5, 25, 250))
    assert medians == [2.5, 25, 250]
    # columns that share a group value make one box
    m.run("figure('Visible', 'off'); boxplot(x, {'a', 'b', 'a'});")
    assert [t.get_text() for t in _axes(m).get_xticklabels()] == ["a", "b"]
    m.run("close all")


def test_set_on_no_objects_does_nothing(m):
    m.run("figure('Visible', 'off'); plot(1:3); txt = findobj(gca, 'Type', 'text');"
          "set(txt, 'FontSize', 12, 'FontName', 'Times'); set([], 'Color', 'r'); close all")
    assert m.num("isempty(txt)")
