"""The standard control plots are drawn headless with the usual titles and labels."""

import numpy as np
import pytest

from mylab.errors import MatlabError


def figure_axes(m):
    st = m.interp.state["plot"]
    return st.figures[st.current].axes


def labels(ax):
    return ax.get_title(), ax.get_xlabel(), ax.get_ylabel()


@pytest.fixture
def fig(m):
    m.run("f = figure('Visible', 'off');")
    yield m
    m.run("close all")


def test_step_plot(fig):
    fig.run("step(tf(1, [1 1]));")
    ax = figure_axes(fig)[0]
    assert labels(ax) == ("Step Response", "Time (seconds)", "Amplitude")
    x, y = ax.lines[0].get_data()
    assert x[0] == 0 and abs(y[-1] - 1) < 5e-3
    fig.run("step(tf(1, [1 1]), tf(1, [1 2]), 'r--', 5);")       # two models, one with a line style
    ax = figure_axes(fig)[0]
    assert len(ax.lines) == 2 and ax.lines[1].get_linestyle() == "--" and ax.lines[1].get_xdata()[-1] == 5


def test_hold_adds_a_second_response(fig):
    fig.run("step(tf(1, [1 1])); hold on; step(tf(1, [1 2])); hold off;")
    assert len(figure_axes(fig)[0].lines) == 2


def test_mimo_step_uses_one_axes_per_channel(fig):
    fig.run("step([tf(1, [1 1]), tf(1, [1 2]); tf(1, [1 3]), tf(1, [1 4])]);")
    axes = figure_axes(fig)
    assert len(axes) == 4
    assert axes[0].get_title() == "From: In(1)" and axes[0].get_ylabel() == "To: Out(1)"
    assert axes[3].get_xlabel() == "Time (seconds)"


def test_impulse_initial_lsim_plots(fig):
    fig.run("impulse(tf(1, [1 1]));")
    assert figure_axes(fig)[0].get_title() == "Impulse Response"
    fig.run("clf; initial(ss(-1, 1, 1, 0), 2);")
    assert figure_axes(fig)[0].get_title() == "Response to Initial Conditions"
    fig.run("clf; t = 0:0.1:5; lsim(tf(1, [1 1]), sin(t), t);")
    ax = figure_axes(fig)[0]
    assert ax.get_title() == "Linear Simulation Results" and len(ax.lines) == 2      # input and output


def test_discrete_step_is_a_staircase(fig):
    fig.run("step(tf(1, [1 -0.5], 0.1));")
    x = figure_axes(fig)[0].lines[0].get_xdata()
    assert np.any(np.diff(x) == 0) or figure_axes(fig)[0].lines[0].get_drawstyle().startswith("steps")


def test_bode_plot(fig):
    fig.run("bode(tf(1, [1 1]));")
    top, bottom = figure_axes(fig)[:2]
    assert top.get_title() == "Bode Diagram" and top.get_ylabel() == "Magnitude (dB)"
    assert bottom.get_ylabel() == "Phase (deg)" and bottom.get_xlabel() == "Frequency  (rad/s)"
    assert top.get_xscale() == "log"
    w, db = top.lines[0].get_data()
    k = int(np.argmin(np.abs(w - 1)))
    assert abs(db[k] + 3.01) < 0.3 and abs(bottom.lines[0].get_ydata()[k] + 45) < 5
    fig.run("clf; bodemag(tf(1, [1 1]), tf(1, [1 2]));")
    axes = figure_axes(fig)
    assert len(axes) == 1 and len(axes[0].lines) == 2


def test_margin_plot_title(fig):
    fig.run("margin(tf(1, [1 3 2 0]));")
    title = figure_axes(fig)[0].get_title()
    assert "Gm = 15.6 dB (at 1.41 rad/s)" in title and "Pm = 53.4 deg (at 0.446 rad/s)" in title


def test_nyquist_nichols_sigma_plots(fig):
    fig.run("nyquist(tf(1, [1 1]));")
    ax = figure_axes(fig)[0]
    assert labels(ax) == ("Nyquist Diagram", "Real Axis", "Imaginary Axis")
    x, y = ax.lines[0].get_data()
    assert np.max(y) > 0.49 and np.min(y) < -0.49            # both halves of the circle of diameter 1
    fig.run("clf; nichols(tf(1, [1 1 0]));")
    assert labels(figure_axes(fig)[0]) == ("Nichols Chart", "Open-Loop Phase (deg)", "Open-Loop Gain (dB)")
    fig.run("clf; sigma([tf(1, [1 1]), tf(1, [1 2]); tf(1, [1 3]), tf(1, [1 4])]);")
    ax = figure_axes(fig)[0]
    assert ax.get_title() == "Singular Values" and len(ax.lines) == 2


def test_pzmap_and_rlocus_plots(fig):
    fig.run("pzmap(tf([1 2], [1 2 5]));")
    ax = figure_axes(fig)[0]
    assert ax.get_title() == "Pole-Zero Map"
    assert ax.get_xlabel() in ("Real Axis (seconds^{-1})", "Real Axis (seconds$^{\\mathrm{-1}}$)")
    assert ax.lines[0].get_marker() == "x" and ax.lines[1].get_marker() == "o"
    assert sorted(ax.lines[0].get_ydata().tolist()) == [-2, 2]
    fig.run("clf; rlocus(tf(1, [1 3 2 0])); sgrid;")
    ax = figure_axes(fig)[0]
    assert ax.get_title() == "Root Locus" and len(ax.lines) >= 4
    fig.run("clf; pzmap(tf(1, [1 -0.5], 0.1)); zgrid;")
    assert figure_axes(fig)[0].get_xlabel() == "Real Axis"


def test_plot_functions_reject_bad_input(fig):
    with pytest.raises(MatlabError, match="one model at a time"):
        fig.run("[y, t] = step(tf(1, [1 1]), tf(1, [1 2]));")
    with pytest.raises(MatlabError, match="same numbers of inputs and outputs"):
        fig.run("step(tf(1, [1 1]), [tf(1, [1 1]), tf(1, [1 2])]);")
    with pytest.raises(MatlabError, match="control-system model"):
        fig.run("step(1, [1 1]);")
    with pytest.raises(MatlabError, match="only draws"):
        fig.run("m = bodemag(tf(1, [1 1]));")
