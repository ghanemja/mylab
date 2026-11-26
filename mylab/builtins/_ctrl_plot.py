"""Drawing of the standard control-system plots, on top of the plotting built-ins.

Each function receives ready-made data, so the analysis code stays free of graphics. Plots go
through ``plot``, ``semilogx``, ``subplot`` and friends, so figure numbering, ``hold on`` and
off-screen rendering behave as for any other plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..values import char


def call(interp, name: str, *args: Any, nargout: int = 0):
    conv = []
    for a in args:
        if isinstance(a, str):
            conv.append(char(a))
        elif isinstance(a, (bool, int, float)):
            conv.append(np.array([[float(a)]]))
        elif isinstance(a, np.ndarray):
            conv.append(np.atleast_2d(a))
        else:
            conv.append(a)
    return interp.call_builtin(name, interp.builtins[name], conv, nargout)


def held(interp) -> bool:
    """True when the current axes exist and have ``hold on`` (never creates a figure)."""
    st = interp.state.get("plot")
    if st is None or st.current is None or st.current not in st.figures:
        return False
    if not st.figures[st.current].axes:
        return False
    res = call(interp, "ishold", nargout=1)
    return bool(res and np.asarray(res[0]).any())


def time_unit_label(unit: str) -> str:
    return f"Time ({unit})"


def freq_unit_label(unit: str) -> str:
    short = {"seconds": "s", "minutes": "minute", "hours": "hour", "days": "day", "milliseconds": "ms",
             "microseconds": "us", "nanoseconds": "ns", "weeks": "week", "months": "month", "years": "year"}
    return f"rad/{short.get(unit, unit)}"


def _column(v) -> np.ndarray:
    return np.asarray(v, dtype=float).reshape(-1, 1)


def _draw(interp, fn: str, x, y, style: str | None, first: bool, hold: bool) -> None:
    if not first or hold:
        call(interp, "hold", "on")
    args = [_column(x), _column(y)]
    if style:
        args.append(style)
    call(interp, fn, *args)


def _chan(names: list, k: int, word: str) -> str:
    return names[k] if names[k] else f"{word}({k + 1})"


def time_plot(interp, title: str, items: list, in_names: list, out_names: list, unit: str = "seconds",
              ylabel: str = "Amplitude", single_column: bool = False) -> None:
    """items: dicts with t (n,), y (n, ny, nu), discrete (bool) and style."""
    ny, nu = items[0]["y"].shape[1], items[0]["y"].shape[2]
    hold = held(interp)
    multi = ny * nu > 1
    for i in range(ny):
        for j in range(nu):
            if multi:
                call(interp, "subplot", ny, nu, i * nu + j + 1)
            for n, it in enumerate(items):
                fn = "stairs" if it["discrete"] else "plot"
                _draw(interp, fn, it["t"], it["y"][:, i, j], it.get("style"), n == 0, hold)
            if not hold:
                call(interp, "hold", "off")
            if multi:
                if i == 0 and not single_column:
                    call(interp, "title", f"From: {_chan(in_names, j, 'In')}")
                if j == 0:
                    call(interp, "ylabel", f"To: {_chan(out_names, i, 'Out')}")
                if i == ny - 1:
                    call(interp, "xlabel", time_unit_label(unit))
            else:
                call(interp, "title", title)
                call(interp, "xlabel", time_unit_label(unit))
                call(interp, "ylabel", ylabel)
    if multi:
        call(interp, "sgtitle", title)


def lsim_plot(interp, items: list, u: np.ndarray, t: np.ndarray, out_names: list, unit: str, discrete: bool) -> None:
    """Outputs of each simulated system with the input signals drawn in grey behind them."""
    ny = items[0]["y"].shape[1]
    hold = held(interp)
    for i in range(ny):
        if ny > 1:
            call(interp, "subplot", ny, 1, i + 1)
        first = True
        if not hold:
            for k in range(u.shape[1]):
                _draw(interp, "stairs" if discrete else "plot", t, u[:, k], None, first, hold)
                _recolor_last(interp, (0.6, 0.6, 0.6))
                first = False
        for it in items:
            _draw(interp, "stairs" if it["discrete"] else "plot", it["t"], it["y"][:, i, 0], it.get("style"),
                  first, hold)
            first = False
        if not hold:
            call(interp, "hold", "off")
        if ny > 1:
            call(interp, "ylabel", f"To: {_chan(out_names, i, 'Out')}")
        else:
            call(interp, "ylabel", "Amplitude")
        if i == ny - 1:
            call(interp, "xlabel", time_unit_label(unit))
    if ny > 1:
        call(interp, "sgtitle", "Linear Simulation Results")
    else:
        call(interp, "title", "Linear Simulation Results")


def _recolor_last(interp, rgb: tuple) -> None:
    st = interp.state.get("plot")
    try:
        ax = st.axes.get(st.current)
        if ax is not None and ax.lines:
            ax.lines[-1].set_color(rgb)
    except (AttributeError, KeyError):
        pass


def bode_plot(interp, items: list, in_names: list, out_names: list, unit: str, mag_only: bool = False,
              title: str = "Bode Diagram", marks: dict | None = None) -> None:
    """items: dicts with w (n,), mag (ny, nu, n) absolute, phase (ny, nu, n) degrees, style."""
    ny, nu = items[0]["mag"].shape[0], items[0]["mag"].shape[1]
    hold = held(interp)
    rows = ny * (1 if mag_only else 2)
    xlabel = f"Frequency  ({freq_unit_label(unit)})"
    for i in range(ny):
        for j in range(nu):
            for part in range(1 if mag_only else 2):
                row = i * (1 if mag_only else 2) + part
                if rows * nu > 1:
                    call(interp, "subplot", rows, nu, row * nu + j + 1)
                for n, it in enumerate(items):
                    w = it["w"]
                    with np.errstate(divide="ignore"):
                        data = 20 * np.log10(it["mag"][i, j]) if part == 0 else it["phase"][i, j]
                    ok = np.isfinite(data) & (w > 0)
                    if not np.any(ok):
                        ok = w > 0
                        data = np.where(np.isfinite(data), data, np.nan)
                    _draw(interp, "semilogx", w[ok], data[ok], it.get("style"), n == 0, hold)
                if marks is not None:
                    _margin_marks(interp, marks, part)
                if not hold:
                    call(interp, "hold", "off")
                call(interp, "grid", "on")
                label = "Magnitude (dB)" if part == 0 else "Phase (deg)"
                if ny * nu > 1 and j == 0:
                    label = f"To: {_chan(out_names, i, 'Out')}\n{label}"
                if j == 0 or ny * nu == 1:
                    call(interp, "ylabel", label)
                if row == 0:
                    call(interp, "title", title if nu == 1 else f"From: {_chan(in_names, j, 'In')}")
                if row == rows - 1:
                    call(interp, "xlabel", xlabel)
    if nu > 1:
        call(interp, "sgtitle", title)


def _margin_marks(interp, marks: dict, part: int) -> None:
    """Vertical lines at the crossover frequencies of a margin plot."""
    call(interp, "hold", "on")
    wcg, wcp = marks.get("wcg"), marks.get("wcp")
    if part == 0:
        call(interp, "yline", 0.0, "k:")
        if wcg is not None and np.isfinite(wcg) and wcg > 0 and np.isfinite(marks.get("gm_db", np.nan)):
            call(interp, "plot", np.array([[wcg], [wcg]]), np.array([[0.0], [-marks["gm_db"]]]), "k-")
    else:
        call(interp, "yline", -180.0, "k:")
        if wcp is not None and np.isfinite(wcp) and wcp > 0 and np.isfinite(marks.get("pm", np.nan)):
            base = marks.get("phase_at_wcp", -180.0 + marks["pm"])
            call(interp, "plot", np.array([[wcp], [wcp]]), np.array([[base - marks["pm"]], [base]]), "k-")


def nyquist_plot(interp, items: list, in_names: list, out_names: list) -> None:
    """items: dicts with re, im of shape (ny, nu, n) and style; the mirror image is drawn too."""
    ny, nu = items[0]["re"].shape[0], items[0]["re"].shape[1]
    hold = held(interp)
    for i in range(ny):
        for j in range(nu):
            if ny * nu > 1:
                call(interp, "subplot", ny, nu, i * nu + j + 1)
            for n, it in enumerate(items):
                re, im = it["re"][i, j], it["im"][i, j]
                ok = np.isfinite(re) & np.isfinite(im)
                x = np.concatenate([re[ok][::-1], re[ok]])
                y = np.concatenate([-im[ok][::-1], im[ok]])
                _draw(interp, "plot", x, y, it.get("style"), n == 0, hold)
            call(interp, "hold", "on")
            call(interp, "plot", np.array([[-1.0]]), np.array([[0.0]]), "r+")
            if not hold:
                call(interp, "hold", "off")
            if ny * nu > 1:
                if i == 0:
                    call(interp, "title", f"From: {_chan(in_names, j, 'In')}")
                if j == 0:
                    call(interp, "ylabel", f"To: {_chan(out_names, i, 'Out')}")
            else:
                call(interp, "title", "Nyquist Diagram")
                call(interp, "xlabel", "Real Axis")
                call(interp, "ylabel", "Imaginary Axis")
    if ny * nu > 1:
        call(interp, "sgtitle", "Nyquist Diagram")


def nichols_plot(interp, items: list) -> None:
    hold = held(interp)
    for n, it in enumerate(items):
        with np.errstate(divide="ignore"):
            db = 20 * np.log10(it["mag"][0, 0])
        ok = np.isfinite(db)
        _draw(interp, "plot", it["phase"][0, 0][ok], db[ok], it.get("style"), n == 0, hold)
    if not hold:
        call(interp, "hold", "off")
    call(interp, "title", "Nichols Chart")
    call(interp, "xlabel", "Open-Loop Phase (deg)")
    call(interp, "ylabel", "Open-Loop Gain (dB)")


def sigma_plot(interp, items: list, unit: str) -> None:
    hold = held(interp)
    first = True
    for it in items:
        with np.errstate(divide="ignore"):
            db = 20 * np.log10(it["sv"])
        for k in range(db.shape[0]):
            ok = np.isfinite(db[k]) & (it["w"] > 0)
            _draw(interp, "semilogx", it["w"][ok], db[k][ok], it.get("style"), first, hold)
            first = False
    if not hold:
        call(interp, "hold", "off")
    call(interp, "grid", "on")
    call(interp, "title", "Singular Values")
    call(interp, "xlabel", f"Frequency ({freq_unit_label(unit)})")
    call(interp, "ylabel", "Singular Values (dB)")


def pz_plot(interp, items: list, unit: str, discrete: bool, title: str = "Pole-Zero Map") -> None:
    """items: dicts with p and z (complex vectors)."""
    hold = held(interp)
    first = True
    for it in items:
        for vals, marker in ((it["p"], "x"), (it["z"], "o")):
            vals = np.asarray(vals, dtype=complex)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            style = it.get("style")
            spec = marker if not style else "".join(ch for ch in style if ch.isalpha() and ch not in "xo") + marker
            _draw(interp, "plot", vals.real, vals.imag, spec, first, hold)
            first = False
    if first:
        _draw(interp, "plot", np.zeros(0), np.zeros(0), None, True, hold)
    call(interp, "hold", "on")
    if discrete:
        th = np.linspace(0, 2 * np.pi, 361)
        call(interp, "plot", _column(np.cos(th)), _column(np.sin(th)), "k:")
        call(interp, "axis", "equal")
    else:
        call(interp, "xline", 0.0, "k:")
    call(interp, "yline", 0.0, "k:")
    if not hold:
        call(interp, "hold", "off")
    call(interp, "title", title)
    suffix = "" if discrete else f" ({unit}^{{-1}})"
    call(interp, "xlabel", f"Real Axis{suffix}")
    call(interp, "ylabel", f"Imaginary Axis{suffix}")


def rlocus_plot(interp, r: np.ndarray, poles: np.ndarray, zeros: np.ndarray, unit: str, discrete: bool,
                style: str | None = None) -> None:
    hold = held(interp)
    first = True
    for branch in r:
        ok = np.isfinite(branch)
        if not np.any(ok):
            continue
        _draw(interp, "plot", branch[ok].real, branch[ok].imag, style, first, hold)
        first = False
    call(interp, "hold", "on")
    if poles.size:
        call(interp, "plot", _column(poles.real), _column(poles.imag), "kx")
    if zeros.size:
        call(interp, "plot", _column(zeros.real), _column(zeros.imag), "ko")
    if discrete:
        th = np.linspace(0, 2 * np.pi, 361)
        call(interp, "plot", _column(np.cos(th)), _column(np.sin(th)), "k:")
    if not hold:
        call(interp, "hold", "off")
    call(interp, "title", "Root Locus")
    suffix = "" if discrete else f" ({unit}^{{-1}})"
    call(interp, "xlabel", f"Real Axis{suffix}")
    call(interp, "ylabel", f"Imaginary Axis{suffix}")
