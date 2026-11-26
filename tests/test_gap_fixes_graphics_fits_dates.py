"""view(3) on 2-D axes, shape functions on graphics arrays, lowess/loess and interpolant surface fits,
and dates in other languages."""

import math

import numpy as np
import pytest

from mylab import Interpreter, MatlabError
from mylab.values import to_str


@pytest.fixture
def m():
    return Interpreter(out=lambda s: None)


def num(m, expr):
    return np.asarray(m.eval_expression(expr))


def txt(m, expr):
    return to_str(m.eval_expression(expr))


# -- view(3) on a 2-D plot -----------------------------------------------------------------------------------

def test_view3_turns_a_2d_plot_into_a_3d_view_keeping_handles(m):
    m.execute("figure; h = plot(1:3, [2 4 3]); ax = gca; title('T'); xlabel('X'); t = text(2, 3, 'hi');"
              "view(3); [az, el] = view;")
    assert num(m, "az").item() == -37.5 and num(m, "el").item() == 30
    assert num(m, "isvalid(h)").item() and num(m, "ax == gca").item()
    assert txt(m, "class(h)") == "matlab.graphics.chart.primitive.Line"
    assert num(m, "h.XData").tolist() == [[1, 2, 3]] and num(m, "h.YData").tolist() == [[2, 4, 3]]
    assert num(m, "h.ZData").tolist() == [[0, 0, 0]]
    assert txt(m, "ax.Title.String") == "T" and txt(m, "get(get(gca, 'XLabel'), 'String')") == "X"
    assert num(m, "xlim").tolist() == [[1, 3]] and num(m, "zlim").tolist() == [[-1, 1]]
    assert num(m, "t.Position").tolist() == [[2, 3, 0]]
    assert num(m, "ax.View").tolist() == [[-37.5, 30]]
    m.execute("h.Color = [1 0 0];")  # the handle drives the line now drawn in 3-D
    assert num(m, "get(h, 'Color')").tolist() == [[1, 0, 0]]


def test_view2_returns_to_a_2d_axes(m):
    m.execute("figure; h = plot(1:3, [2 4 3]); ax = gca; hold on; grid on; view(3); view(2); [az, el] = view;")
    assert (num(m, "az").item(), num(m, "el").item()) == (0, 90)
    assert num(m, "isvalid(h)").item() and num(m, "ax == gca").item() and num(m, "ishold").item()
    assert num(m, "h.YData").tolist() == [[2, 4, 3]] and num(m, "xlim").tolist() == [[1, 3]]
    ax = m.get_var("ax").obj
    assert ax.name == "rectilinear" and any(g.get_visible() for g in ax.xaxis.get_gridlines())


@pytest.mark.parametrize("call, want", [
    ("view(30, 45)", [30, 45]), ("view([30 45])", [30, 45]), ("view([1 0 0])", [90, 0]),
    ("view([0 -1 0])", [0, 0]), ("set(gca, 'View', [10 20])", [10, 20]),
])
def test_view_angles_on_a_2d_axes(m, call, want):
    m.execute(f"figure; plot(1:4); {call}; v = view;")
    assert np.allclose(num(m, "v"), [want])
    x, y, z = 1.0, 1.0, 1.0  # the direction form agrees with the angles: az from -y, el above the plane
    m.execute(f"view([{x} {y} {z}]); [a, e] = view;")
    assert math.isclose(num(m, "a").item(), math.degrees(math.atan2(x, -y)))
    assert math.isclose(num(m, "e").item(), math.degrees(math.atan2(z, math.hypot(x, y))))


def test_view_of_a_2d_axes_seen_from_above_stays_2d(m):
    m.execute("figure; plot(1:3); view(2); [a, e] = view; v = get(gca, 'View');")
    assert (num(m, "a").item(), num(m, "e").item()) == (0, 90) and num(m, "v").tolist() == [[0, 90]]
    assert m.get_var("v") is not None and txt(m, "class(gca)") == "matlab.graphics.axis.Axes"


def test_plot3_after_plot_with_hold_keeps_the_2d_line(m):
    m.execute("figure; p = plot(1:3); hold on; q = plot3(1:3, 1:3, [1 2 3]); n = numel(get(gca, 'Children'));")
    assert num(m, "n").item() == 2 and num(m, "isvalid(p)").item()
    assert num(m, "p.ZData").tolist() == [[0, 0, 0]]


def test_view3_moves_patches_scatter_and_images(m):
    m.execute("figure; hold on; b = bar([1 2 3]); f = fill([0 1 1], [0 0 1], 'g'); s = scatter(1:3, 3:-1:1);"
              "legend('a', 'b', 'c'); view(3); n = numel(get(gca, 'Children')); drawnow;")
    assert num(m, "n").item() == 3
    assert num(m, "[isvalid(b) isvalid(f) isvalid(s)]").tolist() == [[True, True, True]]
    m.execute("view(2); n2 = numel(get(gca, 'Children')); sx = s.XData;")
    assert num(m, "n2").item() == 3 and num(m, "sx").tolist() == [[1, 2, 3]]
    m.execute("figure; imagesc(magic(4)); i = findobj(gca, 'Type', 'image'); view(3); drawnow; ok = isvalid(i);"
              "c = i.CData; view(2); xl = xlim;")
    assert num(m, "ok").item() and num(m, "c").tolist() == num(m, "magic(4)").tolist()
    assert num(m, "xl").tolist() == [[0.5, 4.5]]


def test_hand_set_limits_survive_view3(m):
    m.execute("figure; plot(1:3); xlim([0 5]); view(3); x = xlim; view(2); x2 = xlim;")
    assert num(m, "x").tolist() == [[0, 5]] and num(m, "x2").tolist() == [[0, 5]]


# -- graphics arrays: transpose, reshape, flip -------------------------------------------------------------

def test_transposing_graphics_arrays(m):
    m.execute("figure; h = plot(rand(3)); a = h'; b = h.'; c = h(:)'; d = transpose(h); e = ctranspose(h);")
    for name in "abcde":
        assert num(m, f"size({name})").tolist() == [[1, 3]]
    assert num(m, "a(3) == h(3)").item() and num(m, "size(a')").tolist() == [[3, 1]]
    m.execute("figure; h1 = plot(1:3); hold on; h2 = plot(2:4); g = [h1; h2]'; k = [h1 h2].';")
    assert num(m, "size(g)").tolist() == [[1, 2]] and num(m, "size(k)").tolist() == [[2, 1]]
    assert num(m, "g(2) == h2").item()


def test_reshape_flip_and_concatenation_of_graphics_arrays(m):
    m.execute("figure; h = plot(rand(3)); r = reshape(h, 1, 3); f = flip(h); w = [h h]; q = reshape(w, 2, []);"
              "p = permute(h, [2 1]); lr = fliplr(h'); t = w';")
    assert num(m, "size(r)").tolist() == [[1, 3]] and num(m, "size(f)").tolist() == [[3, 1]]
    assert num(m, "f(1) == h(3)").item() and num(m, "lr(1) == h(3)").item()
    assert num(m, "size(w)").tolist() == [[3, 2]] and num(m, "size(t)").tolist() == [[2, 3]]
    assert num(m, "size(q)").tolist() == [[2, 3]] and num(m, "q(2, 3) == h(3)").item()
    assert num(m, "size(p)").tolist() == [[1, 3]] and num(m, "size(w(2, :))").tolist() == [[1, 2]]
    assert num(m, "isequal(size(h([1 3])), [2 1])").item()
    with pytest.raises(MatlabError):
        m.execute("bad = [w; h'];")


# -- surface fits ---------------------------------------------------------------------------------------------

def _surface_data(n=40, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 4, n)
    y = rng.uniform(0, 3, n)
    z = np.sin(x) + np.cos(y) + 0.05 * rng.standard_normal(n)
    return x, y, z


def _put(m, **arrays):
    for k, v in arrays.items():
        m.set_var(k, np.asarray(v, dtype=float).reshape(-1, 1))


def _local_regression(x, y, z, q, span, quadratic):
    """Reference lowess/loess value at q: normal equations of a tricube-weighted polynomial in x and y."""
    n = x.size
    k = min(n, max(math.ceil(span * n), (6 if quadratic else 3) + 1))
    d = np.hypot(x - q[0], y - q[1])
    idx = np.argsort(d)[:k]
    w = (1 - (d[idx] / d[idx].max()) ** 3) ** 3
    terms = [np.ones(k), x[idx], y[idx]]
    at = [1.0, q[0], q[1]]
    if quadratic:
        terms += [x[idx] ** 2, x[idx] * y[idx], y[idx] ** 2]
        at += [q[0] ** 2, q[0] * q[1], q[1] ** 2]
    A = np.column_stack(terms)
    beta = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * z[idx]))
    return float(np.dot(at, beta))


@pytest.mark.parametrize("model, span", [("lowess", 0.25), ("loess", 0.5), ("lowess", 0.6)])
def test_lowess_and_loess_match_a_reference(m, model, span):
    x, y, z = _surface_data()
    _put(m, x=x, y=y, z=z)
    m.execute(f"f = fit([x y], z, '{model}', 'Span', {span}); v = f([1 2.5 3.2], [0.5 1.5 2.4]);")
    got = num(m, "v").ravel()
    want = [_local_regression(x, y, z, q, span, model == "loess") for q in [(1, 0.5), (2.5, 1.5), (3.2, 2.4)]]
    assert np.allclose(got, want, rtol=1e-9, atol=1e-12)
    assert txt(m, "class(f)") == "sfit" and txt(m, "type(f)") == model
    assert ("quadratic" if model == "loess" else "linear") in txt(m, "formula(f)")


def test_lowess_default_span_and_gof(m):
    x, y, z = _surface_data()
    _put(m, x=x, y=y, z=z)
    m.execute("[f, g] = fit([x y], z, 'lowess'); r = z - f(x, y);")
    want = [_local_regression(x, y, z, (a, b), 0.25, False) for a, b in zip(x, y)]
    assert np.allclose(num(m, "r").ravel(), z - want)
    assert math.isclose(num(m, "g.sse").item(), float(np.sum((z - want) ** 2)))


def test_robust_lowess_resists_an_outlier(m):
    x, y, z = _surface_data(60, seed=5)
    z = np.sin(x) + np.cos(y)
    z_bad = z.copy()
    k = int(np.argmin(np.hypot(x - 2, y - 1.5)))
    z_bad[k] += 5.0
    _put(m, x=x, y=y, z=z_bad)
    m.execute(f"p = fit([x y], z, 'lowess', 'Span', 0.3); r = fit([x y], z, 'lowess', 'Span', 0.3, "
              f"'Robust', 'Bisquare'); ep = abs(p(x({k + 1}), y({k + 1})) - {float(z[k])!r});"
              f"er = abs(r(x({k + 1}), y({k + 1})) - {float(z[k])!r});")
    assert num(m, "er").item() < 0.5 * num(m, "ep").item()


@pytest.mark.parametrize("model, method", [("linearinterp", "linear"), ("cubicinterp", "cubic"),
                                           ("nearestinterp", "nearest")])
def test_triangulation_interpolants_match_griddata(m, model, method):
    from scipy.interpolate import griddata
    x, y, z = _surface_data()
    _put(m, x=x, y=y, z=z)
    qx, qy = np.array([1.5, 2.2, 3.0]), np.array([1.0, 1.6, 2.0])
    m.execute(f"f = fit([x y], z, '{model}'); v = f([1.5 2.2 3.0]', [1.0 1.6 2.0]'); d = f(x, y);")
    want = griddata(np.column_stack([x, y]), z, (qx, qy), method=method)
    assert np.allclose(num(m, "v").ravel(), want)
    assert np.allclose(num(m, "d").ravel(), z)  # an interpolant passes through the data


def test_interpolants_are_nan_outside_the_data(m):
    x, y, z = _surface_data()
    _put(m, x=x, y=y, z=z)
    for model in ("linearinterp", "nearestinterp", "cubicinterp"):
        m.execute(f"f = fit([x y], z, '{model}'); v = f(10, 10);")
        assert np.isnan(num(m, "v").item())
    m.execute("f = fit([x y], z, 'thinplateinterp'); v = f(10, 10);")
    assert np.isfinite(num(m, "v").item())


def test_thin_plate_and_biharmonic_interpolants(m):
    from scipy.interpolate import RBFInterpolator
    x, y, z = _surface_data()
    _put(m, x=x, y=y, z=z)
    P = np.column_stack([x, y])
    Q = np.array([[1.5, 1.0], [2.2, 1.6], [3.7, 0.2]])
    m.execute("t = fit([x y], z, 'thinplateinterp'); vt = t([1.5 1; 2.2 1.6; 3.7 0.2]);"
              "b = fit([x y], z, 'biharmonicinterp'); vb = b([1.5 1; 2.2 1.6; 3.7 0.2]); db = b(x, y);")
    ref_t = RBFInterpolator(P, z, kernel="thin_plate_spline", degree=1)(Q)
    assert np.allclose(num(m, "vt").ravel(), ref_t)
    # biharmonic spline: z(p) = sum w_j r_j^2 (log r_j - 1), the weights solving the system at the data
    def green(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 0, r ** 2 * (np.log(r) - 1), 0.0)
    G = green(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2))
    wts = np.linalg.solve(G, z)
    ref_b = green(np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2)) @ wts
    assert np.allclose(num(m, "vb").ravel(), ref_b)
    assert np.allclose(num(m, "db").ravel(), z)
    assert txt(m, "type(b)") == "biharmonicinterp" and txt(m, "class(t)") == "sfit"


def test_repeated_points_are_averaged_by_interpolants(m):
    _put(m, x=[0, 1, 0, 1, 0], y=[0, 0, 1, 1, 0], z=[1, 2, 3, 4, 3])
    m.execute("f = fit([x y], z, 'linearinterp'); v = f(0, 0);")
    assert num(m, "v").item() == 2.0


def test_surface_fit_display_plot_and_errors(m):
    x, y, z = _surface_data()
    _put(m, x=x, y=y, z=z)
    out = []
    m2 = Interpreter(out=out.append)
    _put(m2, x=x, y=y, z=z)
    m2.execute("f = fit([x y], z, 'loess')")
    assert "Locally weighted smoothing quadratic regression" in "".join(out)
    m.execute("f = fit([x y], z, 'cubicinterp'); figure; h = plot(f, [x y], z);")
    assert txt(m, "class(h)") == "matlab.graphics.chart.primitive.Surface"
    with pytest.raises(MatlabError, match="does not apply"):
        m.execute("fit([x y], z, 'lowess', 'SmoothingParam', 0.5);")
    with pytest.raises(MatlabError, match="Span"):
        m.execute("fit([x y], z, 'lowess', 'Span', 1.5);")
    with pytest.raises(MatlabError, match="surface model"):
        m.execute("fit(x, z, 'thinplateinterp');")
    with pytest.raises(MatlabError, match="curve model"):
        m.execute("fit([x y], z, 'pchipinterp');")


# -- dates in other languages ----------------------------------------------------------------------------------

# Monday 15 March 2021 written as 'eeee d MMMM yyyy' in each language (facts of each calendar's names)
_MONDAY = {
    "fr_FR": "lundi 15 mars 2021", "de_DE": "Montag 15 März 2021", "es_ES": "lunes 15 marzo 2021",
    "it_IT": "lunedì 15 marzo 2021", "pt_BR": "segunda-feira 15 março 2021", "nl_NL": "maandag 15 maart 2021",
    "sv_SE": "måndag 15 mars 2021", "pl_PL": "poniedziałek 15 marca 2021", "ru_RU": "понедельник 15 марта 2021",
    "ja_JP": "月曜日 15 3月 2021", "zh_CN": "星期一 15 三月 2021", "en_US": "Monday 15 March 2021",
}


def test_the_reported_french_date(m):
    m.execute("d = datetime('1 janvier 2020', 'InputFormat', 'd MMMM yyyy', 'Locale', 'fr_FR');")
    assert num(m, "[year(d) month(d) day(d)]").tolist() == [[2020, 1, 1]]
    assert txt(m, "char(d, 'eeee d MMMM yyyy')") == "mercredi 1 janvier 2020"


@pytest.mark.parametrize("loc", sorted(_MONDAY))
def test_display_and_parse_in_each_language(m, loc):
    m.execute(f"t = datetime(2021, 3, 15, 'Locale', '{loc}', 'Format', 'eeee d MMMM yyyy'); s = char(t);")
    assert txt(m, "s") == _MONDAY[loc]
    m.execute(f"u = datetime('{_MONDAY[loc]}', 'InputFormat', 'eeee d MMMM yyyy', 'Locale', '{loc}');")
    assert num(m, "u == t").item()


@pytest.mark.parametrize("loc, text, hour", [
    ("es_ES", "14 julio 2020 3:00 p. m.", 15), ("nl_NL", "4 augustus 2020 3:15 p.m.", 15),
    ("ja_JP", "14 7月 2020 3:00 午後", 15), ("zh_CN", "14 七月 2020 9:00 上午", 9),
    ("sv_SE", "14 juli 2020 3:00 em", 15),
])
def test_am_pm_markers(m, loc, text, hour):
    m.execute(f"t = datetime('{text}', 'InputFormat', 'd MMMM yyyy h:mm a', 'Locale', '{loc}');")
    assert num(m, "hour(t)").item() == hour
    assert txt(m, "char(t, 'd MMMM yyyy h:mm a')") == text


def test_month_and_day_names_follow_the_locale(m):
    m.execute("d = datetime(2022, 5, 5, 'Locale', 'ru_RU'); mn = month(d, 'name'); dn = day(d, 'name');"
              "ms = month(d, 'shortname'); s = char(d, 'd MMMM yyyy');")
    assert to_str(m.get_var("mn").data[0, 0]) == "май"  # the month on its own: nominative
    assert txt(m, "s") == "5 мая 2022"  # next to a day number: genitive
    assert to_str(m.get_var("dn").data[0, 0]) == "четверг"
    assert to_str(m.get_var("ms").data[0, 0]) == "мая"
    m.execute("e = datetime(2022, 5, 5); en = month(e, 'name'); ed = day(e, 'shortname');")
    assert to_str(m.get_var("en").data[0, 0]) == "May" and to_str(m.get_var("ed").data[0, 0]) == "Thu"


def test_string_and_char_take_a_locale(m):
    m.execute("t = datetime(2020, 7, 14, 15, 0, 0);")
    assert txt(m, "string(t, 'eeee d MMMM yyyy h:mm a', 'es_ES')") == "martes 14 julio 2020 3:00 p. m."
    assert txt(m, "char(t, 'EEE d MMM', 'de_DE')") == "Di. 14 Juli"
    assert txt(m, "char(t, 'LLLL', 'pl_PL')") == "lipiec"
    assert txt(m, "char(datetime(2020, 9, 1, 'Locale', 'it_IT'), 'MMMM', 'en_US')") == "September"
    m.execute("c = cellstr(t, 'MMMM', 'nl_NL');")
    assert to_str(m.get_var("c").data[0, 0]) == "juli"


def test_default_locale_is_english(m):
    m.execute("t = datetime('15 March 2021', 'InputFormat', 'd MMMM yyyy'); s = char(t);")
    assert txt(m, "s") == "15-Mar-2021"
    m.execute("u = datetime('12 octobre 2021', 'Locale', 'fr_FR'); v = datetime(u, 'Format', 'MMMM');")
    assert num(m, "[year(u) month(u) day(u)]").tolist() == [[2021, 10, 12]] and txt(m, "char(v)") == "octobre"


def test_unknown_locale_is_an_error(m):
    with pytest.raises(MatlabError, match="locale"):
        m.execute("datetime('1 janvier 2020', 'InputFormat', 'd MMMM yyyy', 'Locale', 'xx_YY');")
    with pytest.raises(MatlabError, match="locale"):
        m.execute("char(datetime(2020, 1, 1), 'MMMM', 'klingon');")
