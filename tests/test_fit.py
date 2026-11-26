"""fit, fittype, fitoptions, cfit and sfit, checked against values computed here with NumPy/SciPy.

CDATE/POP are the U.S. decennial census counts 1790-1990 in millions (U.S. Census Bureau, public domain).
"""

import numpy as np
import pytest
from scipy import stats
from scipy.interpolate import CubicSpline, PchipInterpolator, make_smoothing_spline
from scipy.optimize import curve_fit

from mylab.errors import MatlabError

CDATE = np.arange(1790.0, 1991.0, 10.0)
POP = np.array([3.9, 5.3, 7.2, 9.6, 12.9, 17.1, 23.1, 31.4, 38.6, 50.2, 62.9, 76.0, 92.0, 105.7, 122.8, 131.7,
                150.7, 179.0, 205.0, 226.5, 248.7])


def _set(m, **arrays):
    for k, v in arrays.items():
        a = np.asarray(v, dtype=float)
        m.interp.set_var(k, a.reshape(-1, 1) if a.ndim == 1 else a)


def _poly_ci(x, y, deg, level=0.95, w=None):
    c, cov = np.polyfit(x, y, deg, cov="unscaled", w=None if w is None else np.sqrt(w))
    r = y - np.polyval(c, x)
    dfe = x.size - deg - 1
    mse = np.sum((1 if w is None else w) * r ** 2) / dfe
    t = stats.t.ppf((1 + level) / 2, dfe)
    se = np.sqrt(np.diag(cov) * mse)
    return c, np.vstack([c - t * se, c + t * se]), cov, mse, dfe


@pytest.fixture
def census(m):
    _set(m, cdate=CDATE, pop=POP)
    m.run("[f, gof, out] = fit(cdate, pop, 'poly2');")
    return m


def test_poly2_census_coefficients_bounds_and_display(census):
    m = census
    c, ci, *_ = _poly_ci(CDATE, POP, 2)
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), c, rtol=1e-9)
    np.testing.assert_allclose(m.num("confint(f)"), ci, rtol=1e-7)
    out = m.run("f")
    assert "Linear model Poly2:" in out
    assert "f(x) = p1*x^2 + p2*x + p3" in out
    assert "Coefficients (with 95% confidence bounds):" in out
    # one line per coefficient: the value right-aligned in 11 columns, then its bounds, all with %.4g
    for k, name in enumerate(("p1", "p2", "p3")):
        assert f"{name} = {c[k]:11.4g}  ({ci[0, k]:.4g}, {ci[1, k]:.4g})" in out
    assert m.num("f.p1").item() == pytest.approx(c[0], rel=1e-9)
    assert m.str("class(f)") == "cfit" and m.num("isa(f, 'cfit')").item()


def test_gof_and_output(census):
    m = census
    c = np.polyfit(CDATE, POP, 2)
    r = POP - np.polyval(c, CDATE)
    sse = r @ r
    sst = np.sum((POP - POP.mean()) ** 2)
    n, p = POP.size, 3
    assert m.num("gof.sse").item() == pytest.approx(sse, rel=1e-8)
    assert m.num("gof.rsquare").item() == pytest.approx(1 - sse / sst, rel=1e-10)
    assert m.num("gof.dfe").item() == n - p
    assert m.num("gof.adjrsquare").item() == pytest.approx(1 - (sse / sst) * (n - 1) / (n - p), rel=1e-10)
    assert m.num("gof.rmse").item() == pytest.approx(np.sqrt(sse / (n - p)), rel=1e-8)
    assert m.num("out.numobs").item() == n and m.num("out.numparam").item() == p
    np.testing.assert_allclose(m.num("out.residuals").ravel(), r, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(m.num("out.Jacobian"), np.column_stack([CDATE ** 2, CDATE, np.ones(n)]))
    assert m.num("out.exitflag").item() == 1


def test_predint_variants(census):
    m = census
    c, _, cov, mse, dfe = _poly_ci(CDATE, POP, 2)
    xq = np.array([1800.0, 1905.0, 2000.0])
    J = np.column_stack([xq ** 2, xq, np.ones(3)])
    yq = np.polyval(c, xq)
    vf = np.einsum("ij,jk,ik->i", J, cov, J) * mse
    _set(m, xq=xq)
    cases = {
        "predint(f, xq)": stats.t.ppf(0.975, dfe) * np.sqrt(vf + mse),
        "predint(f, xq, 0.9, 'functional')": stats.t.ppf(0.95, dfe) * np.sqrt(vf),
        "predint(f, xq, 0.95, 'functional', 'on')": np.sqrt(3 * stats.f.ppf(0.95, 3, dfe)) * np.sqrt(vf),
        "predint(f, xq, 0.95, 'observation', 'on')": np.sqrt(4 * stats.f.ppf(0.95, 4, dfe)) * np.sqrt(vf + mse),
    }
    for code, half in cases.items():
        np.testing.assert_allclose(m.num(code), np.column_stack([yq - half, yq + half]), rtol=1e-7, err_msg=code)
    m.run("[b, y] = predint(f, xq);")
    np.testing.assert_allclose(m.num("y").ravel(), yq, rtol=1e-9)
    np.testing.assert_allclose(m.num("confint(f, 0.99)"), _poly_ci(CDATE, POP, 2, 0.99)[1], rtol=1e-7)


def test_evaluate_differentiate_integrate_poly(census):
    m = census
    c = np.polyfit(CDATE, POP, 2)
    _set(m, xq=np.array([1850.0, 1950.0]))
    np.testing.assert_allclose(m.num("f(xq)").ravel(), np.polyval(c, [1850, 1950]), rtol=1e-9)
    np.testing.assert_allclose(m.num("feval(f, [1850 1950])"), [np.polyval(c, [1850, 1950])], rtol=1e-9)
    assert m.num("f([1850 1950])").shape == (1, 2)
    m.run("[d1, d2] = differentiate(f, xq);")
    np.testing.assert_allclose(m.num("d1").ravel(), np.polyval(np.polyder(c), [1850, 1950]), rtol=1e-8)
    np.testing.assert_allclose(m.num("d2").ravel(), [2 * c[0]] * 2, rtol=1e-8)
    ci = np.polyint(c)
    np.testing.assert_allclose(m.num("integrate(f, xq, 1800)").ravel(),
                               np.polyval(ci, [1850, 1950]) - np.polyval(ci, 1800), rtol=1e-8)


def test_normalize(m):
    _set(m, cdate=CDATE, pop=POP)
    m.run("f = fit(cdate, pop, 'poly3', 'Normalize', 'on'); g = fit(cdate, pop, 'poly3');")
    z = (CDATE - CDATE.mean()) / CDATE.std(ddof=1)
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), np.polyfit(z, POP, 3), rtol=1e-9)
    np.testing.assert_allclose(m.num("f(cdate)"), m.num("g(cdate)"), rtol=1e-7)
    out = m.run("f")
    assert f"where x is normalized by mean 1890 and std {CDATE.std(ddof=1):.4g}" in out


def test_weights_exclude_and_robust(m):
    x = np.linspace(0, 10, 15)
    y = 2 * x + 1 + 0.05 * np.cos(3 * x)
    y[4] = 40.0
    w = np.linspace(1, 3, 15)
    _set(m, x=x, y=y, w=w)
    c, ci, *_ = _poly_ci(x, y, 1, w=w)
    m.run("fw = fit(x, y, 'poly1', 'Weights', w);")
    np.testing.assert_allclose(m.num("coeffvalues(fw)").ravel(), c, rtol=1e-9)
    np.testing.assert_allclose(m.num("confint(fw)"), ci, rtol=1e-7)
    keep = np.arange(15) != 4
    m.run("fe = fit(x, y, 'poly1', 'Exclude', 5); fl = fit(x, y, 'poly1', 'Exclude', x > 2.8 & x < 2.9);")
    np.testing.assert_allclose(m.num("coeffvalues(fe)").ravel(), np.polyfit(x[keep], y[keep], 1), rtol=1e-9)
    np.testing.assert_allclose(m.num("coeffvalues(fl)").ravel(), np.polyfit(x[keep], y[keep], 1), rtol=1e-9)
    for robust in ("Bisquare", "LAR", "on"):
        m.run(f"fr = fit(x, y, 'poly1', 'Robust', '{robust}');")
        np.testing.assert_allclose(m.num("coeffvalues(fr)").ravel(), [2, 1], atol=0.05)


def test_bounds_on_linear_model(m):
    x = np.linspace(0, 1, 20)
    _set(m, x=x, y=3 * x - 0.5)
    m.run("f = fit(x, y, 'poly1', 'Lower', [0 0], 'Upper', [10 10]);")
    from scipy.optimize import lsq_linear
    want = lsq_linear(np.column_stack([x, np.ones_like(x)]), 3 * x - 0.5, bounds=([0, 0], [10, 10])).x
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), want, atol=1e-8)


def test_exp1_closed_form(m):
    x = np.linspace(0, 4, 30)
    _set(m, x=x, y=2.5 * np.exp(-0.7 * x))
    m.run("[f, gof] = fit(x, y, 'exp1'); [d1, d2] = differentiate(f, [1 2]); I = integrate(f, 2, 0);")
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), [2.5, -0.7], rtol=1e-7)
    assert m.num("gof.sse").item() < 1e-15
    np.testing.assert_allclose(m.num("d1").ravel(), -0.7 * 2.5 * np.exp(-0.7 * np.array([1, 2])), rtol=1e-6)
    np.testing.assert_allclose(m.num("d2").ravel(), 0.49 * 2.5 * np.exp(-0.7 * np.array([1, 2])), rtol=1e-4)
    assert m.num("I").item() == pytest.approx(2.5 / 0.7 * (1 - np.exp(-1.4)), rel=1e-8)
    out = m.run("f")
    assert "General model Exp1:" in out and "f(x) = a*exp(b*x)" in out


def _noisy(x, seed=3, scale=0.02):
    return scale * np.random.default_rng(seed).standard_normal(x.size)


def _check_against_curve_fit(m, model, fn, p_true, x, y, rel=1e-4):
    """The fit reaches the least-squares optimum that SciPy's curve_fit finds from the true parameters."""
    _set(m, x=x, y=y)
    m.run(f"[f, gof] = fit(x, y, '{model}');")
    got = m.num("coeffvalues(f)").ravel()
    ref, pcov = curve_fit(fn, x, y, p0=p_true, maxfev=20000)
    sse_ref = float(np.sum((fn(x, *ref) - y) ** 2))
    assert m.num("gof.sse").item() <= sse_ref * (1 + 1e-6) + 1e-14
    np.testing.assert_allclose(got, ref, rtol=rel, atol=1e-6)
    dfe = x.size - len(p_true)
    half = stats.t.ppf(0.975, dfe) * np.sqrt(np.diag(pcov))
    np.testing.assert_allclose(m.num("confint(f)"), np.vstack([ref - half, ref + half]), rtol=1e-3, atol=1e-6)


def test_gauss2(m):
    x = np.linspace(0, 10, 80)

    def g2(x, a1, b1, c1, a2, b2, c2):
        return a1 * np.exp(-((x - b1) / c1) ** 2) + a2 * np.exp(-((x - b2) / c2) ** 2)
    p = [3, 4, 1.5, 1.5, 7.5, 0.8]
    _check_against_curve_fit(m, "gauss2", g2, p, x, g2(x, *p) + _noisy(x))


def test_sin1_and_fourier1(m):
    x = np.linspace(0, 12, 90)

    def s1(x, a, b, c):
        return a * np.sin(b * x + c)
    _check_against_curve_fit(m, "sin1", s1, [2, 1.3, 0.4], x, s1(x, 2, 1.3, 0.4) + _noisy(x))

    def f1(x, a0, a1, b1, w):
        return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w)
    _check_against_curve_fit(m, "fourier1", f1, [1, 2, -0.5, 0.9], x, f1(x, 1, 2, -0.5, 0.9) + _noisy(x))


def test_exp2_power2_rat11(m):
    x = np.linspace(0.5, 10, 50)

    def e2(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d * x)
    _check_against_curve_fit(m, "exp2", e2, [3, -0.4, 2, -0.05], x, e2(x, 3, -0.4, 2, -0.05) + _noisy(x, 1, 0.001))

    def p2(x, a, b, c):
        return a * x ** b + c
    _check_against_curve_fit(m, "power2", p2, [2, 1.5, 3], x, p2(x, 2, 1.5, 3) + _noisy(x))

    def r11(x, p1, p2, q1):
        return (p1 * x + p2) / (x + q1)
    _check_against_curve_fit(m, "rat11", r11, [2, 1, 3], x, r11(x, 2, 1, 3) + _noisy(x, 2, 0.001))


def test_custom_equation_and_problem_parameter(m):
    x = np.linspace(0, 5, 25)
    y = 2 * np.exp(-0.3 * x) + 0.5 + _noisy(x, 5, 0.01)
    _set(m, x=x, y=y)
    m.run("ft = fittype('a*exp(-b*x)+c'); f = fit(x, y, ft, 'StartPoint', [1 1 1]);")
    ref, _ = curve_fit(lambda x, a, b, c: a * np.exp(-b * x) + c, x, y, p0=[1, 1, 1])
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), ref, rtol=1e-6)
    assert m.str("formula(f)") == "a*exp(-b*x)+c"
    assert m.str("type(f)") == "customnonlinear" and m.str("category(f)") == "custom"
    assert not m.num("islinear(f)").item()
    out = m.run("ft")
    assert "General model:" in out and "ft(a,b,c,x) = a*exp(-b*x)+c" in out
    np.testing.assert_allclose(m.num("ft(2, 0.3, 0.5, [0 1])"), [[2.5, 2 * np.exp(-0.3) + 0.5]])
    # a problem parameter is fixed during the fit
    m.run("g = fittype('a*x^n + b', 'problem', 'n'); h = fit(x, 3*x.^2 + 1, g, 'problem', 2, 'StartPoint', [1 1]);")
    np.testing.assert_allclose(m.num("coeffvalues(h)").ravel(), [3, 1], rtol=1e-8)
    assert m.num("probvalues(h)").item() == 2
    assert m.run("disp(argnames(g))").split() == ["{'a'}", "{'b'}", "{'n'}", "{'x'}"]
    assert "Problem parameters:" in m.run("h")


def test_custom_options_and_handle(m):
    x = np.linspace(1, 3, 12)
    _set(m, t=x, y=4 * x + 2)
    m.run("ft = fittype('k*t + d', 'independent', 't', 'coefficients', {'k', 'd'}); f = fit(t, y, ft, "
          "'StartPoint', [1 1]); h = fittype(@(a, b, x) a*x + b); g = fit(t, y, h, 'StartPoint', [0 0]);")
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), [4, 2], rtol=1e-8)
    assert m.run("disp(coeffnames(f))").split() == ["{'k'}", "{'d'}"]
    assert m.run("disp(indepnames(f))").split() == ["{'t'}"]
    np.testing.assert_allclose(m.num("coeffvalues(g)").ravel(), [4, 2], rtol=1e-8)
    assert m.num("numcoeffs(h)").item() == 2


def test_random_start_warns(m):
    _set(m, x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 10) * 2)
    out = m.run("f = fit(x, y, 'a*x');")
    assert "Start point not provided, choosing random start point." in out
    assert m.num("f.a").item() == pytest.approx(2, rel=1e-8)


def test_linear_model_terms(m):
    x = np.linspace(0, 6, 30)
    y = 0.5 * x - 1.2 * np.sin(x) + 3 + _noisy(x)
    _set(m, x=x, y=y)
    m.run("ft = fittype({'x', 'sin(x)', '1'}); f = fit(x, y, ft);")
    A = np.column_stack([x, np.sin(x), np.ones_like(x)])
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), np.linalg.lstsq(A, y, rcond=None)[0], rtol=1e-9)
    assert m.num("islinear(ft)").item()
    assert "f(x) = a*x + b*sin(x) + c" in m.run("f")
    assert "Linear model:" in m.run("f")


def test_interpolants_and_smoothing_spline(m):
    x = np.array([0.0, 0.4, 1.1, 1.5, 2.3, 3.0, 3.8, 4.4])
    y = np.sin(x) + 0.1 * x
    xq = np.array([0.2, 1.3, 2.9, 4.1])
    _set(m, x=x, y=y, xq=xq)
    np.testing.assert_allclose(m.num("feval(fit(x, y, 'linearinterp'), xq)").ravel(), np.interp(xq, x, y))
    np.testing.assert_allclose(m.num("feval(fit(x, y, 'pchipinterp'), xq)").ravel(), PchipInterpolator(x, y)(xq))
    np.testing.assert_allclose(m.num("feval(fit(x, y, 'cubicinterp'), xq)").ravel(), CubicSpline(x, y)(xq))
    m.run("[s, gof, out] = fit(x, y, 'smoothingspline', 'SmoothingParam', 0.8);")
    ref = make_smoothing_spline(x, y, lam=(1 - 0.8) / 0.8)
    np.testing.assert_allclose(m.num("s(xq)").ravel(), ref(xq), rtol=1e-8)
    assert m.num("out.p").item() == 0.8
    out = m.run("s")
    assert "Smoothing spline:" in out and "s(x) = piecewise polynomial computed from p" in out
    m.run("[li, g2] = fit(x, y, 'linearinterp');")
    assert m.num("g2.sse").item() == 0 and m.num("g2.dfe").item() == 0
    assert m.str("category(li)") == "interpolant"


def test_surface_poly22(m):
    rng = np.random.default_rng(7)
    x, y = rng.uniform(-2, 2, 40), rng.uniform(0, 3, 40)
    z = 1 + 2 * x - y + 0.5 * x ** 2 + 0.3 * x * y - 0.2 * y ** 2 + 0.01 * rng.standard_normal(40)
    _set(m, x=x, y=y, z=z)
    m.run("[f, gof] = fit([x, y], z, 'poly22');")
    A = np.column_stack([np.ones_like(x), x, y, x ** 2, x * y, y ** 2])
    c = np.linalg.lstsq(A, z, rcond=None)[0]
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), c, rtol=1e-8)
    assert m.str("class(f)") == "sfit"
    np.testing.assert_allclose(m.num("f(0.5, 1)").item(), c @ [1, 0.5, 1, 0.25, 0.5, 1], rtol=1e-10)
    np.testing.assert_allclose(m.num("f([0.5 1; 1 2])").ravel(),
                               [c @ [1, 0.5, 1, 0.25, 0.5, 1], c @ [1, 1, 2, 1, 2, 4]], rtol=1e-10)
    out = m.run("f")
    assert "Linear model Poly22:" in out
    assert "f(x,y) = p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2" in out
    assert m.run("disp(coeffnames(fittype('poly23')))").split() == [
        "{'p00'}", "{'p10'}", "{'p01'}", "{'p20'}", "{'p11'}", "{'p02'}", "{'p21'}", "{'p12'}", "{'p03'}"]
    m.run("[fx, fy] = differentiate(f, 0.5, 1);")
    assert m.num("fx").item() == pytest.approx(c[1] + 2 * c[3] * 0.5 + c[4] * 1, rel=1e-6)
    assert m.num("fy").item() == pytest.approx(c[2] + c[4] * 0.5 + 2 * c[5] * 1, rel=1e-6)


def test_library_formulas_and_names(m):
    cases = {
        "poly3": "p1*x^3 + p2*x^2 + p3*x + p4",
        "exp2": "a*exp(b*x) + c*exp(d*x)",
        "power1": "a*x^b",
        "gauss1": "a1*exp(-((x-b1)/c1)^2)",
        "sin2": "a1*sin(b1*x+c1) + a2*sin(b2*x+c2)",
        "fourier2": "a0 + a1*cos(x*w) + b1*sin(x*w) + a2*cos(2*x*w) + b2*sin(2*x*w)",
        "rat23": "(p1*x^2 + p2*x + p3) / (x^3 + q1*x^2 + q2*x + q3)",
        "weibull": "a*b*x^(b-1)*exp(-a*x^b)",
    }
    for name, formula in cases.items():
        assert m.str(f"formula(fittype('{name}'))") == formula
    assert m.num("numcoeffs(fittype('rat55'))").item() == 11
    assert m.num("numcoeffs(fittype('fourier8'))").item() == 18
    assert m.num("numcoeffs(fittype('gauss8'))").item() == 24
    assert m.str("type(fittype('poly9'))") == "poly9" and m.str("category(fittype('exp1'))") == "library"
    assert m.str("category(fittype('smoothingspline'))") == "spline"


def test_fitoptions(m):
    out = m.run("opts = fitoptions('gauss2')")
    assert "nlsqoptions" in out and "Method: 'NonlinearLeastSquares'" in out
    np.testing.assert_array_equal(m.num("opts.Lower"), [[-np.inf, -np.inf, 0, -np.inf, -np.inf, 0]])
    m.run("o = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', [1 1], 'Lower', [0 -Inf]);"
          "o.MaxIter = 50; x = linspace(0, 2, 15)'; y = 1.5*exp(0.4*x); f = fit(x, y, 'exp1', o);")
    assert m.num("o.MaxIter").item() == 50
    np.testing.assert_allclose(m.num("coeffvalues(f)").ravel(), [1.5, 0.4], rtol=1e-7)
    assert m.str("fitoptions('poly2').Method") == "LinearLeastSquares"
    assert m.str("class(fitoptions('poly2'))") == "curvefit.llsqoptions"
    with pytest.raises(MatlabError, match="NoSuchOption__"):
        m.run("fitoptions('Method', 'SmoothingSpline', 'NoSuchOption__', 2);")
    with pytest.raises(MatlabError, match="not an option"):
        m.run("fitoptions('poly2', 'StartPoint', [1 2 3]);")


def test_cfit_constructor_and_coefficient_assignment(m):
    m.run("f = cfit(fittype('exp1'), 2, -0.5); v = f(1); f.a = 4; w = f(1);")
    assert m.num("v").item() == pytest.approx(2 * np.exp(-0.5))
    assert m.num("w").item() == pytest.approx(4 * np.exp(-0.5))
    assert "Coefficients:" in m.run("f")


def test_plot_draws_data_and_fit(m):
    _set(m, x=np.linspace(0, 3, 12), y=np.linspace(0, 3, 12) ** 2)
    m.run("figure('Visible', 'off'); f = fit(x, y, 'poly2'); h = plot(f, x, y); n = numel(h);"
          "lg = legend; s = lg.String; ax = gca; nl = numel(ax.Children);")
    assert m.num("n").item() == 2 and m.num("nl").item() == 2
    assert m.run("disp(s)").split() == ["{'data'}", "{'fitted", "curve'}"]
    m.run("clf; plot(f, x, y, 'residuals'); plot(f); close all;")


def test_errors(m):
    _set(m, x=np.linspace(0, 1, 6), y=np.linspace(0, 1, 6))
    with pytest.raises(MatlabError, match="one or two columns"):
        m.run("fit(x', y', 'poly1');")
    with pytest.raises(MatlabError, match="column vector"):
        m.run("fit(x, y', 'poly1');")
    with pytest.raises(MatlabError, match="not a library model"):
        m.run("fit(x, y, 'poly42x');")
    with pytest.raises(MatlabError, match="NoSuchOption__"):
        m.run("fit(x, y, 'poly1', 'NoSuchOption__', 1);")
    with pytest.raises(MatlabError, match="does not apply"):
        m.run("fit(x, y, 'poly1', 'StartPoint', [1 1]);")
    with pytest.raises(MatlabError, match="bound constraints"):
        m.run("fit(x, y, 'exp1', 'Algorithm', 'Levenberg-Marquardt', 'Lower', [0 0]);")
    with pytest.raises(MatlabError, match="NaN or Inf"):
        m.run("y2 = y; y2(2) = NaN; fit(x, y2, 'poly1');")
    with pytest.raises(MatlabError, match="positive x"):
        m.run("fit(x, y + 1, 'power1');")
    m.run("y3 = y; y3(2) = NaN; [xx, yy] = prepareCurveData(x, y3);")
    assert m.num("numel(xx)").item() == 5
