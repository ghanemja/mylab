"""griddedInterpolant: values checked against scipy.interpolate and closed forms computed here."""

import numpy as np
import pytest
from scipy.interpolate import (Akima1DInterpolator, CubicSpline, PchipInterpolator, RegularGridInterpolator,
                               interp1d)

from mylab.errors import MatlabError


def _set(m, **arrays):
    for k, v in arrays.items():
        m.interp.set_var(k, np.asarray(v, dtype=float))


X1 = np.array([0.0, 0.7, 1.5, 2.0, 3.2, 4.0, 5.5])
Y1 = np.sin(X1) + 0.3 * X1
XQ = np.array([-0.5, 0.0, 0.35, 1.1, 2.6, 3.9, 5.5, 6.2])


def test_1d_linear_and_default_extrapolation(m):
    _set(m, x=X1.reshape(-1, 1), v=Y1.reshape(-1, 1), xq=XQ.reshape(1, -1))
    m.run("F = griddedInterpolant(x, v);")
    got = m.num("F(xq)")
    assert got.shape == (1, XQ.size)
    want = interp1d(X1, Y1, fill_value="extrapolate")(XQ)
    np.testing.assert_allclose(got.ravel(), want, rtol=1e-12)
    inside = (XQ >= X1[0]) & (XQ <= X1[-1])
    np.testing.assert_allclose(got.ravel()[inside], np.interp(XQ[inside], X1, Y1), rtol=1e-12)
    assert m.str("F.Method") == "linear" and m.str("F.ExtrapolationMethod") == "linear"


@pytest.mark.parametrize("method,ref", [
    ("spline", lambda x, y: CubicSpline(x, y, bc_type="not-a-knot")),
    ("pchip", lambda x, y: PchipInterpolator(x, y)),
    ("makima", lambda x, y: Akima1DInterpolator(x, y, method="makima", extrapolate=True)),
])
def test_1d_piecewise_cubic_methods(m, method, ref):
    _set(m, x=X1, v=Y1, xq=XQ)
    m.run(f"F = griddedInterpolant(x, v, '{method}'); r = F(xq);")
    got = m.num("r").ravel()
    np.testing.assert_allclose(got, ref(X1, Y1)(XQ), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", ["nearest", "next", "previous"])
def test_1d_piecewise_constant_methods(m, method):
    q = np.array([0.1, 0.69, 0.71, 1.9, 3.3, 5.4])  # no ties, all inside
    _set(m, x=X1, v=Y1, xq=q)
    m.run(f"F = griddedInterpolant(x, v, '{method}'); r = F(xq);")
    np.testing.assert_allclose(m.num("r").ravel(), interp1d(X1, Y1, kind=method)(q))


def test_1d_next_previous_outside_range(m):
    _set(m, x=X1, v=Y1)
    m.run("F = griddedInterpolant(x, v, 'next'); a = F([-1 7]);"
          "G = griddedInterpolant(x, v, 'previous'); b = G([-1 7]);")
    a, b = m.num("a").ravel(), m.num("b").ravel()
    assert a[0] == Y1[0] and np.isnan(a[1])
    assert np.isnan(b[0]) and b[1] == Y1[-1]


def test_extrapolation_method_and_none(m):
    _set(m, x=X1, v=Y1)
    m.run("F = griddedInterpolant(x, v, 'spline', 'nearest'); a = F([-1 2.6 9]);"
          "G = griddedInterpolant(x, v, 'linear', 'none'); b = G([-1 2.6 9]);")
    a, b = m.num("a").ravel(), m.num("b").ravel()
    assert a[0] == Y1[0] and a[2] == Y1[-1]
    assert a[1] == pytest.approx(float(CubicSpline(X1, Y1)(2.6)), rel=1e-12)
    assert np.isnan(b[0]) and np.isnan(b[2]) and b[1] == pytest.approx(np.interp(2.6, X1, Y1))
    assert m.str("G.ExtrapolationMethod") == "none"


def test_default_grid_from_values(m):
    m.run("v = [3 1 4 1 5]; F = griddedInterpolant(v); r = F(2.5); g = F.GridVectors;")
    assert m.num("r").item() == pytest.approx(2.5)
    np.testing.assert_array_equal(m.num("g{1}"), [[1, 2, 3, 4, 5]])


xg = np.array([0.0, 1.0, 2.5, 3.0, 4.5])
yg = np.array([-1.0, 0.0, 0.5, 2.0])
XG, YG = np.meshgrid(xg, yg, indexing="ij")
VG = np.cos(XG) * YG + 0.2 * XG ** 2
QP = np.array([[0.3, -0.5], [2.2, 1.7], [4.4, 0.1], [1.0, 0.5], [-0.3, 2.4], [5.0, -1.5]])


@pytest.fixture
def grid2(m):
    _set(m, X=XG, Y=YG, V=VG, Q=QP, xq=QP[:, 0], yq=QP[:, 1])
    return m


@pytest.mark.parametrize("method,scipy_method", [("linear", "linear"), ("spline", "cubic")])
def test_2d_against_regular_grid_interpolator(grid2, method, scipy_method):
    m = grid2
    m.run(f"F = griddedInterpolant(X, Y, V, '{method}'); a = F(Q); b = F(xq, yq);")
    rgi = RegularGridInterpolator((xg, yg), VG, method=scipy_method, bounds_error=False, fill_value=None)
    inside = (QP[:, 0] >= 0) & (QP[:, 0] <= 4.5) & (QP[:, 1] >= -1) & (QP[:, 1] <= 2)
    a = m.num("a")
    assert a.shape == (QP.shape[0], 1)
    np.testing.assert_allclose(a.ravel()[inside], rgi(QP[inside]), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(m.num("b").ravel(), a.ravel())
    if method == "linear":  # linear extrapolation of the bilinear patch
        np.testing.assert_allclose(a.ravel()[~inside], rgi(QP[~inside]), rtol=1e-10)


def test_2d_cell_grid_vectors_and_grid_query(grid2):
    m = grid2
    m.run("F = griddedInterpolant({X(:,1), Y(1,:)}, V); G = F({[0.5 1.5 4], [-0.5 1]});"
          "[Xq, Yq] = ndgrid([0.5 1.5 4], [-0.5 1]); H = F(Xq, Yq);")
    G = m.num("G")
    assert G.shape == (3, 2)
    np.testing.assert_allclose(G, m.num("H"), rtol=1e-12)
    rgi = RegularGridInterpolator((xg, yg), VG)
    np.testing.assert_allclose(G, rgi(np.stack(np.meshgrid([0.5, 1.5, 4], [-0.5, 1], indexing="ij"), -1)))


def test_2d_pchip_and_makima_on_separable_data(m):
    # for f(x) + g(y) the tensor-product interpolant is the sum of the 1-D interpolants
    fx, gy = np.exp(-xg) + xg, np.array([0.0, 2.0, 2.5, 7.0])
    _set(m, V=fx[:, None] + gy[None, :], xv=xg, yv=yg, xq=QP[:3, 0], yq=QP[:3, 1])
    m.run("F = griddedInterpolant({xv, yv}, V, 'pchip'); a = F(xq, yq); F.Method = 'makima'; b = F(xq, yq);")
    want = PchipInterpolator(xg, fx)(QP[:3, 0]) + PchipInterpolator(yg, gy)(QP[:3, 1])
    np.testing.assert_allclose(m.num("a").ravel(), want, rtol=1e-12)
    mk = Akima1DInterpolator(xg, fx, method="makima")(QP[:3, 0]) + Akima1DInterpolator(yg, gy, method="makima")(
        QP[:3, 1])
    np.testing.assert_allclose(m.num("b").ravel(), mk, rtol=1e-12)


def test_cubic_convolution_reproduces_quadratics(m):
    # cubic convolution on a uniform grid is exact for quadratics (inside and, with its end conditions, outside)
    m.run("[X, Y] = ndgrid(0:0.5:3, -1:1:3); V = 2*X.^2 - X.*Y + 3*Y.^2 + 1;"
          "F = griddedInterpolant(X, Y, V, 'cubic'); r = F([0.3 1.7 2.9], [-0.4 2.2 0.1]);")
    x, y = np.array([0.3, 1.7, 2.9]), np.array([-0.4, 2.2, 0.1])
    np.testing.assert_allclose(m.num("r").ravel(), 2 * x ** 2 - x * y + 3 * y ** 2 + 1, rtol=1e-12)


def test_3d_linear(m):
    x, y, z = np.linspace(0, 1, 4), np.linspace(0, 2, 5), np.array([0.0, 0.3, 1.0])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    V = X + 2 * Y ** 2 - Z * X
    q = np.array([[0.2, 1.1, 0.5], [0.9, 0.1, 0.05]])
    _set(m, V=V, xv=x, yv=y, zv=z, q=q)
    m.run("F = griddedInterpolant({xv, yv, zv}, V); r = F(q);")
    np.testing.assert_allclose(m.num("r").ravel(), RegularGridInterpolator((x, y, z), V)(q), rtol=1e-12)


def test_multivalued_values(m):
    V2 = np.stack([VG, 2 * VG + 1], axis=2)
    _set(m, X=XG, Y=YG, V2=V2, V=VG)
    m.run("F = griddedInterpolant(X, Y, V2); G = griddedInterpolant(X, Y, V); r = F([0.5 1.2]', [0.1 1.9]');"
          "s = G([0.5 1.2]', [0.1 1.9]');")
    r, s = m.num("r"), m.num("s").ravel()
    assert r.shape == (2, 2)
    np.testing.assert_allclose(r[:, 0], s)
    np.testing.assert_allclose(r[:, 1], 2 * s + 1)
    m.run("H = griddedInterpolant((1:4)', [1 2; 3 5; 4 4; 0 1]); h = H([1.5; 3.5]);")
    np.testing.assert_allclose(m.num("h"), [[2, 3.5], [2, 2.5]])


def test_set_properties(grid2):
    m = grid2
    m.run("F = griddedInterpolant(X, Y, V); a = F(1.2, 0.3); F.Values = 3*V; b = F(1.2, 0.3);"
          "F.Method = 'nearest'; c = F(1.2, 0.3); F.GridVectors = {2*X(:,1), Y(1,:)}; d = F(2.4, 0.3);"
          "F.ExtrapolationMethod = 'none'; e = F(100, 0);")
    assert m.num("b").item() == pytest.approx(3 * m.num("a").item())
    assert m.num("c").item() == pytest.approx(3 * VG[1, 2])
    assert m.num("d").item() == pytest.approx(3 * VG[1, 2])
    assert np.isnan(m.num("e").item())
    with pytest.raises(MatlabError):
        m.run("F.Method = 'quintic';")
    with pytest.raises(MatlabError):
        m.run("F.Values = ones(3, 3);")


def test_identity_and_display(grid2):
    m = grid2
    m.run("F = griddedInterpolant(1:5, (1:5).^2);")
    assert m.str("class(F)") == "griddedInterpolant"
    assert m.num("isa(F, 'griddedInterpolant')").item()
    out = m.run("F")
    assert "griddedInterpolant with properties:" in out
    assert "GridVectors: {[1 2 3 4 5]}" in out and "Values: [1 4 9 16 25]" in out
    assert "Method: 'linear'" in out and "ExtrapolationMethod: 'linear'" in out
    assert "GridVectors" in m.run("properties(F)")


def test_errors(m):
    with pytest.raises(MatlabError, match="strictly monotonically increasing"):
        m.run("griddedInterpolant([1 3 2 4], [1 2 3 4]);")
    with pytest.raises(MatlabError, match="unique points"):
        m.run("griddedInterpolant([1 2 2 4], [1 2 3 4]);")
    with pytest.raises(MatlabError, match="meshgrid format"):
        m.run("[X, Y] = meshgrid(1:3, 1:4); griddedInterpolant(X, Y, X + Y);")
    with pytest.raises(MatlabError, match="does not match"):
        m.run("griddedInterpolant({1:3, 1:4}, ones(4, 3));")
    with pytest.raises(MatlabError, match="not supported"):
        m.run("griddedInterpolant(1:3, 1:3, 'quadratic');")
    with pytest.raises(MatlabError, match="at least two"):
        m.run("griddedInterpolant(1, 1);")
