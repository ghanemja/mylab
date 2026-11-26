"""Numerical integration: one, two and three dimensions. Expected values are exact integrals."""

import numpy as np
import pytest

from mylab.errors import MatlabError


def val(m, expr):
    return m.num(expr)[0, 0]


def test_integral_basic_and_tolerances(m):
    assert abs(val(m, "integral(@(x) x.^2, 0, 1)") - 1 / 3) < 1e-12
    assert abs(val(m, "integral(@(x) sin(x), 0, pi)") - 2) < 1e-10
    assert abs(val(m, "integral(@(x) x, 0, 1, 'RelTol', 1e-12, 'AbsTol', 1e-14)") - 0.5) < 1e-14
    assert abs(val(m, "integral(@(x) exp(x), 1, 0)") + (np.e - 1)) < 1e-10      # reversed limits
    assert val(m, "integral(@(x) x, 2, 2)") == 0
    # a looser tolerance is honoured: far fewer evaluations for an oscillatory integrand
    m.run("global N; N = 0;")
    m.run("function y = osc(x)\n global N\n N = N + numel(x);\n y = cos(200 * x);\nend")
    m.run("q1 = integral(@osc, 0, 1, 'RelTol', 1e-2, 'AbsTol', 1e-2); n1 = N; N = 0;")
    m.run("q2 = integral(@osc, 0, 1, 'RelTol', 1e-12, 'AbsTol', 1e-13); n2 = N;")
    assert abs(val(m, "q2") - np.sin(200) / 200) < 1e-12
    assert val(m, "n1") <= val(m, "n2")


def test_integral_infinite_limits_and_singularities(m):
    assert abs(val(m, "integral(@(x) exp(-x.^2), -Inf, Inf)") - np.sqrt(np.pi)) < 1e-8
    assert abs(val(m, "integral(@(x) 1 ./ (1 + x.^2), 0, Inf)") - np.pi / 2) < 1e-8
    assert abs(val(m, "integral(@(x) exp(x), -Inf, 0)") - 1) < 1e-8
    assert abs(val(m, "integral(@(x) 1 ./ sqrt(x), 0, 1)") - 2) < 1e-6
    assert abs(val(m, "integral(@(x) log(x), 0, 1)") + 1) < 1e-6
    assert abs(val(m, "integral(@(x) 1 ./ sqrt(1 - x.^2), -1, 1)") - np.pi) < 1e-5


def test_integral_array_valued(m):
    assert np.max(np.abs(m.num("integral(@(x) [x, x.^2], 0, 1, 'ArrayValued', true)") - [0.5, 1 / 3])) < 1e-12
    q = m.num("integral(@(x) [sin(x) 1; x exp(x)], 0, 1, 'ArrayValued', true)")
    assert np.max(np.abs(q - [[1 - np.cos(1), 1], [0.5, np.e - 1]])) < 1e-10
    # a scalar-only integrand works with ArrayValued and fails without it
    assert abs(val(m, "integral(@(x) x^2, 0, 1, 'ArrayValued', true)") - 1 / 3) < 1e-12
    with pytest.raises(MatlabError):
        m.run("integral(@(x) 1, 0, 1);")


def test_integral_waypoints_and_complex(m):
    assert abs(val(m, "integral(@(x) abs(x - 0.3), 0, 1, 'Waypoints', 0.3)") - (0.045 + 0.245)) < 1e-12
    # residue theorem: the integral of 1/z around the square with corners +-1 +-i is 2 pi i
    q = m.num("integral(@(z) 1 ./ z, 1, 1, 'Waypoints', [1+1i, -1+1i, -1-1i, 1-1i])")[0, 0]
    assert abs(q - 2j * np.pi) < 1e-9
    assert abs(m.num("integral(@(x) exp(1i * x), 0, pi)")[0, 0] - 2j) < 1e-10
    assert abs(m.num("integral(@(z) z, 0, 1 + 1i)")[0, 0] - (1 + 1j) ** 2 / 2) < 1e-12
    with pytest.raises(MatlabError):
        m.run("integral(@(x) x, 0, 1, 'Waypoints', 5);")
    with pytest.raises(MatlabError):
        m.run("integral(@(x) x, 0, 1, 'NoSuchOption', 5);")


def test_quadgk(m):
    m.run("[q, e] = quadgk(@(x) sin(x), 0, pi);")
    assert abs(val(m, "q") - 2) < 1e-12 and 0 <= val(m, "e") < 1e-8
    # integral of x^5 exp(-x) sin(x) over [0, Inf) is -15
    assert abs(val(m, "quadgk(@(x) x.^5 .* exp(-x) .* sin(x), 0, Inf, 'RelTol', 1e-8, 'AbsTol', 1e-12)") + 15) < 1e-8
    assert abs(val(m, "quadgk(@(x) abs(x), -1, 2, 'Waypoints', 0)") - 2.5) < 1e-12
    out = m.run("q = quadgk(@(x) sin(1 ./ x) ./ x, 0, 1, 'MaxIntervalCount', 20);")
    assert "Warning" in out


def test_quad_quadl_quadv(m):
    m.run("[q, n] = quad(@(x) sin(x), 0, pi);")
    assert abs(val(m, "q") - 2) < 1e-6 and val(m, "n") > 4
    m.run("[q, n] = quadl(@(x) sin(x), 0, pi);")
    assert abs(val(m, "q") - 2) < 1e-6 and val(m, "n") > 4
    assert abs(val(m, "quad(@(x) 1 ./ (1 + x.^2), 0, 1, 1e-10)") - np.pi / 4) < 1e-9
    assert abs(val(m, "quadl(@(x) exp(-x.^2), 0, 2, 1e-12)") - 0.8820813907624217) < 1e-11
    assert abs(val(m, "quad(@(x, a) a * x, 0, 1, [], [], 4)") - 2) < 1e-10        # extra parameter
    Q = m.num("quadv(@(x) [x, x^2; sin(x), 1], 0, 1)")
    assert np.max(np.abs(Q - [[0.5, 1 / 3], [1 - np.cos(1), 1]])) < 1e-6
    assert abs(val(m, "quad(@(x) x.^2, 1, 0)") + 1 / 3) < 1e-8
    with pytest.raises(MatlabError):
        m.run("quad(@(x) x, 0, Inf);")


def test_integral2(m):
    assert abs(val(m, "integral2(@(x, y) x .* y, 0, 1, 0, 2)") - 1) < 1e-10
    assert abs(val(m, "integral2(@(x, y) x + y, 0, 1, 0, @(x) x)") - 0.5) < 1e-10
    for method in ("tiled", "iterated", "auto"):
        # the unit disc: area pi and second moment pi/4
        q = val(m, f"integral2(@(x, y) x.^2, -1, 1, @(x) -sqrt(1 - x.^2), @(x) sqrt(1 - x.^2), 'Method', '{method}')")
        assert abs(q - np.pi / 4) < 1e-6, method
    assert abs(val(m, "integral2(@(x, y) exp(-x.^2 - y.^2), -Inf, Inf, -Inf, Inf)") - np.pi) < 1e-7
    assert abs(val(m, "integral2(@(x, y) 1 ./ sqrt(x + y), 0, 1, 0, 1)") - (8 / 3) * (np.sqrt(2) - 1)) < 1e-6
    assert abs(val(m, "integral2(@(x, y) x .* y, 0, 1, 0, 2, 'RelTol', 1e-12, 'AbsTol', 1e-14)") - 1) < 1e-13
    with pytest.raises(MatlabError):
        m.run("integral2(@(x, y) x, 0, Inf, 0, 1, 'Method', 'tiled');")
    with pytest.raises(MatlabError):
        m.run("integral2(@(x, y) x, 0, 1, 0, 1, 'Method', 'fast');")


def test_integral3(m):
    assert abs(val(m, "integral3(@(x, y, z) x + y + z, 0, 1, 0, 1, 0, 1)") - 1.5) < 1e-9
    # variable limits: 0 <= y <= x, 0 <= z <= x + y; the integral of x y z is 17/144
    q = val(m, "integral3(@(x, y, z) x .* y .* z, 0, 1, 0, @(x) x, 0, @(x, y) x + y)")
    assert abs(q - 17 / 144) < 1e-9
    # volume of the unit ball
    q = val(m, "integral3(@(x, y, z) ones(size(x)), -1, 1, @(x) -sqrt(1 - x.^2), @(x) sqrt(1 - x.^2), "
               "@(x, y) -sqrt(max(1 - x.^2 - y.^2, 0)), @(x, y) sqrt(max(1 - x.^2 - y.^2, 0)))")
    assert abs(q - 4 * np.pi / 3) < 1e-5


def test_quad2d_dblquad_triplequad(m):
    m.run("[q, e] = quad2d(@(x, y) x .* y, 0, 1, 0, @(x) x);")
    assert abs(val(m, "q") - 1 / 8) < 1e-8 and val(m, "e") < 1e-5
    assert abs(val(m, "quad2d(@(x, y) cos(x + y), 0, 1, 0, 1, 'AbsTol', 1e-10, 'Singular', false)")
               - (2 * np.cos(1) - np.cos(2) - 1)) < 1e-9
    assert abs(val(m, "dblquad(@(x, y) x .* y, 0, 1, 0, 1)") - 0.25) < 1e-7
    assert abs(val(m, "dblquad(@(x, y) y * sin(x) + x * cos(y), pi, 2 * pi, 0, pi, 1e-8, @quadl)") + np.pi ** 2) < 1e-6
    assert abs(val(m, "triplequad(@(x, y, z) y * sin(x) + z * cos(x), 0, pi, 0, 1, -1, 1)") - 2) < 1e-6
    with pytest.raises(MatlabError):
        m.run("dblquad(@(x, y) x, 0, Inf, 0, 1);")
