"""Boundary value problems, delay equations, 1-D PDEs and the ode object.

Expected values are closed-form solutions (or a power series for the pantograph equation)."""

import numpy as np
import pytest

from mylab.errors import MatlabError


# -- boundary value problems ------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["bvp4c", "bvp5c"])
def test_bvp_two_point(m, solver):
    # y'' = -y, y(0) = 0, y(pi/2) = 1  ->  y = sin x
    m.run(f"sol = {solver}(@(x, y) [y(2); -y(1)], @(ya, yb) [ya(1); yb(1) - 1], "
          "bvpinit(linspace(0, pi / 2, 5), [0 1]));")
    assert m.str("sol.solver") == solver
    assert abs(m.num("deval(sol, pi / 4, 1)")[0, 0] - np.sin(np.pi / 4)) < 1e-3
    m.run("[yq, ypq] = deval(sol, [0.3 0.9]);")
    assert np.max(np.abs(m.num("yq(1, :)") - np.sin([0.3, 0.9]))) < 1e-3
    assert np.max(np.abs(m.num("ypq(1, :)") - np.cos([0.3, 0.9]))) < 1e-3
    assert m.num("isfield(sol, 'yp') && isfield(sol.stats, 'nmeshpoints')")[0, 0]


def test_bvp_tolerance_and_stats(m):
    out = m.run("sol = bvp4c(@(x, y) [y(2); -y(1)], @(ya, yb) [ya(1); yb(1) - 1], "
                "bvpinit(linspace(0, pi / 2, 5), [0 1]), bvpset('RelTol', 1e-7, 'Stats', 'on'));")
    assert "mesh" in out
    assert abs(m.num("deval(sol, 1, 1)")[0, 0] - np.sin(1)) < 1e-6
    assert m.num("numel(sol.x)")[0, 0] > 5


def test_bvp_unknown_parameter(m):
    # y'' + lambda y = 0, y(0) = 0, y(pi) = 0, y'(0) = 1  ->  lambda = 1
    m.run("init = bvpinit(linspace(0, pi, 10), @(x) [sin(x); cos(x)], 0.8);")
    m.run("sol = bvp4c(@(x, y, lam) [y(2); -lam * y(1)], @(ya, yb, lam) [ya(1); yb(1); ya(2) - 1], init);")
    assert abs(m.num("sol.parameters")[0, 0] - 1) < 1e-3


def test_bvp_nonlinear_and_function_guess(m):
    # y'' = 2 y^3 with y(1) = 1, y(2) = 1/2  ->  y = 1/x
    m.run("sol = bvp4c(@(x, y) [y(2); 2 * y(1)^3], @(ya, yb) [ya(1) - 1; yb(1) - 0.5], "
          "bvpinit(linspace(1, 2, 8), @(x) [1; 0]), bvpset('RelTol', 1e-6));")
    assert abs(m.num("deval(sol, 1.5, 1)")[0, 0] - 1 / 1.5) < 1e-5


def test_bvp_jacobians_and_vectorized(m):
    m.run("o = bvpset('FJacobian', @(x, y) [0 1; -1 0], 'BCJacobian', @(ya, yb) deal([1 0; 0 0], [0 0; 1 0]), "
          "'Vectorized', 'on', 'RelTol', 1e-6);")
    m.run("sol = bvp4c(@(x, y) [y(2, :); -y(1, :)], @(ya, yb) [ya(1); yb(1) - 1], "
          "bvpinit(linspace(0, pi / 2, 5), [0 1]), o);")
    assert abs(m.num("deval(sol, 1, 1)")[0, 0] - np.sin(1)) < 1e-5


def test_bvpinit_and_bvpset(m):
    m.run("s = bvpinit([0 1 2], [3; 4]);")
    assert np.array_equal(m.num("s.y"), [[3, 3, 3], [4, 4, 4]]) and m.num("s.x").shape == (1, 3)
    m.run("s2 = bvpinit(s, [0 3]);")
    assert np.array_equal(m.num("s2.x").ravel(), [0, 1, 2, 3])
    m.run("o = bvpset('NMax', 50, 'reltol', 1e-4);")
    assert m.num("bvpget(o, 'NMax')")[0, 0] == 50 and m.num("o.RelTol")[0, 0] == 1e-4
    with pytest.raises(MatlabError):
        m.run("bvpset('Nonsense', 1);")
    with pytest.raises(MatlabError):
        m.run("bvp4c(@(x, y) y, @(ya, yb) ya, 5);")


# -- delay differential equations ---------------------------------------------------------------------

def test_dde23_method_of_steps(m):
    # y' = -y(t - 1), y = 1 for t <= 0: y = 1 - t on [0, 1], y = t^2/2 - 2 t + 3/2 on [1, 2]
    m.run("sol = dde23(@(t, y, Z) -Z, 1, 1, [0 2]);")
    assert abs(m.num("sol.y(end)")[0, 0] + 0.5) < 1e-6
    assert abs(m.num("deval(sol, 0.5)")[0, 0] - 0.5) < 1e-8
    assert abs(m.num("deval(sol, 1.5)")[0, 0] - (1.5 ** 2 / 2 - 3 + 1.5)) < 1e-6
    assert m.str("sol.solver") == "dde23"
    assert np.any(np.isclose(m.num("sol.x").ravel(), 1.0))     # the discontinuity at t = 1 is a mesh point
    assert m.num("isfield(sol, 'yp') && isfield(sol, 'discont') && isfield(sol.stats, 'nfevals')")[0, 0]


def test_dde23_history_function_and_system(m):
    # y' = -y(t - pi/2) with history sin t has the solution y = sin t
    m.run("sol = dde23(@(t, y, Z) -Z, pi / 2, @(t) sin(t), [0 4], ddeset('RelTol', 1e-7, 'AbsTol', 1e-9));")
    assert abs(m.num("sol.y(end)")[0, 0] - np.sin(4)) < 1e-5
    # two components and two lags with solution y1 = sin t, y2 = cos t:
    # y1' = y1(t - 3 pi / 2) = cos t and y2' = y1(t - pi) = -sin t
    m.run("f = @(t, y, Z) [Z(1, 2); Z(1, 1)];")
    m.run("sol = dde23(f, [pi, 3 * pi / 2], @(t) [sin(t); cos(t)], [0 3], ddeset('RelTol', 1e-7, 'AbsTol', 1e-9));")
    assert np.max(np.abs(m.num("sol.y(:, end)").ravel() - [np.sin(3), np.cos(3)])) < 1e-5


def test_dde23_events_and_options(m):
    m.run("sol = dde23(@(t, y, Z) -Z, 1, 1, [0 5], ddeset('Events', @(t, y, Z) deal(y, 1, 0)));")
    assert abs(m.num("sol.xe")[0, 0] - 1) < 1e-8 and abs(m.num("sol.x(end)")[0, 0] - 1) < 1e-8
    m.run("sol = dde23(@(t, y, Z) -Z, 1, 1, [0 2], ddeset('InitialY', 2, 'MaxStep', 0.05));")
    assert abs(m.num("sol.y(1)")[0, 0] - 2) < 1e-12           # y(0) = 2, history 1: y = 2 - t on [0, 1]
    assert abs(m.num("deval(sol, 1)")[0, 0] - 1) < 1e-8
    assert m.num("max(diff(sol.x))")[0, 0] <= 0.05 + 1e-12
    out = m.run("sol = dde23(@(t, y, Z) -Z, 1, 1, [0 2], ddeset('Stats', 'on'));")
    assert "successful steps" in out


def test_dde23_continues_from_a_solution(m):
    m.run("s1 = dde23(@(t, y, Z) -Z, 1, 1, [0 1]); s2 = dde23(@(t, y, Z) -Z, 1, s1, [1 2]);")
    assert abs(m.num("s2.y(end)")[0, 0] + 0.5) < 1e-5


def test_ddesd(m):
    m.run("s = ddesd(@(t, y, Z) -Z, 1, 1, [0 2]);")                       # constant lag
    assert abs(m.num("s.y(end)")[0, 0] + 0.5) < 1e-5
    m.run("s = ddesd(@(t, y, Z) -Z, @(t, y) t - 1, @(t) 1, [0 2]);")      # the same as a delay function
    assert abs(m.num("s.y(end)")[0, 0] + 0.5) < 1e-3
    # pantograph equation y' = -y + y(t/2)/2, y(0) = 1, compared with its power series
    a, total = 1.0, 1.0
    for k in range(60):
        a = a * (-1 + 0.5 * 2.0 ** (-k)) / (k + 1)
        total += a * 2.0 ** (k + 1)
    m.run("s = ddesd(@(t, y, Z) -y + 0.5 * Z, @(t, y) t / 2, 1, [0 2], ddeset('RelTol', 1e-7, 'AbsTol', 1e-9));")
    assert abs(m.num("s.y(end)")[0, 0] - total) < 1e-5


# -- partial differential equations -------------------------------------------------------------------

def test_pdepe_heat_dirichlet(m):
    m.run("x = linspace(0, 1, 41); t = linspace(0, 0.2, 5);")
    m.run("u = pdepe(0, @(x, t, u, dudx) deal(1, dudx, 0), @(x) sin(pi * x), "
          "@(xl, ul, xr, ur, t) deal(ul, 0, ur, 0), x, t);")
    assert m.num("u").shape == (5, 41)
    x, t = np.linspace(0, 1, 41), np.linspace(0, 0.2, 5)
    exact = np.exp(-np.pi ** 2 * t)[:, None] * np.sin(np.pi * x)[None, :]
    assert np.max(np.abs(m.num("u") - exact)) < 2e-3


def test_pdepe_neumann_and_source(m):
    # u_t = u_xx + 1 with zero flux at both ends and u0 = cos(pi x): u = t + exp(-pi^2 t) cos(pi x)
    m.run("x = linspace(0, 1, 41); t = [0 0.05 0.1 0.2];")
    m.run("u = pdepe(0, @(x, t, u, dudx) deal(1, dudx, 1), @(x) cos(pi * x), "
          "@(xl, ul, xr, ur, t) deal(0, 1, 0, 1), x, t, odeset('RelTol', 1e-5, 'AbsTol', 1e-8));")
    x, t = np.linspace(0, 1, 41), np.array([0, 0.05, 0.1, 0.2])
    exact = t[:, None] + np.exp(-np.pi ** 2 * t)[:, None] * np.cos(np.pi * x)[None, :]
    assert np.max(np.abs(m.num("u") - exact)) < 2e-3


def test_pdepe_spherical_and_cylindrical(m):
    # sphere: u = exp(-pi^2 t) sin(pi r) / (pi r), u(1) = 0
    m.run("r = linspace(0, 1, 41); t = linspace(0, 0.1, 3);")
    m.run("u = pdepe(2, @(x, t, u, dudx) deal(1, dudx, 0), @(x) sinc_(x), "
          "@(xl, ul, xr, ur, t) deal(0, 1, ur, 0), r, t);\n"
          "function v = sinc_(r)\n if r == 0, v = 1; else, v = sin(pi * r) / (pi * r); end\nend")
    r, t = np.linspace(0, 1, 41), np.linspace(0, 0.1, 3)
    exact = np.exp(-np.pi ** 2 * t)[:, None] * np.sinc(r)[None, :]
    assert np.max(np.abs(m.num("u") - exact)) < 3e-3
    # cylinder: steady state of u_t = (1/r)(r u_r)_r + 4 with u(1) = 0 is u = 1 - r^2
    m.run("u = pdepe(1, @(x, t, u, dudx) deal(1, dudx, 4), @(x) 1 - x^2, "
          "@(xl, ul, xr, ur, t) deal(0, 1, ur, 0), r, [0 0.5 1]);")
    assert np.max(np.abs(m.num("u(end, :)").ravel() - (1 - r ** 2))) < 2e-3


def test_pdepe_system_and_pdeval(m):
    m.run("x = linspace(0, 1, 31); t = linspace(0, 0.1, 3);")
    m.run("u = pdepe(0, @(x, t, u, dudx) deal([1; 1], [dudx(1); 2 * dudx(2)], [0; 0]), "
          "@(x) [sin(pi * x); sin(pi * x)], @(xl, ul, xr, ur, t) deal(ul, [0; 0], ur, [0; 0]), x, t);")
    assert m.num("u").shape == (3, 31, 2)
    x = np.linspace(0, 1, 31)
    assert np.max(np.abs(m.num("u(end, :, 1)").ravel() - np.exp(-np.pi ** 2 * 0.1) * np.sin(np.pi * x))) < 3e-3
    assert np.max(np.abs(m.num("u(end, :, 2)").ravel() - np.exp(-2 * np.pi ** 2 * 0.1) * np.sin(np.pi * x))) < 3e-3
    m.run("[uq, dq] = pdeval(0, x, u(1, :, 1), [0.25 0.5]);")
    assert np.max(np.abs(m.num("uq").ravel() - np.sin(np.pi * np.array([0.25, 0.5])))) < 5e-3
    assert abs(m.num("dq(1)")[0, 0] - np.pi * np.cos(np.pi * 0.25)) < 0.2


def test_pdepe_argument_checks(m):
    for code in ("pdepe(3, @f, @g, @h, 0:0.1:1, 0:0.1:1)", "pdepe(0, @f, @g, @h, [0 1], 0:0.1:1)",
                 "pdepe(0, @f, @g, @h, 0:0.1:1, [0 1])", "pdepe(1, @f, @g, @h, -1:0.1:1, 0:0.1:1)"):
        with pytest.raises(MatlabError):
            m.run(code + ";")


# -- the ode object -------------------------------------------------------------------------------------

def test_ode_object_solve(m):
    m.run("F = ode(ODEFcn = @(t, y) -y, InitialValue = 1); S = solve(F, 0, 1);")
    assert abs(m.num("S.Solution(end)")[0, 0] - np.exp(-1)) < 1e-3
    assert m.num("S.Time").shape[0] == 1 and m.num("S.Solution").shape == m.num("S.Time").shape
    assert m.str("F.SelectedSolver") == "ode45" and m.str("class(F)") == "ode"
    m.run("F.Solver = \"ode15s\"; F.RelativeTolerance = 1e-8; F.AbsoluteTolerance = 1e-10;")
    m.run("S = solve(F, [0 0.5 1]);")
    assert np.array_equal(m.num("S.Time").ravel(), [0, 0.5, 1])
    assert np.max(np.abs(m.num("S.Solution").ravel() - np.exp([-0, -0.5, -1]))) < 1e-6
    assert m.str("F.SelectedSolver") == "ode15s"
    m.run("S = solve(F, [0.25 0.75]);")
    assert np.max(np.abs(m.num("S.Solution").ravel() - np.exp([-0.25, -0.75]))) < 1e-6
    assert "RelativeTolerance" in m.run("F")


def test_ode_object_parameters_initial_time_and_solution_fcn(m):
    m.run("F = ode(ODEFcn = @(t, y, p) p(1) * y, InitialValue = 2, InitialTime = 1, Parameters = -3, "
          "RelativeTolerance = 1e-8, AbsoluteTolerance = 1e-10);")
    m.run("S = solve(F, 1, 2); fh = solutionFcn(F, 0, 2);")
    assert abs(m.num("S.Solution(end)")[0, 0] - 2 * np.exp(-3)) < 1e-6
    tq = np.array([0, 0.5, 1, 1.5, 2])
    assert np.max(np.abs(m.num("fh([0 0.5 1 1.5 2])").ravel() - 2 * np.exp(-3 * (tq - 1)))) < 1e-4
    assert m.str("class(fh)") == "function_handle"
    m.run("S = solve(F, 0, 2);")                 # InitialTime inside the interval: both directions
    assert m.num("S.Time(1)")[0, 0] == 0 and m.num("S.Time(end)")[0, 0] == 2
    assert abs(m.num("S.Solution(1)")[0, 0] - 2 * np.exp(3)) < 1e-4


def test_ode_object_events_mass_and_options(m):
    m.run("F = ode(ODEFcn = @(t, y) [y(2); -y(1)], InitialValue = [0; 1], RelativeTolerance = 1e-8, "
          "AbsoluteTolerance = 1e-10);")
    m.run("F.EventDefinition = odeEvent(EventFcn = @(t, y) y(1), Direction = \"descending\", Response = \"stop\");")
    m.run("S = solve(F, 0, 10);")
    assert abs(m.num("S.EventTime")[0, 0] - np.pi) < 1e-6 and abs(m.num("S.Time(end)")[0, 0] - np.pi) < 1e-6
    assert m.num("S.EventIndex")[0, 0] == 1 and m.num("S.EventSolution").shape == (2, 1)
    m.run("F.EventDefinition.Response = \"proceed\"; S = solve(F, 0, 10);")
    assert m.num("S.Time(end)")[0, 0] == 10 and m.num("numel(S.EventTime)")[0, 0] == 2
    m.run("G = ode(ODEFcn = @(t, y) [-2 * y(1); -y(2)], InitialValue = [1; 1], "
          "MassMatrix = odeMassMatrix(MassMatrix = [2 0; 0 1]));")
    m.run("G.SolverOptions.MaxStep = 0.05; S = solve(G, 0, 1);")
    assert abs(m.num("S.Solution(1, end)")[0, 0] - np.exp(-1)) < 1e-3
    assert m.num("max(diff(S.Time))")[0, 0] <= 0.05 + 1e-12
    assert m.str("G.SelectedSolver") == "ode15s"


def test_ode_object_callback_event(m):
    # a bouncing ball: reverse the velocity at each impact and stop after the third
    m.run("""
function [stop, y] = bounce(t, y)
    global NB
    NB = NB + 1;
    y(2) = -0.5 * y(2);
    y(1) = 0;
    stop = NB >= 3;
end
""")
    m.run("global NB; NB = 0;")
    m.run("F = ode(ODEFcn = @(t, y) [y(2); -10], InitialValue = [5; 0], RelativeTolerance = 1e-8, "
          "AbsoluteTolerance = 1e-10);")
    m.run("F.EventDefinition = odeEvent(EventFcn = @(t, y) y(1), Direction = \"descending\", "
          "Response = \"callback\", CallbackFcn = @bounce);")
    m.run("S = solve(F, 0, 10);")
    # impacts at t = 1, then every 2 * v / g with v halving: 1, 2, 2.5
    assert np.max(np.abs(m.num("S.EventTime").ravel() - [1, 2, 2.5])) < 1e-5
    assert abs(m.num("S.Time(end)")[0, 0] - 2.5) < 1e-5


def test_ode_object_errors(m):
    with pytest.raises(MatlabError):
        m.run("F = ode(NoSuchProperty = 1);")
    with pytest.raises(MatlabError):
        m.run("F = ode(ODEFcn = @(t, y) -y); solve(F, 0, 1);")
    with pytest.raises(MatlabError):
        m.run("F = ode(ODEFcn = @(t, y) -y, InitialValue = 1); F.Solver = \"ode99\";")
    with pytest.raises(MatlabError):
        m.run("F = ode(ODEFcn = @(t, y) -y, InitialValue = 1); F.SelectedSolver = \"ode23\";")
