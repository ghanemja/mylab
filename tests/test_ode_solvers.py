"""Initial value solvers: accuracy, options, mass matrices, events, solution structures.

Expected values are closed-form solutions of the test problems."""

import numpy as np
import pytest

from mylab.errors import MatlabError

ALL = ["ode45", "ode23", "ode113", "ode78", "ode89", "ode15s", "ode23s", "ode23t", "ode23tb"]
STIFF = ["ode15s", "ode23s", "ode23t", "ode23tb"]


@pytest.mark.parametrize("solver", ALL)
def test_decay_default_tolerances(m, solver):
    m.run(f"[t, y] = {solver}(@(t, y) -y, [0 1], 1);")
    t, y = m.num("t"), m.num("y")
    assert t.shape[1] == 1 and y.shape == t.shape
    assert t[0, 0] == 0 and t[-1, 0] == 1
    assert abs(y[-1, 0] - np.exp(-1)) < 2e-3
    assert np.all(np.diff(t[:, 0]) > 0)


def test_ode45_refine_and_tspan_points(m):
    m.run("[t1, y1] = ode45(@(t, y) -y, [0 1], 1, odeset('Refine', 1));")
    m.run("[t4, y4] = ode45(@(t, y) -y, [0 1], 1);")
    n1, n4 = m.num("t1").size, m.num("t4").size
    assert n4 == 4 * (n1 - 1) + 1          # three extra points per step by default
    m.run("ts = linspace(0, 2, 9); [t, y] = ode45(@(t, y) -y, ts, 1);")
    assert np.array_equal(m.num("t").ravel(), np.linspace(0, 2, 9))
    assert np.max(np.abs(m.num("y").ravel() - np.exp(-np.linspace(0, 2, 9)))) < 1e-3


def test_high_order_solvers_are_accurate(m):
    for s in ("ode78", "ode89", "ode113"):
        m.run(f"[t, y] = {s}(@(t, y) [y(2); -y(1)], [0 2*pi], [0; 1], odeset('RelTol', 1e-10, 'AbsTol', 1e-12));")
        assert np.max(np.abs(m.num("y(end, :)") - [0, 1])) < 1e-8


def test_tolerances_and_vector_abstol(m):
    m.run("o = odeset('RelTol', 1e-9, 'AbsTol', [1e-11 1e-12]);")
    m.run("[t, y] = ode45(@(t, y) [y(2); -4 * y(1)], [0 1], [1; 0], o);")
    assert abs(m.num("y(end, 1)")[0, 0] - np.cos(2)) < 1e-7
    with pytest.raises(MatlabError):
        m.run("ode45(@(t, y) [y(2); -y(1)], [0 1], [1; 0], odeset('AbsTol', [1 2 3]));")


def test_maxstep_and_initialstep(m):
    m.run("[t, y] = ode45(@(t, y) 1, [0 1], 0, odeset('MaxStep', 0.01, 'Refine', 1));")
    assert np.max(np.diff(m.num("t").ravel())) <= 0.01 + 1e-12
    m.run("[t, y] = ode23(@(t, y) -y, [0 1], 1, odeset('InitialStep', 1e-4));")
    assert abs(m.num("t(2)")[0, 0] - 1e-4) < 1e-12


def test_backward_integration_and_complex(m):
    m.run("[t, y] = ode45(@(t, y) -y, [1 0], exp(-1), odeset('RelTol', 1e-8, 'AbsTol', 1e-10));")
    assert m.num("t(end)")[0, 0] == 0 and abs(m.num("y(end)")[0, 0] - 1) < 1e-6
    m.run("[t, y] = ode45(@(t, y) 1i * y, [0 pi], 1, odeset('RelTol', 1e-8, 'AbsTol', 1e-10));")
    assert abs(m.num("y(end)")[0, 0] + 1) < 1e-6


def test_extra_parameters_after_options(m):
    m.run("[t, y] = ode45(@(t, y, a, b) a * y + b, [0 1], 0, [], -1, 2);")
    assert abs(m.num("y(end)")[0, 0] - 2 * (1 - np.exp(-1))) < 1e-3
    m.run("[t, y] = ode15s(@(t, y, a) a * y, [0 1], 1, odeset('RelTol', 1e-6), -2);")
    assert abs(m.num("y(end)")[0, 0] - np.exp(-2)) < 1e-4


# -- mass matrices ---------------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ALL)
def test_constant_mass_matrix(m, solver):
    m.run("o = odeset('Mass', [2 0; 0 1], 'RelTol', 1e-6, 'AbsTol', 1e-9);")
    m.run(f"[t, y] = {solver}(@(t, y) [-2 * y(1); -y(2)], [0 1], [1; 1], o);")
    assert np.max(np.abs(m.num("y(end, :)") - np.exp(-1))) < 1e-4


def test_coupled_mass_matrix(m):
    # [2 1; 1 2] y' = [-3 y1 - 3 y2 ... ] chosen so that y' = -y: M*(-y) = -M*y
    m.run("M = [2 1; 1 2]; o = odeset('Mass', M, 'RelTol', 1e-8, 'AbsTol', 1e-10);")
    for s in ("ode45", "ode15s", "ode23s", "ode23tb"):
        m.run(f"[t, y] = {s}(@(t, y) -M * y, [0 1], [1; 3], o);")
        assert np.max(np.abs(m.num("y(end, :)") - np.array([1, 3]) * np.exp(-1))) < 2e-5


@pytest.mark.parametrize("solver", [s for s in ALL if s != "ode23s"])
def test_time_and_state_dependent_mass(m, solver):
    # (1 + t) y' = -y  ->  y = 1 / (1 + t)
    m.run("o = odeset('Mass', @(t) 1 + t, 'MStateDependence', 'none', 'RelTol', 1e-6, 'AbsTol', 1e-9);")
    m.run(f"[t, y] = {solver}(@(t, y) -y, [0 1], 1, o);")
    assert abs(m.num("y(end)")[0, 0] - 0.5) < 1e-4
    # y y' = 1  ->  y = sqrt(1 + 2 t)
    for dep in ("weak", "strong"):
        m.run(f"o = odeset('Mass', @(t, y) y, 'MStateDependence', '{dep}', 'RelTol', 1e-6, 'AbsTol', 1e-9);")
        m.run(f"[t, y] = {solver}(@(t, y) 1, [0 1], 1, o);")
        assert abs(m.num("y(end)")[0, 0] - np.sqrt(3)) < 1e-4


def test_ode23s_rejects_variable_mass(m):
    with pytest.raises(MatlabError, match="constant mass"):
        m.run("ode23s(@(t, y) -y, [0 1], 1, odeset('Mass', @(t, y) 1 + t));")


@pytest.mark.parametrize("solver", ["ode15s", "ode23t", "ode23tb"])
def test_singular_mass_dae(m, solver):
    # y1' = -y1 + y2, 0 = y1 - 2 y2  ->  y1 = exp(-t/2), y2 = y1 / 2
    m.run("o = odeset('Mass', [1 0; 0 0], 'RelTol', 1e-6, 'AbsTol', 1e-9);")
    m.run(f"[t, y] = {solver}(@(t, y) [-y(1) + y(2); y(1) - 2 * y(2)], [0 2], [1; 0.5], o);")
    assert np.max(np.abs(m.num("y(end, :)") - np.exp(-1) * np.array([1, 0.5]))) < 1e-4
    # an inconsistent algebraic variable is corrected at the start
    m.run(f"[t, y] = {solver}(@(t, y) [-y(1) + y(2); y(1) - 2 * y(2)], [0 2], [1; 0], o);")
    assert abs(m.num("y(1, 2)")[0, 0] - 0.5) < 1e-8
    assert np.max(np.abs(m.num("y(end, :)") - np.exp(-1) * np.array([1, 0.5]))) < 1e-4


def test_explicit_solver_refuses_singular_mass(m):
    with pytest.raises(MatlabError, match="singular"):
        m.run("ode45(@(t, y) [-y(1); y(1) - y(2)], [0 1], [1; 1], odeset('Mass', [1 0; 0 0]));")


def test_mass_singular_no_is_an_error_when_wrong(m):
    with pytest.raises(MatlabError):
        m.run("ode15s(@(t, y) [-y(1); y(1) - y(2)], [0 1], [1; 1], "
              "odeset('Mass', [1 0; 0 0], 'MassSingular', 'no'));")


# -- stiff solvers -----------------------------------------------------------------------------------

@pytest.mark.parametrize("solver", STIFF)
def test_stiff_relaxation(m, solver):
    # y' = -1000 (y - cos t): after the transient y = (1e6 cos t + 1000 sin t) / (1e6 + 1)
    m.run(f"[t, y] = {solver}(@(t, y) -1000 * (y - cos(t)), [0 1], 0);")
    exact = (1e6 * np.cos(1) + 1000 * np.sin(1)) / (1e6 + 1)
    assert abs(m.num("y(end)")[0, 0] - exact) < 1e-3
    assert m.num("numel(t)")[0, 0] < 400


@pytest.mark.parametrize("solver", STIFF)
def test_jacobian_options(m, solver):
    m.run("J = [-1000 0; 1 -1];")
    exact1 = np.exp(-1)   # y2' = y1 - y2 with y1 = 0 after the transient, y1(0) = 0
    for opt in ("odeset('Jacobian', J)", "odeset('Jacobian', @(t, y) J)", "odeset('JPattern', [1 0; 1 1])",
                "odeset('Vectorized', 'on')", "odeset('Jacobian', @(t, y) J, 'JConstant', 'on')"):
        m.run(f"[t, y] = {solver}(@(t, y) J * y, [0 1], [0; 1], {opt});")
        assert abs(m.num("y(end, 2)")[0, 0] - exact1) < 2e-3, opt


def test_ode15s_bdf_and_maxorder(m):
    m.run("[t, y] = ode15s(@(t, y) -y, [0 2], 1, odeset('BDF', 'on', 'MaxOrder', 2, 'RelTol', 1e-5));")
    assert abs(m.num("y(end)")[0, 0] - np.exp(-2)) < 1e-4
    m.run("[t1, y1] = ode15s(@(t, y) -y, [0 2], 1, odeset('MaxOrder', 1, 'RelTol', 1e-5));")
    assert m.num("numel(t1)")[0, 0] > m.num("numel(t)")[0, 0]   # order 1 needs more steps
    with pytest.raises(MatlabError):
        m.run("ode15s(@(t, y) -y, [0 2], 1, odeset('MaxOrder', 7));")


def test_van_der_pol_stiff_agrees_between_solvers(m):
    m.run("f = @(t, y) [y(2); 100 * (1 - y(1)^2) * y(2) - y(1)];")
    m.run("[t, y] = ode15s(f, [0 100], [2; 0], odeset('RelTol', 1e-7, 'AbsTol', 1e-9)); ref = y(end, 1);")
    for s in STIFF:
        m.run(f"[t, y] = {s}(f, [0 100], [2; 0], odeset('RelTol', 1e-4, 'AbsTol', 1e-6));")
        assert abs(m.num("y(end, 1) - ref")[0, 0]) < 2e-2, s


# -- events ---------------------------------------------------------------------------------------------

def test_event_as_anonymous_deal(m):
    m.run("ev = @(t, y) deal(y(1), 1, -1); o = odeset('Events', ev);")
    m.run("[t, y, te, ye, ie] = ode45(@(t, y) [y(2); -9.81], [0 5], [10; 0], o);")
    te = np.sqrt(20 / 9.81)
    assert abs(m.num("te")[0, 0] - te) < 1e-6
    assert abs(m.num("t(end)")[0, 0] - te) < 1e-6          # terminal: the integration stops there
    assert abs(m.num("ye(1)")[0, 0]) < 1e-6 and m.num("ie")[0, 0] == 1
    assert m.num("ye").shape == (1, 2)


@pytest.mark.parametrize("solver", ALL)
def test_events_direction_and_terminal(m, solver):
    m.run("o = odeset('Events', @(t, y) deal([y(1) - 0.5; y(1)], [0; 1], [1; -1]), 'RelTol', 1e-8, 'AbsTol', 1e-10);")
    m.run(f"[t, y, te, ye, ie] = {solver}(@(t, y) [y(2); -y(1)], [0 10], [0; 1], o);")
    # y1 = sin t: rises through 0.5 at pi/6 (the fall at 5 pi/6 is filtered out), falls through 0 at pi
    assert np.array_equal(m.num("ie").ravel(), [1, 2])
    assert np.max(np.abs(m.num("te").ravel() - [np.pi / 6, np.pi])) < 1e-4
    assert abs(m.num("t(end)")[0, 0] - np.pi) < 1e-4


def test_event_outputs_empty_without_events(m):
    m.run("[t, y, te, ye, ie] = ode45(@(t, y) -y, [0 1], 1);")
    assert m.num("te").size == 0 and m.num("ie").size == 0


def test_events_in_solution_structure(m):
    m.run("sol = ode45(@(t, y) [y(2); -y(1)], [0 10], [0; 1], odeset('Events', @(t, y) deal(y(1), 0, 0)));")
    xe = m.num("sol.xe").ravel()
    assert np.max(np.abs(xe - np.pi * np.arange(1, 4))) < 5e-3
    assert m.num("sol.ye").shape == (2, 3) and m.num("sol.ie").shape == (1, 3)


# -- output functions, statistics, nonnegativity ----------------------------------------------------------

def test_output_function_can_stop(m):
    m.run("o = odeset('OutputFcn', @(t, y, flag) ~isempty(t) && t(end) > 0.5 && isempty(flag));")
    m.run("[t, y] = ode45(@(t, y) -y, [0 1], 1, o);")
    assert 0.5 < m.num("t(end)")[0, 0] < 1


def test_output_function_flags_and_outputsel(m):
    m.run("""
function status = recorder(t, y, flag)
    global LOG
    LOG{end + 1} = {flag, size(t), size(y)};
    status = false;
end
""")
    m.run("global LOG; LOG = {};")
    m.run("[t, y] = ode45(@(t, y) [y(2); -y(1)], [0 1], [0; 1], odeset('OutputFcn', @recorder, 'OutputSel', 2));")
    assert m.str("LOG{1}{1}") == "init" and m.str("LOG{end}{1}") == "done"
    assert np.array_equal(m.num("LOG{1}{2}").ravel(), [1, 2])        # [t0 tfinal]
    assert np.array_equal(m.num("LOG{1}{3}").ravel(), [1, 1])        # one selected component
    assert np.array_equal(m.num("LOG{2}{3}").ravel(), [1, 4])        # four refined points per step


def test_stats_are_printed(m):
    out = m.run("[t, y] = ode15s(@(t, y) -y, [0 1], 1, odeset('Stats', 'on'));")
    assert "successful steps" in out and "function evaluations" in out and "LU decompositions" in out
    out = m.run("[t, y] = ode45(@(t, y) -y, [0 1], 1, odeset('Stats', 'on'));")
    assert "failed attempts" in out and "LU" not in out


def test_nonnegative(m):
    for s in ("ode45", "ode23", "ode15s", "ode23t"):
        m.run(f"[t, y] = {s}(@(t, y) -abs(y) - 0.5, [0 5], 1, odeset('NonNegative', 1));")
        assert m.num("min(y)")[0, 0] >= 0, s
    m.run("[t, y] = ode45(@(t, y) -abs(y) - 0.5, [0 5], 1);")
    assert m.num("min(y)")[0, 0] < 0
    with pytest.raises(MatlabError, match="NonNegative"):
        m.run("ode23s(@(t, y) -y, [0 1], 1, odeset('NonNegative', 1));")


def test_normcontrol(m):
    m.run("[t, y] = ode45(@(t, y) [-y(1); -100 * y(2)], [0 1], [1; 1e-8], odeset('NormControl', 'on'));")
    assert abs(m.num("y(end, 1)")[0, 0] - np.exp(-1)) < 1e-3


# -- options structures -----------------------------------------------------------------------------------

def test_odeset_and_odeget(m):
    m.run("o = odeset('reltol', 1e-5);")
    assert m.num("o.RelTol")[0, 0] == 1e-5 and m.num("isempty(o.AbsTol)")[0, 0]
    assert m.num("isfield(o, 'MStateDependence') && isfield(o, 'Events')")[0, 0]
    m.run("o2 = odeset(o, 'AbsT', 1e-9, 'Stats', 'on');")
    assert m.num("o2.RelTol")[0, 0] == 1e-5 and m.num("o2.AbsTol")[0, 0] == 1e-9
    assert m.num("odeget(o2, 'RelTol')")[0, 0] == 1e-5
    assert m.num("odeget(o2, 'MaxStep', 7)")[0, 0] == 7 and m.num("isempty(odeget(o2, 'MaxStep'))")[0, 0]
    m.run("o3 = odeset(o, odeset('RelTol', 1e-3, 'MaxStep', 2));")
    assert m.num("o3.RelTol")[0, 0] == 1e-3 and m.num("o3.MaxStep")[0, 0] == 2
    with pytest.raises(MatlabError, match="Unrecognized property"):
        m.run("odeset('NoSuchOption', 1);")
    with pytest.raises(MatlabError, match="Ambiguous"):
        m.run("odeset('M', 1);")
    assert "RelTol" in m.run("odeset")


def test_bad_arguments(m):
    for code in ("ode45(@(t, y) -y, 1, 1)", "ode45(@(t, y) -y, [0 0], 1)", "ode45(@(t, y) -y, [0 2 1], 1)",
                 "ode45(@(t, y) [y; y], [0 1], 1)", "ode45(@(t, y) -y, [0 1], 1, odeset('RelTol', -1))",
                 "ode45(@(t, y) -y, [0 1], 1, odeset('MaxStep', 0))",
                 "ode45(@(t, y) -y, [0 1], 1, odeset('Refine', 2.5))",
                 "ode45(@(t, y) -y, [0 1], 1, odeset('Mass', [1 2; 3 4]))"):
        with pytest.raises(MatlabError):
            m.run(code + ";")


# -- solution structures -----------------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ALL)
def test_deval(m, solver):
    m.run(f"sol = {solver}(@(t, y) [y(2); -y(1)], [0 3], [0; 1], odeset('RelTol', 1e-7, 'AbsTol', 1e-9));")
    assert m.str("sol.solver") == solver
    m.run("tq = [0 0.3 1.7 3]; [yq, ypq] = deval(sol, tq);")
    tq = np.array([0, 0.3, 1.7, 3])
    assert m.num("yq").shape == (2, 4)
    tol = 2e-3 if solver in ("ode23s", "ode23t") else 1e-4
    assert np.max(np.abs(m.num("yq") - np.vstack([np.sin(tq), np.cos(tq)]))) < tol
    assert np.max(np.abs(m.num("ypq") - np.vstack([np.cos(tq), -np.sin(tq)]))) < 30 * tol
    assert np.allclose(m.num("deval(sol, tq, 2)"), m.num("yq(2, :)"))
    assert np.allclose(m.num("deval(tq, sol)"), m.num("yq"))
    assert np.allclose(m.num("deval(sol, sol.x)"), m.num("sol.y"), atol=1e-12)


def test_deval_errors(m):
    m.run("sol = ode45(@(t, y) -y, [0 1], 1);")
    with pytest.raises(MatlabError, match="outside the interval"):
        m.run("deval(sol, 2);")
    with pytest.raises(MatlabError):
        m.run("deval(sol, 0.5, 3);")


def test_solution_structure_fields(m):
    m.run("sol = ode15s(@(t, y) -y, [0 1], 1);")
    names = m.str("strjoin(fieldnames(sol)', ',')")
    assert names == "solver,extdata,x,y,stats,idata"
    assert m.num("isfield(sol.stats, 'npds') && isfield(sol.stats, 'nfevals')")[0, 0]
    assert m.num("size(sol.x, 1)")[0, 0] == 1 and m.num("size(sol.y, 2) == numel(sol.x)")[0, 0]


def test_odextend(m):
    m.run("sol = ode45(@(t, y) -y, [0 1], 1, odeset('RelTol', 1e-8, 'AbsTol', 1e-10));")
    m.run("sol2 = odextend(sol, [], 2);")
    assert m.num("sol2.x(1)")[0, 0] == 0 and m.num("sol2.x(end)")[0, 0] == 2
    assert abs(m.num("sol2.y(end)")[0, 0] - np.exp(-2)) < 1e-7
    assert np.max(np.abs(m.num("deval(sol2, [0.5 1.5])").ravel() - np.exp([-0.5, -1.5]))) < 1e-6
    m.run("sol3 = odextend(sol, @(t, y) -2 * y, 2, 5);")          # new function and restart value
    assert abs(m.num("sol3.y(end)")[0, 0] - 5 * np.exp(-2)) < 1e-5
    with pytest.raises(MatlabError):
        m.run("odextend(sol, [], 0.5);")


# -- fully implicit --------------------------------------------------------------------------------------

def test_ode15i(m):
    m.run("[t, y] = ode15i(@(t, y, yp) yp + y, [0 1], 1, -1);")
    assert abs(m.num("y(end)")[0, 0] - np.exp(-1)) < 1e-2
    m.run("o = odeset('RelTol', 1e-7, 'AbsTol', 1e-9);")
    m.run("[t, y] = ode15i(@(t, y, yp) [yp(1) - y(2); yp(2) + y(1)], [0 2], [0; 1], [1; 0], o);")
    assert np.max(np.abs(m.num("y(end, :)") - [np.sin(2), np.cos(2)])) < 1e-5
    m.run("sol = ode15i(@(t, y, yp) yp + y, [0 1], 1, -1, o); v = deval(sol, 0.5);")
    assert abs(m.num("v")[0, 0] - np.exp(-0.5)) < 1e-5
    # analytic Jacobian and an event with three inputs
    m.run("o2 = odeset(o, 'Jacobian', @(t, y, yp) deal(1, 1), 'Events', @(t, y, yp) deal(y - 0.5, 1, 0));")
    m.run("[t, y, te] = ode15i(@(t, y, yp) yp + y, [0 2], 1, -1, o2);")
    assert abs(m.num("te")[0, 0] - np.log(2)) < 1e-5


def test_decic(m):
    m.run("[y0, yp0, r] = decic(@(t, y, yp) yp + 2 * y, 0, 3, 1, 0, 0);")
    assert m.num("y0")[0, 0] == 3 and abs(m.num("yp0")[0, 0] + 6) < 1e-8 and m.num("r")[0, 0] < 1e-8
    # y fixed in the first component, the algebraic constraint y1 + y2 = 1 fixes the second
    m.run("f = @(t, y, yp) [yp(1) + y(1); y(1) + y(2) - 1];")
    m.run("[y0, yp0] = decic(f, 0, [0.25; 0], [1 0], [0; 0], []);")
    assert np.allclose(m.num("y0").ravel(), [0.25, 0.75], atol=1e-8)
    assert abs(m.num("yp0(1)")[0, 0] + 0.25) < 1e-8
    with pytest.raises(MatlabError):
        m.run("decic(@(t, y, yp) yp + y, 0, 1, 1, 0, 1);")      # nothing left to change, residual 1


def test_no_output_arguments_plots(m):
    m.run("ode45(@(t, y) -y, [0 1], 1); n = numel(get(gca, 'Children')); close all")
    assert m.num("n")[0, 0] >= 1
