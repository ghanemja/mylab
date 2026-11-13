"""mylab Blocks: simulations checked against analytic solutions and against ode45."""

import math

import numpy as np
import pytest

from mylab.api import Session
from mylab.blocks import SimError, build, compile_model, linearize, simulate, trim

TIGHT = dict(RelTol="1e-8", AbsTol="1e-10")


def run(model, **kw):
    return simulate(model, **kw)


def out(res, k=0):
    return res.y[:, k]


def first_order(**cfg):
    # dx/dt = 2*(1 - x)
    return build("fo", [("Step", "Step", {"Time": "0"}), ("Sum", "Sum", {"Inputs": "+-"}), ("K", "Gain", {"Gain": "2"}),
                        ("Int", "Integrator"), ("Out", "Outport")],
                 ["Step/1 -> Sum/1", "Sum/1 -> K/1", "K/1 -> Int/1", "Int/1 -> Sum/2", "Int/1 -> Out/1"], **cfg)


# -- linear systems against analytic solutions --------------------------------------------------

def test_first_order_step_response():
    res = run(first_order(StopTime="3", **TIGHT))
    exact = 1 - np.exp(-2 * res.t)
    assert res.t[0] == 0 and res.t[-1] == 3
    assert np.max(np.abs(out(res) - exact)) < 1e-7
    assert np.all(np.diff(res.t) > 0)


def test_default_tolerance_is_reasonable():
    res = run(first_order(StopTime="3"))
    assert np.max(np.abs(out(res) - (1 - np.exp(-2 * res.t)))) < 1e-3
    assert res.t.size >= 51  # MaxStep defaults to a fiftieth of the run


def test_transfer_function_first_order():
    m = build("tf1", [("Step", "Step", {"Time": "0"}), ("G", "TransferFcn", {"Numerator": "[3]", "Denominator": "[2 1]"}),
                      ("Out", "Outport")], ["Step/1 -> G/1", "G/1 -> Out/1"], StopTime="8", **TIGHT)
    res = run(m)
    assert np.max(np.abs(out(res) - 3 * (1 - np.exp(-res.t / 2)))) < 1e-7


def second_order_exact(t, wn, zeta):
    wd = wn * math.sqrt(1 - zeta ** 2)
    return 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta * wn / wd * np.sin(wd * t))


def test_transfer_function_second_order_underdamped():
    wn, zeta = 3.0, 0.2
    m = build("tf2", [("Step", "Step", {"Time": "0"}),
                      ("G", "TransferFcn", {"Numerator": "[wn^2]", "Denominator": "[1 2*zeta*wn wn^2]"}), ("Out", "Outport")],
              ["Step/1 -> G/1", "G/1 -> Out/1"], StopTime="6", **TIGHT)
    s = Session()
    s.run("wn = 3; zeta = 0.2;")
    res = simulate(m, s.interp)
    assert np.max(np.abs(out(res) - second_order_exact(res.t, wn, zeta))) < 1e-6
    assert abs(np.max(out(res)) - (1 + math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2)))) < 2e-3  # overshoot


def msd_model(**cfg):
    # m x'' + c x' + k x = F, built from integrators
    return build("msd", [("F", "Step", {"Time": "0", "After": "F0"}), ("Sum", "Sum", {"Inputs": "+--"}),
                         ("invm", "Gain", {"Gain": "1/m"}), ("v", "Integrator"), ("x", "Integrator"),
                         ("c", "Gain", {"Gain": "c"}), ("k", "Gain", {"Gain": "k"}), ("Out", "Outport"),
                         ("Vout", "Outport", {"Port": "2"})],
                 ["F/1 -> Sum/1", "Sum/1 -> invm/1", "invm/1 -> v/1", "v/1 -> x/1", "v/1 -> c/1", "c/1 -> Sum/2",
                  "x/1 -> k/1", "k/1 -> Sum/3", "x/1 -> Out/1", "v/1 -> Vout/1"], **cfg)


def test_mass_spring_damper_matches_analytic_and_ode45():
    s = Session()
    s.run("m = 2; c = 1.2; k = 18; F0 = 9;")
    res = simulate(msd_model(StopTime="10", **TIGHT), s.interp)
    wn, zeta = math.sqrt(18 / 2), 1.2 / (2 * math.sqrt(18 * 2))
    exact = 9 / 18 * second_order_exact(res.t, wn, zeta)
    assert np.max(np.abs(out(res) - exact)) < 1e-6
    r = s.run("opts = odeset('RelTol',1e-9,'AbsTol',1e-11);"
              "[t, y] = ode45(@(t, y) [y(2); (F0 - c*y(2) - k*y(1))/m], [0 10], [0; 0], opts); yend = y(end, 1);")
    assert r.error is None
    assert abs(out(res)[-1] - float(s.interp.get_var("yend")[0, 0])) < 1e-6
    assert res.y.shape[1] == 2
    assert np.max(np.abs(np.gradient(out(res), res.t)[5:-5] - out(res, 1)[5:-5])) < 0.05


def test_state_space_and_zero_pole_agree_with_transfer_function():
    blocks = [("Step", "Step", {"Time": "0"}),
              ("TF", "TransferFcn", {"Numerator": "[1 3]", "Denominator": "[1 3 2]"}),
              ("SS", "StateSpace", {"A": "[0 1; -2 -3]", "B": "[0; 1]", "C": "[3 1]", "D": "0"}),
              ("ZP", "ZeroPole", {"Zeros": "[-3]", "Poles": "[-1 -2]", "Gain": "1"}),
              ("O1", "Outport"), ("O2", "Outport", {"Port": "2"}), ("O3", "Outport", {"Port": "3"})]
    lines = ["Step/1 -> TF/1", "Step/1 -> SS/1", "Step/1 -> ZP/1", "TF/1 -> O1/1", "SS/1 -> O2/1", "ZP/1 -> O3/1"]
    res = run(build("lin", blocks, lines, StopTime="5", **TIGHT))
    exact = 1.5 - 2 * np.exp(-res.t) + 0.5 * np.exp(-2 * res.t)
    for k in range(3):
        assert np.max(np.abs(out(res, k) - exact)) < 1e-7


def test_transfer_function_with_direct_term():
    # (s + 2)/(s + 1) has a jump of 1 at the step
    m = build("lead", [("Step", "Step", {"Time": "0"}), ("G", "TransferFcn", {"Numerator": "[1 2]", "Denominator": "[1 1]"}),
                       ("Out", "Outport")], ["Step/1 -> G/1", "G/1 -> Out/1"], StopTime="4", **TIGHT)
    res = run(m)
    assert np.max(np.abs(out(res) - (2 - np.exp(-res.t)))) < 1e-7
    assert out(res)[0] == pytest.approx(1.0)


def test_improper_transfer_function_is_an_error():
    m = build("bad", [("Step", "Step"), ("G", "TransferFcn", {"Numerator": "[1 0 0]", "Denominator": "[1 1]"}),
                      ("Out", "Outport")], ["Step/1 -> G/1", "G/1 -> Out/1"])
    with pytest.raises(SimError, match="proper"):
        run(m)


# -- solvers ------------------------------------------------------------------------------------

@pytest.mark.parametrize("solver,tol", [("ode45", 1e-4), ("ode23", 2e-3), ("ode113", 1e-4), ("ode15s", 5e-3),
                                        ("ode23s", 5e-3), ("VariableStepAuto", 1e-4)])
def test_variable_step_solvers(solver, tol):
    res = run(first_order(StopTime="3", Solver=solver))
    assert np.max(np.abs(out(res) - (1 - np.exp(-2 * res.t)))) < tol


@pytest.mark.parametrize("solver,order", [("ode1", 1), ("ode2", 2), ("ode3", 3), ("ode4", 4), ("ode5", 5)])
def test_fixed_step_solvers_have_their_order(solver, order):
    errs = []
    for h in (0.1, 0.05):
        res = run(first_order(StopTime="2", Solver=solver, FixedStep=str(h)))
        assert np.allclose(np.diff(res.t), h)
        errs.append(np.max(np.abs(out(res) - (1 - np.exp(-2 * res.t)))))
    assert errs[0] / errs[1] == pytest.approx(2 ** order, rel=0.35)


def test_euler_matches_hand_computation():
    res = run(first_order(StopTime="0.3", Solver="ode1", FixedStep="0.1"))
    x, expect = 0.0, [0.0]
    for _ in range(3):
        x += 0.1 * 2 * (1 - x)
        expect.append(x)
    assert np.allclose(out(res), expect)


def test_stiff_system_with_bdf():
    # x' = -1000 (x - cos t): explicit steps would need h < 0.002
    m = build("stiff", [("C", "Clock"), ("cos", "TrigonometricFunction", {"Operator": "cos"}), ("Sum", "Sum", {"Inputs": "+-"}),
                        ("K", "Gain", {"Gain": "1000"}), ("Int", "Integrator"), ("Out", "Outport")],
              ["C/1 -> cos/1", "cos/1 -> Sum/1", "Int/1 -> Sum/2", "Sum/1 -> K/1", "K/1 -> Int/1", "Int/1 -> Out/1"],
              StopTime="2", Solver="ode15s", RelTol="1e-5", AbsTol="1e-8", MaxStep="0.5")
    res = run(m)
    lam = 1000.0
    exact = (lam ** 2 * np.cos(res.t) + lam * np.sin(res.t)) / (lam ** 2 + 1)
    assert np.max(np.abs(out(res)[res.t > 0.1] - exact[res.t > 0.1])) < 1e-3
    assert res.t.size < 400


def test_unknown_solver_and_bad_settings():
    with pytest.raises(SimError, match="Unknown solver"):
        run(first_order(Solver="ode99"))
    with pytest.raises(SimError, match="StopTime"):
        run(first_order(StopTime="-1"))
    with pytest.raises(SimError, match="finite"):
        run(first_order(StopTime="inf"))
    with pytest.raises(SimError, match="without continuous states"):
        run(first_order(Solver="FixedStepDiscrete"))


def test_max_step_limits_the_step():
    res = run(first_order(StopTime="2", MaxStep="0.01"))
    assert np.max(np.diff(res.t)) <= 0.01 + 1e-12


def test_stop_time_expression_and_start_time():
    s = Session()
    s.run("T = 2;")
    res = simulate(first_order(StartTime="1", StopTime="T + 1"), s.interp)
    assert res.t[0] == 1 and res.t[-1] == 3


# -- nonlinear continuous models ----------------------------------------------------------------

def vdp(mu):
    return build("vdp", [("x1", "Integrator", {"InitialCondition": "2"}), ("x2", "Integrator", {"InitialCondition": "0"}),
                         ("f", "Fcn", {"Expr": f"{mu}*(1 - u(1)^2)*u(2) - u(1)"}), ("Mux", "Mux"),
                         ("O1", "Outport"), ("O2", "Outport", {"Port": "2"})],
                 ["x2/1 -> x1/1", "x1/1 -> Mux/1", "x2/1 -> Mux/2", "Mux/1 -> f/1", "f/1 -> x2/1", "x1/1 -> O1/1",
                  "x2/1 -> O2/1"], StopTime="20", **TIGHT)


def test_van_der_pol_against_ode45():
    res = run(vdp(1))
    s = Session()
    r = s.run("opts = odeset('RelTol',1e-10,'AbsTol',1e-12);"
              "[t, y] = ode45(@(t, y) [y(2); (1 - y(1)^2)*y(2) - y(1)], [0 20], [2; 0], opts); ye = y(end, :);")
    assert r.error is None
    ye = np.ravel(s.interp.get_var("ye"))
    assert abs(out(res, 0)[-1] - ye[0]) < 1e-5 and abs(out(res, 1)[-1] - ye[1]) < 1e-5
    assert 2.0 < np.max(np.abs(out(res, 0))) < 2.02  # limit-cycle amplitude for mu = 1
    # period of the limit cycle (about 6.663 for mu = 1): spacing of upward zero crossings
    x = out(res, 0)
    up = [res.t[i] - x[i] * (res.t[i + 1] - res.t[i]) / (x[i + 1] - x[i]) for i in range(x.size - 1) if x[i] < 0 <= x[i + 1]]
    assert np.diff(up)[-1] == pytest.approx(6.6633, abs=2e-3)


def test_pendulum_energy_is_conserved():
    m = build("pend", [("th", "Integrator", {"InitialCondition": "1"}), ("w", "Integrator"),
                       ("sin", "TrigonometricFunction"), ("g", "Gain", {"Gain": "-9.81"}),
                       ("O1", "Outport"), ("O2", "Outport", {"Port": "2"})],
              ["w/1 -> th/1", "th/1 -> sin/1", "sin/1 -> g/1", "g/1 -> w/1", "th/1 -> O1/1", "w/1 -> O2/1"],
              StopTime="10", **TIGHT)
    res = run(m)
    energy = 0.5 * out(res, 1) ** 2 - 9.81 * np.cos(out(res, 0))
    assert np.max(np.abs(energy - energy[0])) < 1e-6


# -- PID ----------------------------------------------------------------------------------------

def pid_loop(pid, plant_num="[1]", plant_den="[1 1]", **cfg):
    return build("pid", [("r", "Step", {"Time": "0"}), ("e", "Sum", {"Inputs": "+-"}), ("PID", "PIDController", pid),
                         ("G", "TransferFcn", {"Numerator": plant_num, "Denominator": plant_den}), ("Out", "Outport"),
                         ("U", "Outport", {"Port": "2"})],
                 ["r/1 -> e/1", "e/1 -> PID/1", "PID/1 -> G/1", "G/1 -> e/2", "G/1 -> Out/1", "PID/1 -> U/1"], **cfg)


def test_pi_closed_loop_matches_analytic():
    # plant 1/(s+1), PI with P = 1, I = 1: closed loop (s+1)/(s^2+2s+1) = 1/(s+1)
    res = run(pid_loop({"Controller": "PI", "P": "1", "I": "1"}, StopTime="6", **TIGHT))
    assert np.max(np.abs(out(res) - (1 - np.exp(-res.t)))) < 1e-7


def test_pid_closed_loop_matches_equivalent_transfer_function():
    P, I, D, N = 2.0, 3.0, 0.5, 20.0
    res = run(pid_loop({"P": str(P), "I": str(I), "D": str(D), "N": str(N)}, "[1]", "[1 2 1]", StopTime="8", MaxStep="0.004", **TIGHT))
    # C(s) = P + I/s + D N s/(s + N) and G = 1/(s+1)^2, closed loop as one transfer function
    cnum = np.polyadd(np.polyadd(np.polymul([P], np.polymul([1, 0], [1, N])), np.polymul([I], [1, N])),
                      np.polymul([D * N, 0], [1, 0]))
    cden = np.polymul([1, 0], [1, N])
    num = cnum
    den = np.polyadd(np.polymul(cden, [1, 2, 1]), cnum)
    ref = build("ref", [("r", "Step", {"Time": "0"}),
                        ("T", "TransferFcn", {"Numerator": "[" + " ".join(repr(float(v)) for v in num) + "]",
                                              "Denominator": "[" + " ".join(repr(float(v)) for v in den) + "]"}), ("Out", "Outport")],
                ["r/1 -> T/1", "T/1 -> Out/1"], StopTime="8", MaxStep="0.004", **TIGHT)
    ref_res = run(ref)
    assert np.max(np.abs(np.interp(ref_res.t, res.t, out(res)) - out(ref_res))) < 5e-5
    assert out(res)[-1] == pytest.approx(1.0, abs=0.05)


def test_pid_anti_windup_reduces_overshoot():
    base = {"Controller": "PI", "P": "2", "I": "4", "LimitOutput": "on", "UpperSaturationLimit": "1.2",
            "LowerSaturationLimit": "-1.2"}
    peak = {}
    for mode in ("none", "clamping", "back-calculation"):
        res = run(pid_loop({**base, "AntiWindupMode": mode, "Kb": "5"}, "[1]", "[5 1]", StopTime="30"))
        assert np.max(out(res, 1)) <= 1.2 + 1e-12 and np.min(out(res, 1)) >= -1.2 - 1e-12
        assert out(res)[-1] == pytest.approx(1.0, abs=2e-2)
        peak[mode] = np.max(out(res))
    assert peak["clamping"] < peak["none"] - 0.02
    assert peak["back-calculation"] < peak["none"] - 0.02


def test_pid_ideal_form_and_p_only():
    res = run(pid_loop({"Controller": "P", "P": "4"}, StopTime="5", **TIGHT))
    assert np.max(np.abs(out(res) - 0.8 * (1 - np.exp(-5 * res.t)))) < 1e-7
    a = run(pid_loop({"Controller": "PI", "Form": "Ideal", "P": "2", "I": "0.5"}, StopTime="5", MaxStep="0.004", **TIGHT))
    b = run(pid_loop({"Controller": "PI", "Form": "Parallel", "P": "2", "I": "1"}, StopTime="5", MaxStep="0.004", **TIGHT))
    assert np.max(np.abs(np.interp(b.t, a.t, out(a)) - out(b))) < 1e-6


# -- events -------------------------------------------------------------------------------------

def ball(restitution="0.8", **cfg):
    return build("ball", [("g", "Constant", {"Value": "-9.81"}),
                          ("v", "Integrator", {"InitialCondition": "0", "ExternalReset": "rising",
                                               "InitialConditionSource": "external"}),
                          ("x", "Integrator", {"InitialCondition": "10", "LimitOutput": "on", "LowerSaturationLimit": "0"}),
                          ("hit", "CompareToZero", {"relop": "<="}), ("bounce", "Gain", {"Gain": "-" + restitution}),
                          ("X", "Outport"), ("V", "Outport", {"Port": "2"})],
                 ["g/1 -> v/1", "v/1 -> x/1", "x/1 -> hit/1", "hit/1 -> v/2", "v/1 -> bounce/1", "bounce/1 -> v/3",
                  "x/1 -> X/1", "v/1 -> V/1"], **cfg)


def test_bouncing_ball_impact_times_and_heights():
    res = run(ball(StopTime="8", RelTol="1e-8", AbsTol="1e-10", MaxStep="0.05"))
    x, v, t = out(res, 0), out(res, 1), res.t
    t1 = math.sqrt(2 * 10 / 9.81)
    # impacts: velocity jumps from negative to positive
    hits = [t[i] for i in range(1, t.size) if v[i - 1] < -1 and v[i] > 1]
    expect = [t1]
    v0 = 9.81 * t1
    for _ in range(3):
        v0 *= 0.8
        expect.append(expect[-1] + 2 * v0 / 9.81)
    assert len(hits) >= 4
    for h, e in zip(hits, expect):
        assert h == pytest.approx(e, abs=2e-6)
    assert np.min(x) >= -1e-9
    # apex after the first bounce is 0.64 of the drop height
    first, second = hits[0], hits[1]
    seg = (t > first) & (t < second)
    assert np.max(x[seg]) == pytest.approx(6.4, abs=2e-3)


def test_bouncing_ball_without_zero_crossing_detection_is_less_accurate():
    exact = math.sqrt(2 * 10 / 9.81)

    def first_hit(**cfg):
        res = run(ball(StopTime="2", MaxStep="0.1", **cfg))
        v = out(res, 1)
        return next(res.t[i] for i in range(1, res.t.size) if v[i - 1] < -1 and v[i] > 1)

    assert abs(first_hit() - exact) < 1e-5
    assert abs(first_hit(ZeroCrossings="off") - exact) > 1e-3


def relay_model(**cfg):
    # thermostat: dT/dt = (-(T - 5) + 30*heater)/10, heater on below 19, off above 21
    return build("thermo", [("set", "Constant", {"Value": "20"}), ("err", "Sum", {"Inputs": "+-"}),
                            ("relay", "Relay", {"OnSwitchValue": "1", "OffSwitchValue": "-1"}),
                            ("heat", "Gain", {"Gain": "30"}), ("amb", "Constant", {"Value": "5"}),
                            ("sum", "Sum", {"Inputs": "++-"}), ("tau", "Gain", {"Gain": "0.1"}),
                            ("T", "Integrator", {"InitialCondition": "15"}), ("Out", "Outport"),
                            ("H", "Outport", {"Port": "2"})],
                 ["set/1 -> err/1", "T/1 -> err/2", "err/1 -> relay/1", "relay/1 -> heat/1", "heat/1 -> sum/1",
                  "amb/1 -> sum/2", "T/1 -> sum/3", "sum/1 -> tau/1", "tau/1 -> T/1", "T/1 -> Out/1", "relay/1 -> H/1"], **cfg)


def test_relay_limit_cycle_period_and_bounds():
    res = run(relay_model(StopTime="60", RelTol="1e-8", AbsTol="1e-10"))
    T, h, t = out(res, 0), out(res, 1), res.t
    settled = t > 10
    assert np.max(T[settled]) == pytest.approx(21, abs=1e-6)
    assert np.min(T[settled]) == pytest.approx(19, abs=1e-6)
    # heating 19 -> 21 towards 35, cooling 21 -> 19 towards 5
    period = 10 * math.log((35 - 19) / (35 - 21)) + 10 * math.log((21 - 5) / (19 - 5))
    on = [t[i] for i in range(1, t.size) if h[i - 1] == 0 and h[i] == 1 and t[i] > 5]
    assert np.diff(on) == pytest.approx(period, abs=1e-5)
    assert set(np.unique(h)) == {0.0, 1.0}


def test_relay_chattering_is_reported():
    m = relay_model(StopTime="60")
    m.root.block("relay").params.update({"OnSwitchValue": "0", "OffSwitchValue": "0"})
    with pytest.raises(SimError, match="chattering"):
        run(m)


def test_step_time_is_hit_exactly():
    m = build("st", [("Step", "Step", {"Time": "0.7351"}), ("Int", "Integrator"), ("Out", "Outport")],
              ["Step/1 -> Int/1", "Int/1 -> Out/1"], StopTime="2")
    res = run(m)
    assert np.any(np.abs(res.t - 0.7351) < 1e-12)
    assert np.max(np.abs(out(res) - np.maximum(res.t - 0.7351, 0))) < 1e-9


def test_integrator_limits_and_release():
    # integrate a sine: without limits x = 1 - cos t (0..2); the upper limit 1.5 holds it, then lets go when the input turns
    m = build("lim", [("S", "SineWave"), ("Int", "Integrator", {"LimitOutput": "on", "UpperSaturationLimit": "1.5",
                                                                 "LowerSaturationLimit": "-inf"}), ("Out", "Outport")],
              ["S/1 -> Int/1", "Int/1 -> Out/1"], StopTime="7", RelTol="1e-8", AbsTol="1e-10")
    res = run(m)
    t, y = res.t, out(res)
    t_hit = math.acos(-0.5)
    assert np.max(y) == pytest.approx(1.5, abs=1e-9)
    before = t < t_hit
    assert np.max(np.abs(y[before] - (1 - np.cos(t[before])))) < 1e-7
    held = (t > t_hit + 1e-6) & (t < math.pi)
    assert np.allclose(y[held], 1.5)
    after = (t > math.pi) & (t < 2 * math.pi)
    assert np.max(np.abs(y[after] - (1.5 - (1 + np.cos(t[after]))))) < 1e-6


def test_integrator_reset_modes():
    def model(mode, period="1"):
        return build("rst", [("one", "Constant"), ("P", "PulseGenerator", {"Period": period, "PulseWidth": "50"}),
                             ("Int", "Integrator", {"ExternalReset": mode}), ("Out", "Outport")],
                     ["one/1 -> Int/1", "P/1 -> Int/2", "Int/1 -> Out/1"], StopTime="3.2")

    rising = run(model("rising"))
    assert 0.9 < np.max(out(rising)) <= 1.0  # reset each time the pulse rises
    assert out(rising)[-1] == pytest.approx(0.2, abs=1e-9)
    assert np.interp(2.5, rising.t, out(rising)) == pytest.approx(0.5, abs=1e-9)
    assert out(rising)[np.argmin(np.abs(rising.t - 2.0))] == 0  # the value at the reset instant is the new one
    either = run(model("either"))
    assert 0.4 < np.max(out(either)) <= 0.5 + 1e-9


def test_hit_crossing_and_stop_simulation():
    m = build("stop", [("C", "Clock"), ("f", "Fcn", {"Expr": "u^2 - 2"}), ("H", "HitCrossing", {"HitCrossingDirection": "rising"}),
                       ("Stop", "StopSimulation"), ("Out", "Outport")],
              ["C/1 -> f/1", "f/1 -> H/1", "H/1 -> Stop/1", "f/1 -> Out/1"], StopTime="10")
    res = run(m)
    assert res.t[-1] == pytest.approx(math.sqrt(2), abs=1e-8)
    assert res.stopped_by == "Stop"


def test_switch_event_location():
    # the switch flips when sin(t) crosses 0.5; integrate the switch output
    m = build("sw", [("S", "SineWave"), ("one", "Constant", {"Value": "1"}), ("zero", "Constant", {"Value": "0"}),
                     ("Sw", "Switch", {"Threshold": "0.5"}), ("Int", "Integrator"), ("Out", "Outport")],
              ["one/1 -> Sw/1", "S/1 -> Sw/2", "zero/1 -> Sw/3", "Sw/1 -> Int/1", "Int/1 -> Out/1"],
              StopTime="3", MaxStep="0.5")
    res = run(m)
    assert out(res)[-1] == pytest.approx(5 * math.pi / 6 - math.pi / 6, abs=1e-8)


# -- delays -------------------------------------------------------------------------------------

def test_transport_delay():
    m = build("td", [("S", "SineWave", {"Frequency": "2"}), ("D", "TransportDelay", {"DelayTime": "0.5", "InitialOutput": "0.25"}),
                     ("Out", "Outport")], ["S/1 -> D/1", "D/1 -> Out/1"], StopTime="5", MaxStep="0.01")
    res = run(m)
    t, y = res.t, out(res)
    assert np.all(y[t < 0.5] == 0.25)
    late = t > 0.5
    assert np.max(np.abs(y[late] - np.sin(2 * (t[late] - 0.5)))) < 1e-4


def test_delay_in_feedback_loop_oscillates_as_predicted():
    # x' = -x(t - pi/2) has the exact solution cos(t) for the history cos(t) on [-pi/2, 0]; here the history is 1,
    # so compare with a method-of-steps reference on the first interval: x = 1 - t for t <= pi/2
    m = build("dde", [("Int", "Integrator", {"InitialCondition": "1"}),
                      ("D", "TransportDelay", {"DelayTime": "pi/2", "InitialOutput": "1"}), ("neg", "Gain", {"Gain": "-1"}),
                      ("Out", "Outport")], ["Int/1 -> D/1", "D/1 -> neg/1", "neg/1 -> Int/1", "Int/1 -> Out/1"],
              StopTime="3", MaxStep="0.005")
    res = run(m)
    t, y = res.t, out(res)
    a = t <= math.pi / 2
    assert np.max(np.abs(y[a] - (1 - t[a]))) < 1e-9
    b = (t > math.pi / 2) & (t <= math.pi)
    s = t[b] - math.pi / 2
    exact = (1 - math.pi / 2) - s + s ** 2 / 2
    assert np.max(np.abs(y[b] - exact)) < 1e-4


def test_derivative_block():
    m = build("der", [("S", "SineWave"), ("D", "Derivative"), ("Out", "Outport")], ["S/1 -> D/1", "D/1 -> Out/1"],
              StopTime="6", MaxStep="0.001")
    res = run(m)
    assert np.max(np.abs(out(res)[5:] - np.cos(res.t[5:]))) < 1e-3


# -- structure ----------------------------------------------------------------------------------

def test_algebraic_loop_is_reported_with_the_blocks():
    m = build("loop", [("C", "Constant"), ("Sum", "Sum", {"Inputs": "+-"}), ("K", "Gain", {"Gain": "2"}), ("Out", "Outport")],
              ["C/1 -> Sum/1", "Sum/1 -> K/1", "K/1 -> Sum/2", "K/1 -> Out/1"])
    with pytest.raises(SimError) as info:
        run(m)
    assert "Algebraic loop" in info.value.message
    assert set(info.value.paths) == {"Sum", "K"}
    assert "Memory" in info.value.message
    diags = compile_model(m, strict=False).diagnostics
    loop = [d for d in diags if d["kind"] == "algebraic-loop"]
    assert len(loop) == 1 and loop[0]["level"] == "error"


def test_loop_broken_by_memory_or_integrator_is_fine():
    m = build("ok", [("C", "Constant"), ("Sum", "Sum", {"Inputs": "+-"}), ("K", "Gain", {"Gain": "0.5"}), ("M", "Memory"),
                     ("Out", "Outport")],
              ["C/1 -> Sum/1", "Sum/1 -> K/1", "K/1 -> M/1", "M/1 -> Sum/2", "K/1 -> Out/1"], StopTime="1", MaxStep="0.02")
    res = run(m)
    assert out(res)[-1] == pytest.approx(1 / 3, abs=1e-6)  # fixed point of y = 0.5 (1 - y)


def test_transfer_function_loop_with_direct_feedthrough_is_algebraic():
    m = build("l2", [("G", "TransferFcn", {"Numerator": "[1 1]", "Denominator": "[1 2]"}), ("K", "Gain"), ("T", "Terminator")],
              ["G/1 -> K/1", "K/1 -> G/1", "G/1 -> T/1"])
    with pytest.raises(SimError, match="Algebraic loop"):
        run(m)
    m.root.block("G").params["Numerator"] = "[1]"
    run(m)


def test_subsystem_nesting_gives_the_same_result():
    inner = ([("In", "Inport"), ("K", "Gain", {"Gain": "2"}), ("Out", "Outport")], ["In/1 -> K/1", "K/1 -> Out/1"])
    mid = ([("In1", "Inport"), ("Fb", "Inport", {"Port": "2"}), ("Sum", "Sum", {"Inputs": "+-"}), ("Inner", "Subsystem", {}, inner),
            ("Out1", "Outport")], ["In1/1 -> Sum/1", "Fb/1 -> Sum/2", "Sum/1 -> Inner/1", "Inner/1 -> Out1/1"])
    m = build("nest", [("Step", "Step", {"Time": "0"}), ("Ctl", "Subsystem", {}, mid), ("Int", "Integrator"), ("Out", "Outport")],
              ["Step/1 -> Ctl/1", "Int/1 -> Ctl/2", "Ctl/1 -> Int/1", "Int/1 -> Out/1"], StopTime="3", **TIGHT)
    res = run(m)
    assert np.max(np.abs(out(res) - (1 - np.exp(-2 * res.t)))) < 1e-7
    comp = compile_model(m)
    assert sorted(n.path for n in comp.order) == ["Ctl/Inner/K", "Ctl/Sum", "Int", "Out", "Step"]


def test_goto_from_across_subsystems():
    inner = ([("F", "From", {"GotoTag": "sig"}), ("K", "Gain", {"Gain": "3"}), ("Out", "Outport")], ["F/1 -> K/1", "K/1 -> Out/1"])
    m = build("gf", [("C", "Constant", {"Value": "2"}), ("G", "Goto", {"GotoTag": "sig"}), ("S", "Subsystem", {}, inner),
                     ("Out", "Outport")], ["C/1 -> G/1", "S/1 -> Out/1"], StopTime="1")
    assert np.allclose(out(run(m)), 6)
    m.root.block("G").params["GotoTag"] = "other"
    with pytest.raises(SimError, match="no Goto block"):
        run(m)


def test_mask_parameters_shadow_the_workspace():
    inner = ([("In", "Inport"), ("K", "Gain", {"Gain": "k*2"}), ("Out", "Outport")], ["In/1 -> K/1", "K/1 -> Out/1"])
    m = build("mask", [("C", "Constant", {"Value": "1"}), ("S", "Subsystem", {}, inner), ("Out", "Outport")],
              ["C/1 -> S/1", "S/1 -> Out/1"], StopTime="1")
    m.root.block("S").extra["mask"] = {"k": "a + 1"}
    s = Session()
    s.run("a = 4; k = 100;")
    assert np.allclose(out(simulate(m, s.interp)), 10)


def test_vector_signals_through_integrator_gain_and_mux():
    m = build("vec", [("C", "Constant", {"Value": "[1 2 3]"}), ("K", "Gain", {"Gain": "[1 10 100]"}),
                      ("Int", "Integrator", {"InitialCondition": "0"}), ("D", "Demux", {"Outputs": "3"}),
                      ("M", "Mux", {"Inputs": "2"}), ("Out", "Outport"), ("O2", "Outport", {"Port": "2"})],
              ["C/1 -> K/1", "K/1 -> Int/1", "Int/1 -> Out/1", "Int/1 -> D/1", "D/3 -> M/1", "D/1 -> M/2", "M/1 -> O2/1"],
              StopTime="2")
    res = run(m)
    assert res.y.shape == (res.t.size, 5)
    assert np.allclose(res.y[-1], [2, 40, 600, 600, 2])


def test_matrix_gain_state_space_form():
    # x' = A x with matrix gain feeding a vector integrator
    m = build("mat", [("Int", "Integrator", {"InitialCondition": "[1 0]"}),
                      ("A", "Gain", {"Gain": "[0 1; -4 0]", "Multiplication": "Matrix(K*u)"}), ("Out", "Outport")],
              ["Int/1 -> A/1", "A/1 -> Int/1", "Int/1 -> Out/1"], StopTime="3", **TIGHT)
    res = run(m)
    assert np.max(np.abs(res.y[:, 0] - np.cos(2 * res.t))) < 1e-6
    assert np.max(np.abs(res.y[:, 1] + 2 * np.sin(2 * res.t))) < 1e-6


def test_unconnected_input_is_a_warning_not_an_error():
    m = build("unc", [("Sum", "Sum"), ("C", "Constant", {"Value": "4"}), ("Out", "Outport")], ["C/1 -> Sum/1", "Sum/1 -> Out/1"],
              StopTime="1")
    res = run(m)
    assert np.allclose(out(res), 4)
    assert any("Input 2 of 'Sum'" in w for w in res.warnings)
    d = [x for x in compile_model(m, strict=False).diagnostics if x["kind"] == "unconnected-input"]
    assert d[0]["path"] == "Sum" and d[0]["port"] == 2 and d[0]["level"] == "warning"


def test_errors_name_the_block():
    m = build("err", [("C", "Constant", {"Value": "nosuchvar + 1"}), ("Out", "Outport")], ["C/1 -> Out/1"])
    with pytest.raises(SimError) as info:
        run(m)
    assert info.value.path == "C" and "nosuchvar" in info.value.message and "Value" in info.value.message
    m = build("err2", [("C", "Constant", {"Value": "[1 2 3]"}), ("K", "Gain", {"Gain": "[1 2]"}), ("Out", "Outport")],
              ["C/1 -> K/1", "K/1 -> Out/1"])
    with pytest.raises(SimError) as info:
        run(m)
    assert info.value.path == "K"
    m = build("err3", [("X", "NoSuchBlock"), ("Out", "Outport")], ["X/1 -> Out/1"])
    with pytest.raises(SimError, match="no block type 'NoSuchBlock'"):
        run(m)
    m = build("err4", [("C", "Constant"), ("C2", "Constant"), ("Out", "Outport")], ["C/1 -> Out/1", "C2/1 -> Out/1"])
    with pytest.raises(SimError, match="more than one line"):
        run(m)


def test_divergence_is_reported():
    m = build("blow", [("Int", "Integrator", {"InitialCondition": "1"}), ("sq", "Product"), ("Out", "Outport")],
              ["Int/1 -> sq/1", "Int/1 -> sq/2", "sq/1 -> Int/1", "Int/1 -> Out/1"], StopTime="2")
    with pytest.raises(SimError):  # x' = x^2 blows up at t = 1
        run(m)


# -- linearization and trim ---------------------------------------------------------------------

def test_linearize_known_system():
    s = Session()
    s.run("m = 2; c = 1.2; k = 18; F0 = 1;")
    plant = msd_model()
    plant.root.blocks[0] = type(plant.root.blocks[0])("F", "Inport", {"Port": "1"})
    r = linearize(plant, s.interp)
    # states in block order: v, x
    assert np.allclose(r["A"], [[-0.6, -9], [1, 0]], atol=1e-7)
    assert np.allclose(r["B"], [[0.5], [0]], atol=1e-7)
    assert np.allclose(r["C"], [[0, 1], [1, 0]], atol=1e-7)
    assert np.allclose(r["D"], 0)
    assert r["states"] == ["v", "x"]


def test_linearize_nonlinear_pendulum_at_two_points():
    m = build("pl", [("u", "Inport"), ("th", "Integrator"), ("w", "Integrator"), ("sin", "TrigonometricFunction"),
                     ("sum", "Sum", {"Inputs": "+-"}), ("g", "Gain", {"Gain": "9.81"}), ("y", "Outport")],
              ["w/1 -> th/1", "th/1 -> sin/1", "sin/1 -> g/1", "u/1 -> sum/1", "g/1 -> sum/2", "sum/1 -> w/1", "th/1 -> y/1"])
    down = linearize(m)
    assert np.allclose(down["A"], [[0, 1], [-9.81, 0]], atol=1e-6)
    up = linearize(m, x0=[math.pi, 0])
    assert np.allclose(up["A"], [[0, 1], [9.81, 0]], atol=1e-6)
    assert np.allclose(up["B"], [[0], [1]]) and np.allclose(up["C"], [[1, 0]])


def test_linearize_transfer_function_has_the_right_gain_and_poles():
    m = build("lt", [("u", "Inport"), ("G", "TransferFcn", {"Numerator": "[2 1]", "Denominator": "[1 3 2]"}), ("y", "Outport")],
              ["u/1 -> G/1", "G/1 -> y/1"])
    r = linearize(m)
    assert np.allclose(sorted(np.linalg.eigvals(r["A"]).real), [-2, -1])
    dc = (-r["C"] @ np.linalg.inv(r["A"]) @ r["B"] + r["D"])[0, 0]
    assert dc == pytest.approx(0.5)


def test_trim_finds_the_steady_state():
    # x' = -x^3 + u, with u fixed by a constant 8 added to the input: steady state x = 2 for u = 0
    m = build("tr", [("u", "Inport"), ("c", "Constant", {"Value": "8"}), ("s", "Sum", {"Inputs": "++-"}),
                     ("x", "Integrator", {"InitialCondition": "1"}), ("cube", "Fcn", {"Expr": "u^3"}), ("y", "Outport")],
              ["u/1 -> s/1", "c/1 -> s/2", "cube/1 -> s/3", "s/1 -> x/1", "x/1 -> cube/1", "x/1 -> y/1"])
    r = trim(m)
    assert np.max(np.abs(r["dx"])) < 1e-8
    assert r["x"][0] ** 3 - r["u"][0] == pytest.approx(8, abs=1e-6)
