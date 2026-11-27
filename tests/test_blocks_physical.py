"""mylab Blocks: physical networks (electrical, translational, rotational) against exact solutions."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mylab.api import Session
from mylab.blocks import SimError, build, linearize, load_model, simulate
from mylab.blocks.writer import save_as

TIGHT = dict(RelTol="1e-9", AbsTol="1e-12")


def rc_model(**cfg):
    # a unit step through R = 1 kOhm into C = 1 mF: v_C = 1 - exp(-t / RC), RC = 1 s
    return build(
        "rc",
        [("St", "Step", {"Time": "0"}), ("ToPS", "SimulinkPSConverter"), ("Vs", "ControlledVoltageSource"),
         ("R", "Resistor", {"R": "1000"}), ("C", "Capacitor", {"C": "1e-3"}), ("G", "ElectricalReference"),
         ("VS", "VoltageSensor"), ("FromPS", "PSSimulinkConverter"), ("Out", "Outport"),
         ("Solver", "SolverConfiguration")],
        ["St/1 -> ToPS/1", "ToPS/1 -> Vs/1", "Vs/p -> R/p", "R/n -> C/p", "C/n -> G/p", "Vs/n -> G/p",
         "VS/p -> C/p", "VS/n -> C/n", "VS/1 -> FromPS/1", "FromPS/1 -> Out/1", "Solver/p -> G/p"],
        StopTime="5",
        **cfg,
    )


def test_rc_circuit():
    r = simulate(rc_model(**TIGHT))
    assert np.max(np.abs(r.y[:, 0] - (1 - np.exp(-r.t)))) < 1e-8
    r = simulate(rc_model(Solver="ode15s", RelTol="1e-8", AbsTol="1e-10"))
    assert np.max(np.abs(r.y[:, 0] - (1 - np.exp(-r.t)))) < 1e-6
    r = simulate(rc_model(Solver="ode4", FixedStep="0.01"))
    assert np.max(np.abs(r.y[:, 0] - (1 - np.exp(-r.t)))) < 1e-8


def test_series_rlc_circuit():
    # 1 V into R = 2, L = 1, C = 0.25: alpha = 1, w0 = 2, wd = sqrt(3)
    m = build(
        "rlc",
        [("V", "DCVoltageSource", {"V": "1"}), ("R", "Resistor", {"R": "2"}), ("L", "Inductor", {"L": "1"}),
         ("C", "Capacitor", {"C": "0.25"}), ("G", "ElectricalReference"), ("I", "CurrentSensor"),
         ("VC", "VoltageSensor"), ("O1", "Outport"), ("O2", "Outport", {"Port": "2"})],
        ["V/p -> R/p", "R/n -> I/p", "I/n -> L/p", "L/n -> C/p", "C/n -> G/p", "V/n -> G/p", "VC/p -> C/p",
         "VC/n -> C/n", "I/1 -> O1/1", "VC/1 -> O2/1"],
        StopTime="6",
        **TIGHT,
    )
    r = simulate(m)
    t, wd = r.t, np.sqrt(3)
    vc = 1 - np.exp(-t) * (np.cos(wd * t) + np.sin(wd * t) / wd)
    i = 0.25 * np.exp(-t) * (1 + 1 / wd**2) * wd * np.sin(wd * t)  # C dv_C/dt
    assert np.max(np.abs(r.y[:, 1] - vc)) < 1e-8 and np.max(np.abs(r.y[:, 0] - i)) < 1e-8


def test_current_source_into_parallel_rc():
    # 2 mA into R = 1 kOhm parallel C = 1 mF: v = I R (1 - exp(-t / RC))
    m = build(
        "irc",
        [("I", "DCCurrentSource", {"I": "2e-3"}), ("R", "Resistor", {"R": "1e3"}), ("C", "Capacitor", {"C": "1e-3"}),
         ("G", "ElectricalReference"), ("VS", "VoltageSensor"), ("O", "Outport")],
        ["I/p -> R/p", "R/p -> C/p", "I/n -> G/p", "R/n -> G/p", "C/n -> G/p", "VS/p -> C/p", "VS/n -> G/p",
         "VS/1 -> O/1"],
        StopTime="4",
        **TIGHT,
    )
    r = simulate(m)
    assert np.max(np.abs(r.y[:, 0] - 2 * (1 - np.exp(-r.t)))) < 1e-8


def test_mass_spring_damper_against_the_ode_and_linmod():
    m = build(
        "msd",
        [("F", "Step", {"Time": "0", "After": "5"}), ("Src", "IdealForceSource"), ("M", "Mass", {"m": "2"}),
         ("K", "TranslationalSpring", {"k": "20"}), ("B", "TranslationalDamper", {"b": "1.5"}),
         ("Ref", "MechanicalTranslationalReference"), ("S", "IdealTranslationalMotionSensor"),
         ("FS", "IdealForceSensor"), ("O1", "Outport"), ("O2", "Outport", {"Port": "2"}),
         ("O3", "Outport", {"Port": "3"})],
        ["F/1 -> Src/1", "Src/R -> M/M", "Src/C -> Ref/R", "K/R -> FS/R", "FS/C -> M/M", "K/C -> Ref/R",
         "B/R -> M/M", "B/C -> Ref/R", "S/R -> M/M", "S/C -> Ref/R", "S/1 -> O1/1", "S/2 -> O2/1", "FS/1 -> O3/1"],
        StopTime="10",
        **TIGHT,
    )
    r = simulate(m)
    sol = solve_ivp(lambda t, y: [y[1], (5 - 20 * y[0] - 1.5 * y[1]) / 2], [0, 10], [0, 0], t_eval=r.t,
                    rtol=1e-11, atol=1e-13)
    assert np.max(np.abs(r.y[:, 1] - sol.y[0])) < 1e-8 and np.max(np.abs(r.y[:, 0] - sol.y[1])) < 1e-8
    # the force sensor sits in series with the spring: it measures the spring force, -k x (into the mass)
    assert np.max(np.abs(r.y[:, 2] + 20 * sol.y[0])) < 1e-6
    # replace the force by a top-level input: linmod gives the textbook matrices
    m.root.block("F").type = "Inport"
    m.root.block("F").params = {}
    lin = linearize(m)
    # states: mass velocity, spring stretch, sensor position
    a = lin["A"]
    assert np.allclose(a[:2, :2], [[-0.75, -10], [1, 0]]) and np.allclose(lin["B"][:2, 0], [0.5, 0])


def test_rotational_inertia_spring_and_damper():
    # J = 0.5 with damper b = 0.2, constant torque 1: w = 5 (1 - exp(-0.4 t))
    m = build(
        "rot",
        [("Tq", "Constant", {"Value": "1"}), ("Src", "IdealTorqueSource"), ("J", "Inertia", {"J": "0.5"}),
         ("D", "RotationalDamper", {"b": "0.2"}), ("Ref", "MechanicalRotationalReference"),
         ("S", "IdealRotationalMotionSensor"), ("O", "Outport"), ("O2", "Outport", {"Port": "2"})],
        ["Tq/1 -> Src/1", "Src/F -> J/I", "Src/B -> Ref/R", "D/F -> J/I", "D/B -> Ref/R", "S/F -> J/I",
         "S/B -> Ref/R", "S/1 -> O/1", "S/2 -> O2/1"],
        StopTime="5",
        **TIGHT,
    )
    r = simulate(m)
    t = r.t
    assert np.max(np.abs(r.y[:, 0] - 5 * (1 - np.exp(-0.4 * t)))) < 1e-8
    assert np.max(np.abs(r.y[:, 1] - (5 * t - 12.5 * (1 - np.exp(-0.4 * t))))) < 1e-7
    # an inertia on a torsion spring, started at 1 rad/s: w = cos(sqrt(k/J) t)
    m = build(
        "osc",
        [("J", "Inertia", {"J": "2", "w0": "1"}), ("K", "RotationalSpring", {"k": "8"}),
         ("Ref", "MechanicalRotationalReference"), ("TS", "IdealTorqueSensor"), ("S", "IdealRotationalMotionSensor"),
         ("O", "Outport"), ("O2", "Outport", {"Port": "2"})],
        ["K/F -> TS/F", "TS/B -> J/I", "K/B -> Ref/R", "S/F -> J/I", "S/B -> Ref/R", "S/1 -> O/1", "TS/1 -> O2/1"],
        StopTime="6",
        **TIGHT,
    )
    r = simulate(m)
    assert np.max(np.abs(r.y[:, 0] - np.cos(2 * r.t))) < 1e-8
    assert np.max(np.abs(r.y[:, 1] + 8 * np.sin(2 * r.t) / 2)) < 1e-7  # torque -k theta, theta = sin(2t)/2


def test_velocity_source_and_diode_rectifier():
    m = build(
        "dio",
        [("S", "SineWave", {"Amplitude": "5"}), ("V", "ControlledVoltageSource"),
         ("D", "Diode", {"Vf": "0.7", "Ron": "0.001", "Goff": "0"}), ("R", "Resistor", {"R": "10"}),
         ("G", "ElectricalReference"), ("VS", "VoltageSensor"), ("O", "Outport")],
        ["S/1 -> V/1", "V/p -> D/p", "D/n -> R/p", "R/n -> G/p", "V/n -> G/p", "VS/p -> R/p", "VS/n -> R/n",
         "VS/1 -> O/1"],
        StopTime="10",
        **TIGHT,
    )
    r = simulate(m)
    u = 5 * np.sin(r.t)
    assert np.max(np.abs(r.y[:, 0] - np.where(u > 0.7, (u - 0.7) * 10 / 10.001, 0))) < 1e-9
    # a velocity source drives a damper: force b v
    m = build(
        "vs",
        [("Clk", "Clock"), ("Src", "IdealTranslationalVelocitySource"), ("B", "TranslationalDamper", {"b": "3"}),
         ("Ref", "MechanicalTranslationalReference"), ("FS", "IdealForceSensor"), ("O", "Outport")],
        ["Clk/1 -> Src/1", "Src/R -> FS/R", "FS/C -> B/R", "B/C -> Ref/R", "Src/C -> Ref/R", "FS/1 -> O/1"],
        StopTime="2",
    )
    r = simulate(m)
    assert np.allclose(r.y[:, 0], 3 * r.t)


def test_networks_that_are_not_index_1_or_not_determined_are_errors():
    m = build("bad", [("V", "DCVoltageSource"), ("C", "Capacitor"), ("G", "ElectricalReference")],
              ["V/p -> C/p", "V/n -> G/p", "C/n -> G/p"])
    with pytest.raises(SimError, match="index-1 networks only.*'V', 'C'|index-1 networks only.*'C', 'V'"):
        simulate(m)
    m = build("float", [("R", "Resistor"), ("C", "Capacitor")], ["R/p -> C/p", "R/n -> C/n"])
    with pytest.raises(SimError, match="add a reference"):
        simulate(m)
    m = build("rigid", [("M1", "Mass"), ("M2", "Mass"), ("K", "TranslationalSpring"),
                        ("Ref", "MechanicalTranslationalReference")], ["M1/M -> M2/M", "K/R -> M1/M", "K/C -> Ref/R"])
    with pytest.raises(SimError, match="two masses joined rigidly"):
        simulate(m)
    m = build("mix", [("R", "Resistor"), ("M", "Mass"), ("G", "ElectricalReference")], ["R/p -> M/M", "R/n -> G/p"])
    with pytest.raises(SimError, match="connected to an electrical port"):
        simulate(m)
    m = build("sig", [("R", "Resistor"), ("K", "Gain"), ("G", "ElectricalReference")], ["R/p -> K/1", "R/n -> G/p"])
    with pytest.raises(SimError, match="physical ports connect only to physical ports"):
        simulate(m)


def test_physical_ports_of_subsystems_and_files(tmp_path):
    inner = (
        [("L", "ConnectionPort", {"Port": "1", "Side": "left"}),
         ("Rt", "ConnectionPort", {"Port": "1", "Side": "right"}),
         ("R", "Resistor", {"R": "1000"}), ("C", "Capacitor", {"C": "1e-3"})],
        ["L/x -> R/p", "R/n -> C/p", "C/n -> Rt/x"],
    )
    m = build(
        "rcsub",
        [("St", "Step", {"Time": "0"}), ("Vs", "ControlledVoltageSource"), ("RC", "Subsystem", {}, inner),
         ("G", "ElectricalReference"), ("I", "CurrentSensor"), ("O", "Outport")],
        ["St/1 -> Vs/1", "Vs/p -> I/p", "I/n -> RC/L1", "RC/R1 -> G/p", "Vs/n -> G/p", "I/1 -> O/1"],
        StopTime="3",
        **TIGHT,
    )
    r = simulate(m)
    assert np.max(np.abs(r.y[:, 0] - 1e-3 * np.exp(-r.t))) < 1e-9
    for ext in (".slx", ".mdl"):
        save_as(m, str(tmp_path / ("rcsub" + ext)))
        again = load_model(str(tmp_path / ("rcsub" + ext)))
        assert again.notes == [] and len(again.root.lines) == len(m.root.lines)
        assert np.allclose(simulate(again).y, r.y)


def test_physical_models_from_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Session()
    res = s.run(
        """
        new_system('spring');
        add_block('blocks/Mass','spring/M','m','1','v0','1');
        add_block('blocks/TranslationalSpring','spring/K','k','4');
        add_block('blocks/MechanicalTranslationalReference','spring/Ref');
        add_block('blocks/IdealTranslationalMotionSensor','spring/S');
        add_block('blocks/Outport','spring/x');
        add_line('spring','K/R','M/M'); add_line('spring','K/C','Ref/R');
        add_line('spring','S/R','M/M'); add_line('spring','S/C','Ref/R');
        add_line('spring','S/2','x/1');
        out = sim('spring','StopTime','3','RelTol','1e-9','AbsTol','1e-12');
        err = max(abs(out.yout - sin(2*out.tout)/2));
        """
    )
    assert res.error is None, res.error
    assert float(np.ravel(s.interp.get_var("err"))[0]) < 1e-8
