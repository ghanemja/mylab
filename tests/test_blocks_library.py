"""mylab Blocks: every block type does what its one-line description says."""

import math

import numpy as np
import pytest

from mylab.api import Session
from mylab.blocks import SimError, TYPES, build, catalog, find_type, simulate
from mylab.blocks.library import count_ports, signature


def apply(block_type, params=None, inputs=(), stop="1", n_out=1, **cfg):
    """Feed constants (or (type, params) sources) into one block and return its logged outputs."""
    blocks = [("B", block_type, params or {})]
    lines = []
    for k, src in enumerate(inputs):
        if isinstance(src, tuple):
            blocks.append((f"u{k + 1}", src[0], src[1]))
        else:
            blocks.append((f"u{k + 1}", "Constant", {"Value": str(src)}))
        lines.append(f"u{k + 1}/1 -> B/{k + 1}")
    for k in range(n_out):
        blocks.append((f"y{k + 1}", "Outport", {"Port": str(k + 1)}))
        lines.append(f"B/{k + 1} -> y{k + 1}/1")
    cfg.setdefault("MaxStep", "0.01")
    return simulate(build("t", blocks, lines, StopTime=stop, **cfg))


def last(block_type, params=None, inputs=(), **kw):
    return apply(block_type, params, inputs, **kw).y[-1]


CLOCK = ("Clock", {})


# -- the catalog ----------------------------------------------------------------------------------


def test_library_has_at_least_seventy_documented_blocks():
    cat = catalog()
    assert len(cat["blocks"]) >= 70
    for b in cat["blocks"]:
        assert b["doc"] and b["doc"].endswith("."), b["type"]
        assert "\n" not in b["doc"]
        assert b["category"] in cat["categories"]
        assert b["title"]
    assert len({b["type"] for b in cat["blocks"]}) == len(cat["blocks"])


def test_every_block_type_is_exercised_by_this_file():
    import re

    src = open(__file__, encoding="utf-8").read()
    used = set(re.findall(r'"([A-Z][A-Za-z0-9]+)"', src))
    missing = sorted(t for t in TYPES if t not in used)
    assert missing == []


@pytest.mark.parametrize(
    "name,expected",
    [
        ("simulink/Continuous/Integrator", "Integrator"),
        ("built-in/Gain", "Gain"),
        ("simulink/Continuous/Transfer Fcn", "TransferFcn"),
        ("simulink/Math Operations/Sum", "Sum"),
        ("simulink/Sources/Sine Wave", "SineWave"),
        ("simulink/Sinks/Out1", "Outport"),
        ("simulink/Sources/In1", "Inport"),
        ("simulink/Sinks/To Workspace", "ToWorkspace"),
        ("blocks/Unit Delay", "UnitDelay"),
        ("simulink/Discontinuities/Dead Zone", "DeadZone"),
        ("simulink/Continuous/PID Controller", "PIDController"),
        ("simulink/Signal Routing/Mux", "Mux"),
        ("simulink/Discrete/Zero-Order Hold", "ZeroOrderHold"),
        ("simulink/User-Defined Functions/MATLAB Function", "MATLABFunction"),
        ("simulink/Sinks/XY Graph", "XYGraph"),
        ("simulink/Lookup Tables/1-D Lookup Table", "LookupTable1D"),
        ("simulink/Discontinuities/Coulomb &\nViscous Friction", "CoulombViscousFriction"),
        ("simulink/Logic and Bit Operations/Relational Operator", "RelationalOperator"),
        ("Subsystem", "Subsystem"),
    ],
)
def test_familiar_library_paths_are_aliases(name, expected):
    assert find_type(name).type == expected


def test_port_rules():
    assert count_ports(["signs", "Inputs"], {"Inputs": "+-+"}) == 3
    assert count_ports(["signs", "Inputs"], {"Inputs": "|+-"}) == 2
    assert count_ports(["signs", "Inputs"], {"Inputs": "4"}) == 4
    assert count_ports(["signs", "Outputs"], {"Outputs": "[2 1 3]"}) == 3
    assert count_ports(["int", "N"], {"N": "5"}) == 5
    assert count_ports(["csv", "S"], {"S": "a, b,c"}) == 3
    assert count_ports(["if", "Op", "atan2", 2, 1], {"Op": "atan2"}) == 2
    assert count_ports(["if", "Op", ["pow", "mod"], 2, 1], {"Op": "exp"}) == 1
    assert count_ports(["sum", 1, ["int", "N"]], {"N": "3"}) == 4
    assert count_ports(["sig", "S", "in"], {"S": "function [a, b] = f(x, y, z)\na = x;"}) == 3
    assert signature("function [a, b] = f(x, y, z)") == (["x", "y", "z"], ["a", "b"])
    assert signature("% comment\nfunction y = fcn(u)\ny = u;") == (["u"], ["y"])
    assert signature("function go()") == ([], [])


# -- sources --------------------------------------------------------------------------------------


def test_constant_scalar_vector_matrix():
    assert np.allclose(last("Constant", {"Value": "7"}), 7)
    assert np.allclose(last("Constant", {"Value": "[1 2 3]"}), [1, 2, 3])
    assert np.allclose(last("Constant", {"Value": "[1 2; 3 4]"}), [1, 3, 2, 4])  # matrices are logged column by column


def test_step_ramp_clock_ground():
    r = apply("Step", {"Time": "0.4", "Before": "-1", "After": "3"})
    assert np.all(r.y[r.t < 0.4] == -1) and np.all(r.y[r.t >= 0.4] == 3)
    r = apply("Ramp", {"Slope": "2", "Start": "0.25", "InitialOutput": "1"})
    assert np.allclose(r.y[:, 0], 1 + 2 * np.maximum(r.t - 0.25, 0))
    r = apply("Clock")
    assert np.allclose(r.y[:, 0], r.t)
    assert np.all(apply("Ground").y == 0)


def test_sine_wave_and_chirp():
    r = apply("SineWave", {"Amplitude": "2", "Bias": "1", "Frequency": "3", "Phase": "0.5"}, stop="4")
    assert np.allclose(r.y[:, 0], 2 * np.sin(3 * r.t + 0.5) + 1)
    r = apply("Chirp", {"f1": "1", "T": "2", "f2": "3"}, stop="2", MaxStep="0.001")
    assert np.allclose(r.y[:, 0], np.cos(2 * np.pi * (r.t + 0.5 * r.t**2)))


def test_pulse_generator_edges_are_exact():
    r = apply(
        "PulseGenerator",
        {"Amplitude": "5", "Period": "0.4", "PulseWidth": "25", "PhaseDelay": "0.1"},
        stop="1.3",
        MaxStep="1",
    )
    y, t = r.y[:, 0], r.t
    phase = np.mod(t - 0.1, 0.4)
    expect = np.where((t >= 0.1 - 1e-12) & (phase < 0.1 - 1e-9), 5.0, 0.0)
    expect[np.isclose(phase, 0.4)] = 5.0
    assert np.array_equal(y, expect)
    for edge in (0.1, 0.2, 0.5, 0.6, 0.9, 1.0):
        assert np.any(np.abs(t - edge) < 1e-12)
    # integral of the pulse train: three full pulses of area 0.5
    r = simulate(
        build(
            "pi",
            [
                ("P", "PulseGenerator", {"Amplitude": "5", "Period": "0.4", "PulseWidth": "25", "PhaseDelay": "0.1"}),
                ("I", "Integrator"),
                ("O", "Outport"),
            ],
            ["P/1 -> I/1", "I/1 -> O/1"],
            StopTime="1.3",
        )
    )
    assert r.y[-1, 0] == pytest.approx(1.5, abs=1e-9)


def test_signal_generator_wave_forms():
    r = apply("SignalGenerator", {"WaveForm": "sine", "Amplitude": "2", "Frequency": "0.5"}, stop="3")
    assert np.allclose(r.y[:, 0], 2 * np.sin(np.pi * r.t))
    r = apply("SignalGenerator", {"WaveForm": "square", "Frequency": "1"}, stop="2.2")
    mid = (np.mod(r.t, 0.5) > 0.01) & (np.mod(r.t, 0.5) < 0.49)
    assert np.allclose(r.y[mid, 0], np.sign(np.sin(2 * np.pi * r.t[mid])))
    r = apply("SignalGenerator", {"WaveForm": "sawtooth", "Frequency": "2", "Amplitude": "3"}, stop="1.2")
    inside = np.mod(r.t, 0.5) > 1e-6
    assert np.allclose(r.y[inside, 0], 3 * (2 * np.mod(r.t[inside], 0.5) / 0.5 - 1))
    r = apply("SignalGenerator", {"WaveForm": "random", "Amplitude": "2"}, stop="2")
    assert np.all(np.abs(r.y) <= 2) and np.std(r.y) > 0.5
    r = apply("SignalGenerator", {"WaveForm": "sine", "Frequency": "2", "Units": "rad/sec"}, stop="2")
    assert np.allclose(r.y[:, 0], np.sin(2 * r.t))


def test_random_sources_are_repeatable_and_have_the_right_statistics():
    cfg = dict(stop="400", MaxStep="1")
    a = apply("RandomNumber", {"Mean": "2", "Variance": "9", "Seed": "5", "SampleTime": "0.1"}, **cfg)
    b = apply("RandomNumber", {"Mean": "2", "Variance": "9", "Seed": "5", "SampleTime": "0.1"}, **cfg)
    c = apply("RandomNumber", {"Mean": "2", "Variance": "9", "Seed": "6", "SampleTime": "0.1"}, **cfg)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y)
    assert a.t.size == 4001 and np.allclose(np.diff(a.t), 0.1)
    assert np.mean(a.y) == pytest.approx(2, abs=0.2) and np.var(a.y) == pytest.approx(9, rel=0.1)
    u = apply("UniformRandomNumber", {"Minimum": "3", "Maximum": "5", "SampleTime": "0.1"}, **cfg)
    assert u.y.min() >= 3 and u.y.max() <= 5 and np.mean(u.y) == pytest.approx(4, abs=0.05)
    w = apply("BandLimitedWhiteNoise", {"Cov": "0.2", "Ts": "0.1"}, **cfg)
    assert np.var(w.y) == pytest.approx(0.2 / 0.1, rel=0.1)


def test_from_workspace_matrix_and_struct():
    s = Session()
    s.run("simin = [0 0; 1 2; 3 0]; ts.Time = [0; 2]; ts.Data = [1 10; 3 30];")
    m = build(
        "fw",
        [
            ("A", "FromWorkspace", {"VariableName": "simin"}),
            ("B", "FromWorkspace", {"VariableName": "ts", "OutputAfterFinalValue": "Holding final value"}),
            ("C", "FromWorkspace", {"VariableName": "[0 5; 1 7]", "Interpolate": "off"}),
            ("O1", "Outport"),
            ("O2", "Outport", {"Port": "2"}),
            ("O3", "Outport", {"Port": "3"}),
        ],
        ["A/1 -> O1/1", "B/1 -> O2/1", "C/1 -> O3/1"],
        StopTime="4",
        MaxStep="0.25",
    )
    r = simulate(m, s.interp)
    assert np.allclose(r.y[:, 0], np.interp(r.t, [0, 1, 3], [0, 2, 0]) - np.maximum(r.t - 3, 0))  # extrapolates
    assert np.allclose(r.y[:, 1], np.interp(r.t, [0, 2], [1, 3]))
    assert np.allclose(r.y[:, 2], np.interp(r.t, [0, 2], [10, 30]))
    assert np.allclose(r.y[r.t < 1, 3], 5) and np.allclose(r.y[r.t >= 1, 3], 7)
    with pytest.raises(SimError, match="time in the first column"):
        apply("FromWorkspace", {"VariableName": "[1 2 3]'"})


def test_repeating_sequences():
    r = apply("RepeatingSequence", {"rep_seq_t": "[0 1 1.5]", "rep_seq_y": "[0 2 0]"}, stop="4", MaxStep="0.05")
    tau = np.mod(r.t, 1.5)
    ok = (tau > 1e-6) & (tau < 1.5 - 1e-6)
    assert np.allclose(r.y[ok, 0], np.interp(tau[ok], [0, 1, 1.5], [0, 2, 0]))
    r = apply("RepeatingSequenceStair", {"OutValues": "[3 1 4]", "tsamp": "0.5"}, stop="2.5", MaxStep="1")
    assert np.array_equal(r.y[:, 0], [3, 1, 4, 3, 1, 4]) and np.allclose(r.t, np.arange(6) * 0.5)


def test_root_inport_takes_external_input():
    m = build(
        "ext",
        [("In", "Inport"), ("K", "Gain", {"Gain": "2"}), ("O", "Outport")],
        ["In/1 -> K/1", "K/1 -> O/1"],
        StopTime="2",
        MaxStep="0.1",
    )
    assert np.all(simulate(m).y == 0)
    r = simulate(m, external_input=np.array([[0, 0], [2, 4.0]]))
    assert np.allclose(r.y[:, 0], 4 * r.t)


# -- sinks ----------------------------------------------------------------------------------------


def test_scope_to_workspace_display_xy_graph_terminator():
    m = build(
        "sinks",
        [
            ("S", "SineWave"),
            ("C", "Clock"),
            ("Scope", "Scope", {"NumInputPorts": "2"}),
            ("W", "ToWorkspace", {"VariableName": "wave", "Decimation": "2"}),
            ("D", "Display"),
            ("XY", "XYGraph"),
            ("T", "Terminator"),
        ],
        [
            "S/1 -> Scope/1 : wave",
            "C/1 -> Scope/2",
            "S/1 -> W/1",
            "S/1 -> D/1",
            "C/1 -> XY/1",
            "S/1 -> XY/2",
            "C/1 -> T/1",
        ],
        StopTime="2",
        MaxStep="0.1",
    )
    r = simulate(m)
    sc = r.scopes["Scope"]
    assert len(sc["signals"]) == 2 and sc["signals"][0]["name"] == "wave"
    assert np.allclose(sc["signals"][0]["data"][:, 0], np.sin(sc["t"])) and np.allclose(
        sc["signals"][1]["data"][:, 0], sc["t"]
    )
    w = r.workspace["wave"]
    assert w["t"].size == (r.t.size + 1) // 2 and np.allclose(w["data"][:, 0], np.sin(w["t"]))
    assert r.displays["D"][0] == pytest.approx(math.sin(2))
    assert np.allclose(r.xy["XY"]["y"][:, 0], np.sin(r.xy["XY"]["x"][:, 0]))
    assert r.y.shape == (r.t.size, 0)


def test_named_logged_line():
    m = build(
        "logs",
        [("S", "SineWave"), ("K", "Gain", {"Gain": "2"}), ("T", "Terminator")],
        ["S/1 -> K/1 : raw", "K/1 -> T/1"],
        StopTime="1",
    )
    m.root.lines[0].log = True
    r = simulate(m)
    assert np.allclose(r.logs["raw"]["data"][:, 0], np.sin(r.logs["raw"]["t"]))


# -- math -----------------------------------------------------------------------------------------


def test_gain_modes():
    assert last("Gain", {"Gain": "3"}, [4]) == 12
    assert np.allclose(last("Gain", {"Gain": "[1 2 3]"}, [2]), [2, 4, 6])
    assert np.allclose(last("Gain", {"Gain": "[1 2; 3 4]", "Multiplication": "Matrix(K*u)"}, ["[1 1]"]), [3, 7])
    assert np.allclose(last("Gain", {"Gain": "[1 2; 3 4]", "Multiplication": "Matrix(u*K)"}, ["[1 1]"]), [4, 6])
    assert np.allclose(last("Gain", {"Gain": "[1 2 3]", "Multiplication": "Matrix(K*u)"}, ["[1 1 1]"]), 6)


def test_sum_product_divide():
    assert last("Sum", {"Inputs": "+-+"}, [5, 2, 10]) == 13
    assert last("Sum", {"Inputs": "|+-"}, [5, 2]) == 3
    assert last("Sum", {"Inputs": "3"}, [1, 2, 3]) == 6
    assert last("Sum", {"Inputs": "+"}, ["[1 2 3]"]) == 6  # one input: sum of its elements
    assert np.allclose(last("Sum", {"Inputs": "++"}, ["[1 2]", 10]), [11, 12])
    assert last("Product", {"Inputs": "3"}, [2, 3, 4]) == 24
    assert last("Product", {"Inputs": "*/"}, [9, 3]) == 3
    assert last("Divide", {}, [9, 2]) == 4.5
    assert np.allclose(
        last("Product", {"Inputs": "2", "Multiplication": "Matrix(*)"}, ["[1 2; 3 4]", "[1; 1]"]), [3, 7]
    )
    with pytest.raises(SimError, match=r"only contain \+ and -"):
        last("Sum", {"Inputs": "+*"}, [1, 2])


def test_elementwise_math_blocks():
    v = "[-2.5 0 1.5]"
    assert np.allclose(last("Abs", {}, [v]), [2.5, 0, 1.5])
    assert np.allclose(last("Sign", {}, [v]), [-1, 0, 1])
    assert np.allclose(last("UnaryMinus", {}, [v]), [2.5, 0, -1.5])
    assert np.allclose(last("Bias", {"Bias": "10"}, [v]), [7.5, 10, 11.5])
    assert last("SumOfElements", {}, [v]) == -1
    assert last("ProductOfElements", {}, ["[2 3 4]"]) == 24
    assert np.allclose(last("Polynomial", {"Coefs": "[1 0 -1]"}, [v]), [5.25, -1, 1.25])
    assert last("Width", {}, [v]) == 3
    assert np.allclose(last("DataTypeConversion", {}, [v]), [-2.5, 0, 1.5])
    assert last("DotProduct", {}, ["[1 2 3]", "[4 5 6]"]) == 32


def test_math_function_and_trigonometry():
    assert last("MathFunction", {"Operator": "exp"}, [1]) == pytest.approx(math.e)
    assert last("MathFunction", {"Operator": "log10"}, [1000]) == pytest.approx(3)
    assert last("MathFunction", {"Operator": "square"}, [-3]) == 9
    assert last("MathFunction", {"Operator": "reciprocal"}, [4]) == 0.25
    assert last("MathFunction", {"Operator": "pow"}, [2, 10]) == 1024
    assert last("MathFunction", {"Operator": "mod"}, [-1, 3]) == 2
    assert last("MathFunction", {"Operator": "rem"}, [-1, 3]) == -1
    assert last("MathFunction", {"Operator": "hypot"}, [3, 4]) == 5
    assert np.allclose(last("MathFunction", {"Operator": "transpose"}, ["[1 2; 3 4]"]), [1, 2, 3, 4])
    assert last("TrigonometricFunction", {"Operator": "cos"}, [0]) == 1
    assert last("TrigonometricFunction", {"Operator": "atan2"}, [1, -1]) == pytest.approx(3 * math.pi / 4)
    assert last("TrigonometricFunction", {"Operator": "asinh"}, [1]) == pytest.approx(math.asinh(1))
    assert np.allclose(
        last("TrigonometricFunction", {"Operator": "sincos"}, [0.3], n_out=2), [math.sin(0.3), math.cos(0.3)]
    )
    with pytest.raises(SimError, match="unknown function"):
        last("MathFunction", {"Operator": "gamma"}, [1])


def test_sqrt_minmax_rounding():
    assert last("Sqrt", {}, [16]) == 4
    assert last("Sqrt", {"Operator": "signedSqrt"}, [-16]) == -4
    assert last("Sqrt", {"Operator": "rSqrt"}, [16]) == 0.25
    assert last("MinMax", {"Function": "max", "Inputs": "3"}, [1, 7, 3]) == 7
    assert last("MinMax", {"Function": "min", "Inputs": "1"}, ["[4 2 9]"]) == 2
    assert np.allclose(last("MinMax", {"Function": "min", "Inputs": "2"}, ["[4 2 9]", 3]), [3, 2, 3])
    v = "[-2.5 -0.4 0.5 2.5]"
    assert np.allclose(last("Rounding", {"Operator": "floor"}, [v]), [-3, -1, 0, 2])
    assert np.allclose(last("Rounding", {"Operator": "ceil"}, [v]), [-2, 0, 1, 3])
    assert np.allclose(last("Rounding", {"Operator": "round"}, [v]), [-3, 0, 1, 3])
    assert np.allclose(last("Rounding", {"Operator": "fix"}, [v]), [-2, 0, 0, 2])


def test_reshape_and_concatenate():
    assert np.allclose(
        last("Reshape", {"OutputDimensionality": "Customize", "OutputDimensions": "[2 2]"}, ["[1 2 3 4]"]), [1, 2, 3, 4]
    )
    assert np.allclose(last("Reshape", {}, ["[1 2; 3 4]"]), [1, 3, 2, 4])
    assert np.allclose(last("Concatenate", {"NumInputs": "3"}, [1, "[2 3]", 4]), [1, 2, 3, 4])
    r = last("Concatenate", {"Mode": "Multidimensional array", "ConcatenateDimension": "2"}, ["[1; 2]", "[3; 4]"])
    assert np.allclose(r, [1, 2, 3, 4])
    with pytest.raises(SimError, match="cannot be reshaped"):
        last("Reshape", {"OutputDimensionality": "Customize", "OutputDimensions": "[2 3]"}, ["[1 2 3 4]"])


# -- logic ----------------------------------------------------------------------------------------


def test_relational_and_compare_blocks():
    for op, expect in (
        ("==", [0, 1, 0]),
        ("~=", [1, 0, 1]),
        ("<", [1, 0, 0]),
        ("<=", [1, 1, 0]),
        (">=", [0, 1, 1]),
        (">", [0, 0, 1]),
    ):
        assert np.array_equal(last("RelationalOperator", {"Operator": op}, ["[1 2 3]", 2]), expect)
        assert np.array_equal(last("CompareToConstant", {"relop": op, "const": "2"}, ["[1 2 3]"]), expect)
        assert np.array_equal(last("CompareToZero", {"relop": op}, ["[-1 0 1]"]), expect)


def test_logical_and_bitwise_operators():
    a, b = "[0 0 1 1]", "[0 1 0 1]"
    table = {
        "AND": [0, 0, 0, 1],
        "OR": [0, 1, 1, 1],
        "NAND": [1, 1, 1, 0],
        "NOR": [1, 0, 0, 0],
        "XOR": [0, 1, 1, 0],
        "NXOR": [1, 0, 0, 1],
    }
    for op, expect in table.items():
        assert np.array_equal(last("LogicalOperator", {"Operator": op}, [a, b]), expect)
    assert np.array_equal(last("LogicalOperator", {"Operator": "NOT"}, [a]), [1, 1, 0, 0])
    assert last("LogicalOperator", {"Operator": "AND", "Inputs": "3"}, [1, 2, 0]) == 0
    assert last("BitwiseOperator", {"logicop": "AND", "BitMask": "12"}, [10]) == 8
    assert last("BitwiseOperator", {"logicop": "XOR", "UseBitMask": "off", "NumInputPorts": "2"}, [10, 12]) == 6


def test_switches():
    assert last("Switch", {"Threshold": "2"}, [10, 2, 20]) == 10
    assert last("Switch", {"Threshold": "2", "Criteria": "u2 > Threshold"}, [10, 2, 20]) == 20
    assert last("Switch", {"Criteria": "u2 ~= 0"}, [10, -1, 20]) == 10
    assert last("MultiportSwitch", {"Inputs": "3"}, [2, 10, 20, 30]) == 20
    assert last("MultiportSwitch", {"Inputs": "3", "DataPortOrder": "Zero-based contiguous"}, [2, 10, 20, 30]) == 30
    with pytest.raises(SimError, match="control input"):
        last("MultiportSwitch", {"Inputs": "2"}, [5, 10, 20])
    assert last("ManualSwitch", {"sw": "1"}, [1, 2]) == 1
    assert last("ManualSwitch", {"sw": "0"}, [1, 2]) == 2


def test_hit_crossing_fires_once_per_crossing():
    r = apply(
        "HitCrossing", {"HitCrossingOffset": "0.5"}, [("SineWave", {"Frequency": "2*pi"})], stop="2", MaxStep="0.05"
    )
    assert int(np.sum(r.y)) == 4
    hits = r.t[r.y[:, 0] == 1]
    assert np.allclose(hits, [1 / 12, 5 / 12, 13 / 12, 17 / 12], atol=1e-8)


# -- nonlinear ------------------------------------------------------------------------------------


def static_map(block_type, params):
    r = apply(block_type, params, [("Ramp", {"Slope": "1", "InitialOutput": "-2"})], stop="4", MaxStep="0.05")
    return r.t - 2, r.y[:, 0], r


def test_saturation_static_map_and_corner_times():
    u, y, r = static_map("Saturation", {"UpperLimit": "1", "LowerLimit": "-0.5"})
    assert np.allclose(y, np.clip(u, -0.5, 1), atol=1e-9)
    assert np.any(np.abs(r.t - 1.5) < 1e-8) and np.any(np.abs(r.t - 3.0) < 1e-8)  # the corners are located
    with pytest.raises(SimError, match="lower limit is above"):
        last("Saturation", {"UpperLimit": "0", "LowerLimit": "1"}, [0])


def test_dead_zone_static_map():
    u, y, _ = static_map("DeadZone", {"LowerValue": "-0.5", "UpperValue": "1"})
    assert np.allclose(y, np.where(u > 1, u - 1, np.where(u < -0.5, u + 0.5, 0)), atol=1e-9)


def test_quantizer_friction_and_dynamic_saturation():
    u, y, _ = static_map("Quantizer", {"QuantizationInterval": "0.5"})
    assert np.allclose(y, 0.5 * np.sign(u) * np.floor(np.abs(u) / 0.5 + 0.5))
    u, y, _ = static_map("CoulombViscousFriction", {"offset": "2", "gain": "3"})
    away = np.abs(u) > 1e-6
    assert np.allclose(y[away], np.sign(u[away]) * (3 * np.abs(u[away]) + 2))
    assert last("SaturationDynamic", {}, [1, 5, -1]) == 1
    assert last("SaturationDynamic", {}, [1, -5, -1]) == -1
    assert last("SaturationDynamic", {}, [1, 0.3, -1]) == 0.3


def test_relay_hysteresis_map():
    r = apply(
        "Relay",
        {"OnSwitchValue": "0.5", "OffSwitchValue": "-0.5", "OnOutputValue": "2", "OffOutputValue": "-2"},
        [("SineWave", {})],
        stop="7",
        MaxStep="0.05",
    )
    u, y = np.sin(r.t), r.y[:, 0]
    assert set(np.unique(y)) == {-2.0, 2.0}
    assert np.all(y[u > 0.5 + 1e-6] == 2) and np.all(y[u < -0.5 - 1e-6] == -2)
    falling = (r.t > math.pi / 2) & (u > -0.5 + 1e-6) & (r.t < math.pi + 0.5)
    assert np.all(y[falling] == 2)  # stays on until the input falls below the switch-off point


def test_rate_limiter_and_backlash():
    r = apply(
        "RateLimiter",
        {"RisingSlewLimit": "2", "FallingSlewLimit": "-1"},
        [("Step", {"Time": "0.2", "After": "1"})],
        stop="2",
        MaxStep="0.01",
    )
    expect = np.clip(2 * (r.t - 0.2), 0, 1)
    assert np.max(np.abs(r.y[:, 0] - expect)) < 0.021
    r = apply("Backlash", {"BacklashWidth": "1"}, [("SineWave", {"Amplitude": "2"})], stop="8", MaxStep="0.01")
    y = r.y[:, 0]
    assert np.max(y) == pytest.approx(1.5, abs=1e-3) and np.min(y) == pytest.approx(-1.5, abs=1e-3)
    assert np.all(np.abs(y - 2 * np.sin(r.t)) <= 0.5 + 1e-9)


def test_lookup_tables():
    p = {"BreakpointsForDimension1": "[0 1 3]", "Table": "[0 10 30]"}
    assert last("LookupTable1D", p, [2]) == 20
    assert last("LookupTable1D", p, [4]) == 40  # linear extrapolation
    assert last("LookupTable1D", {**p, "ExtrapMethod": "Clip"}, [4]) == 30
    assert np.allclose(last("LookupTable1D", p, ["[-1 0.5]"]), [-10, 5])
    p2 = {"BreakpointsForDimension1": "[0 1]", "BreakpointsForDimension2": "[0 1 2]", "Table": "[0 1 2; 10 11 12]"}
    assert last("LookupTable2D", p2, [0.5, 1.5]) == pytest.approx(6.5)
    assert last("LookupTable2D", {**p2, "ExtrapMethod": "Clip"}, [5, 5]) == 12
    with pytest.raises(SimError, match="breakpoints"):
        last("LookupTable1D", {"BreakpointsForDimension1": "[0 1]", "Table": "[1 2 3]"}, [0])
    with pytest.raises(SimError, match="table is 2x3"):
        last("LookupTable2D", {**p2, "BreakpointsForDimension1": "[0 1 2]"}, [0, 0])


# -- continuous blocks not covered by the validation file -----------------------------------------


def test_integrator_initial_value_and_state_port():
    r = apply("Integrator", {"InitialCondition": "3", "ShowStatePort": "on"}, [2], n_out=2, stop="2")
    assert np.allclose(r.y[:, 0], 3 + 2 * r.t) and np.allclose(r.y[:, 1], r.y[:, 0])
    r = apply("Integrator", {"InitialConditionSource": "external"}, [1, 5], stop="1")
    assert np.allclose(r.y[:, 0], 5 + r.t)


def test_derivative_transport_delay_and_lti_blocks_run():
    assert last("Derivative", {}, [CLOCK], stop="1") == pytest.approx(1)
    assert last("TransportDelay", {"DelayTime": "0.25"}, [CLOCK], stop="1") == pytest.approx(0.75)
    assert last("TransferFcn", {"Numerator": "[2]", "Denominator": "[1 1]"}, [1], stop="20") == pytest.approx(
        2, abs=1e-3
    )
    assert last("StateSpace", {"A": "-1", "B": "1", "C": "3", "D": "0"}, [1], stop="20") == pytest.approx(3, abs=1e-3)
    assert last("ZeroPole", {"Zeros": "[]", "Poles": "[-1 -2]", "Gain": "4"}, [1], stop="25") == pytest.approx(
        2, abs=1e-3
    )
    assert last("PIDController", {"Controller": "P", "P": "3"}, [2]) == 6
    with pytest.raises(SimError, match="scalar input"):
        last("PIDController", {}, ["[1 2]"])
    with pytest.raises(SimError, match="do not fit together"):
        last("StateSpace", {"A": "[1 2; 3 4]", "B": "1", "C": "1", "D": "0"}, [1])


# -- discrete blocks ------------------------------------------------------------------------------


def test_unit_delay_memory_delay():
    r = apply("UnitDelay", {"InitialCondition": "-1", "SampleTime": "0.25"}, [CLOCK], stop="1", MaxStep="1")
    assert np.allclose(r.t, [0, 0.25, 0.5, 0.75, 1.0]) and np.allclose(r.y[:, 0], [-1, 0, 0.25, 0.5, 0.75])
    r = apply(
        "Delay", {"DelayLength": "3", "InitialCondition": "9", "SampleTime": "0.25"}, [CLOCK], stop="1", MaxStep="1"
    )
    assert np.allclose(r.y[:, 0], [9, 9, 9, 0, 0.25])
    r = apply("Memory", {"InitialCondition": "5"}, [CLOCK], stop="1", Solver="ode1", FixedStep="0.25")
    assert np.allclose(r.y[:, 0], [5, 0, 0.25, 0.5, 0.75])


def test_zero_order_hold_and_rate_transition():
    for kind, p in (("ZeroOrderHold", {"SampleTime": "0.3"}), ("RateTransition", {"OutPortSampleTime": "0.3"})):
        m = build(
            "zoh",
            [("C", "Clock"), ("H", kind, p), ("Sc", "Scope"), ("O", "Outport")],
            ["C/1 -> H/1", "H/1 -> O/1", "C/1 -> Sc/1"],
            StopTime="1",
            MaxStep="0.05",
        )
        r = simulate(m)
        o = r.outports[0]
        assert np.allclose(o["t"], [0, 0.3, 0.6, 0.9]) and np.allclose(o["data"][:, 0], [0, 0.3, 0.6, 0.9])
        assert np.allclose(r.y[:, 0], np.floor(r.t / 0.3 + 1e-9) * 0.3)  # held between samples in yout


def test_discrete_time_integrator_methods():
    def run(method):
        return apply(
            "DiscreteTimeIntegrator",
            {"IntegratorMethod": method, "SampleTime": "0.5", "gainval": "2", "InitialCondition": "1"},
            [CLOCK],
            stop="2",
            MaxStep="1",
        ).y[:, 0]

    u = np.arange(5) * 0.5
    fwd = 1 + np.concatenate([[0], np.cumsum(2 * 0.5 * u[:-1])])
    assert np.allclose(run("Integration: Forward Euler"), fwd)
    assert np.allclose(run("Integration: Backward Euler"), 1 + np.cumsum(2 * 0.5 * u))
    trap = 1 + np.concatenate([[0], np.cumsum(2 * 0.5 * (u[:-1] + u[1:]) / 2)])
    assert np.allclose(run("Integration: Trapezoidal"), trap)
    r = apply(
        "DiscreteTimeIntegrator",
        {"SampleTime": "0.5", "LimitOutput": "on", "UpperSaturationLimit": "1.2"},
        [1],
        stop="3",
        MaxStep="1",
    )
    assert np.max(r.y) == pytest.approx(1.2)


def test_discrete_transfer_function_filter_fir_state_space():
    step = [1]
    # y[k+1] = 0.5 y[k] + u[k]
    expect = [0, 1, 1.5, 1.75, 1.875]
    r = apply(
        "DiscreteTransferFcn",
        {"Numerator": "[1]", "Denominator": "[1 -0.5]", "SampleTime": "1"},
        step,
        stop="4",
        MaxStep="4",
    )
    assert np.allclose(r.y[:, 0], expect)
    r = apply(
        "DiscreteStateSpace", {"A": "0.5", "B": "1", "C": "1", "D": "0", "SampleTime": "1"}, step, stop="4", MaxStep="4"
    )
    assert np.allclose(r.y[:, 0], expect)
    # in powers of 1/z: y[k] = 0.5 y[k-1] + u[k]
    r = apply(
        "DiscreteFilter",
        {"Numerator": "[1]", "Denominator": "[1 -0.5]", "SampleTime": "1"},
        step,
        stop="4",
        MaxStep="4",
    )
    assert np.allclose(r.y[:, 0], [1, 1.5, 1.75, 1.875, 1.9375])
    r = apply("DiscreteFir", {"Coefficients": "[0.5 0.3 0.2]", "SampleTime": "1"}, [CLOCK], stop="4", MaxStep="4")
    assert np.allclose(r.y[:, 0], [0, 0.5, 1.3, 2.3, 3.3])
    r = apply("DiscreteDerivative", {"SampleTime": "0.5"}, [("Ramp", {"Slope": "3"})], stop="2", MaxStep="2")
    assert np.allclose(r.y[1:, 0], 3)


# -- routing --------------------------------------------------------------------------------------


def test_mux_demux_selector_assignment():
    assert np.allclose(last("Mux", {"Inputs": "3"}, [1, "[2 3]", 4]), [1, 2, 3, 4])
    assert np.allclose(last("Demux", {"Outputs": "3"}, ["[7 8 9]"], n_out=3), [7, 8, 9])
    assert np.allclose(last("Demux", {"Outputs": "[2 1]"}, ["[7 8 9]"], n_out=2), [7, 8, 9])
    assert np.allclose(last("Demux", {"Outputs": "2"}, ["[1 2 3 4]"], n_out=2), [1, 2, 3, 4])
    with pytest.raises(SimError, match="cannot be split evenly"):
        last("Demux", {"Outputs": "2"}, ["[7 8 9]"], n_out=2)
    assert np.allclose(last("Selector", {"Indices": "[3 1]"}, ["[7 8 9]"]), [9, 7])
    assert np.allclose(last("Selector", {"IndexMode": "Zero-based", "Indices": "0"}, ["[7 8 9]"]), 7)
    assert last("Selector", {"IndexOptions": "Index vector (port)"}, ["[7 8 9]", 2]) == 8
    p2 = {"NumberOfDimensions": "2", "IndexOptions": "Index vector (dialog),Select all", "Indices": "[2],[1]"}
    assert np.allclose(last("Selector", p2, ["[1 2 3; 4 5 6]"]), [4, 5, 6])
    with pytest.raises(SimError, match="outside the input"):
        last("Selector", {"Indices": "5"}, ["[7 8 9]"])
    assert np.allclose(last("Assignment", {"Indices": "2"}, ["[7 8 9]", 0]), [7, 0, 9])
    assert np.allclose(last("Assignment", {"Indices": "[1 3]"}, ["[7 8 9]", "[1 2]"]), [1, 8, 2])


def test_bus_creator_and_selector_use_signal_names():
    m = build(
        "bus",
        [
            ("a", "Constant", {"Value": "1"}),
            ("b", "Constant", {"Value": "[2 3]"}),
            ("c", "Constant", {"Value": "4"}),
            ("BC", "BusCreator", {"Inputs": "3"}),
            ("BS", "BusSelector", {"OutputSignals": "speed,signal3"}),
            ("O1", "Outport"),
            ("O2", "Outport", {"Port": "2"}),
        ],
        ["a/1 -> BC/1 : pos", "b/1 -> BC/2 : speed", "c/1 -> BC/3", "BC/1 -> BS/1", "BS/1 -> O1/1", "BS/2 -> O2/1"],
        StopTime="1",
    )
    assert np.allclose(simulate(m).y[-1], [2, 3, 4])
    m.root.block("BS").params["OutputSignals"] = "nope,pos"
    with pytest.raises(SimError, match="no signal 'nope'.*pos, speed, signal3"):
        simulate(m)


def test_goto_from_in_one_system():
    m = build(
        "gf",
        [("C", "Clock"), ("G", "Goto", {"GotoTag": "t"}), ("F", "From", {"GotoTag": "t"}), ("O", "Outport")],
        ["C/1 -> G/1", "F/1 -> O/1"],
        StopTime="1",
    )
    r = simulate(m)
    assert np.allclose(r.y[:, 0], r.t)


def enabled_model(kind, port_block, params=None):
    inner = (
        [
            ("In", "Inport"),
            ("ctl", port_block, params or {}),
            ("K", "Gain", {"Gain": "2"}),
            ("Int", "Integrator"),
            ("Out", "Outport"),
            ("Out2", "Outport", {"Port": "2"}),
        ],
        ["In/1 -> K/1", "K/1 -> Out/1", "In/1 -> Int/1", "Int/1 -> Out2/1"],
    )
    return build(
        "en",
        [
            ("C", "Clock"),
            ("P", "PulseGenerator", {"Period": "1", "PulseWidth": "50", "Amplitude": "2"}),
            ("bias", "Bias", {"Bias": "-1"}),
            ("S", "Subsystem", {}, inner),
            ("O1", "Outport"),
            ("O2", "Outport", {"Port": "2"}),
        ],
        ["C/1 -> S/1", "P/1 -> bias/1", f"bias/1 -> S/{kind}", "S/1 -> O1/1", "S/2 -> O2/1"],
        StopTime="2",
        MaxStep="0.01",
    )


def test_enabled_subsystem_holds_outputs_and_states():
    r = simulate(enabled_model("enable", "EnablePort"))
    t, y, x = r.t, r.y[:, 0], r.y[:, 1]
    on = np.mod(t, 1) < 0.5 - 1e-9
    assert np.allclose(y[on], 2 * t[on])
    off = (t > 0.5 + 1e-9) & (t < 1 - 1e-9)
    assert np.allclose(y[off], 1.0)  # held at the last value: 2 * 0.5
    assert np.allclose(x[off], 0.125)  # integral of t over [0, 0.5], frozen while disabled
    assert x[-1] == pytest.approx(0.125 + (1.5**2 - 1) / 2, abs=1e-6)


def test_triggered_subsystem_runs_once_per_edge():
    r = simulate(enabled_model("trigger", "TriggerPort", {"TriggerType": "either"}))
    t, y = r.t, r.y[:, 0]
    # edges of the pulse at 0.5, 1.0, 1.5, 2.0: the output is 2 * (time of the last edge)
    assert np.allclose(y[(t > 0.5) & (t < 1.0 - 1e-9)], 1.0)
    assert np.allclose(y[(t > 1.0) & (t < 1.5 - 1e-9)], 2.0)
    assert np.allclose(y[(t > 1.5) & (t < 2.0 - 1e-9)], 3.0)
    rising = simulate(enabled_model("trigger", "TriggerPort", {"TriggerType": "rising"}))
    assert np.allclose(rising.y[(rising.t > 1.0) & (rising.t < 2.0 - 1e-9), 0], 2.0)


def test_merge_takes_the_branch_that_ran():
    def branch(gain):
        return (
            [("In", "Inport"), ("en", "EnablePort"), ("K", "Gain", {"Gain": gain}), ("Out", "Outport")],
            ["In/1 -> K/1", "K/1 -> Out/1"],
        )

    m = build(
        "mrg",
        [
            ("C", "Clock"),
            ("S", "SineWave", {"Frequency": "2*pi"}),
            ("neg", "Gain", {"Gain": "-1"}),
            ("A", "Subsystem", {}, branch("1")),
            ("B", "Subsystem", {}, branch("-1")),
            ("M", "Merge"),
            ("O", "Outport"),
        ],
        [
            "C/1 -> A/1",
            "C/1 -> B/1",
            "S/1 -> A/enable",
            "S/1 -> neg/1",
            "neg/1 -> B/enable",
            "A/1 -> M/1",
            "B/1 -> M/2",
            "M/1 -> O/1",
        ],
        StopTime="2",
        MaxStep="0.01",
    )
    r = simulate(m)
    t, y = r.t, r.y[:, 0]
    pos = np.sin(2 * np.pi * t) > 1e-3
    neg = np.sin(2 * np.pi * t) < -1e-3
    assert np.allclose(y[pos], t[pos]) and np.allclose(y[neg], -t[neg])


# -- user-defined ---------------------------------------------------------------------------------


def test_fcn_block():
    assert last("Fcn", {"Expr": "u(1)^2 + 3*u(2)"}, ["[2 5]"]) == 19
    assert last("Fcn", {"Expr": "u[1]*2"}, [4]) == 8
    r = apply("Fcn", {"Expr": "sin(u) + t"}, [1], stop="1")
    assert np.allclose(r.y[:, 0], math.sin(1) + r.t)
    s = Session()
    s.run("gain = 7;")
    m = build(
        "f",
        [("C", "Constant", {"Value": "2"}), ("F", "Fcn", {"Expr": "gain*u"}), ("O", "Outport")],
        ["C/1 -> F/1", "F/1 -> O/1"],
        StopTime="1",
    )
    assert simulate(m, s.interp).y[-1, 0] == 14
    with pytest.raises(SimError, match="not valid"):
        last("Fcn", {"Expr": "u +* 2"}, [1])


def test_matlab_function_block_runs_interpreter_code():
    code = "function [s, p] = both(a, b)\n% sum and product\ns = a + b;\nif a > b\n    p = a * b;\nelse\n    p = -a * b;\nend\n"  # noqa: E501
    assert np.allclose(last("MATLABFunction", {"Script": code}, [3, 4], n_out=2), [7, -12])
    assert np.allclose(last("MATLABFunction", {"Script": "function y = f(u)\ny = cumsum(u);"}, ["[1 2 3]"]), [1, 3, 6])
    counter = "function y = count(u)\npersistent n\nif isempty(n), n = 0; end\nn = n + u;\ny = n;\n"
    r = apply(
        "MATLABFunction",
        {"Script": counter},
        [("PulseGenerator", {"Period": "1"})],
        stop="3",
        Solver="ode1",
        FixedStep="0.5",
    )
    assert r.y[-1, 0] >= 3
    with pytest.raises(SimError) as info:
        last("MATLABFunction", {"Script": "function y = f(u)\ny = undefined_thing(u);"}, [1])
    assert info.value.path == "B" and "undefined_thing" in info.value.message
    with pytest.raises(SimError, match="function line"):
        last("MATLABFunction", {"Script": "y = 1;"}, [])


def test_matlab_function_block_in_a_feedback_loop():
    # x' = f(x) with f written as code: logistic growth, exact solution known
    code = "function dx = rhs(x)\nr = 2; K = 5;\ndx = r * x * (1 - x / K);\n"
    m = build(
        "logi",
        [
            ("F", "MATLABFunction", {"Script": code}),
            ("Int", "Integrator", {"InitialCondition": "0.5"}),
            ("O", "Outport"),
        ],
        ["Int/1 -> F/1", "F/1 -> Int/1", "Int/1 -> O/1"],
        StopTime="4",
        RelTol="1e-8",
        AbsTol="1e-10",
    )
    r = simulate(m)
    exact = 5 / (1 + (5 / 0.5 - 1) * np.exp(-2 * r.t))
    assert np.max(np.abs(r.y[:, 0] - exact)) < 1e-6


def test_interpreted_function_block():
    assert last("InterpretedFunction", {"MATLABFcn": "@(u) u.^2 + 1"}, ["[1 2 3]"]).tolist() == [2, 5, 10]
    assert last("InterpretedFunction", {"MATLABFcn": "sqrt"}, [16]) == 4
    assert last("InterpretedFunction", {"MATLABFcn": "u(1)*u(2)"}, ["[3 4]"]) == 12
    s = Session()
    s.run("h = @(u) 10*u;")
    m = build(
        "h",
        [("C", "Constant", {"Value": "2"}), ("F", "InterpretedFunction", {"MATLABFcn": "h"}), ("O", "Outport")],
        ["C/1 -> F/1", "F/1 -> O/1"],
        StopTime="1",
    )
    assert simulate(m, s.interp).y[-1, 0] == 20


def test_structure_blocks_are_in_the_catalog():
    for name in (
        "Subsystem",
        "Inport",
        "Outport",
        "EnablePort",
        "TriggerPort",
        "Goto",
        "From",
        "Scope",
        "ToWorkspace",
        "Display",
        "XYGraph",
        "Terminator",
        "StopSimulation",
        "Ground",
        "Clock",
        "Chirp",
        "Merge",
        "HitCrossing",
        "Derivative",
        "TransportDelay",
        "TransferFcn",
        "ZeroPole",
        "StateSpace",
        "Memory",
        "UnitDelay",
        "Delay",
    ):
        assert find_type(name) is not None


def test_stop_simulation_block():
    r = apply("StopSimulation", {}, [("Step", {"Time": "0.4"})], n_out=0, stop="5")
    assert r.t[-1] == pytest.approx(0.4) and r.stopped_by == "B"


# -- blocks added for imported models ---------------------------------------------------------------


def test_if_block_runs_the_action_subsystem_of_the_first_true_condition():
    def action(gain):
        return (
            [("In", "Inport"), ("Act", "ActionPort"), ("K", "Gain", {"Gain": gain}), ("Out", "Outport")],
            ["In/1 -> K/1", "K/1 -> Out/1"],
        )

    m = build(
        "ifelse",
        [
            ("Ramp", "Ramp", {"Slope": "1", "InitialOutput": "-1"}),
            ("If", "If", {"IfExpression": "u1 < 0", "ElseIfExpressions": "u1 < 0.5", "ShowElse": "on"}),
            ("Neg", "Subsystem", {}, action("-1")),
            ("Mid", "Subsystem", {}, action("10")),
            ("Pos", "Subsystem", {}, action("100")),
            ("Merge", "Merge", {"Inputs": "3"}),
            ("y", "Outport"),
        ],
        [
            "Ramp/1 -> If/1",
            "Ramp/1 -> Neg/1",
            "Ramp/1 -> Mid/1",
            "Ramp/1 -> Pos/1",
            "If/1 -> Neg/enable",
            "If/2 -> Mid/enable",
            "If/3 -> Pos/enable",
            "Neg/1 -> Merge/1",
            "Mid/1 -> Merge/2",
            "Pos/1 -> Merge/3",
            "Merge/1 -> y/1",
        ],
        StopTime="2",
        Solver="ode4",
        FixedStep="0.25",
    )
    r = simulate(m)
    u = r.t - 1
    expected = np.where(u < 0, -u, np.where(u < 0.5, 10 * u, 100 * u))
    assert np.allclose(r.y[:, 0], expected)
    assert count_ports(find_type("If").nout, {"ElseIfExpressions": "u1 > 2, u1 > 3", "ShowElse": "off"}) == 3


def test_lookup_table_with_three_dimensions():
    p = {
        "NumberOfTableDimensions": "3",
        "Table": "reshape(0:26, 3, 3, 3)",  # value = i + 3 j + 9 k for 0-based indices
        "Breakpoints": "[0 1 2], [0 10 20], [0 100 200]",
    }
    assert last("LookupTableND", p, [1, 10, 100])[0] == pytest.approx(13)
    assert last("LookupTableND", p, [0.5, 5, 150])[0] == pytest.approx(0.5 + 1.5 + 13.5)
    assert last("LookupTableND", p, [3, 0, 0])[0] == pytest.approx(3)  # a straight line continues outside
    assert last("LookupTableND", {**p, "ExtrapMethod": "Clip"}, [3, 0, 0])[0] == pytest.approx(2)
    with pytest.raises(SimError, match="3 lists of breakpoints"):
        last("LookupTableND", {**p, "Breakpoints": "[0 1 2], [0 10 20]"}, [1, 1, 1])


def test_direct_lookup_and_prelookup():
    tab = {"Table": "[4 5 6; 16 19 20; 10 18 23]"}
    assert last("DirectLookupTable", tab, [1, 2])[0] == 20  # row 1, column 2, counted from 0
    assert last("DirectLookupTable", tab, [9, -3])[0] == 10  # outside: the nearest edge
    col = last("DirectLookupTable", {**tab, "InputsSelectThisObjectFromTable": "Vector"}, [1])
    assert list(col) == [5, 19, 18]
    r = apply("Prelookup", {"BreakpointsData": "[10 20 40]"}, [25], n_out=2)
    assert r.y[-1, 0] == 1 and r.y[-1, 1] == pytest.approx(0.25)
    assert last("Prelookup", {"BreakpointsData": "[10 20 40]", "OutputSelection": "Index only"}, [40])[0] == 2


def test_rate_limiter_dynamic_and_running_minimum():
    r = apply("RateLimiterDynamic", {}, [2, ("Step", {"Time": "0", "After": "10"}), -1], stop="1", Solver="ode4", FixedStep="0.1")  # noqa: E501
    assert np.allclose(r.y[:, 0], 2 * r.t, atol=1e-9)  # rises by at most 2 per second
    r = apply("MinMaxRunningResettable", {"Function": "min", "vinit": "5"}, [("SineWave", {"Frequency": "2*pi"}), 0], stop="1")  # noqa: E501
    assert r.y[-1, 0] == pytest.approx(-1, abs=1e-3) and r.y[0, 0] == 0
    r = apply(
        "MinMaxRunningResettable",
        {"Function": "max", "vinit": "0"},
        [("SineWave", {"Frequency": "2*pi"}), ("Step", {"Time": "0.6"})],
        stop="1",
    )
    assert r.y[-1, 0] == pytest.approx(0, abs=1e-6)  # reset after the peak: back to the initial value


def test_permute_dimensions_and_viewer():
    y = apply("PermuteDimensions", {"Order": "[2 1]"}, ["[1 2 3; 4 5 6]"])
    assert list(y.y[-1]) == [1, 2, 3, 4, 5, 6]  # the columns of the transposed matrix (the input logs as 1 4 2 5 3 6)
    with pytest.raises(SimError, match="two-dimensional"):
        last("PermuteDimensions", {"Order": "[3 1 2]"}, ["[1 2; 3 4]"])
    r = apply("Viewer", {"NumInputPorts": "2", "Kind": "Video Viewer"}, [1, 2], n_out=0)
    assert r.t[-1] == 1  # a sink that draws nothing


SFUN_L1 = """
function [sys, x0, str, ts] = decay1(t, x, u, flag, rate)
switch flag
    case 0
        s = simsizes; s.NumContStates = 1; s.NumOutputs = 2; s.NumInputs = 1;
        s.DirFeedthrough = 1; s.NumSampleTimes = 1;
        sys = simsizes(s); x0 = 0; str = []; ts = [0 0];
    case 1
        sys = rate * (u - x);
    case 3
        sys = [x; 2 * u];
    otherwise
        sys = [];
end
"""

SFUN_L2 = """
function count2(block)
setup(block);

function setup(block)
block.NumInputPorts = 1;
block.NumOutputPorts = 1;
block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;
block.InputPort(1).Dimensions = 1;
block.InputPort(1).DirectFeedthrough = false;
block.OutputPort(1).Dimensions = 1;
block.NumDialogPrms = 1;
block.SampleTimes = [0.5 0];
block.RegBlockMethod('PostPropagationSetup', @DoPost);
block.RegBlockMethod('Start', @Start);
block.RegBlockMethod('Outputs', @Outputs);
block.RegBlockMethod('Update', @Update);

function DoPost(block)
block.NumDworks = 1;
block.Dwork(1).Name = 'total';
block.Dwork(1).Dimensions = 1;

function Start(block)
block.Dwork(1).Data = block.DialogPrm(1).Data;

function Outputs(block)
block.OutputPort(1).Data = block.Dwork(1).Data;

function Update(block)
block.Dwork(1).Data = block.Dwork(1).Data + block.InputPort(1).Data;
"""


def test_s_functions_written_in_the_matlab_language(tmp_path):
    (tmp_path / "decay1.m").write_text(SFUN_L1)
    (tmp_path / "count2.m").write_text(SFUN_L2)
    s = Session()
    s.interp.path = [str(tmp_path)]
    m = build(
        "sf",
        [
            ("u", "Constant", {"Value": "3"}),
            ("B", "SFunction", {"FunctionName": "decay1", "Parameters": "4", "NumInputs": "1", "NumOutputs": "1"}),
            ("D", "Demux", {"Outputs": "2"}),
            ("y1", "Outport", {"Port": "1"}),
            ("y2", "Outport", {"Port": "2"}),
        ],
        ["u/1 -> B/1", "B/1 -> D/1", "D/1 -> y1/1", "D/2 -> y2/1"],
        StopTime="1",
        RelTol="1e-8",
    )
    r = simulate(m, s.interp)
    assert r.y[-1, 0] == pytest.approx(3 * (1 - math.exp(-4)), abs=1e-6) and r.y[-1, 1] == 6
    m = build(
        "sf2",
        [
            ("u", "Constant", {"Value": "2"}),
            ("B", "MSFunction", {"FunctionName": "count2", "Parameters": "10", "NumInputs": "1", "NumOutputs": "1"}),
            ("y", "Outport"),
        ],
        ["u/1 -> B/1", "B/1 -> y/1"],
        StopTime="2",
    )
    r = simulate(m, s.interp)
    assert r.y[0, 0] == 10 and r.y[-1, 0] == 10 + 2 * 4  # one update every 0.5 s
    m.root.block("B").params["FunctionName"] = "compiled_thing"
    with pytest.raises(SimError, match="compiled_thing"):
        simulate(m, s.interp)


def test_options_added_for_imported_models():
    # a sine wave driven by an input signal instead of the simulation time
    assert last("SineWave", {"TimeSource": "Use external signal", "Frequency": "1"}, [math.pi / 2])[0] == pytest.approx(1)  # noqa: E501
    # a transfer function that starts away from rest
    r = apply("TransferFcn", {"Numerator": "[1]", "Denominator": "[1 1]", "InitialStates": "2"}, [0], RelTol="1e-8")
    assert r.y[0, 0] == pytest.approx(2) and r.y[-1, 0] == pytest.approx(2 * math.exp(-1), abs=1e-5)
    # a discrete integrator with a reset input
    r = apply(
        "DiscreteIntegrator",
        {"SampleTime": "0.1", "ExternalReset": "rising"},
        [1, ("Step", {"Time": "0.45"})],
        stop="1",
    )
    assert r.y[-1, 0] == pytest.approx(0.5, abs=0.051) and np.max(r.y[:, 0]) < 0.61
    # one random stream per seed
    r = apply("RandomNumber", {"Seed": "[1 2 3]", "SampleTime": "0.1"}, [])
    assert r.y.shape[1] == 3 and len({round(v, 9) for v in r.y[-1]}) == 3
    # a scalar initial value takes the size of the signal
    r = apply("Merge", {"Inputs": "2"}, ["[1 2 3]", "[4 5 6]"])
    assert r.y.shape[1] == 3


def test_a_vector_counts_as_a_column_in_matrix_products():
    # column times row is a full matrix (the outer product), row times column one number
    m = build(
        "outer",
        [
            ("v", "Constant", {"Value": "[1 2 3]"}),
            ("T", "MathFunction", {"Operator": "transpose"}),
            ("Outer", "Product", {"Inputs": "2", "Multiplication": "Matrix(*)"}),
            ("Inner", "Product", {"Inputs": "2", "Multiplication": "Matrix(*)"}),
            ("Back", "MathFunction", {"Operator": "transpose"}),
            ("y1", "Outport", {"Port": "1"}),
            ("y2", "Outport", {"Port": "2"}),
            ("y3", "Outport", {"Port": "3"}),
        ],
        ["v/1 -> T/1", "v/1 -> Outer/1", "T/1 -> Outer/2", "T/1 -> Inner/1", "v/1 -> Inner/2", "T/1 -> Back/1"]
        + ["Outer/1 -> y1/1", "Inner/1 -> y2/1", "Back/1 -> y3/1"],
        StopTime="1",
    )
    r = simulate(m)
    assert list(r.y[-1][:9]) == [1, 2, 3, 2, 4, 6, 3, 6, 9] and r.y[-1][9] == 14
    assert list(r.y[-1][10:]) == [1, 2, 3]  # transposing twice gives the vector back
