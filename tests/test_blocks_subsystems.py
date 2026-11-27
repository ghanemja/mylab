"""mylab Blocks: iterator, function-call, switch-case and resettable subsystems, buses, model references."""

import numpy as np
import pytest

from mylab.api import Session
from mylab.blocks import Model, SimError, build, load_model, simulate

FIXED = dict(Solver="ode4", FixedStep="0.5")


def sum_loop(n="4", reset="reset", gain="2"):
    """A For Iterator subsystem that adds k*u for k = 1..N (a Unit Delay carries the sum from one iteration on)."""
    inner = (
        [
            ("u", "Inport"),
            ("For", "ForIterator", {"IterationLimit": n, "ResetStates": reset}),
            ("P", "Product", {"Inputs": "2"}),
            ("S", "Sum", {"Inputs": "++"}),
            ("D", "UnitDelay", {"InitialCondition": "0"}),
            ("y", "Outport"),
        ],
        ["u/1 -> P/1", "For/1 -> P/2", "P/1 -> S/1", "D/1 -> S/2", "S/1 -> D/1", "S/1 -> y/1"],
    )
    return build(
        "it",
        [("C", "Constant", {"Value": gain}), ("Loop", "Subsystem", {}, inner), ("Out", "Outport")],
        ["C/1 -> Loop/1", "Loop/1 -> Out/1"],
        StopTime="1",
        **FIXED,
    )


# -- For and While Iterator subsystems ------------------------------------------------------------


def test_for_iterator_runs_the_inside_n_times_per_step():
    r = simulate(sum_loop())
    assert np.allclose(r.y[:, 0], 2 * (1 + 2 + 3 + 4))  # states start again in every step
    r = simulate(sum_loop(reset="held"))
    assert np.allclose(r.y[:, 0], [20, 40, 60])  # states carry on from the last step


def test_for_iterator_with_zero_based_index_and_external_limit():
    inner = (
        [
            ("n", "Inport"),
            ("For", "ForIterator", {"IterationSource": "external", "IndexMode": "Zero-based", "ResetStates": "reset"}),
            ("S", "Sum", {"Inputs": "++"}),
            ("D", "Memory"),
            ("y", "Outport"),
        ],
        ["n/1 -> For/1", "For/1 -> S/1", "D/1 -> S/2", "S/1 -> D/1", "S/1 -> y/1"],
    )
    m = build(
        "ext",
        [("Clk", "Clock"), ("G", "Gain", {"Gain": "4"}), ("Loop", "Subsystem", {}, inner), ("Out", "Outport")],
        ["Clk/1 -> G/1", "G/1 -> Loop/1", "Loop/1 -> Out/1"],
        StopTime="1.5",
        **FIXED,
    )
    r = simulate(m)
    n = 4 * r.t
    assert np.allclose(r.y[:, 0], n * (n - 1) / 2)  # 0 + 1 + ... + (n-1); no iterations at t = 0


def test_iterated_assignment_builds_a_vector():
    inner = (
        [
            ("y0", "Inport"),
            ("For", "ForIterator", {"IterationLimit": "5", "ResetStates": "reset"}),
            ("Sq", "MathFunction", {"Operator": "square"}),
            ("A", "Assignment", {"IndexOptions": "Index vector (port)"}),
            ("y", "Outport"),
        ],
        ["y0/1 -> A/1", "For/1 -> Sq/1", "Sq/1 -> A/2", "For/1 -> A/3", "A/1 -> y/1"],
    )
    m = build(
        "asg",
        [("Z", "Constant", {"Value": "zeros(5,1)"}), ("Loop", "Subsystem", {}, inner), ("Out", "Outport")],
        ["Z/1 -> Loop/1", "Loop/1 -> Out/1"],
        StopTime="1",
        **FIXED,
    )
    assert np.allclose(simulate(m).y[-1], [1, 4, 9, 16, 25])


def while_loop(kind):
    blocks = [
        ("u", "Inport"),
        ("W", "WhileIterator", {"WhileBlockType": kind, "ShowIterationPort": "on", "MaxIters": "100"}),
        ("Sq", "MathFunction", {"Operator": "square"}),
        ("R", "RelationalOperator", {"Operator": "<"}),
        ("y", "Outport"),
    ]
    lines = ["W/1 -> Sq/1", "Sq/1 -> R/1", "u/1 -> R/2", "R/1 -> W/1", "W/1 -> y/1"]
    if kind == "while":
        blocks.append(("ic", "Inport", {"Port": "2"}))
        lines.append("ic/1 -> W/2")
    return blocks, lines


def test_while_iterator_counts_until_the_condition_fails():
    # the first k with k^2 >= u
    m = build(
        "wh",
        [("R", "Ramp", {"Slope": "10"}), ("Loop", "Subsystem", {}, while_loop("do-while")), ("Out", "Outport")],
        ["R/1 -> Loop/1", "Loop/1 -> Out/1"],
        StopTime="2",
        **FIXED,
    )
    r = simulate(m)
    expect = [max(1, int(np.ceil(np.sqrt(u)))) for u in 10 * r.t]
    assert np.allclose(r.y[:, 0], expect)
    # a while loop tests its initial condition first: with IC false nothing runs and the output holds
    m = build(
        "wh2",
        [
            ("R", "Constant", {"Value": "10"}),
            ("IC", "Step", {"Time": "1", "Before": "0", "After": "1"}),
            ("Loop", "Subsystem", {}, while_loop("while")),
            ("Out", "Outport"),
        ],
        ["R/1 -> Loop/1", "IC/1 -> Loop/2", "Loop/1 -> Out/1"],
        StopTime="2",
        **FIXED,
    )
    assert np.allclose(simulate(m).y[:, 0], [0, 0, 4, 4, 4])


def test_while_iterator_limit_and_continuous_states_are_refused():
    blocks, lines = while_loop("do-while")
    blocks[1] = ("W", "WhileIterator", {"WhileBlockType": "do-while", "ShowIterationPort": "on", "MaxIters": "3"})
    m = build(
        "lim",
        [("C", "Constant", {"Value": "100"}), ("Loop", "Subsystem", {}, (blocks, lines)), ("O", "Outport")],
        ["C/1 -> Loop/1", "Loop/1 -> O/1"],
        StopTime="1",
        **FIXED,
    )
    assert np.allclose(simulate(m).y[:, 0], 3)
    bad = ([("For", "ForIterator"), ("I", "Integrator"), ("y", "Outport")], ["For/1 -> I/1", "I/1 -> y/1"])
    m = build("bad", [("Loop", "Subsystem", {}, bad), ("O", "Outport")], ["Loop/1 -> O/1"], StopTime="1")
    with pytest.raises(SimError, match="continuous states cannot be inside an iterator"):
        simulate(m)


def test_iterator_inside_a_continuous_model_is_evaluated_in_minor_steps_too():
    # y = sum_{k=1..3} k * x = 6 x feeds an integrator: dx/dt = -6 x, x(0) = 1
    inner = (
        [("u", "Inport"), ("For", "ForIterator", {"IterationLimit": "3", "ResetStates": "reset"}),
         ("P", "Product", {"Inputs": "2"}), ("S", "Sum", {"Inputs": "++"}), ("D", "UnitDelay"), ("y", "Outport")],
        ["u/1 -> P/1", "For/1 -> P/2", "P/1 -> S/1", "D/1 -> S/2", "S/1 -> D/1", "S/1 -> y/1"],
    )
    m = build(
        "cl",
        [("I", "Integrator", {"InitialCondition": "1"}), ("Loop", "Subsystem", {}, inner),
         ("N", "Gain", {"Gain": "-1"}), ("Out", "Outport")],
        ["I/1 -> Loop/1", "Loop/1 -> N/1", "N/1 -> I/1", "I/1 -> Out/1"],
        StopTime="1", RelTol="1e-8", AbsTol="1e-10",
    )
    r = simulate(m)
    assert np.max(np.abs(r.y[:, 0] - np.exp(-6 * r.t))) < 1e-6


# -- function-call subsystems ---------------------------------------------------------------------


def counter(calls="1", ts="0.5"):
    inner = (
        [("u", "Inport"), ("T", "TriggerPort", {"TriggerType": "function-call"}), ("S", "Sum", {"Inputs": "++"}),
         ("D", "UnitDelay"), ("Sc", "Scope"), ("y", "Outport")],
        ["u/1 -> S/1", "D/1 -> S/2", "S/1 -> D/1", "S/1 -> y/1", "S/1 -> Sc/1"],
    )
    return build(
        "fc",
        [("G", "FunctionCallGenerator", {"SampleTime": ts, "NumberOfIterations": calls}),
         ("C", "Constant", {"Value": "1"}), ("F", "Subsystem", {}, inner), ("Out", "Outport")],
        ["C/1 -> F/1", "G/1 -> F/trigger", "F/1 -> Out/1"],
        StopTime="2",
    )


def test_function_call_subsystem_runs_once_per_call():
    r = simulate(counter())
    for tt, n in ((0, 1), (0.25, 1), (0.5, 2), (1.9, 4), (2, 5)):
        k = np.searchsorted(r.t, tt + 1e-12, side="right") - 1
        assert r.y[k, 0] == n
    # a scope inside logs at the calls only
    sc = r.scopes["F/Sc"]
    assert np.allclose(sc["t"], [0, 0.5, 1, 1.5, 2]) and np.allclose(sc["signals"][0]["data"][:, 0], [1, 2, 3, 4, 5])
    r = simulate(counter(calls="3"))
    assert r.y[-1, 0] == 15


def test_function_call_subsystem_needs_its_caller():
    m = counter()
    m.root.lines = [ln for ln in m.root.lines if ln.dst != ("F", "trigger")]
    from mylab.blocks import diagnose

    msgs = [d["message"] for d in diagnose(m)]
    assert any("Input trigger of 'F' is not connected" in s for s in msgs)


# -- switch case, resets ------------------------------------------------------------------------


def action(value):
    return ([("A", "ActionPort"), ("c", "Constant", {"Value": value}), ("y", "Outport")], ["c/1 -> y/1"])


def test_switch_case_picks_the_matching_action_subsystem():
    m = build(
        "sc",
        [("Clk", "Clock"), ("SC", "SwitchCase", {"CaseConditions": "{0,[1 2]}"}),
         ("S1", "Subsystem", {}, action("10")), ("S2", "Subsystem", {}, action("20")),
         ("S3", "Subsystem", {}, action("-1")), ("M", "Merge", {"Inputs": "3"}), ("Out", "Outport")],
        ["Clk/1 -> SC/1", "SC/1 -> S1/enable", "SC/2 -> S2/enable", "SC/3 -> S3/enable",
         "S1/1 -> M/1", "S2/1 -> M/2", "S3/1 -> M/3", "M/1 -> Out/1"],
        StopTime="4", **FIXED,
    )
    r = simulate(m)
    assert np.allclose(r.y[:, 0], [10, 10, 20, 20, 20, 20, -1, -1, -1])  # 1.5 counts as 1: cut toward zero


def integrator_in(port, params):
    blocks = [("u", "Inport"), ("Ctl", port, params), ("I", "Integrator"), ("y", "Outport")]
    return blocks, ["u/1 -> I/1", "I/1 -> y/1"]


def test_resettable_subsystem_starts_again_from_its_initial_states():
    m = build(
        "rs",
        [("C", "Constant"), ("St", "Step", {"Time": "1"}), ("Sub", "Subsystem", {}, integrator_in("ResetPort", {})),
         ("Out", "Outport")],
        ["C/1 -> Sub/1", "St/1 -> Sub/reset", "Sub/1 -> Out/1"],
        StopTime="2",
    )
    r = simulate(m)
    ok = np.abs(r.t - 1) > 1e-9
    assert np.max(np.abs(r.y[ok, 0] - np.where(r.t < 1, r.t, r.t - 1)[ok])) < 1e-9
    k = int(np.flatnonzero(r.t >= 1)[-1 if False else 0])
    assert r.y[k, 0] == pytest.approx(0.0, abs=1e-12)  # the logged value at the reset is the new one


def enabled(params, out_params=None):
    blocks, lines = integrator_in("EnablePort", params)
    if out_params:
        blocks[-1] = ("y", "Outport", out_params)
    return build(
        "en",
        [("C", "Constant"), ("P", "PulseGenerator", {"Period": "2", "PulseWidth": "50"}),
         ("Sub", "Subsystem", {}, (blocks, lines)), ("Out", "Outport")],
        ["C/1 -> Sub/1", "P/1 -> Sub/enable", "Sub/1 -> Out/1"],
        StopTime="4", MaxStep="0.05",
    )


def at(r, t):
    return float(np.interp(t, r.t, r.y[:, 0]))


def test_states_when_enabling_held_or_reset_and_output_when_disabled():
    held = simulate(enabled({}))
    assert [round(at(held, t), 6) for t in (0.5, 1.5, 2.5, 3.5)] == [0.5, 1.0, 1.5, 2.0]
    reset = simulate(enabled({"StatesWhenEnabling": "reset"}))
    assert [round(at(reset, t), 6) for t in (0.5, 1.5, 2.5, 3.5)] == [0.5, 1.0, 0.5, 1.0]
    zero = simulate(enabled({}, {"OutputWhenDisabled": "reset", "InitialOutput": "-1"}))
    assert [round(at(zero, t), 6) for t in (0.5, 1.5, 2.5, 3.5)] == [0.5, -1.0, 1.5, -1.0]


def test_action_subsystem_can_reset_its_states():
    sub = integrator_in("ActionPort", {"InitializeStates": "reset"})
    m = build(
        "act",
        [("C", "Constant"), ("P", "PulseGenerator", {"Period": "2", "PulseWidth": "50"}),
         ("If", "If", {"IfExpression": "u1 > 0", "ShowElse": "off"}), ("Sub", "Subsystem", {}, sub),
         ("Out", "Outport")],
        ["C/1 -> Sub/1", "P/1 -> If/1", "If/1 -> Sub/enable", "Sub/1 -> Out/1"],
        StopTime="4", MaxStep="0.05",
    )
    r = simulate(m)
    assert [round(at(r, t), 6) for t in (0.5, 2.5)] == [0.5, 0.5]


# -- buses ----------------------------------------------------------------------------------------


def test_bus_assignment_selector_nested_buses_and_struct_functions():
    s = Session()
    res = s.run(
        """
        e1 = Simulink.BusElement; e1.Name = 'pos';
        e2 = Simulink.BusElement; e2.Name = 'vel'; e2.Dimensions = 2;
        B = Simulink.Bus; B.Elements = [e1 e2];
        new_system('bm');
        add_block('blocks/Clock','bm/Clk');
        add_block('blocks/Constant','bm/V','Value','[3;4]');
        add_block('blocks/BusCreator','bm/BC','OutDataTypeStr','Bus: B');
        add_block('blocks/Gain','bm/G','Gain','2');
        add_block('blocks/BusAssignment','bm/BA','AssignedSignals','pos');
        add_block('blocks/UnitDelay','bm/UD','SampleTime','0.5');
        add_block('blocks/BusSelector','bm/BS','OutputSignals','vel,pos');
        add_block('blocks/MATLABFunction','bm/F','Script',sprintf('function y = f(b)\\ny = b.pos + sum(b.vel);'));
        add_block('blocks/ToWorkspace','bm/TW','VariableName','busy','SaveFormat','Array');
        add_block('blocks/Outport','bm/O1'); add_block('blocks/Outport','bm/O2'); add_block('blocks/Outport','bm/O3');
        add_line('bm','Clk/1','BC/1'); add_line('bm','V/1','BC/2');
        add_line('bm','Clk/1','G/1'); add_line('bm','BC/1','BA/1'); add_line('bm','G/1','BA/2');
        add_line('bm','BA/1',{'UD/1','F/1','TW/1'});
        add_line('bm','UD/1','BS/1');
        add_line('bm','BS/1','O1/1'); add_line('bm','BS/2','O2/1'); add_line('bm','F/1','O3/1');
        out = sim('bm','StopTime','1');
        y = out.yout(end,:);
        p = out.busy.pos(end); v = out.busy.vel(end,:);
        """
    )
    assert res.error is None, res.error
    assert np.allclose(s.interp.get_var("y"), [3, 4, 1, 9])  # pos through the unit delay is 2*0.5
    assert np.allclose(s.interp.get_var("p"), 2) and np.allclose(s.interp.get_var("v"), [3, 4])


def test_nested_buses_struct_constants_and_bus_output():
    m = build(
        "nb",
        [
            ("K", "Constant", {"Value": "struct('a', 1, 'b', struct('c', [2;3], 'd', 4))"}),
            ("F", "MATLABFunction", {"Script": "function s = f(b)\ns.total = b.a + sum(b.b.c) + b.b.d;\ns.c = b.b.c;"}),
            ("BS", "BusSelector", {"OutputSignals": "b.c,b.d,a"}),
            ("BS2", "BusSelector", {"OutputSignals": "total,c", "OutputAsBus": "on"}),
            ("BS3", "BusSelector", {"OutputSignals": "c"}),
            ("O1", "Outport"), ("O2", "Outport", {"Port": "2"}), ("O3", "Outport", {"Port": "3"}),
            ("O4", "Outport", {"Port": "4"}),
        ],
        ["K/1 -> BS/1", "K/1 -> F/1", "F/1 -> BS2/1", "BS2/1 -> BS3/1", "BS/1 -> O1/1", "BS/2 -> O2/1",
         "BS/3 -> O3/1", "BS3/1 -> O4/1"],
        StopTime="1",
    )
    assert np.allclose(simulate(m).y[-1], [2, 3, 4, 1, 2, 3])
    m.root.block("BS").params["OutputSignals"] = "b.zz,b.d,a"
    with pytest.raises(SimError, match="no signal 'b.zz'.*a, b, b.c, b.d"):
        simulate(m)


def test_bus_selector_says_where_a_non_bus_comes_from():
    m = build("nb", [("C", "Constant"), ("BS", "BusSelector", {"OutputSignals": "a"}), ("O", "Outport")],
              ["C/1 -> BS/1", "BS/1 -> O/1"])
    with pytest.raises(SimError, match="not a bus .*'C'"):
        simulate(m)


# -- Selector and Assignment ----------------------------------------------------------------------


def one(block_type, params, inputs):
    blocks = [("B", block_type, params), ("y", "Outport")]
    lines = ["B/1 -> y/1"]
    for k, v in enumerate(inputs):
        blocks.append((f"u{k}", "Constant", {"Value": v, "VectorParams1D": "off"}))
        lines.append(f"u{k}/1 -> B/{k + 1}")
    return simulate(build("one", blocks, lines, StopTime="0")).y[-1]


def test_selector_starting_index_ports_and_more_dimensions():
    a = "[1 2 3; 4 5 6; 7 8 9]"
    p = {"NumberOfDimensions": "2", "IndexOptions": "Starting index (dialog),Index vector (port)",
         "Indices": "2,", "OutputSizes": "2,1"}
    assert np.allclose(one("Selector", p, [a, "[3 1]"]), [6, 9, 4, 7])  # rows 2:3, columns [3 1]
    p = {"IndexOptions": "Starting index (port)", "OutputSizes": "3"}
    assert np.allclose(one("Selector", p, ["[5 6 7 8 9]", "2"]), [6, 7, 8])
    p = {"NumberOfDimensions": "3", "IndexOptions": "Select all,Index vector (dialog),Index vector (dialog)",
         "Indices": ",2,2"}
    assert np.allclose(one("Selector", p, ["reshape(1:8, 2, 2, 2)"]), [7, 8])
    p = {"NumberOfDimensions": "4", "IndexOptions": "Index vector (dialog),Select all,Select all,Index vector (dialog)",
         "Indices": "1,,,2"}
    assert np.allclose(one("Selector", p, ["reshape(1:16, 2, 2, 2, 2)"]), [9, 11, 13, 15])
    p = {"IndexMode": "Zero-based", "IndexOptions": "Starting index (dialog)", "Indices": "4", "OutputSizes": "2"}
    with pytest.raises(SimError, match="outside the input"):
        one("Selector", p, ["[1 2 3 4 5]"])


def test_assignment_with_index_ports_starting_index_and_given_size():
    p = {"NumberOfDimensions": "2", "IndexOptions": "Index vector (port),Starting index (dialog)", "Indices": ",2"}
    y = one("Assignment", p, ["zeros(3)", "[1 2; 3 4]", "[3 1]"])
    assert np.allclose(np.reshape(y, (3, 3), order="F"), [[0, 3, 4], [0, 0, 0], [0, 1, 2]])
    p = {"OutputInitialize": "Specify size for each dimension", "OutputSizes": "5",
         "IndexOptions": "Index vector (port)"}
    assert np.allclose(one("Assignment", p, ["7", "[2 4]"]), [0, 7, 0, 7, 0])


# -- model references -----------------------------------------------------------------------------


def test_model_block_runs_a_model_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Session()
    res = s.run(
        """
        new_system('plant');
        add_block('blocks/Inport','plant/u'); add_block('blocks/Gain','plant/K','Gain','k');
        add_block('blocks/Integrator','plant/I'); add_block('blocks/Outport','plant/y');
        add_line('plant','u/1','K/1'); add_line('plant','K/1','I/1'); add_line('plant','I/1','y/1');
        save_system('plant'); close_system('plant');
        k = 3;
        new_system('top');
        add_block('blocks/Step','top/S','Time','0');
        add_block('simulink/Ports & Subsystems/Model','top/P','ModelName','plant');
        add_block('blocks/ModelReference','top/P2','ModelNameDialog','plant.mlb.json');
        add_block('blocks/Outport','top/O'); add_block('blocks/Outport','top/O2');
        add_line('top','S/1','P/1'); add_line('top','P/1','P2/1'); add_line('top','P/1','O/1');
        add_line('top','P2/1','O2/1');
        out = sim('top','StopTime','2');
        y = out.yout(end,:);
        save_system('top');
        """
    )
    assert res.error is None, res.error
    assert np.allclose(s.interp.get_var("y"), [6, 18])
    top = load_model(str(tmp_path / "top.mlb.json"))
    assert top.root.block("P").extra["ports"] == [1, 1]
    assert np.allclose(simulate(top, s.interp, config={"StopTime": "2"}).y[-1], [6, 18])


def test_model_reference_errors(tmp_path):
    m = build("loop1", [("M", "ModelReference", {"ModelName": "loop1"})], [])
    m.file = str(tmp_path / "loop1.mlb.json")
    from mylab.blocks import save_model

    save_model(m, m.file)
    with pytest.raises(SimError, match="refers to itself"):
        simulate(load_model(m.file))
    m = build("lost", [("M", "ModelReference", {"ModelName": "nowhere_to_be_found"})], [])
    with pytest.raises(SimError, match="'nowhere_to_be_found' is not open"):
        simulate(m)


def test_new_subsystem_kinds_survive_the_json_file():
    m = sum_loop()
    again = Model.from_json(m.to_json())
    assert np.allclose(simulate(again).y[:, 0], 20)
