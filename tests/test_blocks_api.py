"""mylab Blocks: the MATLAB-side functions, model files, discrete and multi-rate models."""

import json
import math
import os

import numpy as np
import pytest

from mylab.api import Session
from mylab.blocks import Model, build, compile_model, load_model, save_model, simulate


@pytest.fixture
def s(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Session()


def ok(session, code):
    r = session.run(code)
    assert r.error is None, r.error
    return r


def val(session, name):
    return session.interp.get_var(name)


FIRST_ORDER = """
new_system('fo');
add_block('simulink/Sources/Step', 'fo/Step', 'Time', '0');
add_block('simulink/Math Operations/Sum', 'fo/Sum', 'Inputs', '+-');
add_block('built-in/Gain', 'fo/K', 'Gain', 'K');
add_block('simulink/Continuous/Integrator', 'fo/Int');
add_block('simulink/Sinks/Out1', 'fo/Out');
add_block('simulink/Sinks/To Workspace', 'fo/W', 'VariableName', 'simout');
add_line('fo', 'Step/1', 'Sum/1');
add_line('fo', 'Sum/1', 'K/1');
add_line('fo', 'K/1', 'Int/1', 'autorouting', 'on');
add_line('fo', 'Int/1', 'Sum/2');
add_line('fo', 'Int/1', 'Out/1');
add_line('fo', 'Int/1', 'W/1');
"""


# -- building and simulating from code ------------------------------------------------------------


def test_build_and_simulate_from_code(s):
    ok(
        s,
        "K = 2;" + FIRST_ORDER + "out = sim('fo', 'StopTime', '3', 'RelTol', '1e-8', 'AbsTol', '1e-10');"
        "t = out.tout; y = out.yout; e1 = max(abs(y - (1 - exp(-2*t))));"
        "e2 = max(abs(out.simout.Data - (1 - exp(-2*out.simout.Time))));",
    )
    assert val(s, "e1")[0, 0] < 1e-7 and val(s, "e2")[0, 0] < 1e-7
    assert val(s, "t").shape[1] == 1 and val(s, "y").shape == val(s, "t").shape
    ok(s, "K = 5; out2 = sim('fo', 2); tend = out2.tout(end); y5 = out2.yout(end);")
    assert val(s, "tend")[0, 0] == 2
    assert val(s, "y5")[0, 0] == pytest.approx(1 - math.exp(-10), abs=1e-3)


def test_sim_output_forms(s):
    ok(s, "K = 1;" + FIRST_ORDER)
    ok(s, "[t, x, y] = sim('fo', [0 2]);")
    assert val(s, "x").shape == val(s, "t").shape and np.allclose(val(s, "x"), val(s, "y"))
    ok(s, "sim('fo', 'StopTime', '1');")
    assert val(s, "tout")[-1, 0] == 1 and "simout" in s.interp.base.vars and "out" in s.interp.base.vars
    ok(
        s,
        "o = sim('fo', 'StopTime', '1', 'Solver', 'ode4', 'FixedStep', '0.25'); n = numel(o.tout); name = o.SimulationMetadata.Solver;",  # noqa: E501
    )
    assert val(s, "n")[0, 0] == 5
    ok(s, "p.StopTime = '0.5'; o = sim('fo', p); te = o.tout(end);")
    assert val(s, "te")[0, 0] == 0.5
    ok(s, "opts = simset('Solver', 'ode1', 'FixedStep', 0.1); [t2, ~, y2] = sim('fo', 1, opts); n2 = numel(t2);")
    assert val(s, "n2")[0, 0] == 11


def test_to_workspace_formats(s):
    ok(
        s,
        "new_system('w'); add_block('blocks/Clock', 'w/C');"
        "add_block('blocks/ToWorkspace', 'w/A', 'VariableName', 'a', 'SaveFormat', 'Array');"
        "add_block('blocks/ToWorkspace', 'w/B', 'VariableName', 'b', 'SaveFormat', 'Structure With Time');"
        "add_block('blocks/ToWorkspace', 'w/T', 'VariableName', 'c');"
        "add_line('w', 'C/1', 'A/1'); add_line('w', 'C/1', 'B/1'); add_line('w', 'C/1', 'T/1');"
        "o = sim('w', 1); a = o.a; bt = o.b.time; bv = o.b.signals.values; ct = o.c.Time; cd = o.c.Data;",
    )
    assert np.allclose(val(s, "a"), val(s, "bv")) and np.allclose(val(s, "bt"), val(s, "ct"))
    assert np.allclose(val(s, "cd"), val(s, "ct"))


def test_scope_and_logged_signals_in_output(s):
    ok(
        s,
        "new_system('sc'); add_block('blocks/SineWave', 'sc/S'); add_block('blocks/Scope', 'sc/My Scope');"
        "h = add_line('sc', 'S/1', 'My Scope/1'); set_param(h, 'Name', 'wave', 'DataLogging', 'on');"
        "o = sim('sc', 2); d = o.scopes.My_Scope.Data; t = o.scopes.My_Scope.Time; w = o.logsout.wave.Data;",
    )
    assert np.allclose(val(s, "d"), np.sin(val(s, "t"))) and np.allclose(val(s, "w"), val(s, "d"))


def test_add_block_errors_and_unique_names(s):
    ok(s, "new_system('e'); add_block('built-in/Gain', 'e/G');")
    r = s.run("add_block('built-in/Gain', 'e/G');")
    assert "already a block named 'G'" in r.error
    ok(s, "add_block('built-in/Gain', 'e/G', 'MakeNameUnique', 'on'); names = get_param('e', 'Blocks');")
    assert [str("".join(c[0])) for c in val(s, "names").data.flat] == ["G", "G1"]
    assert "no block 'simulink/Nothing/Wobble'" in s.run("add_block('simulink/Nothing/Wobble', 'e/W');").error
    assert "no parameter 'Gian'" in s.run("add_block('built-in/Gain', 'e/H', 'Gian', '2');").error
    assert "There is no model named 'nomodel'" in s.run("add_block('built-in/Gain', 'nomodel/H');").error
    assert "already open" in s.run("new_system('e');").error
    assert "not a valid model name" in s.run("new_system('1 bad');").error


def test_add_line_forms_and_errors(s):
    ok(
        s,
        "new_system('l'); add_block('blocks/Constant', 'l/C'); add_block('blocks/Gain', 'l/G'); add_block('blocks/Mux', 'l/M');"  # noqa: E501
        "add_block('blocks/Outport', 'l/O');",
    )
    ok(
        s,
        "pc = get_param('l/C', 'PortHandles'); pg = get_param('l/G', 'PortHandles'); pm = get_param('l/M', 'PortHandles');"  # noqa: E501
        "add_line('l', pc.Outport(1), pg.Inport(1)); add_line('l', pg.Outport(1), pm.Inport(2));"
        "add_line('l', 'C/1', 'M/1'); add_line('l', 'M/1', 'O/1'); n = numel(pm.Inport);",
    )
    assert val(s, "n")[0, 0] == 2
    assert "already has a line" in s.run("add_line('l', 'C/1', 'G/1');").error
    assert "there is no port 3" in s.run("add_line('l', 'C/1', 'M/3');").error
    assert "no block 'Nope'" in s.run("add_line('l', 'Nope/1', 'M/1');").error
    assert "wrong way round" in s.run("add_line('l', pg.Inport(1), pc.Outport(1));").error
    ok(s, "o = sim('l', 1); y = o.yout(end, :);")
    assert np.allclose(val(s, "y"), [1, 1])
    ok(s, "delete_line('l', 'C/1', 'M/1'); set_param('l/G', 'Gain', '4'); add_line('l', 'G/1', 'M/1');")
    assert "more than one" not in (s.run("o = sim('l', 1); y = o.yout(end, :);").error or "")
    assert "There is no line" in s.run("delete_line('l', 'C/1', 'M/1');").error


def test_branching_to_several_inputs_with_a_cell(s):
    ok(
        s,
        "new_system('br'); add_block('blocks/Constant', 'br/C', 'Value', '3'); add_block('blocks/Product', 'br/P');"
        "add_block('blocks/Outport', 'br/O'); add_line('br', 'C/1', {'P/1', 'P/2'}); add_line('br', 'P/1', 'O/1');"
        "o = sim('br', 1); y = o.yout(end);",
    )
    assert val(s, "y")[0, 0] == 9


def test_set_param_get_param_delete_block_find_system(s):
    ok(s, "K = 2;" + FIRST_ORDER)
    ok(
        s,
        "g = get_param('fo/K', 'Gain'); bt = get_param('fo/Int', 'BlockType'); par = get_param('fo/K', 'Parent');"
        "st = get_param('fo', 'StopTime'); set_param('fo', 'StopTime', '4', 'Solver', 'ode23'); st2 = get_param('fo', 'StopTime');"  # noqa: E501
        "set_param('fo/K', 'Gain', 7); g2 = get_param('fo/K', 'Gain'); set_param('fo/K', 'Position', [100 50 140 90]);"
        "pos = get_param('fo/K', 'Position'); ports = get_param('fo/Sum', 'Ports');",
    )
    assert (
        "".join(val(s, "g")[0]) == "K"
        and "".join(val(s, "bt")[0]) == "Integrator"
        and "".join(val(s, "par")[0]) == "fo"
    )
    assert "".join(val(s, "st")[0]) == "10" and "".join(val(s, "st2")[0]) == "4" and "".join(val(s, "g2")[0]) == "7"
    assert np.allclose(val(s, "pos"), [[100, 50, 140, 90]]) and np.allclose(val(s, "ports"), [[2, 1]])
    ok(
        s,
        "all = find_system('fo'); gains = find_system('fo', 'BlockType', 'Gain'); top = find_system('fo', 'SearchDepth', 1);"  # noqa: E501
        "byname = find_system('fo', 'Name', 'Int'); n = numel(all);",
    )
    assert val(s, "n")[0, 0] == 7
    assert ["".join(c[0]) for c in val(s, "gains").data.flat] == ["fo/K"]
    assert ["".join(c[0]) for c in val(s, "byname").data.flat] == ["fo/Int"]
    ok(
        s,
        "set_param('fo/K', 'Name', 'Kp'); g3 = get_param('fo/Kp', 'Gain'); o = sim('fo', 'StopTime', '1'); delete_block('fo/W');"  # noqa: E501
        "left = find_system('fo', 'BlockType', 'ToWorkspace'); nl = numel(left);",
    )
    assert val(s, "nl")[0, 0] == 0
    assert "no parameter 'Nope'" in s.run("get_param('fo/Kp', 'Nope')").error
    assert "There is no block 'Zed'" in s.run("get_param('fo/Zed', 'Gain')").error
    ok(s, "r = bdroot; c = gcs; b = gcb; r2 = bdroot('fo/Kp'); loaded = bdIsLoaded('fo'); nope = bdIsLoaded('zz');")
    assert "".join(val(s, "r")[0]) == "fo" and "".join(val(s, "r2")[0]) == "fo"
    assert bool(val(s, "loaded")[0, 0]) and not bool(val(s, "nope")[0, 0])


def test_subsystems_from_code(s):
    ok(
        s,
        "new_system('top'); add_block('simulink/Ports & Subsystems/Subsystem', 'top/Ctl');"
        "add_block('blocks/Inport', 'top/Ctl/In'); add_block('blocks/Gain', 'top/Ctl/K', 'Gain', '3');"
        "add_block('blocks/Outport', 'top/Ctl/Out'); add_line('top/Ctl', 'In/1', 'K/1'); add_line('top/Ctl', 'K/1', 'Out/1');"  # noqa: E501
        "add_block('blocks/Constant', 'top/C', 'Value', '2'); add_block('blocks/Outport', 'top/Y');"
        "add_line('top', 'C/1', 'Ctl/1'); add_line('top', 'Ctl/1', 'Y/1'); o = sim('top', 1); y = o.yout(end);"
        "inner = find_system('top/Ctl'); deep = find_system('top', 'BlockType', 'Gain'); p = get_param('top/Ctl', 'Ports');",  # noqa: E501
    )
    assert val(s, "y")[0, 0] == 6
    assert ["".join(c[0]) for c in val(s, "deep").data.flat] == ["top/Ctl/K"]
    assert np.allclose(val(s, "p"), [[1, 1]])
    assert "is not a subsystem" in s.run("add_block('blocks/Gain', 'top/C/X');").error


def test_save_load_close_round_trip(s, tmp_path):
    ok(s, "K = 2;" + FIRST_ORDER + "save_system('fo'); close_system('fo'); isopen = bdIsLoaded('fo');")
    assert not bool(val(s, "isopen")[0, 0])
    path = tmp_path / "fo.mlb.json"
    data = json.loads(path.read_text())
    assert (
        data["format"] == "mylab-blocks-1"
        and data["name"] == "fo"
        and len(data["blocks"]) == 6
        and len(data["lines"]) == 6
    )
    assert {"from": ["Step", 1], "to": ["Sum", 1]} == data["lines"][0]
    assert '"from": ["Step", 1]' in path.read_text()  # short lists stay on one line
    ok(s, "o = sim('fo', 'StopTime', '1'); y = o.yout(end);")  # loads the file by name
    assert val(s, "y")[0, 0] == pytest.approx(1 - math.exp(-2), abs=1e-3)
    ok(s, "load_system('fo'); set_param('fo/K', 'Gain', '1'); save_system('fo', 'copy'); c = bdroot;")
    assert "".join(val(s, "c")[0]) == "copy" and (tmp_path / "copy.mlb.json").exists()
    ok(s, "close_system('copy', 0); bdclose('all'); open_system('copy'); g = get_param('copy/K', 'Gain');")
    assert "".join(val(s, "g")[0]) == "1"


def test_model_json_round_trip_keeps_everything():
    m = build(
        "rt",
        [
            ("C", "Constant", {"Value": "[1 2]"}),
            ("S", "Subsystem", {}, ([("In", "Inport"), ("Out", "Outport")], ["In/1 -> Out/1 : inner"])),
            ("O", "Outport"),
        ],
        ["C/1 -> S/1 : sig", "S/1 -> O/1"],
        StopTime="3",
    )
    m.root.blocks[0].rot, m.root.blocks[0].flip = 1, True
    m.root.lines[0].points = [[10, 20.5], [30, 20.5]]
    m.root.lines[0].log = True
    m.root.block("S").extra["mask"] = {"k": "2"}
    again = Model.from_json(json.loads(m.dumps()))
    assert again.to_json() == m.to_json()
    assert again.root.block("S").system.lines[0].name == "inner"
    assert again.config["StopTime"] == "3" and again.root.blocks[0].rot == 1


def test_bad_model_files_give_clear_errors(tmp_path):
    from mylab.blocks import ModelError

    p = tmp_path / "bad.mlb.json"
    p.write_text("{not json")
    with pytest.raises(ModelError, match="not valid JSON"):
        load_model(str(p))
    p.write_text('{"format": "mylab-blocks-9", "blocks": []}')
    with pytest.raises(ModelError, match="Unknown model format"):
        load_model(str(p))
    p.write_text('{"blocks": [{"name": "A", "type": "Gain"}, {"name": "A", "type": "Gain"}]}')
    with pytest.raises(ModelError, match="Two blocks are named 'A'"):
        load_model(str(p))
    p.write_text('{"hello": 1}')
    with pytest.raises(ModelError, match="not a mylab Blocks model"):
        load_model(str(p))
    with pytest.raises(ModelError, match=".mdl"):
        load_model(str(tmp_path / "old.mdl"))


def test_errors_reach_matlab_code_with_block_names(s):
    ok(
        s,
        "new_system('a'); add_block('blocks/Constant', 'a/C'); add_block('blocks/Sum', 'a/S', 'Inputs', '+-');"
        "add_block('blocks/Gain', 'a/G'); add_block('blocks/Outport', 'a/O'); add_line('a', 'C/1', 'S/1'); add_line('a', 'S/1', 'G/1');"  # noqa: E501
        "add_line('a', 'G/1', 'S/2'); add_line('a', 'G/1', 'O/1');",
    )
    ok(s, "o = sim('a'); y = o.yout(end);")  # y = 1 - y is solved at every step
    assert val(s, "y")[0, 0] == pytest.approx(0.5)
    ok(s, "set_param('a', 'AlgebraicLoop', 'error');")
    r = s.run("sim('a');")
    assert "Algebraic loop" in r.error and "'S'" in r.error and "'G'" in r.error
    ok(s, "try, sim('a'); msg = ''; catch err, msg = err.message; id = err.identifier; end")
    assert "Algebraic loop" in "".join(val(s, "msg")[0]) and "".join(val(s, "id")[0]) == "mylab:blocks:simulation"
    ok(s, "set_param('a/G', 'Gain', 'undefined_gain'); delete_line('a', 'G/1', 'S/2');")
    r = s.run("sim('a');")
    assert "'G': parameter Gain = 'undefined_gain'" in r.error
    assert "Input 2 of 'S' is not connected" in ok(s, "set_param('a/G', 'Gain', '2'); o = sim('a', 1);").output


def test_parameters_come_from_the_callers_workspace_inside_functions(s, tmp_path):
    (tmp_path / "runit.m").write_text(
        "function y = runit(gain)\nKlocal = gain;\no = sim('fn', 1);\ny = o.yout(end);\nend\n"
    )
    ok(
        s,
        "new_system('fn'); add_block('blocks/Constant', 'fn/C'); add_block('blocks/Gain', 'fn/G', 'Gain', 'Klocal');"
        "add_block('blocks/Outport', 'fn/O'); add_line('fn', 'C/1', 'G/1'); add_line('fn', 'G/1', 'O/1'); y = runit(6);",  # noqa: E501
    )
    assert val(s, "y")[0, 0] == 6


def test_linmod_trim_and_simget(s):
    ok(
        s,
        "m = 2; c = 1.2; k = 18; new_system('pl'); add_block('blocks/Inport', 'pl/F'); add_block('blocks/Sum', 'pl/S', 'Inputs', '+--');"  # noqa: E501
        "add_block('blocks/Gain', 'pl/im', 'Gain', '1/m'); add_block('blocks/Integrator', 'pl/v'); add_block('blocks/Integrator', 'pl/x');"  # noqa: E501
        "add_block('blocks/Gain', 'pl/c', 'Gain', 'c'); add_block('blocks/Gain', 'pl/k', 'Gain', 'k'); add_block('blocks/Outport', 'pl/Y');"  # noqa: E501
        "add_line('pl','F/1','S/1'); add_line('pl','S/1','im/1'); add_line('pl','im/1','v/1'); add_line('pl','v/1','x/1');"  # noqa: E501
        "add_line('pl','v/1','c/1'); add_line('pl','c/1','S/2'); add_line('pl','x/1','k/1'); add_line('pl','k/1','S/3');"  # noqa: E501
        "add_line('pl','x/1','Y/1'); [A, B, C, D] = linmod('pl'); sys = linearize('pl'); p = sort(eig(A)); a2 = sys.a;"
        "[xs, us, ys, dxs] = trim('pl', [0; 1], 9); st = simget('pl', 'StopTime'); all = simget('pl');",
    )
    assert np.allclose(val(s, "A"), [[-0.6, -9], [1, 0]]) and np.allclose(val(s, "B"), [[0.5], [0]])
    assert np.allclose(val(s, "C"), [[0, 1]]) and np.allclose(val(s, "D"), 0) and np.allclose(val(s, "a2"), val(s, "A"))
    assert np.allclose(np.asarray(val(s, "p")).real, -0.3)
    assert np.max(np.abs(val(s, "dxs"))) < 1e-8
    assert 18 * val(s, "xs")[1, 0] == pytest.approx(val(s, "us")[0, 0], abs=1e-6)
    assert "".join(val(s, "st")[0]) == "10"


def test_open_system_prints_a_note_once_outside_the_browser(s):
    r = ok(s, "new_system('o'); open_system('o'); open_system('o');")
    assert r.output.count("diagram editor is part of the mylab web app") == 1


def test_help_exists_for_the_new_functions():
    from mylab.help_data import HELP

    for name in (
        "sim",
        "new_system",
        "open_system",
        "load_system",
        "save_system",
        "close_system",
        "add_block",
        "add_line",
        "delete_block",
        "delete_line",
        "set_param",
        "get_param",
        "find_system",
        "bdroot",
        "gcs",
        "gcb",
        "simset",
        "simget",
        "linmod",
        "linearize",
        "trim",
        "bdclose",
        "bdIsLoaded",
    ):
        assert name in HELP, name
        usage, text = HELP[name]
        assert usage and len(text) > 20


# -- discrete and multi-rate ----------------------------------------------------------------------


def test_discrete_loop_matches_difference_equation():
    # y[k+1] = 0.9 y[k] + 0.1 r, Ts = 0.1
    m = build(
        "d",
        [
            ("r", "Constant", {"Value": "1"}),
            ("S", "Sum", {"Inputs": "++"}),
            ("a", "Gain", {"Gain": "0.9"}),
            ("b", "Gain", {"Gain": "0.1"}),
            ("z", "UnitDelay", {"SampleTime": "0.1"}),
            ("O", "Outport"),
        ],
        ["r/1 -> b/1", "b/1 -> S/1", "z/1 -> a/1", "a/1 -> S/2", "S/1 -> z/1", "z/1 -> O/1"],
        StopTime="2",
    )
    r = simulate(m)
    o = r.outports[0]
    k = np.arange(o["t"].size)
    assert np.allclose(o["t"], 0.1 * k) and np.allclose(o["data"][:, 0], 1 - 0.9**k)
    comp = compile_model(m)
    assert comp.by_path["a"].rate == (0.1, 0.0) and comp.by_path["S"].rate == (0.1, 0.0)  # inherited through the loop
    assert comp.by_path["r"].rate == "const"
    assert simulate(m, config={"Solver": "FixedStepDiscrete"}).t.size == 21


def test_sampled_data_control_matches_exact_discretization():
    # plant 1/(s+1) behind a zero-order hold, discrete PI controller at Ts
    Ts, kp, ki = 0.2, 1.5, 2.0
    m = build(
        "sd",
        [
            ("r", "Step", {"Time": "0"}),
            ("e", "Sum", {"Inputs": "+-"}),
            ("zoh", "ZeroOrderHold", {"SampleTime": str(Ts)}),
            ("kp", "Gain", {"Gain": str(kp)}),
            ("I", "DiscreteTimeIntegrator", {"gainval": str(ki), "SampleTime": str(Ts)}),
            ("u", "Sum", {"Inputs": "++"}),
            ("G", "TransferFcn", {"Numerator": "[1]", "Denominator": "[1 1]"}),
            ("Y", "Outport"),
            ("U", "Outport", {"Port": "2"}),
        ],
        [
            "r/1 -> e/1",
            "G/1 -> e/2",
            "e/1 -> zoh/1",
            "zoh/1 -> kp/1",
            "zoh/1 -> I/1",
            "kp/1 -> u/1",
            "I/1 -> u/2",
            "u/1 -> G/1",
            "G/1 -> Y/1",
            "u/1 -> U/1",
        ],
        StopTime="6",
        RelTol="1e-9",
        AbsTol="1e-11",
    )
    r = simulate(m)
    a = math.exp(-Ts)
    y, xi, ys, us = 0.0, 0.0, [], []
    for _ in range(31):
        e = 1 - y
        u = kp * e + xi
        ys.append(y)
        us.append(u)
        xi += ki * Ts * e
        y = a * y + (1 - a) * u
    hits = np.array([np.argmin(np.abs(r.t - k * Ts)) for k in range(31)])
    assert np.allclose(r.t[hits], np.arange(31) * Ts, atol=1e-12)
    assert np.max(np.abs(r.y[hits, 0] - ys)) < 1e-7
    uo = r.outports[1]
    assert np.allclose(uo["data"][:31, 0], us, atol=1e-7)
    between = (r.t > 0.25) & (r.t < 0.35)
    assert np.allclose(r.y[between, 1], us[1])  # the control signal is held between samples


def test_multi_rate_model():
    # a fast counter (0.1 s) and a slow one (0.25 s, offset 0.05) through rate transitions into one sum
    def counter(name, ts):
        return [
            (name + "s", "Sum", {"Inputs": "++"}),
            (name + "one", "Constant"),
            (name, "UnitDelay", {"SampleTime": ts}),
        ], [f"{name}one/1 -> {name}s/1", f"{name}/1 -> {name}s/2", f"{name}s/1 -> {name}/1"]

    b1, l1 = counter("fast", "0.1")
    b2, l2 = counter("slow", "[0.25 0.05]")
    m = build(
        "mr",
        b1 + b2 + [("F", "Outport"), ("S", "Outport", {"Port": "2"}), ("sum", "Sum"), ("T", "Outport", {"Port": "3"})],
        l1 + l2 + ["fast/1 -> F/1", "slow/1 -> S/1", "fast/1 -> sum/1", "slow/1 -> sum/2", "sum/1 -> T/1"],
        StopTime="1",
    )
    r = simulate(m)
    f, sl = r.outports[0], r.outports[1]
    assert np.allclose(f["t"], np.arange(11) * 0.1) and np.allclose(f["data"][:, 0], np.arange(11))
    assert np.allclose(sl["t"], 0.05 + 0.25 * np.arange(4)) and np.allclose(sl["data"][:, 0], np.arange(4))
    assert np.all(
        np.isin(np.round(np.concatenate([f["t"], sl["t"]]), 12), np.round(r.t, 12))
    )  # every hit is a major step
    total = r.y[:, 2]
    expect = np.floor(r.t / 0.1 + 1e-9) + np.maximum(np.floor((r.t - 0.05) / 0.25 + 1e-9), 0)
    assert np.allclose(total, expect)
    comp = compile_model(m)
    assert comp.by_path["sum"].rate is None  # mixed rates: runs at every step
    assert comp.by_path["fasts"].rate == (0.1, 0.0) and comp.by_path["slows"].rate == (0.25, 0.05)


def test_invalid_sample_time_is_reported():
    from mylab.blocks import SimError

    m = build(
        "bad",
        [("C", "Constant"), ("z", "UnitDelay", {"SampleTime": "[0.1 0.5]"}), ("O", "Outport")],
        ["C/1 -> z/1", "z/1 -> O/1"],
    )
    with pytest.raises(SimError, match="not a valid sample time"):
        simulate(m)
    m = build(
        "bad2",
        [("C", "Clock"), ("I", "DiscreteTimeIntegrator", {"SampleTime": "-1"}), ("O", "Outport")],
        ["C/1 -> I/1", "I/1 -> O/1"],
    )
    with pytest.raises(SimError, match="needs a discrete sample time"):
        simulate(m)


def test_fixed_step_with_discrete_blocks_uses_the_sample_time():
    m = build(
        "fs",
        [("C", "Clock"), ("z", "ZeroOrderHold", {"SampleTime": "0.2"}), ("I", "Integrator"), ("O", "Outport")],
        ["C/1 -> z/1", "z/1 -> I/1", "I/1 -> O/1"],
        StopTime="1",
        Solver="ode4",
    )
    r = simulate(m)
    assert np.allclose(np.diff(r.t), 0.2)
    assert r.y[-1, 0] == pytest.approx(0.2 * (0 + 0.2 + 0.4 + 0.6 + 0.8))


# -- example models -------------------------------------------------------------------------------

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "blocks")


@pytest.mark.parametrize(
    "name",
    ["mass_spring_damper", "pid_speed_control", "bouncing_ball", "thermostat", "sampled_data_control", "van_der_pol"],
)
def test_example_models_run_and_have_no_errors(name):
    m = load_model(os.path.join(EXAMPLES, name + ".mlb.json"))
    assert [d for d in compile_model(m, strict=False).diagnostics if d["level"] != "info"] == []
    r = simulate(m)
    assert r.t[-1] == pytest.approx(float(m.config["StopTime"])) and r.scopes
    for b in m.root.blocks:
        assert b.pos[2] > 0 and b.pos[3] > 0, f"{b.name} has no size"
    again = Model.from_json(json.loads(m.dumps()))
    assert again.to_json() == m.to_json()


def test_example_results():
    def scope(name, which=None):
        r = simulate(load_model(os.path.join(EXAMPLES, name + ".mlb.json")))
        sc = r.scopes[which] if which else next(iter(r.scopes.values()))
        return sc["t"], [sig["data"] for sig in sc["signals"]], r

    t, d, _ = scope("mass_spring_damper")
    assert d[0][-1, 0] == pytest.approx(0.25, abs=5e-3)  # F/k = 5/20
    t, d, _ = scope("pid_speed_control")
    assert d[0][-1, 0] == pytest.approx(1.0, abs=1e-2)
    t, d, _ = scope("bouncing_ball")
    assert d[0].min() >= -1e-9 and 9.9 < d[0].max() <= 10.0
    t, d, _ = scope("thermostat")
    late = t > 20
    assert d[0][late, 0].max() == pytest.approx(21, abs=1e-3) and d[0][late, 0].min() == pytest.approx(19, abs=1e-3)
    t, d, _ = scope("sampled_data_control")
    assert d[0][-1, 0] == pytest.approx(1.0, abs=2e-2)
    t, d, _ = scope("van_der_pol")
    assert 1.9 < np.abs(d[0][:, 0]).max() < 2.1


def test_generated_editor_data_is_up_to_date():
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(EXAMPLES))
    r = subprocess.run(
        [sys.executable, os.path.join(root, "scripts", "gen_blocks_data.py"), "--check"],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=root),
    )
    assert r.returncode == 0, "run scripts/gen_blocks_data.py: " + r.stdout + r.stderr


def test_save_model_and_load_model_helpers(tmp_path):
    m = build("h", [("C", "Constant"), ("O", "Outport")], ["C/1 -> O/1"])
    path = str(tmp_path / "h.mlb.json")
    save_model(m, path)
    assert load_model(path).to_json() == m.to_json() and not m.dirty


def test_the_web_app_runs_a_model_with_its_callbacks_and_reports_loops(s):
    from mylab.blocks import web
    from mylab.blocks.model import build

    m = build(
        "cb",
        [("C", "Constant", {"Value": "level"}), ("S", "Sum", {"Inputs": "+-"}), ("K", "Gain"), ("Scope", "Scope")],
        ["C/1 -> S/1", "S/1 -> K/1", "K/1 -> S/2", "K/1 -> Scope/1"],
        StopTime="1",
    )
    m.callbacks["InitFcn"] = "level = 8;"
    marks = web.check(s, {"model": m.dumps()})["diagnostics"]  # checking a model does not run its code
    assert any("level" in d["message"] and d["level"] == "error" for d in marks)
    out = web.run(s, {"model": m.dumps()})
    assert out["ok"] and out["error"] is None, out
    assert out["scopes"][0]["signals"][0][-1] == [4.0] or out["scopes"][0]["signals"][0][-1] == 4.0  # y = 8 - y
    loop = [d for d in out["diagnostics"] if d["kind"] == "algebraic-loop"]
    assert len(loop) == 1 and loop[0]["level"] == "warning"
    m.callbacks["InitFcn"] = "level = no_such_thing;"
    ok(s, "clear level")
    out = web.run(s, {"model": m.dumps()})
    assert not out["ok"] and "InitFcn callback stopped" in out["error"]["message"]
