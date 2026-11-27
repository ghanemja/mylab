"""mylab Blocks: state charts (the Chart block)."""

import os

import numpy as np
import pytest

from mylab.api import Session
from mylab.blocks import SimError, build, load_model, simulate
from mylab.blocks.chart import ChartBuilder, parse_state_label, parse_transition_label
from mylab.blocks.writer import save_as


def chart_model(chart, sources=(), outs=1, ts="1", stop="10", extra=(), lines=()):
    """A Chart fed by the given (name, type, params) sources in order, its outputs on Out blocks."""
    blocks = list(sources) + [("C", "Chart", {"SampleTime": ts} if ts else {})]
    ln = [f"{s[0]}/1 -> C/{k + 1}" for k, s in enumerate(sources)]
    for k in range(outs):
        blocks.append((f"O{k + 1}", "Outport", {"Port": str(k + 1)}))
        ln.append(f"C/{k + 1} -> O{k + 1}/1")
    m = build("ch", blocks + list(extra), ln + list(lines), StopTime=stop)
    m.root.block("C").extra["chart"] = chart.to_json() if isinstance(chart, ChartBuilder) else chart
    return m


def at_samples(r, times, col=0):
    return [r.y[np.searchsorted(r.t, t - 1e-12), col] for t in times]


def test_labels():
    s = parse_state_label("Heating\nentry: y = 1;\ndu: n = n + 1;\nexit, en: z = 2;\nw = 3;")
    assert s == {"name": "Heating", "entry": " y = 1;\n z = 2;\nw = 3;", "during": " n = n + 1;",
                 "exit": " z = 2;\nw = 3;"}
    assert parse_state_label("A / x = 1;")["entry"] == "x = 1;"
    t = parse_transition_label("go | stop[x > [1 2](1)]{c = c + 1;}/y = 2; z = {1};")
    assert t == {"events": ["go", "stop"], "condition": "x > [1 2](1)", "conditionAction": "c = c + 1;",
                 "action": "y = 2; z = {1};"}
    assert parse_transition_label("after(3, sec)")["events"] == ["after(3, sec)"]
    assert parse_transition_label("[a || b]")["condition"] == "a || b"


def test_two_state_chart_with_a_condition_and_after_ticks():
    c = ChartBuilder()
    c.data("u", "input").data("y", "output", "0").data("n", "local", "0")
    c.state("Off", "entry: y = 0;").state("On", "entry: y = 1;")
    c.transition("", "Off").transition("Off", "On", "[u > 0.5]").transition("On", "Off", "after(3, tick)")
    m = chart_model(c, [("S", "SineWave", {"Frequency": "1"})], ts="0.5")
    r = simulate(m)
    ts = np.arange(0, 10.01, 0.5)
    expect, state, entered = [], "Off", 0
    for k, t in enumerate(ts, start=1):  # the same rules, by hand
        if k > 1:
            if state == "Off" and np.sin(t) > 0.5:
                state, entered = "On", k
            elif state == "On" and k - entered >= 3:
                state, entered = "Off", k
        expect.append(1.0 if state == "On" else 0.0)
    assert at_samples(r, ts) == expect


def test_entry_exit_order_through_the_hierarchy():
    # y collects one digit per action: entry Run 1, entry Slow 2, exit Slow 3, exit Run 4, transition 5, entry Stop 6
    c = ChartBuilder()
    c.data("y", "output", "0")
    c.state("Run", "entry: y = 10*y + 1;\nexit: y = 10*y + 4;")
    c.state("Run.Slow", "entry: y = 10*y + 2;\nexit: y = 10*y + 3;")
    c.state("Run.Fast")
    c.state("Stop", "entry: y = 10*y + 6;")
    c.transition("", "Run").transition("", "Run.Slow").transition("Run", "Stop", "after(2, tick)/y = 10*y + 5;")
    c.transition("Run.Slow", "Run.Fast", "[false]")
    r = simulate(chart_model(c, ts="1", stop="3"))
    assert at_samples(r, [0, 1, 2, 3]) == [12, 12, 123456, 123456]


def test_parallel_states_and_during_actions():
    c = ChartBuilder("parallel")
    c.data("a", "output", "0").data("b", "output", "0")
    c.state("Count", "during: a = a + 1;").state("Twice", "entry: b = 100;\nduring: b = b + 2;")
    r = simulate(chart_model(c, outs=2, ts="1", stop="4"))
    assert at_samples(r, [0, 1, 4], 0) == [0, 1, 4] and at_samples(r, [0, 1, 4], 1) == [100, 102, 108]


def test_junctions_backtracking_and_condition_actions():
    c = ChartBuilder()
    c.data("x", "input").data("y", "output", "0").data("tries", "output", "0")
    c.state("A", "entry: y = 1;").state("B", "entry: y = 2;").state("C", "entry: y = 3;").state("D", "entry: y = 4;")
    c.junction("j")
    c.transition("", "A")
    c.transition("A", "j", "{tries = tries + 1;}")  # tried first; its condition action runs even when the path fails
    c.transition("A", "D", "[x < -5]")
    c.transition("j", "B", "[x > 5]")
    c.transition("j", "C", "[x > 2]")
    for x, y, tries in ((10, 2, 1), (3, 3, 1), (0, 1, 2), (-10, 4, 1)):
        r = simulate(chart_model(c, [("X", "Constant", {"Value": str(x)})], outs=2, stop="2"))
        assert at_samples(r, [2], 0) == [y] and at_samples(r, [2], 1) == [tries], x


def test_flow_chart_runs_its_default_paths_every_step():
    c = ChartBuilder()
    c.data("u", "input").data("y", "output", "0")
    c.junction("j").junction("end")
    c.transition("", "j")
    c.transition("j", "end", "[u > 0]{y = 1;}")
    c.transition("j", "end", "[u < 0]{y = -1;}")
    c.transition("j", "end", "{y = 0;}")
    r = simulate(chart_model(c, [("S", "SineWave", {})], ts="0.5", stop="7"))
    ts = np.arange(0, 7.01, 0.5)
    assert at_samples(r, ts) == list(np.sign(np.round(np.sin(ts), 12)))


def test_input_events_wake_the_chart():
    c = ChartBuilder()
    c.data("n", "output", "0").event("tick_up", "input", "rising")
    c.state("S")  # an entry action here would run again at each self-transition
    c.transition("", "S").transition("S", "S", "tick_up/n = n + 1;")
    m = chart_model(c, [("P", "PulseGenerator", {"Period": "1", "PulseWidth": "50"})], ts="", stop="4.2")
    r = simulate(m)
    # rising edges at 1, 2, 3, 4 (the pulse is already high at t = 0, which is not an edge); the first one
    # enters the chart (n = 0), each later one counts
    assert at_samples(r, [0.5, 1.5, 2.5, 4.1]) == [0, 0, 1, 3]


def test_output_function_call_event_drives_a_subsystem():
    c = ChartBuilder()
    c.data("u", "input").event("hit", "output", "function-call")
    c.state("Wait").state("Busy")
    c.transition("", "Wait")
    c.transition("Wait", "Busy", "[u > 0]{send(hit);}")
    c.transition("Busy", "Wait", "[u <= 0]{hit;}")  # a bare event name sends it too
    counter = (
        [("u", "Inport"), ("T", "TriggerPort", {"TriggerType": "function-call"}), ("S", "Sum", {"Inputs": "++"}),
         ("D", "UnitDelay"), ("y", "Outport")],
        ["u/1 -> S/1", "D/1 -> S/2", "S/1 -> D/1", "S/1 -> y/1"],
    )
    m = chart_model(
        c,
        [("Sq", "PulseGenerator", {"Period": "2", "PulseWidth": "50", "Amplitude": "1"})],
        outs=0,
        ts="0.5",
        stop="6",
        extra=[("One", "Constant"), ("F", "Subsystem", {}, counter), ("Out", "Outport")],
        lines=["C/1 -> F/trigger", "One/1 -> F/1", "F/1 -> Out/1"],
    )
    r = simulate(m)
    # the chart enters Wait at t = 0 and follows the square wave from t = 0.5 on: one call per change of state
    assert at_samples(r, [0, 0.5, 1, 2, 3, 6]) == [0, 1, 2, 3, 4, 7]


def test_local_events_between_parallel_states():
    c = ChartBuilder("parallel")
    c.data("u", "input").data("y", "output", "0").event("go", "local")
    c.state("Watch").state("Act")
    c.state("Watch.Low").state("Watch.High")
    c.state("Act.Idle", "entry: y = 0;").state("Act.Moving", "entry: y = 1;")
    c.transition("", "Watch.Low").transition("", "Act.Idle")
    c.transition("Watch.Low", "Watch.High", "[u > 0]/send(go);")
    c.transition("Watch.High", "Watch.Low", "[u <= 0]")
    c.transition("Act.Idle", "Act.Moving", "go")
    c.transition("Act.Moving", "Act.Idle", "go")
    r = simulate(chart_model(c, [("S", "SineWave", {})], ts="1", stop="13"))
    ts = np.arange(0, 13.01)
    y, high, expect = 0, False, []
    for k, t in enumerate(ts):
        if k and not high and np.sin(t) > 0:
            high, y = True, 1 - y
        elif k and high and np.sin(t) <= 0:
            high = False
        expect.append(y)
    assert at_samples(r, ts) == expect


def test_temporal_logic_in_seconds_every_and_in():
    c = ChartBuilder()
    c.data("y", "output", "0").data("k", "output", "0").data("inB", "output", "0")
    c.state("A", "entry: y = 1;\nduring: inB = in(B);").state("B", "entry: y = 2;\nduring: inB = in(B);")
    c.transition("", "A")
    c.transition("A", "B", "after(1.5, sec)")
    c.transition("B", "B", "every(2, tick)[true]{k = k + 1;}")
    r = simulate(chart_model(c, outs=3, ts="0.5", stop="5"))
    assert at_samples(r, [1.0, 1.5, 2.0], 0) == [1, 2, 2]
    # B entered at t = 1.5 (re-entered by its own every(2) loop): k counts every second tick after entry
    assert at_samples(r, [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], 1) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert at_samples(r, [1.0, 2.0], 2) == [0, 1]


def test_chart_errors_name_the_problem():
    c = {"states": [{"name": "A"}, {"name": "B"}], "data": [{"name": "y", "scope": "output"}]}
    with pytest.raises(SimError, match="no default transition"):
        simulate(chart_model(c))
    c["transitions"] = [{"from": "", "to": "A"}, {"from": "A", "to": "Nowhere"}]
    with pytest.raises(SimError, match="'Nowhere', which is not a state"):
        simulate(chart_model(c))
    c["transitions"] = [{"from": "", "to": "A"}]
    c["states"][0]["label"] = "entry: y = undefined_thing + 1;"
    with pytest.raises(SimError, match="entry action of state 'A'.*undefined_thing"):
        simulate(chart_model(c))
    c["states"][0]["label"] = "entry: y = [1 2;"
    with pytest.raises(SimError, match="cannot be read"):
        simulate(chart_model(c))


def test_charts_from_code_and_in_files(tmp_path):
    os.chdir(tmp_path)
    s = Session()
    res = s.run(
        """
        new_system('therm');
        add_block('blocks/SineWave','therm/T','Amplitude','5','Bias','20');
        add_block('blocks/Chart','therm/Ctl','SampleTime','0.25');
        add_block('blocks/Outport','therm/heater');
        chart_data('therm/Ctl','temp','input');
        chart_data('therm/Ctl','on','output','0');
        chart_state('therm/Ctl','Off','entry: on = 0;');
        chart_state('therm/Ctl','On','entry: on = 1;');
        chart_transition('therm/Ctl','','Off');
        chart_transition('therm/Ctl','Off','On','[temp < 18]');
        chart_transition('therm/Ctl','On','Off','[temp > 22]');
        add_line('therm','T/1','Ctl/1'); add_line('therm','Ctl/1','heater/1');
        out = sim('therm','StopTime','12');
        y = out.yout; t = out.tout;
        save_system('therm');
        save_system('therm', 'therm_copy.slx');
        save_system('therm', 'therm_text.mdl');
        """
    )
    assert res.error is None, res.error
    t = np.ravel(s.interp.get_var("t"))
    y = np.ravel(s.interp.get_var("y"))
    temp = 20 + 5 * np.sin(np.arange(0, 12.01, 0.25))
    state, expect = 0, []
    for k, v in enumerate(temp):
        if k and state == 0 and v < 18:
            state = 1
        elif k and state == 1 and v > 22:
            state = 0
        expect.append(state)
    assert [y[np.searchsorted(t, k * 0.25 - 1e-12)] for k in range(len(temp))] == expect
    base = simulate(load_model(str(tmp_path / "therm.mlb.json")), config={"StopTime": "12"})
    for name in ("therm_copy.slx", "therm_text.mdl"):
        m = load_model(str(tmp_path / name))
        assert m.root.block("Ctl").extra["chart"] == load_model(str(tmp_path / "therm.mlb.json")).root.block(
            "Ctl").extra["chart"]
        assert np.allclose(simulate(m, config={"StopTime": "12"}).y, base.y)


def test_chart_in_files_keeps_junctions_events_and_hierarchy(tmp_path):
    c = ChartBuilder()
    c.data("u", "input").data("y", "output", "0").event("go", "input", "rising").event("done", "output", "either")
    c.state("Top", "entry: y = 1;", decomposition="parallel").state("Top.P1").state("Top.P2", "during: y = y + 1;")
    c.state("Rest")
    c.junction("j", "Top")
    c.transition("", "Top").transition("Top", "j", "go").transition("j", "Rest", "[u > 0]/send(done);")
    m = chart_model(c, [("U", "Constant"), ("G", "Step", {"Time": "2"})], outs=2, ts="")
    for ext in (".slx", ".mdl"):
        save_as(m, str(tmp_path / ("c" + ext)))
        again = load_model(str(tmp_path / ("c" + ext)))
        assert again.root.block("C").extra["chart"] == m.root.block("C").extra["chart"]


def test_function_call_input_event_runs_the_chart_once_per_call():
    c = ChartBuilder()
    c.data("n", "output", "0").event("call", "input", "function-call")
    c.state("S").transition("", "S").transition("S", "S", "call/n = n + 1;")
    m = chart_model(c, [("G", "FunctionCallGenerator", {"SampleTime": "1", "NumberOfIterations": "2"})], ts="",
                    stop="3")
    r = simulate(m)
    # at t = 0 the first call enters the chart, the second counts; then two per second
    assert at_samples(r, [0, 1, 2, 3]) == [1, 3, 5, 7]
