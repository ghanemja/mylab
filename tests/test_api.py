from mylab.api import Session


def test_session_run_output_and_error():
    s = Session()
    r = s.run("x = 2 + 3")
    assert r.ok and r.output == "x = 5\n"
    r = s.run("y = x * 2; undefined_fn(1)")
    assert not r.ok and "undefined_fn" in r.error and r.error_line == 1
    assert s.get("y") == 10


def test_session_figures_and_variables():
    s = Session()
    r = s.run("plot(1:3); title('t')")
    assert len(r.figures) == 1 and r.figures[0][:4] == b"\x89PNG"
    s.run("a = [1 2 3]; b = 'txt'; c = {1};")
    names = {v.name: v for v in s.variables()}
    assert names["a"].size == "1×3" and names["b"].preview == "'txt'" and names["c"].cls == "cell"
    assert "sin" in s.complete("si")
    s.reset()
    assert s.variables() == []


def test_help_lookup():
    s = Session()
    assert "Print the value" in s.help("disp")
    assert "not found" in s.help("no_such_function_zzz")
