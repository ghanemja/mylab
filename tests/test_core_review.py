"""Regression tests for silent wrong answers found while reviewing the merged engine core."""


def test_argument_names_do_not_leak_to_later_calls(m, tmp_path, monkeypatch):
    # The variable names of one call's arguments (for inputname, table, display) used to stay
    # behind after a call to a built-in, so the next call through a handle saw them.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "argname.m").write_text("function r = argname(x)\nr = inputname(1);\nend\n")
    m.run("q = 5; zz = 3; disp(zz); g = @argname; r = g(q);")
    assert m.str("r") != "zz"
    m.run("a = [1;2]; b = [3;4]; numel(b); h = @table; t = h(a);")
    assert m.str("t.Properties.VariableNames{1}") != "b"
    m.run("r2 = argname(q);")
    assert m.str("r2") == "q"


def test_rounding_and_parts_keep_integer_class(m):
    # floor/ceil/fix/real/imag/conj of an integer array used to return double.
    for fn in ("floor", "ceil", "fix", "real", "imag", "conj"):
        assert m.str(f"class({fn}(int8([3 -4])))") == "int8"
        assert m.str(f"class({fn}(uint16(5)))") == "uint16"
    assert m.num("floor(int8([3 -4]))").tolist() == [[3, -4]]
    assert m.num("imag(int8([3 -4]))").tolist() == [[0, 0]]
    assert m.str("class(floor(true))") == "double"
    assert m.str("class(floor(single(2.5)))") == "single"
    assert m.num("floor(-2.5)") == -3


def test_helper_interpreter_keeps_the_display_format(m, monkeypatch):
    # The first indexed assignment into a sparse matrix makes a helper interpreter; its start
    # reset the process-wide display format, so 'format long' was silently lost.
    from mylab.builtins import sparse
    monkeypatch.setitem(sparse._REF, "baseline", None)  # calibrate again inside this run
    m.run("format long")
    try:
        out = m.run("S = sparse(2, 2); S(1, 1) = 1; x = pi")
        assert "3.141592653589793" in out
    finally:
        m.run("format short")
