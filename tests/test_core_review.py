"""Regression tests for silent wrong answers found while reviewing the merged engine core."""

import numpy as np
import pytest  # noqa: F401

from mylab.errors import MatlabError  # noqa: F401


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
