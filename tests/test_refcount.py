"""The in-place fast paths must never change a value another variable can see.

They decide "unshared" from reference counts, which differ between Python versions (3.14 lends a
caller's reference to the callee without counting it). The thresholds are measured at import in
mylab/_refcount.py; these tests run the sharing patterns that would expose a wrong threshold.
CI runs them on the oldest and the newest supported Python.
"""

import sys

import numpy as np
import pytest

from mylab import _refcount
from mylab.interpreter import Interpreter


def run(code: str) -> Interpreter:
    m = Interpreter(out=lambda s: None)
    m.execute(code)
    return m


def num(m: Interpreter, expr: str) -> list:
    return np.asarray(m.eval_expression(expr)).tolist()


def test_thresholds_are_measured_or_disabled():
    for name in ("DICT_AND_LOCAL", "ELEMENT", "CALLER_LOCAL", "FRESH_LEFT", "FRESH_RIGHT"):
        value = getattr(_refcount, name)
        assert value == -1 or value >= 1, name


def test_a_second_holder_is_always_counted():
    if _refcount.DICT_AND_LOCAL == -1:
        pytest.skip("in-place assignment is off on this Python")
    store = {"x": np.empty((1, 1))}
    store["y"] = store["x"]
    arr = store.get("x")
    assert sys.getrefcount(arr) > _refcount.DICT_AND_LOCAL


def test_binop_values_never_writes_into_an_argument():
    # A helper passing on arrays it was given: on Python 3.14 their count looks like a temporary's.
    m = Interpreter(out=lambda s: None)
    a = np.ones((1, 200000))
    b = np.full((1, 200000), 2.0)
    for op in ("+", "-", ".*", "./"):
        r = m.binop_values(op, a, b)
        assert r is not a and r is not b
    s = np.array([[3.0]])
    for op in ("+", ".*"):
        assert m.binop_values(op, s, b) is not b
    assert (a == 1).all() and (b == 2).all() and s[0, 0] == 3.0


@pytest.mark.parametrize("code, name, expected", [
    # x(i) = v on a shared array
    ("a = zeros(1, 3); b = a; b(2) = 5;", "a", [[0, 0, 0]]),
    ("a = zeros(1, 3); c = {a}; a(1) = 7; d = c{1};", "d", [[0, 0, 0]]),
    ("s.f = zeros(1, 3); a = s.f; a(3) = 1;", "s.f", [[0, 0, 0]]),
    # temporaries reused in place, operands still visible
    ("x = ones(1, 200000); y = x .* 2 + 1; z = sum(x);", "z", [[200000]]),
    ("x = ones(1, 200000); y = 2 .* x + x; z = sum(x);", "z", [[200000]]),
    ("x = ones(1, 200000); y = (x + 0) .* x; z = sum(x);", "z", [[200000]]),
    # loop compiler: element stores into a shared array, a shared struct, a shared field
    ("a = zeros(1, 50); b = a; for k = 1:50, b(k) = k; end", "a", [[0] * 50]),
    ("s.v = zeros(1, 50); t = s; for k = 1:50, s.v(k) = k; end; z = sum(t.v);", "z", [[0]]),
    ("v = zeros(1, 50); s.v = v; for k = 1:50, s.v(k) = k; end; z = sum(v);", "z", [[0]]),
])
def test_shared_values_are_left_alone(code, name, expected):
    assert num(run(code), name) == expected


def test_compiled_function_does_not_change_its_caller():
    m = run("""
a = zeros(1, 40); b = fill_(a); z = sum(a); w = sum(b);
function r = fill_(v)
  for k = 1:numel(v)
    v(k) = k;
  end
  r = v;
end
""".strip())
    assert num(m, "z") == [[0]] and num(m, "w") == [[820]]
