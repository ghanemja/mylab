"""Fused evaluation of large elementwise expressions (optional numexpr accelerator)."""

import numpy as np
import pytest

from mylab import fuse
from mylab.api import Session

pytestmark = pytest.mark.skipif(not fuse.AVAILABLE, reason="numexpr is not installed")


def both(code, monkeypatch, min_size=1):
    """Run code with fusion off and on; return the two workspaces."""
    monkeypatch.setattr(fuse, "MIN_SIZE", min_size)
    out = []
    for on in (False, True):
        s = Session()
        s.interp.fuse_enabled = on
        r = s.run(code)
        assert r.error is None, r.error
        out.append(s.interp.frame.vars)
    return out


@pytest.mark.parametrize("expr", [
    "sin(x).^2 + cos(x).^2 + x.*x - 2*x",
    "3*x + z.*z - x./(z+1)",
    "exp(-x.^2/2)/2.5",
    "-x + abs(z - 0.5).*tanh(x) - atan(z)",
    "x.^3 - x.^-2 + 1",
    "(x + 1).*(z - 1)./(x + z + 2)",
    "x*2 + 2*z - x/4",
    "1 - x - z",
])
def test_fused_matches_ordinary(expr, monkeypatch):
    off, on = both(f"rng(3); x = rand(40,50) + 0.1; z = rand(40,50); y = {expr};", monkeypatch)
    assert on["y"].shape == off["y"].shape == (40, 50)
    assert on["y"].dtype == np.float64
    np.testing.assert_allclose(on["y"], off["y"], rtol=1e-13, atol=0)


def test_fusion_really_runs(monkeypatch):
    calls = []
    real = fuse._ne.evaluate
    monkeypatch.setattr(fuse._ne, "evaluate", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    both("x = rand(1,2e5); y = sin(x).^2 + 3*x - 1;", monkeypatch, min_size=100_000)
    assert len(calls) == 1 and "sin" in calls[0]


@pytest.mark.parametrize("code,check", [
    # a variable called sin is indexed, not called
    ("sin = [10 20 30 40]; x = [1 2 3 4]; y = sin(x) + x.*2 - 1;", [11, 23, 35, 47]),
    # integer class keeps saturating arithmetic
    ("x = int8([100 100 100]); y = x + x - 50 + x;", [127, 127, 127]),
    # complex operands
    ("x = [1+2i, 3]; y = x.*x + 2*x - 1;", [-2 + 8j, 14]),
    # implicit expansion of a row and a column
    ("x = [1 2 3]; c = [10; 20]; y = x + c.*2 - 1;", [[20, 21, 22], [40, 41, 42]]),
    # scalars only
    ("a = 2; y = sin(a).^2 + cos(a).^2 + a - 1;", [2]),
    # logical operands
    ("x = [true false true]; y = x + x - 2*x;", [0, 0, 0]),
    # NaN and Inf follow IEEE rules
    ("x = [0 1 Inf NaN]; y = x./x + x - x;", [np.nan, 1, np.nan, np.nan]),
    # matrix product is not elementwise
    ("A = [1 2; 3 4]; y = A*A + A.*A - A;", [[7, 12], [21, 34]]),
])
def test_unfusable_cases_keep_their_meaning(code, check, monkeypatch):
    off, on = both(code, monkeypatch)
    assert on["y"].dtype == off["y"].dtype
    np.testing.assert_array_equal(on["y"], off["y"])
    np.testing.assert_allclose(np.squeeze(on["y"]), np.squeeze(np.array(check)), equal_nan=True)


def test_user_function_named_like_builtin_wins(monkeypatch, tmp_path):
    (tmp_path / "tanh.m").write_text("function y = tanh(x)\ny = x + 100;\nend\n")
    off, on = both(f"addpath('{tmp_path.as_posix()}'); x = [1 2 3]; y = tanh(x) + x.*2 - 1;", monkeypatch)
    np.testing.assert_array_equal(on["y"], [[102, 105, 108]])
    np.testing.assert_array_equal(on["y"], off["y"])


def test_leaves_are_evaluated_once(monkeypatch):
    code = ("x = rand(1,50); k = bump() + bump();\ny = sin(x).^2 + x.*k - 1;\nn = bump();\n"
            "function v = bump()\npersistent c\nif isempty(c), c = 0; end\nc = c + 1; v = c;\nend\n")
    off, on = both(code, monkeypatch)
    assert on["n"].item() == off["n"].item() == 3


def test_objects_use_their_operators(monkeypatch):
    off, on = both("d = duration(1,0,0); y = hours(d + d - d*0.5) + 1 - 0.5;", monkeypatch)
    assert on["y"].item() == off["y"].item() == 2.0
