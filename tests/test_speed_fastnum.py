"""The fast numeric layer (mylab/fastnum.py) against the general code.

Every shortcut is run with the layer on and off on random and edge-case inputs (NaN, Inf, signed
zeros, empty arrays, other classes).  The results must be the same to the last bit, with the same
class and shape, and the same error when there is one.  Only the LAPACK/FFT back-end choices are
compared to rounding error, as documented in the module.
"""

import itertools

import numpy as np
import pytest

from mylab import Interpreter, fastnum, ops
from mylab.builtins import BUILTINS
from mylab.errors import MatlabError
from mylab.values import Cell, char

NAN, INF = np.nan, np.inf
NEG_NAN = np.array([np.nan]).view(np.int64)
NEG_NAN = float((NEG_NAN | np.int64(-2 ** 63)).view(float)[0])


@pytest.fixture(autouse=True)
def _layer_back_on():
    yield
    fastnum.set_enabled(True)


def call(interp, name, args, nargout):
    """Call a built-in the way the interpreter does (results converted to MATLAB values)."""
    out = interp.call_builtin(name, BUILTINS[name], list(args), nargout)
    return out[0] if len(out) == 1 else tuple(out)


def describe(v):
    """A value as something comparable bit for bit."""
    if isinstance(v, (tuple, list)):
        return tuple(describe(x) for x in v)
    if isinstance(v, np.ndarray):
        return (v.shape, v.dtype.str, v.tobytes())
    if isinstance(v, (float, bool, int)):
        return (type(v).__name__, np.array(v).tobytes())
    return ("other", repr(v))


def on_and_off(fn):
    out = []
    for on in (True, False):
        fastnum.set_enabled(on)
        try:
            with np.errstate(all="ignore"):
                out.append(fn())
        except MatlabError as e:
            out.append(("MatlabError", e.message))
        except Exception as e:  # noqa: BLE001 - the built-in wrapper turns these into MATLAB errors
            out.append((type(e).__name__, str(e)))
    fastnum.set_enabled(True)
    return out


def same(fn, what=""):
    fast, slow = on_and_off(fn)
    assert describe(fast) == describe(slow), f"{what}: fast {fast!r} != general {slow!r}"
    return fast


def edge_values():
    return [0.0, -0.0, 1.0, -1.0, 2.5, -2.5, 0.5, -0.5, 1.5, 3.0, 7.0, -7.0, 1e-320, 1e308, -1e308, 1e-9,
            123456789.25, NAN, NEG_NAN, INF, -INF, 2.0 ** 53, 0.1, -0.3]


def vectors(rng, count=40):
    """Real double vectors of 1..9 elements in both orientations, with awkward values mixed in."""
    out = []
    edges = edge_values()
    for k in range(count):
        n = int(rng.integers(1, 10))
        v = rng.standard_normal(n) * 10.0 ** rng.integers(-6, 7, n)
        if k % 3 == 1:
            v = rng.integers(-4, 5, n).astype(float)
        if k % 4 == 2:
            for j in range(n):
                if rng.random() < 0.4:
                    v[j] = edges[int(rng.integers(len(edges)))]
        out.append(v.reshape(1, -1).copy())
        out.append(v.reshape(-1, 1).copy())
    return out


def other_inputs():
    zeros = [np.full(shape, z) for n in range(1, 10) for shape in ((1, n), (n, 1)) for z in (-0.0, 0.0)]
    return zeros + [np.zeros((0, 0)), np.zeros((1, 0)), np.zeros((0, 3)), np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1, 2, 3]], dtype=np.int32), np.array([[200, 3]], dtype=np.uint8),
            np.array([[True, False, True]]), np.array([[1.5, -2.0]], dtype=np.float32),
            np.array([[1 + 2j, 3.0]]), char("abc"), np.arange(24.0).reshape(2, 3, 4),
            np.arange(1.0, 13.0).reshape(1, 12), np.array([[np.nan, 1.0, np.nan]]), np.array([[-0.0, 0.0]]),
            np.array([[0.0, -0.0]]), np.array([[True]]), np.array([[5]], dtype=np.int8)]


ONE_ARG = ["sum", "prod", "cumsum", "max", "min", "any", "all", "abs", "round", "numel", "length", "norm",
           "floor", "ceil", "fix", "exp", "sin", "cos", "tan", "atan", "sinh", "cosh", "tanh", "asinh", "expm1",
           "sqrt", "log", "log2", "log10", "sort", "unique"]


@pytest.mark.parametrize("name", ONE_ARG)
def test_one_argument_functions(name):
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    interp = Interpreter()
    for v in vectors(rng) + other_inputs() + [np.array([[x]]) for x in edge_values()]:
        same(lambda: call(interp, name, [v], 1), f"{name}({v!r})")


@pytest.mark.parametrize("name", ["max", "min", "mod", "rem", "dot", "ismember", "size"])
def test_two_argument_functions(name):
    rng = np.random.default_rng(11)
    interp = Interpreter()
    scalars = [np.array([[x]]) for x in edge_values()]
    for a, b in itertools.product(scalars, scalars):
        same(lambda: call(interp, name, [a, b], 1), f"{name}({a!r}, {b!r})")
    vecs = vectors(rng, 12)
    for v in vecs + other_inputs():
        for b in scalars[:12] + [v, v[::-1].copy(), np.array([[1.0, 2.0, 3.0]]), np.array([[2]], dtype=np.int8)]:
            same(lambda: call(interp, name, [v, b], 1), f"{name}({v!r}, {b!r})")
            if name == "ismember":
                same(lambda: call(interp, name, [v, b], 2), f"[tf, loc] = ismember({v!r}, {b!r})")


def test_outputs_and_options_fall_through():
    interp = Interpreter()
    v = np.array([[3.0, 1.0, 2.0, 1.0]])
    same(lambda: call(interp, "max", [v], 2))
    same(lambda: call(interp, "sort", [v], 2))
    same(lambda: call(interp, "sort", [v, char("descend")], 1))
    same(lambda: call(interp, "sort", [v, char("DESCEND")], 1))
    same(lambda: call(interp, "sort", [v, np.array([[2.0]])], 1))
    same(lambda: call(interp, "unique", [v], 3))
    same(lambda: call(interp, "unique", [v, char("stable")], 1))
    same(lambda: call(interp, "sum", [v, np.array([[1.0]])], 1))
    same(lambda: call(interp, "sum", [v, char("omitnan")], 1))
    same(lambda: call(interp, "log2", [v], 2))
    same(lambda: call(interp, "norm", [v, np.array([[1.0]])], 1))
    same(lambda: call(interp, "numel", [Cell.from_list([v, v])], 1))
    same(lambda: call(interp, "size", [v, np.array([[3.0]])], 1))
    same(lambda: call(interp, "size", [v, np.array([[0.0]])], 1))
    same(lambda: call(interp, "size", [v, np.array([[1.5]])], 1))


OPS = ["+", "-", "*", "/", "\\", ".*", "./", ".\\", "^", ".^", "==", "~=", "<", "<=", ">", ">=", "&", "|"]


@pytest.mark.parametrize("op", OPS)
def test_operators_on_scalars(op):
    scalars = [np.array([[x]]) for x in edge_values()]
    for a, b in itertools.product(scalars, scalars):
        x, y = a.item(), b.item()
        if op in ("^", ".^") and x < 0 and not np.isfinite(y):
            # The general scalar code used to stop with an internal error here; the result is now
            # the real IEEE value, as in MATLAB ((-2)^Inf is Inf, (-1)^NaN is NaN).
            r = ops.binop(op, a, b)
            assert r.dtype == np.float64 and r.shape == (1, 1)
            assert describe(r) == describe(np.array([[x ** y]]))
            continue
        same(lambda: ops.binop(op, a, b), f"{a!r} {op} {b!r}")


@pytest.mark.parametrize("op", OPS)
def test_operators_on_arrays(op):
    rng = np.random.default_rng(5)
    pool = vectors(rng, 10) + other_inputs()
    scalars = [np.array([[x]]) for x in (2.0, -3.0, 0.5, 0.0, NAN, INF, 2.5)] + [np.array([[True]])]
    for a in pool:
        for b in scalars + [a, a.copy(), a[::-1].copy() if a.ndim == 2 else a, np.array([[1.0, 2.0, 3.0]])]:
            same(lambda: ops.binop(op, a, b), f"{a!r} {op} {b!r}")
            same(lambda: ops.binop(op, b, a), f"{b!r} {op} {a!r}")
    masks = [np.array([[True, False, True]]), np.array([[False, False, True]]), np.array([[True]]),
             np.array([[True], [False]]), np.zeros((0, 0), dtype=bool)]
    for a, b in itertools.product(masks, masks):
        same(lambda: ops.binop(op, a, b), f"{a!r} {op} {b!r}")


def test_matrix_literals():
    code = [
        "x = [1, 2, 3];", "x = [1; 2; 3];", "a = 2; x = [a, a + 1; 5, 6];", "v = [1 2 3]; x = [v, 4, v];",
        "v = [1 2 3]; x = [v; v + 1];", "x = [[1 2]', [3 4]'];", "x = [1, [], 2];", "x = [];", "x = [1, 'a'];",
        "x = [1, true];", "x = [int8(1), 2.7];", "x = [1 2; 3];", "x = [zeros(1, 0), 5];", "x = [single(1), 2];",
        "x = [1, 2i];", "x = [-0, 0, NaN, Inf];", "c = {1, 2}; x = [c{:}];", "x = [1:3; 4:6];",
    ]
    for line in code:
        def run():
            m = Interpreter(out=lambda s: None)
            m.execute(line)
            return m.get_var("x")
        same(run, line)


def test_large_mod_rem_sort_unique_ismember():
    rng = np.random.default_rng(21)
    interp = Interpreter()
    n = 6000
    arrays = [rng.integers(-10 ** 6, 10 ** 6, (1, n)).astype(float), rng.standard_normal((n, 1)) * 1e3,
              np.arange(1.0, n + 1).reshape(1, -1) * 7919, rng.integers(0, 50, (n, 1)) * 0.1,
              rng.standard_normal((60, 100)), rng.integers(-3, 4, (1, n)).astype(float)]
    spiked = rng.integers(-3, 4, (1, n)).astype(float)
    spiked[0, ::7] = NAN
    spiked[0, 3::11] = NEG_NAN
    spiked[0, 5::13] = -0.0
    spiked[0, 1::17] = INF
    arrays.append(spiked)
    for a in arrays:
        for y in (3.0, -3.0, 0.1, 1009.0, 1e-320, 1e300, 0.3, 2.0 ** 40, INF, NAN, 0.0, 7.0):
            same(lambda: call(interp, "mod", [a, np.array([[y]])], 1), f"mod(..., {y})")
            same(lambda: call(interp, "rem", [a, np.array([[y]])], 1), f"rem(..., {y})")
        same(lambda: call(interp, "unique", [a], 1), "unique")
        same(lambda: call(interp, "ismember", [a, a[..., :50]], 2), "ismember")
        same(lambda: call(interp, "ismember", [a[..., :80], a], 1), "ismember")
        if 1 in a.shape:
            same(lambda: call(interp, "sort", [a], 1), "sort")
            same(lambda: call(interp, "sort", [a, char("descend")], 1), "sort descend")
    ints = rng.integers(-50, 50, (1, n)).astype(np.int16)
    same(lambda: call(interp, "sort", [ints, char("descend")], 1))
    same(lambda: call(interp, "unique", [ints], 1))
    same(lambda: call(interp, "sort", [ints > 0], 1))


def test_summation_order_is_verified():
    """The Python sum/prod/cumsum agree with NumPy's order on this installation."""
    for family in ("sum", "prod", "cumsum", "norm", "dot"):
        assert fastnum._verified(family), family
    # and the check notices a wrong order
    real = fastnum._FAST["sum"]
    try:
        fastnum._FAST["sum"] = lambda args, nargout: fastnum.scalar(float(sum(reversed(args[0].ravel().tolist()))))
        assert not fastnum._run_check("sum")
    finally:
        fastnum._FAST["sum"] = real


def assert_same_solution(x, ref, rtol=1e-9):
    """Two LAPACK builds (NumPy's and SciPy's) round differently: on Windows a 130 x 130 solve
    differed by 4e-11 in a component of 0.025. Each component must agree to rtol relative to the
    solution's largest component, the usual measure for comparing solves."""
    np.testing.assert_allclose(x, ref, rtol=rtol, atol=rtol * np.abs(ref).max())


def test_solve_matches_numpy_and_warns_like_matlab():
    rng = np.random.default_rng(2)
    backend = fastnum.lapack_backend()
    for n in (2, 3, 5, 9, 40, 130):
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, 2))
        x = ops.left_divide(a, b)
        if backend is not None and backend.name == "numpy-openblas":
            assert np.array_equal(x, np.linalg.solve(a, b) + 0.0)  # the same library and driver as NumPy
        else:
            assert_same_solution(x, np.linalg.solve(a, b))
        spd = a @ a.T + n * np.eye(n)
        np.testing.assert_allclose(ops.left_divide(spd, b), np.linalg.solve(spd, b), rtol=1e-10)
        for tri in (np.triu(a) + 3 * np.eye(n), np.tril(a) + 3 * np.eye(n), np.diag(np.arange(1.0, n + 1))):
            assert_same_solution(ops.left_divide(tri, b), np.linalg.solve(tri, b))
        np.testing.assert_allclose(ops.right_divide(b.T, a), np.linalg.solve(a.T, b).T, rtol=1e-9)
    c = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    np.testing.assert_allclose(ops.left_divide(c, np.ones((4, 1))), np.linalg.solve(c, np.ones((4, 1))), rtol=1e-12)
    out = []
    m = Interpreter(out=out.append)
    m.execute("x = [1 2; 2 4] \\ [1; 1];")
    assert "Matrix is singular to working precision." in "".join(out)
    assert np.isinf(m.get_var("x")).all()
    out.clear()
    m.execute("y = magic(4) \\ ones(4, 1); [msg, id] = lastwarn;")
    text = "".join(out)
    assert "close to singular or badly scaled" in text and "RCOND = " in text
    assert "".join(m.get_var("id")[0]) == "MATLAB:nearlySingularMatrix"
    assert np.isfinite(m.get_var("y")).all()
    out.clear()
    m.execute("z = [2 1; 1 3] \\ [3; 5]; t = [2 0; 1 4] \\ [2; 9]; e = zeros(0, 0) \\ zeros(0, 3);")
    assert out == []
    np.testing.assert_allclose(m.get_var("z"), [[0.8], [1.4]], rtol=1e-14)
    np.testing.assert_allclose(m.get_var("t"), [[1.0], [2.0]], rtol=1e-14)
    assert m.get_var("e").shape == (0, 3)


def test_solve_falls_back_without_lapack(monkeypatch):
    monkeypatch.setattr(fastnum, "_LAPACK", False)
    a = np.array([[4.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(ops.left_divide(a, np.array([[1.0], [2.0]])), [[0.1], [0.6]], rtol=1e-14)


def test_scipy_lapack_backend_agrees():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(8)
    be = fastnum._ScipyLapack()
    for n in (2, 6, 30):
        a = rng.standard_normal((n, n)) + n * np.eye(n)
        b = rng.standard_normal((n, 2))
        x, info, rcond = be.general(a, b)
        assert info == 0 and 0 < rcond <= 1
        np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-10)
        x, rcond = be.cholesky(a @ a.T, b)
        np.testing.assert_allclose(x, np.linalg.solve(a @ a.T, b), rtol=1e-8)
        x, rcond = be.triangular(np.tril(a), b, True)
        np.testing.assert_allclose(x, np.linalg.solve(np.tril(a), b), rtol=1e-10)
    assert be.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]), np.ones((2, 1))) is None
    x, info, rcond = be.general(np.array([[1.0, 2.0], [2.0, 4.0]]), np.ones((2, 1)))
    assert info > 0 and rcond == 0.0


def test_eigenvalues_and_singular_values_only():
    rng = np.random.default_rng(4)
    interp = Interpreter()
    for n in (2, 3, 6, 25, 60):
        a = rng.standard_normal((n, n))
        for mat in (a, a + a.T, np.triu(a), rng.integers(-3, 4, (n, n)).astype(float)):
            fast, slow = on_and_off(lambda: call(interp, "eig", [mat], 1))
            assert fast.shape == slow.shape == (n, 1)
            assert fast.dtype.kind == slow.dtype.kind
            order = lambda w: np.sort_complex(np.asarray(w, dtype=complex).ravel())  # noqa: E731
            np.testing.assert_allclose(order(fast), order(slow), rtol=1e-8, atol=1e-9 * np.abs(slow).max())
            fast, slow = on_and_off(lambda: call(interp, "svd", [mat], 1))
            np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-13)
    # two outputs, errors and other classes go the general way
    a = rng.standard_normal((4, 4))
    same(lambda: call(interp, "eig", [a], 2))
    same(lambda: call(interp, "eig", [np.array([[1.0, NAN], [2.0, 3.0]])], 1))
    same(lambda: call(interp, "eig", [np.ones((2, 3))], 1))
    same(lambda: call(interp, "eig", [np.array([[5.0]])], 1))
    same(lambda: call(interp, "svd", [np.array([[3.0, 4.0]])], 1))


def test_fft_backends_agree():
    rng = np.random.default_rng(6)
    interp = Interpreter()
    small = rng.standard_normal((1, 64))
    same(lambda: call(interp, "fft", [small], 1))  # small transforms stay with numpy.fft
    same(lambda: call(interp, "ifft", [small + 1j], 1))
    n = 1 << 15
    for x in (rng.standard_normal((1, n)), rng.standard_normal((n, 1)) + 1j * rng.standard_normal((n, 1)),
              rng.standard_normal((n // 8, 8)), rng.standard_normal((1, n - 3))):
        for name in ("fft", "ifft"):
            fast, slow = on_and_off(lambda: BUILTINS[name](interp, [x], 1))
            assert fast.shape == slow.shape and fast.dtype == slow.dtype
            np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-10 * np.abs(slow).max())
    padded = rng.standard_normal((1, n))
    fast, slow = on_and_off(lambda: call(interp, "fft", [padded, np.array([[2.0 * n]])], 1))
    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-8)
    single = rng.standard_normal((1, n)).astype(np.float32)
    fast, slow = on_and_off(lambda: call(interp, "ifft", [single], 1))
    assert fast.dtype == slow.dtype


def test_environment_switch():
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = "from mylab import Interpreter, fastnum; m = Interpreter(); m.execute('x = sum([1 2 3]) + 1;'); " \
           "print(fastnum.ENABLED, float(m.get_var('x')[0, 0]))"
    for value, expected in (("0", "False 7.0"), ("1", "True 7.0")):
        env = dict(os.environ, MYLAB_FASTNUM=value, PYTHONPATH=root)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=root)
        assert out.stdout.strip() == expected, out.stderr
