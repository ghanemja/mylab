"""Shared helpers for the differential fuzz tests of the fast paths (see test_fuzz_fastpaths*.py).

A program is run twice, each time in a fresh interpreter: once with every fast path on (loop
compiler, numexpr fusion, numeric fast layer) and once with all of them off.  The two outcomes
must be identical: the same variables with the same classes, shapes and bits (NaN payload and sign
aside; the sign of zero counts), the same printed output, and the same error message and
identifier.
"""

from __future__ import annotations

import math
import os
import random

import numpy as np

from mylab import Interpreter, MatlabError, fastnum, fuse
from mylab.values import Cell, FunctionHandle, MString, StringArray, Struct, values_equal

# MYLAB_FUZZ_N=2000 runs that many cases per test instead of the quick default.
FUZZ_N = int(os.environ.get("MYLAB_FUZZ_N", "0") or 0)


def count(default: int) -> int:
    return FUZZ_N if FUZZ_N > 0 else default


# -- running ---------------------------------------------------------------------------------------

def run(code: str, fast: bool, path: list[str] | None = None):
    """Run code in a fresh interpreter; return (workspace, output, error)."""
    saved = fastnum.ENABLED
    fastnum.set_enabled(fast)
    out: list[str] = []
    interp = Interpreter(out=out.append, path=list(path or []))
    interp.jit_enabled = fast
    interp.fuse_enabled = fast and fuse.AVAILABLE
    interp.warnings_enabled = False
    error = None
    try:
        interp.execute(code)
    except MatlabError as e:
        error = ("error", e.identifier, e.message)
    except RecursionError:
        error = ("python", "RecursionError")
    except Exception as e:  # noqa: BLE001 - an internal error is a difference too
        error = ("python", f"{type(e).__name__}: {e}")
    finally:
        fastnum.set_enabled(saved)
    return interp.base.vars, "".join(out), error


def canon(v):
    """A hashable, exactly comparable picture of a value."""
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            return ("obj", type(v).__name__, v.shape, tuple(canon(x) for x in v.ravel(order="F")))
        a = np.ascontiguousarray(v)
        if a.dtype.kind in "fc":
            f = a.view(a.real.dtype).copy() if a.dtype.kind == "c" else a.copy()
            f[np.isnan(f)] = np.nan  # NaN payload and sign do not count; the sign of zero does
            data = f.tobytes()
        else:
            data = a.tobytes()
        return ("nd", type(v).__name__, a.dtype.str, a.shape, data)
    if isinstance(v, Cell):
        return ("cell", v.data.shape, tuple(canon(x) for x in v.data.ravel(order="F")))
    if isinstance(v, Struct):
        return ("struct", v.data.shape, tuple(v.fields),
                tuple(tuple((k, canon(d.get(k))) for k in v.fields) for d in v.data.ravel(order="F")))
    if isinstance(v, MString):
        return ("mstring", str(v))
    if isinstance(v, StringArray):
        return ("strings", canon(v.data))
    if isinstance(v, FunctionHandle):
        return ("handle", str(getattr(v, "name", "")), str(getattr(v, "source", "")))
    return ("other", type(v).__name__, v)


def _describe(v) -> str:
    if isinstance(v, np.ndarray) and v.dtype != object:
        with np.printoptions(precision=17, threshold=40):
            return f"{v.dtype}{list(v.shape)} {v!r}"
    return repr(v)


def differences(code: str, path: list[str] | None = None) -> list[str]:
    """Run code both ways; list what differs (empty when the outcomes agree)."""
    ws_on, out_on, err_on = run(code, True, path)
    ws_off, out_off, err_off = run(code, False, path)
    diffs = []
    if err_on != err_off:
        diffs.append(f"error: fast={err_on!r} plain={err_off!r}")
    if out_on != out_off:
        diffs.append(f"output: fast={out_on!r} plain={out_off!r}")
    if sorted(ws_on) != sorted(ws_off):
        diffs.append(f"variables: fast={sorted(ws_on)} plain={sorted(ws_off)}")
    for name in sorted(set(ws_on) & set(ws_off)):
        if name == "err":  # the caught exception object: its message is copied into e<k>
            continue
        a, b = ws_on[name], ws_off[name]
        try:
            same = canon(a) == canon(b)
        except Exception:  # noqa: BLE001 - a value canon() does not know: compare it the usual way
            same = type(a) is type(b) and values_equal(a, b, nan_equal=True)
        if not same:
            diffs.append(f"{name}: fast={_describe(a)} plain={_describe(b)}")
    return diffs


def check_all(programs, path: list[str] | None = None, limit: int = 5) -> None:
    """Fail with the first few programs whose outcomes differ."""
    bad = []
    for code in programs:
        d = differences(code, path)
        if d:
            bad.append(code + "\n    " + "\n    ".join(d))
            if len(bad) >= limit:
                break
    assert not bad, "fast paths differ from the interpreter:\n\n" + "\n\n".join(bad)


# -- values ----------------------------------------------------------------------------------------

DENORMAL = 5e-324
SPECIAL = [math.nan, math.inf, -math.inf, -0.0, 0.0, DENORMAL, 2.2250738585072014e-308 / 4, 1e308, -1e308,
           1e-300, -1e-300, 1.0, -1.0, 0.5, -0.5, 2.5, -2.5, 3.0, 7.0, 2.0 ** 53, -(2.0 ** 53) + 1, 1e15 + 0.5,
           0.1, 255.5, -128.5, 300.0, 65535.0, 4294967296.0]
INT_CLASSES = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
CLASSES = ["double", "single", "logical", "char", "complex", "complex0"] + INT_CLASSES


def num(x: float) -> str:
    """A MATLAB literal for a double."""
    if x != x:
        return "NaN"
    if x == math.inf:
        return "Inf"
    if x == -math.inf:
        return "-Inf"
    if x == 0:
        return "-0" if math.copysign(1.0, x) < 0 else "0"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


def rand_double(rng: random.Random, special: float = 0.5) -> float:
    if rng.random() < special:
        return rng.choice(SPECIAL)
    r = rng.random()
    if r < 0.4:
        return float(rng.randint(-9, 9))
    if r < 0.8:
        return round(rng.uniform(-10, 10), rng.randint(0, 3))
    return rng.uniform(-1, 1) * 10.0 ** rng.randint(-20, 20)


SHAPES = [(1, 1), (1, 1), (1, 1), (1, 2), (1, 3), (3, 1), (1, 8), (8, 1), (2, 2), (2, 3), (3, 2), (2, 2, 2),
          (0, 0), (1, 0), (0, 3), (0, 1)]


def array_literal(values: list[str], shape) -> str:
    n = int(np.prod(shape))
    if n == 0:
        return f"zeros({', '.join(str(s) for s in shape)})"
    body = "[" + ", ".join(values) + "]"
    if len(shape) == 2 and shape[0] == 1:
        return body
    return f"reshape({body}, {', '.join(str(s) for s in shape)})"


def value(rng: random.Random, cls: str | None = None, shape=None, special: float = 0.5) -> str:
    """A MATLAB expression for a random value of the given class and shape."""
    cls = cls or rng.choice(CLASSES)
    shape = shape or rng.choice(SHAPES)
    n = int(np.prod(shape))
    if cls == "char":
        vals = [str(rng.choice([32, 48, 65, 97, 122, 0, 255])) for _ in range(n)]
        return f"char({array_literal(vals, shape)})"
    if cls == "logical":
        vals = [str(rng.randint(0, 1)) for _ in range(n)]
        return f"logical({array_literal(vals, shape)})"
    if cls in ("complex", "complex0"):
        re = [num(rand_double(rng, special)) for _ in range(n)]
        im = ["0" if cls == "complex0" else num(rand_double(rng, special)) for _ in range(n)]
        return f"complex({array_literal(re, shape)}, {array_literal(im, shape)})"
    vals = [num(rand_double(rng, special)) for _ in range(n)]
    lit = array_literal(vals, shape)
    return lit if cls == "double" else f"{cls}({lit})"


def wrap(stmts: list[str]) -> str:
    """Run each statement on its own: r<k> holds its result, e<k> its error."""
    parts = []
    for k, s in enumerate(stmts):
        parts.append(f"try, r{k} = {s}; catch err, e{k} = {{err.identifier, err.message}}; end")
    return "\n".join(parts)
